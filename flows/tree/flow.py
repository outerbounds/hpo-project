from metaflow import step, pypi, current, card, Parameter, kubernetes, Config, trigger
from metaflow.cards import Markdown

# FIXME(Eddie): This database util is generalizable, nothing Optuna-specific.
from metaflow.plugins.optuna import get_db_url
from obproject import ProjectFlow
import os


@trigger(event="tree_model_hpo")
class TreeModelHpoFlow(ProjectFlow):

    objective_function_file = Parameter(
        "objective_function_file",
        default="objective_fn.py",
        help="Relative path to the objective function file",
    )
    n_trials_override = Parameter("n_trials_override", default="", help="Total number of trials for HPO")
    # trials_per_task = Parameter(
    #     "trials_per_task", default=1, help="Number of trials per task"
    # )
    optuna_app_name = Parameter(
        "optuna_app_name", default="hpo-dashboard", help="Name of the Optuna app"
    )
    override_study_name = Parameter(
        "override_study_name", default="", help="Name of the Optuna study"
    )
    config = Config(
        "config", default=os.path.join(os.path.dirname(__file__), "config.json")
    )

    def resolve_direction(self):
        if self.config.get("direction", None) == "maximize":
            return "maximize"
        elif self.config.get("direction", None) == "minimize":
            return "minimize"
        elif isinstance(self.config.get("directions", None), list):
            assert (
                len(self.config.get("directions", ["maximize", "minimize"])) == 2
            ), "Direction of multi-objective optimization must be a list of max two values."
            return self.config.get("directions")
        else:
            docs_ref = "https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html"
            raise ValueError(f"Invalid direction: {self.config.get('direction', 'maximize')}. See {docs_ref} for more information.")

    @pypi(
        python=config.environment.get("python"),
        packages=config.environment.get("packages"),
    )
    @step
    def start(self):
        import optuna
        from sklearn.datasets import load_iris

        # Load and register training data
        self.data = load_iris()
        self.X, self.y = self.data["data"], self.data["target"]
        self.prj.register_data(
            "iris_dataset",
            "data",
            annotations={
                "n_samples": len(self.X),
                "n_features": self.X.shape[1],
                "n_classes": len(set(self.y)),
            },
            tags={"source": "sklearn", "dataset": "iris"},
            description="Iris flower dataset for classification"
        )

        # self.batches = [self.trials_per_task] * (self.n_trials // self.trials_per_task)
        # Handle Argo "null" string issue: if override is empty or "null", use config value
        if self.n_trials_override and self.n_trials_override != "null":
            self.n_trials = int(self.n_trials_override)
        else:
            self.n_trials = self.config.get("n_trials", 10)
        self.workers = list(range(min(self.n_trials, self.config.get("max_parallelism", 10))))
        override_study_name = (
            None
            if self.override_study_name == "" or self.override_study_name == "null"
            else self.override_study_name
        )
        self.study_name = (
            override_study_name
            or self.config.get("study_name", None)
            or "/".join(current.pathspec.split("/")[:2])
        )
        print(f"Study name: {self.study_name}")

        self.study_kwargs = dict(
            storage=get_db_url(self.optuna_app_name),
            study_name=self.study_name,
            load_if_exists=True,
            direction=self.resolve_direction(),
            # Sampler and pruner may be exposed as end user config option, depending on use case.
            sampler=optuna.samplers.GPSampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        _ = optuna.create_study(**self.study_kwargs)
        self.next(self.run_trial, foreach="workers")

    @kubernetes(compute_pool=config.compute_pool)
    @pypi(
        python=config.environment.get("python"),
        packages=config.environment.get("packages"),
    )
    @step
    def run_trial(self):
        import optuna
        from utils import load_objective_function

        self.objective = load_objective_function(self.objective_function_file)
        study = optuna.create_study(**self.study_kwargs)
        study.optimize(self.objective, n_trials=1)
        self.trial_result = {"completed": True}

        # Decision time.
            # 1. num trials to complete
            # 2. num trials already completed
            # 3. num trials in progress
        # if 1 > (2+3) then "yes" else "no"
        # In resume study case, are 1 and 2 inclusive of trials from another flow?
            # Query global state from Optuna DB
        successful_count = sum(
            1 for t in study.trials 
            if t.state in (optuna.trial.TrialState.COMPLETE, 
                        optuna.trial.TrialState.PRUNED)
        )
        self.continue_study = "yes" if successful_count < self.n_trials else "no"
        self.next({"yes": self.run_trial, "no": self.post_trial}, condition="continue_study")

    @kubernetes(compute_pool=config.compute_pool)
    @pypi(python=config.environment.get("python"), packages=config.environment.get("packages"))
    @step
    def post_trial(self):
        self.next(self.join)

    @card(id="best_model")
    @kubernetes(compute_pool=config.compute_pool)
    @pypi(
        python=config.environment.get("python"),
        packages=config.environment.get("packages"),
    )
    @step
    def join(self, inputs):
        from utils import load_study
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np

        self.study_name = inputs[0].study_name
        study = load_study(self.study_name, self.optuna_app_name)
        self.results = study.trials_dataframe()
        self.best_params = study.best_params

        # Train final model with best params
        X = inputs[0].X
        y = inputs[0].y
        self.best_model = ExtraTreeClassifier(**self.best_params)
        self.best_model.fit(X, y)
        self.best_score = study.best_value

        # Register HPO results as data asset
        self.prj.register_data(
            "hpo_results",
            "results",
            annotations={
                "n_trials": len(self.results),
                "best_score": float(self.best_score),
                "study_name": self.study_name,
            },
            tags={"optimization": "optuna", "model_type": "tree"},
            description="Hyperparameter optimization results"
        )

        # Register best model
        self.prj.register_model(
            "iris_classifier",
            "best_model",
            annotations={
                "accuracy": float(self.best_score),
                "model_type": "ExtraTreeClassifier",
                "max_depth": int(self.best_params.get("max_depth", 0)),
                "criterion": self.best_params.get("criterion", ""),
                "n_trials": len(self.results),
            },
            tags={"framework": "sklearn", "optimizer": "optuna"},
            description="Best iris classifier from HPO"
        )

        current.card["best_model"].append(
            Markdown(f"### Best model parameters: {self.best_params}")
        )
        current.card["best_model"].append(
            Markdown(f"### Best cross-validation score: {self.best_score:.4f}")
        )
        self.next(self.end)

    @pypi(
        python=config.environment.get("python"),
        packages=config.environment.get("packages"),
    )
    @step
    def end(self):
        print("Flow completed")


if __name__ == "__main__":
    TreeModelHpoFlow()

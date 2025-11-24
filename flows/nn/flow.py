from metaflow import step, pypi, current, card, Parameter, kubernetes, Config, trigger
from metaflow.cards import Markdown

# FIXME(Eddie): This database util is generalizable, nothing Optuna-specific.
from metaflow.plugins.optuna import get_db_url
from obproject import ProjectFlow
import os


@trigger(event="nn_hpo")
class NeuralNetHpoFlow(ProjectFlow):

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
                len(self.config.get("directions", None)) == 2
            ), "Direction of multi-objective optimization direction list must be a list of two values."
            return self.config.get("directions")
        else:
            docs_ref = "https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection"
            raise ValueError(
                f"Invalid direction: {self.config.get('direction', None)}. See {docs_ref} for more information."
            )

    @pypi(
        python=config.environment.get("python"),
        packages=config.environment.get("packages"),
    )
    @step
    def start(self):
        import optuna
        from utils import cache_mnist

        # Load and register training data
        # Note: MNIST is downloaded in cache_mnist(), registering metadata here
        self.prj.register_external_data(
            "mnist_dataset",
            blobs=[],  # MNIST downloaded from torchvision.datasets
            kind="external",
            annotations={
                "source": "torchvision.datasets.MNIST",
                "train_samples": "60000",
                "test_samples": "10000",
                "n_classes": "10",
            },
            tags={"source": "torchvision", "dataset": "mnist"},
            description="MNIST handwritten digit dataset for neural network training"
        )

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

        ### Cache Dataset
        # cache_mnist()
        # NOTE: Turned off for now as each worker downloads the small DS, see utils.py for details.

        ### Configure Study
        self.study_kwargs = dict(
            storage=get_db_url(self.optuna_app_name),
            study_name=self.study_name,
            load_if_exists=True,
            # Sampler and pruner may be exposed as end user config option, depending on use case.
            sampler=optuna.samplers.GPSampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        diresult = self.resolve_direction()
        if isinstance(diresult, list):
            self.study_kwargs["directions"] = diresult
        else:
            self.study_kwargs["direction"] = diresult
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

        # Each task runs exactly 1 trial for 1:1 task-trial mapping
        study = optuna.create_study(**self.study_kwargs)
        study.optimize(self.objective, n_trials=1)
        self.trial_result = {"completed": True}

        # Query global state from Optuna DB to decide whether to continue
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

        self.study_name = inputs[0].study_name
        study = load_study(self.study_name, self.optuna_app_name)
        self.results = study.trials_dataframe()

        ## Best params is an over-loaded question with multi-objective task.
        # self.best_params = study.best_params
        self.best_params = study.best_trials
        current.card["best_model"].append(
            Markdown(f"### Best model parameters: {self.best_params}")
        )

        # Register HPO results data asset
        self.prj.register_data(
            "nn_hpo_results",
            "results",
            annotations={
                "n_trials": len(self.results),
                "study_name": self.study_name,
            },
            tags={"optimizer": "optuna", "sampler": "gp"},
            description="Neural network HPO trial results from Optuna"
        )

        # Register best model (multi-objective, so registering trial metadata)
        # Note: For multi-objective, best_trials is a list of Pareto-optimal solutions
        n_best = len(self.best_params) if isinstance(self.best_params, list) else 1
        self.prj.register_external_model(
            "nn_classifier",
            blobs=[],  # Model weights would be saved separately
            kind="external",
            annotations={
                "model_type": "PyTorch_Neural_Network",
                "n_pareto_solutions": n_best,
                "study_name": self.study_name,
                "optimization": "multi_objective",
            },
            tags={"framework": "pytorch", "optimizer": "optuna"},
            description="Best neural network classifier from multi-objective HPO"
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
    NeuralNetHpoFlow()

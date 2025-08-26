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
    n_trials = Parameter("n_trials", default=10, help="Total number of trials for HPO")
    trials_per_task = Parameter(
        "trials_per_task", default=1, help="Number of trials per task"
    )
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
            assert len(self.config.get("directions", None)) == 2, "Direction of multi-objective optimization direction list must be a list of two values."
            return self.config.get("directions")
        else:
            docs_ref = "https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection"
            raise ValueError(f"Invalid direction: {self.config.get('direction', None)}. See {docs_ref} for more information.")

    @pypi(python=config.environment.get("python"), packages=config.environment.get("packages"))
    @step
    def start(self):
        import optuna
        from utils import cache_mnist

        num_full_batches = self.n_trials // self.trials_per_task
        remaining_trials = self.n_trials % self.trials_per_task        
        self.batches = [self.trials_per_task] * num_full_batches
        if remaining_trials > 0:
            self.batches.append(remaining_trials)
        
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
        cache_mnist()

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
        self.next(self.run_trial, foreach="batches")

    @kubernetes(compute_pool=config.compute_pool)
    @pypi(python=config.environment.get("python"), packages=config.environment.get("packages"))
    @step
    def run_trial(self):
        import optuna
        from utils import load_objective_function

        self.objective = load_objective_function(self.objective_function_file)
        # Each task runs a batch of trials, may be just 1 for cleanest (trial, model, task) accounting.
        # TODO: Refactor as conditionals/looping DAGs lands. Revisit this.
        batch_size = self.input
        study = optuna.create_study(**self.study_kwargs)
        study.optimize(self.objective, n_trials=batch_size)
        self.trial_result = {"batch_size": batch_size, "completed": True}
        self.next(self.join)

    @card(id="best_model")
    @kubernetes(compute_pool=config.compute_pool)
    @pypi(python=config.environment.get("python"), packages=config.environment.get("packages"))
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
        self.next(self.end)

    @pypi(python=config.environment.get("python"), packages=config.environment.get("packages"))
    @step
    def end(self):
        print("Flow completed")


if __name__ == "__main__":
    NeuralNetHpoFlow()

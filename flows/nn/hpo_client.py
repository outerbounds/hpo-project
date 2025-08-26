import os
import argparse
import asyncio
from metaflow import Runner, Deployer
from metaflow.integrations import ArgoEvent

from utils import (
    load_config,
    extract_flow_name,
    extract_project_name,
    load_objective_function,
    workflow_template_exists,
)

config = load_config()


class HyperparameterTuner(object):
    """
    This class uses the Metaflow Runner and ArgoEvent to launch an HPO flow in one of three modes:
        1. Blocking mode: Run the flow in the current process, block until the flow is complete. Useful for interactive/notebook development and fast-running HPO jobs launched from other flows.
        2. Async mode: Run the flow in the background, do not block the current process on its completion. Useful for launching long-running flows when you don't want to deploy to the production orchestrator.
        3. Trigger mode: Publish an ArgoEvent to trigger the flow. Useful for triggering flows from other systems. Useful for launching long-running flows deployed to the production orchestrator.
    In the steady-state, the ideal UX is to focus design on how/when to use mode 3, aside from leveraging mode 1 for iterative development and debugging.

    All modes require the following parameters:
        - objective_function_file: The path to the Optuna objective function file.
        - override_study_name: The name of the study to use.
        - optuna_app_name: The name of the Optuna app to use.

    Each mode's calling method also support the following optional parameters:
        - override_compute_pool: The name of the compute pool to use. When used, this overrides the compute pool specified in the config.
        - n_trials: The number of trials to run. When used, this overrides the number of trials specified in the config.
        - trials_per_task: The number of trials to run per Metaflow task. When used, this overrides the number of trials per task specified in the config.
    """

    def __init__(
        self,
        objective_function_file,
        override_study_name=None,
        optuna_app_name=None,
    ):
        self.objective_function_file = objective_function_file
        self.override_study_name = override_study_name or config.get("study_name", None)
        self.optuna_app_name = optuna_app_name or config["optuna_app_name"]

        # Fail fast if core elements are not properly defined.
        # Add project-specific constraints below these checks.

        if not os.path.exists(objective_function_file):
            raise FileNotFoundError(
                f"Objective function file {objective_function_file} does not exist"
            )
        try:
            _ = load_objective_function(self.objective_function_file)
        except Exception as e:
            raise Exception(
                f"Failed to load objective function from {self.objective_function_file}: {e}"
            )

        docs_ref = "https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection"
        if config.get("direction", None) and config.get("directions", None):
            raise ValueError(
                "Either direction or directions must be provided in the Optuna config, not both. See {docs_ref} for more information."
            )
        if not config.get("direction", None) and not config.get("directions", []):
            raise ValueError(
                "Either direction or directions must be provided in the config. See {docs_ref} for more information."
            )

    def run_blocking(
        self,
        override_compute_pool=None,
        override_study_name=None,
        n_trials=None,
        trials_per_task=None,
    ):
        run_id = None
        with Runner(
            flow_file=config["flow_file"],
            environment=config["environment_builder"],
            decospecs=(
                [f"kubernetes:compute_pool={override_compute_pool}"]
                if override_compute_pool
                else None
            ),
        ).run(
            objective_function_file=self.objective_function_file,
            override_study_name=override_study_name or self.override_study_name,
            optuna_app_name=self.optuna_app_name,
            n_trials=n_trials or config["n_trials"],
            trials_per_task=trials_per_task or config["trials_per_task"],
        ) as running:
            run_id = running.run.id
        return run_id

    async def run_async(
        self,
        override_compute_pool=None,
        override_study_name=None,
        n_trials=None,
        trials_per_task=None,
    ):
        run_id = None
        with await Runner(
            flow_file=config["flow_file"],
            environment=config["environment_builder"],
            decospecs=(
                [f"kubernetes:compute_pool={override_compute_pool}"]
                if override_compute_pool
                else None
            ),
        ).async_run(
            objective_function_file=self.objective_function_file,
            override_study_name=override_study_name or self.override_study_name,
            optuna_app_name=self.optuna_app_name,
            n_trials=n_trials or config["n_trials"],
            trials_per_task=trials_per_task or config["trials_per_task"],
        ) as running:
            run_id = running.run.id
        return run_id

    def deploy(self, override_compute_pool=None):
        deployer = Deployer(
            flow_file=config["flow_file"],
            environment=config["environment_builder"],
            decospecs=(
                [f"kubernetes:compute_pool={override_compute_pool}"]
                if override_compute_pool
                else None
            ),
        )
        deployed_flow = deployer.argo_workflows().create()
        return deployed_flow

    def trigger(
        self,
        override_study_name=None,
        override_compute_pool=None,
        n_trials=None,
        trials_per_task=None,
        deploy_if_not_exists=True,
        namespace=None,
    ):

        flow_name = extract_flow_name(config["flow_file"])
        project_name = extract_project_name()
        exists = workflow_template_exists(flow_name, project_name, namespace)

        if not exists and deploy_if_not_exists:
            _ = self.deploy(override_compute_pool)
            print(
                f"\nDeployed workflow template {flow_name} in project {project_name}.\n"
            )
        elif not exists and not deploy_if_not_exists:
            raise ValueError(
                f"Workflow template {flow_name} does not exist in project {project_name}."
                f"Set deploy_if_not_exists=True to deploy the workflow template automatically."
            )

        event_publication = ArgoEvent(name="nn_hpo").publish(
            payload=dict(
                objective_function_file=self.objective_function_file,
                override_study_name=override_study_name or self.override_study_name,
                optuna_app_name=self.optuna_app_name,
                n_trials=n_trials or config["n_trials"],
                trials_per_task=trials_per_task or config["trials_per_task"],
            ),
            ignore_errors=False,
        )
        print(f"\nTriggered flow {flow_name} in project {project_name}.\n")
        return event_publication


async def main():
    "Wrapper to call for asyncio.run() invocation."

    tuner = HyperparameterTuner(
        objective_function_file=config["objective_function_file"],
        override_study_name=config.get("study_name", None),
        optuna_app_name=config["optuna_app_name"],
    )
    run_id = await tuner.run_async(
        override_compute_pool=config["compute_pool"],
        n_trials=config["n_trials"],
        trials_per_task=config["trials_per_task"],
    )
    print(f"Run ID: {run_id}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--namespace", "-n", type=str, required=False, default=None)
    parser.add_argument(
        "--resume-study", "-r", action="store_true", required=False, default=False
    )
    args = parser.parse_args()
    mode = args.mode
    override_study_name = args.resume_study or config.get("study_name", None)

    ###############################
    ### Mode 1 - Blocking usage ###
    ###############################
    if mode == 1:
        tuner = HyperparameterTuner(
            objective_function_file=config["objective_function_file"],
            override_study_name=override_study_name,
            optuna_app_name=config["optuna_app_name"],
        )
        tuner.run_blocking(
            override_compute_pool=config["compute_pool"],
            n_trials=config["n_trials"],
            trials_per_task=config["trials_per_task"],
        )

    ############################
    ### Mode 2 - Async usage ###
    ############################
    if mode == 2:
        asyncio.run(main())

    ##############################
    ### Mode 3 - Trigger usage ###
    ##############################
    if mode == 3:
        tuner = HyperparameterTuner(
            objective_function_file=config["objective_function_file"],
            override_study_name=override_study_name,
            optuna_app_name=config["optuna_app_name"],
        )
        tuner.trigger(
            n_trials=config["n_trials"],
            trials_per_task=config["trials_per_task"],
            namespace=args.namespace,
        )

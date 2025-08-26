import importlib
import os
import re
import json
import optuna
import tomllib
from metaflow import get_namespace
from metaflow.plugins.optuna import get_db_url
from metaflow.metaflow_config import KUBERNETES_NAMESPACE
from metaflow.plugins.argo.argo_client import ArgoClient


def load_objective_function(objective_function_file):
    spec = importlib.util.spec_from_file_location(
        "objective_module", objective_function_file
    )
    objective_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objective_module)

    if not hasattr(objective_module, "objective"):
        raise ValueError('Objective function must have an "objective" attribute')
    return objective_module.objective


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_study(study_name, optuna_app_name):
    """
    This function can only be used from a
    FlowSpec task, or from an Outerbounds workstation.
    In other words, it should not be able to authenticate
    against the database from your local machine.
    Because of this, `flow.py` should store any results
    needed for downstream analysis as FlowSpec artifacts.
    """
    study = optuna.create_study(
        storage=get_db_url(optuna_app_name), study_name=study_name, load_if_exists=True
    )
    return study


def get_argo_workflow_template(flow_name, project_name, namespace=None):
    project_name = project_name.lower().replace("-", "").replace("_", "")
    ns = namespace if namespace else get_namespace().replace(":", ".").replace("@", "")
    workflow_template_name = f"{project_name}.{ns}.{flow_name.lower()}"
    client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
    return client.get_workflow_template(workflow_template_name)


def extract_flow_name(flow_file, sanitize=True):
    FLOW_RE = re.compile(
        r"^class\s+(\w+)\s*\([^)]*?\bProjectFlow\b[^)]*\)\s*:", re.MULTILINE
    )
    with open(flow_file, "r") as f:
        match = FLOW_RE.search(f.read())
        if match:
            if sanitize:
                return match.group(1).lower().replace("-", "").replace("_", "")
            else:
                return match.group(1)
        else:
            raise ValueError(f"Could not extract flow name from {flow_file}")


def extract_project_name(sanitize=True):
    cfg_rel_path = os.path.join(os.path.dirname(__file__), "../../obproject.toml")
    with open(cfg_rel_path, "rb") as f:
        project_name = tomllib.load(f)["project"]
    if sanitize:
        return project_name.lower().replace("-", "").replace("_", "")
    else:
        return project_name


def workflow_template_exists(flow_name, project_name, namespace=None):
    template = get_argo_workflow_template(
        flow_name=flow_name, project_name=project_name, namespace=namespace
    )
    return template is not None

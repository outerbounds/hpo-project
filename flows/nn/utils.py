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
from torchvision import datasets
from torchvision import transforms
from metaflow.metaflow_config import DATATOOLS_S3ROOT
from metaflow import S3
import torch

REMOTE_BUCKET_KEY = "sample-datasets/mnist"
LOCAL_DIR = os.getcwd()


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


<<<<<<< HEAD
### NOTE: Temporarily deprecate AWS-specific implementation in favor of simpler cloud-agnostic/dumb approach.
# def cache_mnist(batch_size=128):
#     """
#     This function is used to cache the FashionMNIST dataset in the Outerbounds S3 data plane.
#     """
#     cache_key = os.path.join(DATATOOLS_S3ROOT, REMOTE_BUCKET_KEY)
#     with S3(s3root=cache_key) as s3:
#         cache_contents = s3.list_paths()
#         if not cache_contents:
#             print("Downloading FashionMNIST dataset...")
#             _ = torch.utils.data.DataLoader(
#                 datasets.FashionMNIST(
#                     LOCAL_DIR,
#                     train=True,
#                     download=True,
#                     transform=transforms.ToTensor(),
#                 ),
#                 batch_size=batch_size,
#                 shuffle=True,
#             )
#             _ = torch.utils.data.DataLoader(
#                 datasets.FashionMNIST(
#                     LOCAL_DIR, train=False, transform=transforms.ToTensor()
#                 ),
#                 batch_size=batch_size,
#                 shuffle=True,
#             )

#             data_dir = os.path.join(LOCAL_DIR, "FashionMNIST")
#             s3_key_paths_pairs = []
#             for root, _, files in os.walk(data_dir):
#                 for file in files:
#                     s3_key_paths_pairs.append(
#                         (
#                             os.path.join(os.path.relpath(root, LOCAL_DIR), file),
#                             os.path.join(root, file),
#                         )
#                     )
#             s3.put_files(s3_key_paths_pairs, overwrite=True)

#         return cache_contents

def cache_mnist(batch_size=128):
    """Download MNIST dataset locally (each worker downloads independently)"""
    print("Downloading FashionMNIST dataset locally...")
    _ = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            LOCAL_DIR, train=True, download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size, shuffle=True,
    )
    _ = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            LOCAL_DIR, train=False, transform=transforms.ToTensor()
        ),
        batch_size=batch_size, shuffle=True,
    )
    return []

def get_mnist(batch_size=128):
    """
    Download FashionMNIST locally. Simple and works across all clouds (AWS/Azure/GCP).
    Each worker downloads independently (~30MB dataset, acceptable overhead).
    """
    # PyTorch will download automatically if not present
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            LOCAL_DIR, 
            train=True, 
            download=True,  # Download if not cached
            transform=transforms.ToTensor()
        ),
=======
def cache_mnist(batch_size=128):
    """
    This function is used to cache the FashionMNIST dataset in the Outerbounds S3 data plane.
    """
    cache_key = os.path.join(DATATOOLS_S3ROOT, REMOTE_BUCKET_KEY)
    with S3(s3root=cache_key) as s3:
        cache_contents = s3.list_paths()
        if not cache_contents:
            print("Downloading FashionMNIST dataset...")
            _ = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                    LOCAL_DIR,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            _ = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                    LOCAL_DIR, train=False, transform=transforms.ToTensor()
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            data_dir = os.path.join(LOCAL_DIR, "FashionMNIST")
            s3_key_paths_pairs = []
            for root, _, files in os.walk(data_dir):
                for file in files:
                    s3_key_paths_pairs.append(
                        (
                            os.path.join(os.path.relpath(root, LOCAL_DIR), file),
                            os.path.join(root, file),
                        )
                    )
            s3.put_files(s3_key_paths_pairs, overwrite=True)

        return cache_contents


def get_mnist(batch_size=128):
    # Create the full directory structure in one call
    data_dir = os.path.join(LOCAL_DIR, "FashionMNIST", "raw")
    os.makedirs(data_dir, exist_ok=True)

    cache_key = os.path.join(DATATOOLS_S3ROOT, REMOTE_BUCKET_KEY)
    with S3(s3root=cache_key) as s3:
        for obj in s3.get_all():
            os.rename(obj.path, os.path.join(LOCAL_DIR, obj.key))

    # Return the data loaders as expected by the objective function
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(LOCAL_DIR, train=True, transform=transforms.ToTensor()),
>>>>>>> e568fcf (resolve readme)
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
<<<<<<< HEAD
        datasets.FashionMNIST(
            LOCAL_DIR, 
            train=False, 
            download=True,  # Download if not cached
            transform=transforms.ToTensor()
        ),
=======
        datasets.FashionMNIST(LOCAL_DIR, train=False, transform=transforms.ToTensor()),
>>>>>>> e568fcf (resolve readme)
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader
<<<<<<< HEAD

    # OLD S3 CACHING VERSION (AWS-only, doesn't work on Azure):
    # # Create the full directory structure in one call
    # data_dir = os.path.join(LOCAL_DIR, "FashionMNIST", "raw")
    # os.makedirs(data_dir, exist_ok=True)
    #
    # cache_key = os.path.join(DATATOOLS_S3ROOT, REMOTE_BUCKET_KEY)
    # with S3(s3root=cache_key) as s3:
    #     for obj in s3.get_all():
    #         os.rename(obj.path, os.path.join(LOCAL_DIR, obj.key))
    #
    # # Return the data loaders as expected by the objective function
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(LOCAL_DIR, train=True, transform=transforms.ToTensor()),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(LOCAL_DIR, train=False, transform=transforms.ToTensor()),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    #
    # return train_loader, valid_loader
=======
>>>>>>> e568fcf (resolve readme)

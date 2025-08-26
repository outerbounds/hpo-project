from optuna_dashboard import wsgi
from optuna.storages import RDBStorage
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import json


def get_mf_token():
    with open(os.path.join(os.environ["METAFLOW_HOME"], "config.json"), "r") as f:
        conf = json.loads(f.read())
        return conf["METAFLOW_SERVICE_AUTH_KEY"]


def generate_db_url():
    # This function mirrors the metaflow.plugins.optuna.get_db_url function used in the /flows.
    # FIXME: Reuse/consolidate existing function.
    mf_token = get_mf_token()
    return f"postgresql://userspace_default:{mf_token}@localhost:5432/userspace_default?sslmode=disable"


STORAGE_URL = generate_db_url()

base_app = wsgi(RDBStorage(STORAGE_URL))
app = ProxyFix(base_app, x_for=1, x_proto=1, x_host=1, x_port=1)

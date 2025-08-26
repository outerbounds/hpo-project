## Hyperparameter Optimization Project
<img style="display: block; max-width: 100%; height: auto; margin: auto;" alt="system" src="https://github.com/user-attachments/assets/f9ca6892-879c-46b1-9696-bf691530de1c" />

This repository shows you how to run a hyperparameter optimization (HPO) system as an Outerbounds project.
This `README.md` will explain why you'd want to connect these concepts, and will show you how to launch HPO jobs for:
- classical ML models
- deep learning models
- end-to-end system tuning

If you have never deployed an Outerbounds project, please read the [Outerbounds documentation](https://docs.outerbounds.com/) before continuing.

## Quick start

Change the `platform` in [`obproject.toml`](./obproject.toml) to match your Outerbounds deployment. 

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv init
uv add outerbounds optuna numpy pandas "psycopg[binary]>=3.2.0" scikit-learn torch torchvision
```

Ensure you've run your `outerbounds configure ...` command.
Then, run flows!

```bash
cd flows/tree
uv run python flow.py --environment=fast-bakery run --with kubernetes
```

> For more information about the containerization technology used in this project, see [Fast Bakery: Automatic Containerization](https://outerbounds.com/blog/containerize-with-fast-bakery).

## How to customize this repository for your use cases

### Make a new directory under `/flows`
To begin, copy the structure in `/flows/nn` or `/flows/tree`:
- `config.json` contains system and hyperparameter config options.
- `flow.py` defines the workflow structure. This should change little across use cases.
- `objective_fn.py` this is the key piece of the puzzle for a new use case. See examples at https://github.com/optuna/optuna-examples/tree/main.
- `utils.py` contains small project-specific helpers.
- `interactive.ipynb` is a starter notebook for running and analyzing hyperparameter tuning runs in a REPL.
- Symlink to `obproject.toml` at the root of the repository. 

If desired, you can directly modify one of these sub-directories.

### Define and evolve your own objective function

The key aspect of customization is about defining the objective function. 
Check out the examples and reach out for assistance if you do not know how to parameterize your task as a tunable optimization problem. 
From there, determine the dependencies needed for running the objective function
and update the `config.json` values accordingly, most notable the Python packages 
section which `flow.py` will use when building consistent environments across 
compute backends.

## Advanced

### Detailed set up

#### Deploy the Optuna dashboard application

The Outerbounds app that will run your Optuna dashboard is defined in [`./deployments/optuna-dashboard/config.yml`](./deployments/optuna-dashboard/config.yml).
When you push to the main branch of this repository, the `obproject-deployer` will create the application in your Outerbounds project branch.
If you'd like to manually deploy the application:

```bash
cd deployments/optuna-dashboard
uv run outerbounds app deploy --config-file config.yml
```

#### Local/workstation dependencies

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).

From your laptop or Outerbounds workstation run:
```bash
uv init
uv add outerbounds optuna numpy pandas "psycopg[binary]>=3.2.0" scikit-learn torch torchvision
```

Configure Outerbounds token. Ask in Slack if not sure.

#### Pick a sub-project
```bash
cd flows/tree
# cd flows/nn
```

#### Setting configs
Before running or deploying the workflows, investigate the relationship between the flow and the `config.json` file.

As long as you haven't changed anything when deploying the application hosting the Optuna dashboard, you do not need to change anything in that file, 
but it is useful to be familiar with these contents and the way the configuration files are interacting with Metaflow code. 

#### Run flows
There are two demos implemented within this project base in `flows/tree` and `flows/nn`.
Each workflow template defines:
- a `flow.py` containing a `FlowSpec`, 
- a single `config.json` to set system variables and hyperparameter configurations,
- an `hpo_client.py` containing entrypoints to run and trigger the flow, 
- notebooks showing how to run and analyze results of hyperparameter tuning runs, and
- the templates show how to define a modular, fully customizable objective function.

For the rest of this section, we'll use the `flows/nn` template, as everything else is the same as for `flows/tree`.

```bash
cd flows/nn
```

#### Use Metaflow directly
```bash
uv run python flow.py --environment=fast-bakery run --with kubernetes
uv run python flow.py --environment=fast-bakery argo-workflows create/trigger
```

#### Use `hpo_client`
The examples also include a convenience wrapper around the workflows in the `hpo_client.py`. 
You can use this for:
- running HPO jobs from notebooks, CLI, or other Metaflow flows, or
- as an example for creating your own experiment entrypoint abstractions.

```bash
uv run python hpo_client.py -m 1 # blocking
uv run python hpo_client.py -m 2 # async
uv run python hpo_client.py -m 3 # trigger deployed flow
```

There are three client modes:
1. Blocking - `python hpo_client.py -m 1`
2. Async - `python hpo_client.py -m 2`
3. Trigger - `python hpo_client.py -m 3` 
    - Trigger option also works with a parameter `--namespace/-n`, which determines the namespace within which this code path checks for already-deployed flows.

### Optuna 101
This system is an integration between [Optuna](https://optuna.org/), a feature-rich and open-source hyperparameter optimization framework, and Outerbounds. Using it leverages functionality built-into your Outerbounds deployment to run a persistent relational database that tasks and applications can communicate with. The Optuna dashboard is run as an Outerbounds app, enabling sophisticated analysis of hyperparameter tuning runs.  

The implementation wraps the standard Optuna interface, aiming to balance two goals:
1. Provide full expressiveness and compatibility with open-source Optuna features.
2. Provide an opinionated and streamlined interface for launching HPO studies as Metaflow flows. 

#### The objective function
Typically, Optuna programs are developed in Python scripts. 
An objective function returns 1 or 2 values. 
It's argument is a [`trial`](https://optuna.readthedocs.io/en/stable/reference/trial.html), 
representing a single execution of the objective function; in other words, a sample drawn from the hyperparameter search space.

```python
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    f1 = x**2 + y
    f2 = -((x - 2) ** 2 + y)
    return f1, f2
```

The key task of the user who wishes to use the `from outerbounds.hpo import HPORunner` abstraction this project affords is to determine:
1. How to define the objective function? 
2. What data, model, and code does the objective function depend on?
3. How many trials do you want to run per study?

With answers to these questions, you'll be ready to adapt your objective functions as demonstrated in the example [`flows/`](./flows/) and call the `HPORunner` interface to automate HPO workflows.

#### Note on search spaces
Notice that with Optuna, the user imperatively defines the hyperparameter space in how the `trial` object is used within the `objective` function.
The number of variables for which we have `trial.suggest_*` defines the dimensionality of the search space. 
Be judicious with adding parameters. Many algorithms, especially bayesian optimization suffers performance degradation when there are many more than 5-10 parameters being tuned simultaneously.

[Read more](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#configurations).

#### Studies, samplers, and pruners
To optimize the hyperparameters, we create a study.
Optuna implements many optimization algorithm families, called as [`optuna.samplers`](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html). These include grid, random, tree-structure parzen estimators, evolutionary (CMA-ES, NSGA-II), Gaussian processes, Quasi Monte Carlo methods, and more.

For example, if you wanted to purely random sample - no learning throughout the study - the hyperparameter space 10 times, you'd run:
```python
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())   
study.optimize(objective, n_trials=10)
```

Sometimes it is desirable to early stop unpromising trials. The mechanism for doing this in Optuna is called [`optuna.pruners`](https://optuna.readthedocs.io/en/stable/reference/pruners.html), which uses intermediate objective function state variables of previous trials to determine a boolean representing whether the trial should be pruned.

#### Resuming studies
To resume a study, simply pass in the name of the previous study. 
If leveraging the Metaflow versioning scheme which uses the Metaflow Run pathspec as the study name - in other words not overriding the study name via configs or CLI - then
you can set this value in the config and resume the study. You can also override in the command line using the `hpo_client`'s `--resume-study/-r` option:

```bash
python hpo_client.py -m 1 -r TreeModelHpoFlow/argo-hposystem.prod.treemodelhpoflow-7ntvz
```

## TODO
- Benchmark gRPC vs. pure RDB scaling thresholds. When is it worth it to do gRPC? How hard is that to implement? How do costs scale in each mode? 

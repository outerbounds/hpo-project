"""
Original source: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""
<<<<<<< HEAD

=======
>>>>>>> e568fcf (resolve readme)
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from utils import get_mnist


DEVICE = torch.device("cpu")
# BATCHSIZE = 128
# CLASSES = 10
# EPOCHS = 10
# N_TRAIN_EXAMPLES = BATCHSIZE * 30
# N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_classes = 10
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, n_classes))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def objective(trial):
    import time

<<<<<<< HEAD
    t0 = time.time()
=======
    t0 = time.time() 
>>>>>>> e568fcf (resolve readme)

    model = define_model(trial).to(DEVICE)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size = 128  # trial.suggest_int("batch_size", 32, 128)
    n_train_examples = 30
    n_valid_examples = 10
    epochs = 25

    train_loader, valid_loader = get_mnist(batch_size=batch_size)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
<<<<<<< HEAD
        total = 0
=======
>>>>>>> e568fcf (resolve readme)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
<<<<<<< HEAD
                total += target.size(0)

        accuracy = correct / total

        ### Not supported with multi-objective.
        # trial.report(accuracy, epoch)

        ### Not supported with multi-objective.
=======

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        ### Not supported with multi-objective :(
        # trial.report(accuracy, epoch)

        ### Not supported with multi-objective :(
>>>>>>> e568fcf (resolve readme)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

<<<<<<< HEAD
    return (time.time() - t0, accuracy)  # cost  # profit
=======
    return (
        time.time() - t0, # cost
        accuracy          # profit
    )
>>>>>>> e568fcf (resolve readme)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

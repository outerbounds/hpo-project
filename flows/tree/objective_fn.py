from sklearn.datasets import load_iris
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def objective(trial):
    data = load_iris()
    X, y = data["data"], data["target"]
    max_depth = trial.suggest_int("max_depth", 2, 16)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    model = ExtraTreeClassifier(max_depth=max_depth, criterion=criterion)
    return np.mean(cross_val_score(model, X, y, cv=5))

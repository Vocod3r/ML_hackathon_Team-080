"""
AM321 - Model zoo.

Turns the entries in configs/models.yaml into ready-to-train sklearn models.
Every model returned here supports .fit(X, y) and .predict_proba(X).
"""
from pathlib import Path

import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]          # repo root
CONFIG_PATH = ROOT / "configs" / "models.yaml"
FEATURE_COLS_PATH = ROOT / "data" / "processed" / "feature_cols.csv"

# name in yaml -> (sklearn class, does it need feature scaling?)
#   Distance/gradient-based models need scaling: otherwise a feature measured
#   in metres (0-100) drowns out one measured as a cosine (-1..1).
#   Tree models split on thresholds, so scaling makes no difference to them.
REGISTRY = {
    "logreg":        (LogisticRegression,     True),
    "gaussian_nb":   (GaussianNB,             False),
    "knn":           (KNeighborsClassifier,   True),
    "random_forest": (RandomForestClassifier, False),
    "lightgbm":      (LGBMClassifier,         False),
    "svm_rbf":       (LinearSVC,              True),
    "mlp":           (MLPClassifier,          True),
}

# models whose constructor accepts random_state (for reproducibility)
SEEDED = {"logreg", "random_forest", "lightgbm", "svm_rbf", "mlp"}


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_feature_cols(path=FEATURE_COLS_PATH):
    """The file has NO header row - header=None keeps the first feature ('d')."""
    return pd.read_csv(path, header=None).iloc[:, 0].tolist()


def build_model(name, params=None, random_state=42):
    """Create one untrained model. `params` overrides the yaml (used later by Optuna)."""
    cls, needs_scaling = REGISTRY[name]
    params = dict(params or {})
    if "hidden_layer_sizes" in params:                    # yaml gives a list, sklearn wants a tuple
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    if name in SEEDED:
        params["random_state"] = random_state
    if name == "svm_rbf":
        # 1) Nystroem maps each row into a 300-dim space where a LINEAR boundary
        #    behaves like an RBF-kernel boundary in the original space (fast approximation).
        # 2) LinearSVC finds the max-margin boundary in that space.
        # 3) An SVM outputs a distance from the boundary, not a probability. Calibration
        #    fits a sigmoid on held-out folds to turn that distance into a probability.
        kernel = Nystroem(kernel="rbf", gamma=params.pop("gamma"),
                          n_components=params.pop("n_components"), random_state=random_state)
        svm = CalibratedClassifierCV(cls(**params), method="sigmoid", cv=3, ensemble=False)
        return make_pipeline(StandardScaler(), kernel, svm)
    model = cls(**params)
    return make_pipeline(StandardScaler(), model) if needs_scaling else model


def get_all_models(config=None):
    """Return {name: (untrained model, family, max_train_rows, params)} for every model in the yaml."""
    config = config or load_config()
    seed = config.get("random_state", 42)
    out = {}
    for name, spec in config["models"].items():
        out[name] = (
            build_model(name, spec.get("params"), seed),
            spec["family"],
            spec.get("max_train_rows"),
            spec.get("params", {}),
        )
    return out
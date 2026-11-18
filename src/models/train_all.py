"""
AM321 - Train, cross-validate and compare every model in configs/models.yaml.

Run from the repo root:
    python -m src.models.train_all                       # all models, full data
    python -m src.models.train_all --models logreg lightgbm
    python -m src.models.train_all --sample              # quick test on the 5-trip sample

For each model it does:
  1. 5-fold GroupKFold CV on the 80 training trips (folds split by trip_id)
  2. refit on all 80 training trips
  3. evaluate on the 10 student-testing trips
and writes one JSON per model to results/models/runs/, then rebuilds
results/models/comparison.csv from all JSONs (so models can be run separately).
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import KEYS, evaluate_all, macro_f1, top1_accuracy
from src.models.model_zoo import ROOT, get_all_models, load_config, load_feature_cols

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "models"
KEEP_COLS = KEYS + ["arc_id", "label", "trusted"]      # saved next to the probabilities


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(sample):
    """Return (train_df, test_df, test_partition_name)."""
    if sample:
        # Quick mode: 4 sample trips for training, the 5th as a stand-in test set.
        df = pd.read_parquet(PROCESSED / "sample_candidates.parquet")
        last = sorted(df.trip_id.unique())[-1]
        return df[df.trip_id != last].copy(), df[df.trip_id == last].copy(), "train"
    train = pd.read_parquet(PROCESSED / "train_candidates.parquet")
    test = pd.read_parquet(PROCESSED / "student_test_candidates.parquet")
    return train, test, "student_test"


def load_arcs(trip_ids, partition):
    """{trip_id: (n_arcs, 2) array of [from_node, to_node]} for route continuity. Empty if raw data missing."""
    arcs = {}
    for tid in trip_ids:
        f = RAW / partition / tid / f"{tid}.arcs"
        if f.exists():
            arcs[tid] = np.loadtxt(f, dtype=np.int64, ndmin=2)
    return arcs if len(arcs) == len(trip_ids) else None


def subsample(df, max_rows, seed):
    """Random row subsample for slow models (None = keep everything)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def proba(model, X):
    """Probability that each candidate is the true arc (column 1 = class 1)."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Training histories (only for models that really train epoch-by-epoch)
# ---------------------------------------------------------------------------
def fit_with_history(name, model, X, y, X_val, y_val):
    """Fit the model. Return a training history dict for LightGBM / MLP, else None."""
    if name == "lightgbm":
        # Record log loss after every boosting round on train and student-test.
        # Recording only - no early stopping, so student-test does not influence the fit.
        model.fit(X, y, eval_set=[(X, y), (X_val, y_val)],
                  eval_names=["train", "student_test"], eval_metric="binary_logloss")
        r = model.evals_result_
        return {"x_label": "boosting round",
                "train_loss": r["train"]["binary_logloss"],
                "val_loss": r["student_test"]["binary_logloss"]}
    model.fit(X, y)
    if name == "mlp":
        mlp = model.steps[-1][1]                     # the MLP inside the scaler pipeline
        return {"x_label": "epoch",
                "train_loss": list(map(float, mlp.loss_curve_)),
                # internal 10% validation split; sklearn records ACCURACY here, not loss
                "val_accuracy": list(map(float, mlp.validation_scores_ or []))}
    return None


# ---------------------------------------------------------------------------
# One model, end to end
# ---------------------------------------------------------------------------
def run_model(name, model_factory, family, max_rows, params, train, test, features,
              test_part, seed, n_folds, arcs_test):
    print(f"\n=== {name} ({family}) ===", flush=True)
    fit_rows = train[train["trusted"]]          # untrusted labels are not used for fitting

    # ---- 1. Cross-validation, folds grouped by trip -----------------------
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.full(len(train), np.nan)           # out-of-fold probability for every training row
    folds = []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["trip_id"])):
        tr = subsample(train.iloc[tr_idx].loc[lambda d: d["trusted"]], max_rows, seed)
        va = train.iloc[va_idx]
        m = model_factory()
        t0 = time.time()
        m.fit(tr[features], tr["label"])
        p_va = proba(m, va[features])
        oof[va_idx] = p_va
        va_eval = va.assign(prob=p_va)
        folds.append({"model": name, "fold": k,
                      "n_val_trips": int(va["trip_id"].nunique()),
                      "train_macro_f1": macro_f1(tr["label"], proba(m, tr[features])),
                      "val_macro_f1": macro_f1(va["label"], p_va),
                      "val_top1_accuracy": top1_accuracy(va_eval),
                      "seconds": round(time.time() - t0, 2)})
        print(f"  fold {k}: val F1 {folds[-1]['val_macro_f1']:.4f}  "
              f"top1 {folds[-1]['val_top1_accuracy']:.4f}  ({folds[-1]['seconds']}s)", flush=True)
    folds = pd.DataFrame(folds)

    # ---- 2. Refit on all training trips ----------------------------------
    fit_df = subsample(fit_rows, max_rows, seed)
    model = model_factory()
    t0 = time.time()
    history = fit_with_history(name, model, fit_df[features], fit_df["label"],
                               test[features], test["label"])
    train_time = time.time() - t0

    # ---- 3. Student-test evaluation ---------------------------------------
    t0 = time.time()
    p_test = proba(model, test[features])
    infer_time = time.time() - t0
    n_obs = test[KEYS].drop_duplicates().shape[0]
    test_eval = test.assign(prob=p_test)
    metrics = evaluate_all(test_eval, arcs_by_trip=arcs_test)
    metrics["top1_accuracy_trusted"] = top1_accuracy(test_eval[test_eval["trusted"]])
    train_f1 = macro_f1(fit_df["label"], proba(model, fit_df[features]))

    # ---- 4. Save -----------------------------------------------------------
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    test[KEEP_COLS].assign(prob=p_test).to_parquet(OUT / "probs" / f"{name}_{test_part}.parquet", index=False)
    train[KEEP_COLS].assign(prob=oof).to_parquet(OUT / "probs" / f"{name}_train_oof.parquet", index=False)
    if history is not None:
        (OUT / "histories").mkdir(parents=True, exist_ok=True)
        (OUT / "histories" / f"{name}.json").write_text(json.dumps(history))

    run = {
        "model": name, "family": family, "config": params,
        "max_train_rows": max_rows, "n_fit_rows": int(len(fit_df)), "seed": seed,
        "cv_macro_f1_mean": float(folds["val_macro_f1"].mean()),
        "cv_macro_f1_std": float(folds["val_macro_f1"].std()),
        "cv_top1_mean": float(folds["val_top1_accuracy"].mean()),
        "cv_train_macro_f1_mean": float(folds["train_macro_f1"].mean()),
        "train_macro_f1_full": train_f1,
        **{f"test_{k}": v for k, v in metrics.items()},
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(infer_time, 3),
        "inference_ms_per_obs": round(1000 * infer_time / n_obs, 4),
        "test_partition": test_part,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "runs" / f"{name}.json").write_text(json.dumps(run, indent=2, default=str))
    folds.to_csv(OUT / "runs" / f"{name}_folds.csv", index=False)
    print(f"  -> CV F1 {run['cv_macro_f1_mean']:.4f}±{run['cv_macro_f1_std']:.4f} | "
          f"test top1 {metrics['top1_accuracy']:.4f} F1 {metrics['macro_f1']:.4f} "
          f"logloss {metrics['log_loss']:.4f} | train {train_time:.1f}s", flush=True)
    return run


def rebuild_comparison():
    """Merge every runs/*.json into one table, sorted by CV macro F1."""
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    df = pd.DataFrame(rows)
    df["config"] = df["config"].apply(json.dumps)
    df = df.sort_values("cv_macro_f1_mean", ascending=False)
    df.to_csv(OUT / "comparison.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="subset of model names (default: all in models.yaml)")
    ap.add_argument("--sample", action="store_true", help="quick run on the 5-trip sample")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg.get("random_state", 42)
    features = load_feature_cols()
    train, test, test_part = load_data(args.sample)
    n_folds = min(args.folds, train["trip_id"].nunique())
    arcs_test = load_arcs(sorted(test["trip_id"].unique()), test_part)
    print(f"train {train.shape} ({train.trip_id.nunique()} trips) | "
          f"test {test.shape} ({test.trip_id.nunique()} trips) | {len(features)} features | "
          f"route continuity {'on' if arcs_test else 'OFF (data/raw missing)'}", flush=True)

    zoo = get_all_models(cfg)
    names = args.models or list(zoo)
    for name in names:
        _, family, max_rows, params = zoo[name]
        factory = lambda n=name: get_all_models(cfg)[n][0]      # fresh untrained model each call
        run_model(name, factory, family, max_rows, params, train, test, features,
                  test_part, seed, n_folds, arcs_test)

    (OUT / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "sample_mode": args.sample}, indent=2))
    table = rebuild_comparison()
    cols = ["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "test_top1_accuracy",
            "test_macro_f1", "test_log_loss", "train_time_s", "inference_ms_per_obs"]
    print("\n" + table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
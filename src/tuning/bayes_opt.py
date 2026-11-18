"""Run Optuna TPE with trip-grouped CV and optionally evaluate a held-out table."""
import argparse
from importlib import import_module
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _model(cfg, params):
    if "team_model" in cfg:
        from src.models.model_zoo import build_model

        return build_model(cfg["team_model"], cfg["estimator_params"] | params, cfg["model_seed"])
    module, name = cfg["estimator"].rsplit(".", 1)
    estimator = getattr(import_module(module), name)(**(cfg["estimator_params"] | params))
    if cfg.get("scale_features", False):
        return make_pipeline(StandardScaler(), estimator)
    return estimator


def _with_team_model(cfg, selection_path, models_path):
    from src.models.model_zoo import load_config

    selection = pd.read_csv(selection_path)
    if len(selection) != 1:
        raise ValueError("Expected one model in AM321's selected_model.csv")
    name = selection["model"].iloc[0]
    config = load_config(models_path)
    spec = config["models"][name]
    return cfg | {
        "team_model": name,
        "model_seed": config.get("random_state", 42),
        "estimator_params": json.loads(selection["config"].iloc[0]),
        "max_train_rows": spec.get("max_train_rows"),
        "trusted_only": True,
        "search": cfg["team_search"][name],
    }


def _fit_indices(train, indices, cfg):
    rows = train.iloc[indices]
    if cfg.get("trusted_only", False):
        rows = rows[rows["trusted"]]
    cap = cfg.get("max_train_rows")
    if cap is not None and len(rows) > cap:
        rows = rows.sample(n=cap, random_state=cfg["model_seed"])
    return rows.index.to_numpy()


def _suggest(trial, search):
    params = {}
    for name, spec in search.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec.get("type") == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def _positive_prob(model, features):
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def tune(train, features, cfg, out_dir, student_test=None, artifacts_dir=None):
    """Tune candidate log loss and save group-held-out training probabilities."""
    train = train.reset_index(drop=True)
    X, y = train[features], train["label"]
    splits = list(GroupKFold(n_splits=cfg["folds"]).split(X, y, groups=train["trip_id"]))
    folds = [(_fit_indices(train, fit, cfg), valid) for fit, valid in splits]
    full_fit = _fit_indices(train, np.arange(len(train)), cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cross_validate(params, trial=None):
        probabilities = np.empty(len(train))
        fold_records = []
        for k, (fit, valid) in enumerate(folds):
            model = _model(cfg, params).fit(X.iloc[fit], y.iloc[fit])
            probs = _positive_prob(model, X.iloc[valid])
            probabilities[valid] = probs
            fold_records.append({
                "macro_f1": f1_score(y.iloc[valid], probs >= 0.5, average="macro"),
                "log_loss": log_loss(y.iloc[valid], probs, labels=[0, 1]),
            })
            if trial is not None:
                # Report the running mean so the median pruner can stop clearly bad trials early.
                trial.report(float(np.mean([r["log_loss"] for r in fold_records])), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        records = pd.DataFrame(fold_records)
        summary = records.mean()
        summary["log_loss_std"] = records["log_loss"].std()
        summary["macro_f1_std"] = records["macro_f1"].std()
        return summary, probabilities

    def objective(trial):
        start = perf_counter()
        metrics, _ = cross_validate(_suggest(trial, cfg["search"]), trial)
        trial.set_user_attr("cv_macro_f1", float(metrics["macro_f1"]))
        trial.set_user_attr("cv_macro_f1_std", float(metrics["macro_f1_std"]))
        trial.set_user_attr("cv_log_loss_std", float(metrics["log_loss_std"]))
        trial.set_user_attr("seconds", round(perf_counter() - start, 1))
        print(f"trial {trial.number}: log loss {metrics['log_loss']:.5f} "
              f"F1 {metrics['macro_f1']:.4f} ({perf_counter() - start:.0f}s)", flush=True)
        return float(metrics["log_loss"])

    pruner = optuna.pruners.MedianPruner(n_startup_trials=cfg.get("pruner_startup_trials", 10), n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner,
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study_start = perf_counter()
    study.optimize(objective, n_trials=cfg["trials"])
    cfg = cfg | {"study_seconds": round(perf_counter() - study_start, 1),
                 "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
                 "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)}
    (out_dir / "study_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    (out_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2) + "\n", encoding="utf-8")

    comparison = []
    tuned_model = None
    for name, params in (("untuned", {}), ("tuned", study.best_params)):
        cv, probabilities = cross_validate(params)
        record = {"model": name, "params": json.dumps(cfg["estimator_params"] | params),
                  "cv_macro_f1": cv["macro_f1"], "cv_macro_f1_std": cv["macro_f1_std"],
                  "cv_log_loss": cv["log_loss"], "cv_log_loss_std": cv["log_loss_std"]}
        start = perf_counter()
        model = _model(cfg, params).fit(X.iloc[full_fit], y.iloc[full_fit])
        record["train_time_s"] = perf_counter() - start
        if student_test is not None:
            start = perf_counter()
            test_probs = _positive_prob(model, student_test[features])
            record["test_inference_s"] = perf_counter() - start
            scored = student_test.reset_index(drop=True).assign(probability=test_probs)
            record["test_inference_ms_per_obs"] = 1000 * record["test_inference_s"] / scored.groupby(["trip_id", "obs_idx"]).ngroups
            best = scored.groupby(["trip_id", "obs_idx"])["probability"].idxmax()
            record["test_top1_accuracy"] = scored.loc[best, "label"].mean()
            record["test_macro_f1"] = f1_score(scored["label"], test_probs >= 0.5, average="macro")
            record["test_log_loss"] = log_loss(scored["label"], test_probs, labels=[0, 1])
            if name == "tuned":
                scored.to_parquet(out_dir / "student_test_probabilities.parquet", index=False)
        if name == "tuned":
            tuned_model = model
            train.assign(probability=probabilities).to_parquet(out_dir / "oof_probabilities.parquet", index=False)
        comparison.append(record)

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(tuned_model, artifacts_dir / "final_model.joblib")
        (artifacts_dir / "feature_list.json").write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    results = pd.DataFrame(comparison)
    results.to_csv(out_dir / "comparison.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_candidates.parquet"))
    parser.add_argument("--student-test", type=Path)
    parser.add_argument("--features", type=Path, default=Path("data/processed/feature_cols.csv"))
    parser.add_argument("--selected-model", type=Path, help="AM321 selected_model.csv")
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/hpo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/hpo"))
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.selected_model:
        cfg = _with_team_model(cfg, args.selected_model, args.models_config)
    features = pd.read_csv(args.features, header=None)[0].tolist()
    train = pd.read_parquet(args.train)
    test = pd.read_parquet(args.student_test) if args.student_test else None
    # Limit BLAS only: capping OpenMP too would force LightGBM onto a single core.
    with threadpool_limits(limits=1, user_api="blas"):
        results = tune(train, features, cfg, args.output, test, args.artifacts_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

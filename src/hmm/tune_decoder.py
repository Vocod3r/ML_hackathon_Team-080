"""Tune HMM decoder parameters with Optuna (TPE) on out-of-fold TRAIN probabilities.

Every training row was scored by a model that never saw its trip (GroupKFold), so the
classifier probabilities the decoder sees here look like those on unseen trips. The tuning
objective is pointwise accuracy of the decoded path on a fixed subset of training trips.
student_test is NOT used to pick parameters; evaluate_hmm.py reports it afterwards.

    python -m src.hmm.tune_decoder --probs results/hpo/oof_probabilities.parquet \
        --probability-column probability --output results/hmm/decoder_tuning
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
import yaml

from src.data.io import load_trip
from src.hmm.viterbi import decode
from src.pipeline.predict_trip import build_trip_graph

SPACE = {
    "beta_m": ("float", 5.0, 120.0, True),
    "disconnected_penalty": ("float", 2.0, 40.0, True),
    "emission_weight": ("float", 0.2, 3.0, True),
    "uturn_penalty": ("float", 0.0, 30.0, False),
    "max_gap_s": ("float", 10.0, 120.0, True),
}


def pick_trips(cand, n_trips, seed, max_obs):
    """Deterministic subset of trips, skipping very long ones so a trial stays cheap."""
    sizes = cand.groupby("trip_id")["obs_idx"].nunique()
    eligible = sizes[sizes <= max_obs].index.to_numpy()
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=min(n_trips, len(eligible)), replace=False).tolist())


def load_trips(cand, raw_root, trip_ids):
    out = {}
    for tid in trip_ids:
        trip = load_trip(raw_root / tid, load_route=False)
        rows = cand[cand["trip_id"] == tid].sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
        out[tid] = (rows, build_trip_graph(trip))
    return out


def accuracy(trips, cfg, prob_col):
    correct = total = 0
    for rows, graph in trips.values():
        ref = rows.drop_duplicates("obs_idx").set_index("obs_idx")["ref_arc"]
        pred = decode(rows, rows[prob_col].to_numpy(), graph, cfg).set_index("obs_idx")["arc_id"]
        correct += int((pred == ref.reindex(pred.index)).sum())
        total += len(pred)
    return correct / total


def baseline_accuracy(trips, prob_col):
    correct = total = 0
    for rows, _ in trips.values():
        ref = rows.drop_duplicates("obs_idx").set_index("obs_idx")["ref_arc"]
        best = rows.loc[rows.groupby("obs_idx")[prob_col].idxmax()].set_index("obs_idx")["arc_id"]
        correct += int((best == ref).sum())
        total += len(best)
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=Path("data/processed/train_candidates.parquet"))
    ap.add_argument("--probs", type=Path, required=True, help="parquet with trip_id, obs_idx, arc_id and a probability column")
    ap.add_argument("--probability-column", default="prob")
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--base-config", type=Path, default=Path("configs/hmm.yaml"))
    ap.add_argument("--n-trips", type=int, default=24)
    ap.add_argument("--max-obs", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--output", type=Path, default=Path("results/hmm/decoder_tuning"))
    ap.add_argument("--warm-start", type=Path, help="decoder yaml (e.g. an earlier tuned config) enqueued as the first trial")
    ap.add_argument("--write-config", type=Path, help="write the tuned config here (e.g. configs/hmm_tuned.yaml)")
    args = ap.parse_args()

    col = args.probability_column
    cand = pd.read_parquet(args.candidates)
    probs = pd.read_parquet(args.probs)[["trip_id", "obs_idx", "arc_id", col]]
    cand = cand.merge(probs, on=["trip_id", "obs_idx", "arc_id"], how="inner")
    trip_ids = pick_trips(cand, args.n_trips, args.seed, args.max_obs)
    trips = load_trips(cand, args.raw_root, trip_ids)
    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    clf_acc = baseline_accuracy(trips, col)
    start_acc = accuracy(trips, base_cfg | {"emission_weight": 1.0, "uturn_penalty": 0.0}, col)
    print(f"{len(trip_ids)} trips | classifier-only {clf_acc:.4f} | HMM with starting config {start_acc:.4f}", flush=True)

    def objective(trial):
        cfg = base_cfg | {
            name: trial.suggest_float(name, lo, hi, log=log) for name, (_, lo, hi, log) in SPACE.items()
        }
        t0 = perf_counter()
        acc = accuracy(trips, cfg, col)
        print(f"trial {trial.number}: acc {acc:.4f} ({perf_counter() - t0:.0f}s) "
              + " ".join(f"{k}={trial.params[k]:.3g}" for k in SPACE), flush=True)
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.enqueue_trial({"beta_m": base_cfg["beta_m"], "disconnected_penalty": base_cfg["disconnected_penalty"],
                         "emission_weight": 1.0, "uturn_penalty": 0.0, "max_gap_s": base_cfg["max_gap_s"]})
    if args.warm_start:
        warm = yaml.safe_load(args.warm_start.read_text(encoding="utf-8"))
        study.enqueue_trial({k: warm[k] for k in SPACE})
    study.optimize(objective, n_trials=args.trials)
    study.trials_dataframe().to_csv(args.output / "trials.csv", index=False)

    best = base_cfg | study.best_params
    summary = {"tuning_trips": trip_ids, "classifier_only_accuracy": clf_acc,
               "starting_config_accuracy": start_acc, "tuned_accuracy": study.best_value,
               "best_params": study.best_params, "search_space": {k: v[1:] for k, v in SPACE.items()},
               "seed": args.seed, "trials": args.trials, "probability_source": str(args.probs)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.write_config:
        args.write_config.write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

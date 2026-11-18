"""Final HMM comparison on the 10 student_test trips and export of artifacts/hmm_config.json.

Compares three decoders on the SAME classifier probabilities:
  classifier-only (argmax per observation), HMM with the original config, HMM with the tuned config.
Decoder parameters come from tune_decoder.py (train out-of-fold probabilities only), so student_test
is used here only for reporting.

    python -m src.hmm.final_eval --probs results/hpo/student_test_probabilities.parquet \
        --probability-column probability --tuned-config results/hmm/decoder_tuning/hmm_tuned.yaml
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.hmm.evaluate_hmm import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=Path("data/processed/student_test_candidates.parquet"))
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--probability-column", default="probability")
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/student_test"))
    ap.add_argument("--original-config", type=Path, default=Path("configs/hmm.yaml"))
    ap.add_argument("--tuned-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/hmm"))
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    col = args.probability_column
    cand = pd.read_parquet(args.candidates).drop(columns=[col], errors="ignore")
    probs = pd.read_parquet(args.probs)[["trip_id", "obs_idx", "arc_id", col]]
    cand = cand.merge(probs, on=["trip_id", "obs_idx", "arc_id"], how="inner")

    original = yaml.safe_load(args.original_config.read_text(encoding="utf-8"))
    tuned = yaml.safe_load(args.tuned_config.read_text(encoding="utf-8"))
    frames = []
    for name, cfg in (("hmm_original", original), ("hmm_tuned", tuned)):
        df = evaluate(cand, args.raw_root, cfg, col)
        if not frames:
            frames.append(df[df["method"] == "baseline"].assign(method="classifier_only"))
        frames.append(df[df["method"] == "hmm"].assign(method=name))
    table = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "student_test_comparison.csv", index=False)

    wide = table.pivot(index="trip_id", columns="method", values="pointwise_accuracy")
    wide["tuned_minus_classifier"] = wide["hmm_tuned"] - wide["classifier_only"]
    wide["hmm_hurt"] = wide["tuned_minus_classifier"] < 0
    wide.to_csv(args.output / "student_test_accuracy_by_trip.csv")

    weights = table.groupby("method")["observations"].sum()
    pooled = {m: float((g["pointwise_accuracy"] * g["observations"]).sum() / weights[m])
              for m, g in table.groupby("method")}
    pooled_cont = {m: float((g["continuity"] * g["observations"]).sum() / weights[m])
                   for m, g in table.groupby("method")}
    summary = {
        "pooled_pointwise_accuracy": pooled,
        "pooled_continuity": pooled_cont,
        "mean_route_arc_f1": table.groupby("method")["route_arc_f1"].mean().to_dict(),
        "reverse_direction_errors": table.groupby("method")["reverse_direction_errors"].sum().astype(int).to_dict(),
        "trips_where_tuned_hmm_hurt": wide.index[wide["hmm_hurt"]].tolist(),
        "trips_where_tuned_hmm_helped": wide.index[wide["tuned_minus_classifier"] > 0].tolist(),
    }
    (args.output / "student_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "hmm_config.json").write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    print(wide.round(4).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Compare row-wise distance or classifier scores with HMM decoding on sample trips."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.data.graph import build_graph
from src.data.io import load_trip
from src.hmm.viterbi import decode


def _metrics(prediction, references, arcs):
    predicted = prediction.set_index("obs_idx")["arc_id"]
    actual = references.reindex(predicted.index)
    correct = predicted.eq(actual)
    arc_ids = predicted.to_numpy()
    consecutive = predicted.index.to_series().diff().eq(1).to_numpy()[1:]
    connected = (arc_ids[:-1] == arc_ids[1:]) | (
        arcs[arc_ids[:-1], 1] == arcs[arc_ids[1:], 0]
    )
    wrong = ~correct.to_numpy()
    reverse = (arcs[arc_ids, 0] == arcs[actual.to_numpy(), 1]) & (arcs[arc_ids, 1] == arcs[actual.to_numpy(), 0])
    # Sequence level: compare the SET of arcs on the predicted route with the reference route arcs,
    # plus the longest run of consecutive correct observations (exact match is almost always False).
    predicted_arcs, reference_arcs = set(arc_ids.tolist()), set(actual.to_numpy().tolist())
    hit = len(predicted_arcs & reference_arcs)
    precision, recall = hit / len(predicted_arcs), hit / len(reference_arcs)
    runs = np.diff(np.flatnonzero(np.r_[True, wrong, True])) - 1
    return {
        "observations": len(predicted),
        "pointwise_accuracy": correct.mean(),
        "sequence_exact_match": bool(correct.all()),
        "route_arc_precision": precision,
        "route_arc_recall": recall,
        "route_arc_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "longest_correct_run": int(runs.max()),
        "reverse_direction_errors": int((wrong & reverse).sum()),
        "continuity": connected[consecutive].mean() if consecutive.any() else None,
    }


def evaluate(candidates, raw_root, cfg, probability_column=None):
    records = []
    for trip_id, rows in candidates.groupby("trip_id", sort=True):
        trip = load_trip(raw_root / str(trip_id).zfill(8), load_route=False)
        graph = build_graph(trip["nodes"], trip["arcs"], trip["nxy"])
        graph.graph["arc_endpoints"] = dict(enumerate(trip["arcs"]))
        rows = rows.sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
        references = rows.drop_duplicates("obs_idx").set_index("obs_idx")["ref_arc"]
        probs = None if probability_column is None else rows[probability_column].to_numpy()
        baseline_scores = -rows["d"] if probs is None else rows[probability_column]
        best = baseline_scores.groupby(rows["obs_idx"]).idxmax()
        baseline = rows.loc[best, ["obs_idx", "arc_id"]]
        t0 = perf_counter()
        decoded = decode(rows, probs, graph, cfg)
        decode_seconds = perf_counter() - t0
        for method, prediction in (("baseline", baseline), ("hmm", decoded)):
            records.append({
                "trip_id": trip_id,
                "method": method,
                **_metrics(prediction, references, trip["arcs"]),
                "decode_seconds": decode_seconds if method == "hmm" else 0.0,
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/sample_candidates.parquet"))
    parser.add_argument("--raw-root", type=Path, required=True, help="Partition folder containing raw trip directories")
    parser.add_argument("--config", type=Path, default=Path("configs/hmm.yaml"))
    parser.add_argument("--probability-column")
    parser.add_argument("--probs", type=Path, help="parquet with trip_id, obs_idx, arc_id, <probability column>")
    parser.add_argument("--output", type=Path, default=Path("results/hmm/sample_comparison.csv"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidates = pd.read_parquet(args.candidates)
    if args.probs:
        probs = pd.read_parquet(args.probs)[["trip_id", "obs_idx", "arc_id", args.probability_column]]
        candidates = candidates.drop(columns=[args.probability_column], errors="ignore").merge(
            probs, on=["trip_id", "obs_idx", "arc_id"], how="inner")
    results = evaluate(candidates, args.raw_root, cfg, args.probability_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

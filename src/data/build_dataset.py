"""Build data/processed/{train,student_test}_candidates.parquet (+ a 5-trip sample).
Run from repo root:  python -m src.data.build_dataset"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .io import load_trip, list_trips
from .features import build_candidates, FEATURE_COLS
from .labels import reference_alignment

REPO = Path(__file__).resolve().parents[2]
RAW, OUT = REPO / "data" / "raw", REPO / "data" / "processed"


def labelled_table(trip):
    """build_candidates (label-free) + label columns derived from the reference route."""
    c = build_candidates(trip)
    al = reference_alignment(trip).set_index("obs_idx")
    c["ref_arc"] = al["ref_arc"].values[c["obs_idx"].values]
    c["label"] = (c["arc_id"].values == c["ref_arc"].values).astype(np.int8)   # candidate is the reference arc
    c["trusted"] = al["trusted"].values[c["obs_idx"].values]
    c["in_route"] = np.isin(c["arc_id"].values, trip["route"]).astype(np.int8)
    return c


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("train", "student_test"):
        t0, tabs = time.time(), []
        for tid in list_trips(RAW / part):
            tabs.append(labelled_table(load_trip(RAW / part / tid)))
        df = pd.concat(tabs, ignore_index=True)
        df.to_parquet(OUT / f"{part}_candidates.parquet", index=False)
        print(part, df.shape, "positives", round(df.label.mean(), 4), "trusted", round(df.trusted.mean(), 4),
              f"{time.time()-t0:.0f}s")
        if part == "train":
            first5 = sorted(df.trip_id.unique())[:5]
            df[df.trip_id.isin(first5)].to_parquet(OUT / "sample_candidates.parquet", index=False)
    pd.Series(FEATURE_COLS).to_csv(OUT / "feature_cols.csv", index=False, header=False)

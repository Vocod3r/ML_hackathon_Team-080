"""
Evaluation metrics for PathSnap.

Input format (agreed with AM320): a "candidate table" with ONE ROW PER
(GPS observation, candidate road segment):

    trip_id | obs_idx | arc_id | label | <features...>

label = 1 if this candidate is the reference (true) arc for that observation,
else 0. After a model runs, we add a column `prob` = model's predicted
probability that the candidate is the true arc.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss as sk_log_loss,
    confusion_matrix as sk_confusion_matrix,
)

KEYS = ["trip_id", "obs_idx"]  # together these identify one GPS observation


# ---------------------------------------------------------------------------
# 1. Candidate-level metrics (each row is a binary classification: true arc or not)
# ---------------------------------------------------------------------------
def macro_f1(y_true, prob, threshold=0.5):
    """F1 for class 0 and class 1, averaged equally (so the rare class-1 counts as much)."""
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def precision_recall(y_true, prob, threshold=0.5):
    """Macro precision and recall, same averaging as macro_f1."""
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def log_loss(y_true, prob):
    """Penalises confident wrong probabilities. Lower is better. Needs real probabilities."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-15, 1 - 1e-15)
    return float(sk_log_loss(y_true, p, labels=[0, 1]))


def confusion_matrix(y_true, prob, threshold=0.5):
    """2x2 matrix: rows = true class (0,1), columns = predicted class (0,1)."""
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    return sk_confusion_matrix(y_true, y_pred, labels=[0, 1])


# ---------------------------------------------------------------------------
# 2. Observation-level metrics (each GPS point gets exactly one predicted arc)
# ---------------------------------------------------------------------------
def top1_predictions(df, prob_col="prob"):
    """For every GPS observation keep only the candidate with the highest probability."""
    best_rows = df.groupby(KEYS)[prob_col].idxmax()
    return (
        df.loc[best_rows, KEYS + ["arc_id", prob_col]]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )


def top1_accuracy(df, prob_col="prob", ref=None):
    """
    Fraction of GPS observations whose highest-probability candidate is the true arc.

    ref (optional): DataFrame [trip_id, obs_idx, ref_arc_id] covering ALL observations,
    including ones that got no candidates at all. If given, those count as wrong.
    If not given, the true arc is read from rows where label == 1.
    """
    top = top1_predictions(df, prob_col)
    if ref is None:
        ref = df.loc[df["label"] == 1, KEYS + ["arc_id"]].rename(columns={"arc_id": "ref_arc_id"})
        # observations where the true arc is not among candidates -> counted as wrong
        all_obs = df[KEYS].drop_duplicates()
        ref = all_obs.merge(ref, on=KEYS, how="left")
    merged = ref.merge(top, on=KEYS, how="left")
    return float((merged["arc_id"] == merged["ref_arc_id"]).mean())


def candidate_miss_rate(df, ref=None):
    """
    Fraction of GPS observations whose true arc is NOT in the candidate set.
    If this is 10%, no model can ever exceed 90% Top-1 accuracy.
    """
    has_true = df.groupby(KEYS)["label"].max()
    if ref is None:
        return float((has_true == 0).mean())
    merged = ref[KEYS].merge(has_true.rename("has_true").reset_index(), on=KEYS, how="left")
    return float((merged["has_true"].fillna(0) == 0).mean())


# ---------------------------------------------------------------------------
# 3. Route-level metric
# ---------------------------------------------------------------------------
def route_continuity(top_df, arcs_by_trip):
    """
    How "drivable" the predicted route is.

    top_df       : output of top1_predictions (one predicted arc per observation)
    arcs_by_trip : dict trip_id -> numpy array of shape (n_arcs, 2) = [from_node, to_node]
                   (each trip has its own road network)

    Steps per trip:
      1. Take predicted arcs in time order and collapse repeats
         (A A A B B C -> A B C), because GPS is 1 Hz and a car stays on one arc
         for many points; counting those would inflate the score.
      2. For each change A -> B, it is "continuous" if A's end node == B's start node.
    Returns the fraction of continuous changes over all trips (1.0 = perfect).
    """
    connected, total = 0, 0
    for trip_id, g in top_df.sort_values(KEYS).groupby("trip_id"):
        arcs = arcs_by_trip[trip_id]
        seq = g["arc_id"].to_numpy()
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]  # collapse consecutive duplicates
        if len(seq) < 2:
            continue
        ends = arcs[seq[:-1], 1]
        starts = arcs[seq[1:], 0]
        connected += int((ends == starts).sum())
        total += len(seq) - 1
    return float(connected / total) if total else float("nan")


# ---------------------------------------------------------------------------
# 4. Everything at once
# ---------------------------------------------------------------------------
def evaluate_all(df, prob_col="prob", arcs_by_trip=None, ref=None, threshold=0.5):
    """Return a dict with every metric, ready to become one row of comparison.csv."""
    y, p = df["label"].to_numpy(), df[prob_col].to_numpy()
    out = {
        "top1_accuracy": top1_accuracy(df, prob_col, ref),
        "macro_f1": macro_f1(y, p, threshold),
        **precision_recall(y, p, threshold),
        "log_loss": log_loss(y, p),
        "candidate_miss_rate": candidate_miss_rate(df, ref),
    }
    if arcs_by_trip is not None:
        out["route_continuity"] = route_continuity(top1_predictions(df, prob_col), arcs_by_trip)
    return out
"""Run candidate generation, classifier scoring, and HMM decoding for one trip.

The model is injected by the caller so this module does not depend on artifact
loading policy. It must implement sklearn's ``predict_proba`` interface.
"""

from pathlib import Path

import networkx as nx
import numpy as np

from src.data.features import build_candidates
from src.data.graph import arc_lengths
from src.data.io import load_trip
from src.hmm.viterbi import decode


def build_trip_graph(trip):
    """Build the directed, length-weighted graph expected by the HMM decoder."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(trip["nodes"])))
    lengths = arc_lengths(trip)
    endpoints = {}
    for arc_id, (start, end) in enumerate(trip["arcs"]):
        start, end = int(start), int(end)
        graph.add_edge(start, end, arc_id=arc_id, length=float(lengths[arc_id]))
        endpoints[arc_id] = (start, end)
    graph.graph["arc_endpoints"] = endpoints
    return graph


def _positive_probabilities(model, candidates, feature_cols):
    """Select class-1 probabilities from a fitted sklearn-style classifier."""
    missing = [name for name in feature_cols if name not in candidates]
    if missing:
        raise ValueError(f"Candidate frame is missing model features: {missing}")
    probabilities = np.asarray(model.predict_proba(candidates[feature_cols]))
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(positive) != 1 or probabilities.shape[1] != len(classes):
        raise ValueError("Model must return predict_proba columns for classes including 1")
    return probabilities[:, positive[0]]


def predict_trip(trip_dir, model, feature_cols, hmm_config, *, candidate_options=None):
    """Predict a trip and return intermediates useful to an API or map renderer.

    ``load_route=False`` guarantees route ground truth is not read in the
    inference path. ``feature_cols`` should come from the model artifact.
    Returns the trip, candidate rows with classifier probabilities, classifier
    argmax path, HMM-decoded path, and graph.
    """
    trip = load_trip(Path(trip_dir), load_route=False)
    candidates = build_candidates(trip, **(candidate_options or {}))
    if candidates.empty:
        empty = candidates[["obs_idx", "arc_id"]].copy()
        return {"trip": trip, "candidates": candidates, "classifier_path": empty,
                "predicted_path": empty.copy(), "graph": build_trip_graph(trip)}

    probabilities = _positive_probabilities(model, candidates, list(feature_cols))
    candidates = candidates.copy()
    candidates["probability"] = probabilities
    order = candidates.groupby("obs_idx")["probability"].idxmax()
    classifier_path = candidates.loc[order, ["obs_idx", "arc_id"]].sort_values("obs_idx").reset_index(drop=True)

    graph = build_trip_graph(trip)
    hmm_path = decode(candidates, probabilities, graph, hmm_config)
    return {"trip": trip, "candidates": candidates, "classifier_path": classifier_path,
            "predicted_path": hmm_path, "graph": graph}


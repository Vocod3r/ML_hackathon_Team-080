"""Decode a single trip using candidate arc emissions and network transitions."""
from functools import lru_cache

import networkx as nx
import numpy as np


def _gps_step(previous, current):
    lon1, lat1 = np.radians(previous)
    lon2, lat2 = np.radians(current)
    h = np.sin((lat2 - lat1) / 2) ** 2
    h += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 12_742_000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _emissions(candidates, probs, cfg):
    scores = np.zeros(len(candidates))
    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        if probs.shape != scores.shape or not np.all((probs >= 0) & (probs <= 1)):
            raise ValueError("Supply one positive-class probability per candidate row")
        scores = cfg.get("emission_weight", 1.0) * np.log(np.maximum(probs, cfg.get("prob_floor", 1e-12)))
    if probs is None or cfg.get("combine_gps_distance", False):
        sigma = cfg.get("gps_sigma_m", 20.0)
        if sigma <= 0:
            raise ValueError("gps_sigma_m must be positive")
        scores -= 0.5 * (candidates["d"].to_numpy() / sigma) ** 2
    return scores


def _transitions(graph, cfg):
    beta = cfg.get("beta_m", 30.0)
    penalty = cfg.get("disconnected_penalty", 12.0)
    cutoff = cfg.get("search_cutoff_m", 600.0)
    uturn = cfg.get("uturn_penalty", 0.0)
    if not (beta > 0 and penalty >= 0 and cutoff > 0):
        raise ValueError("Invalid transition configuration")
    endpoints = graph.graph.get("arc_endpoints")
    if endpoints is None:
        endpoints = {
            data["arc_id"]: (start, end)
            for start, end, data in graph.edges(data=True)
            if "arc_id" in data
        }

    @lru_cache(maxsize=cfg.get("cache_size", 256))
    def distances(source):
        return nx.single_source_dijkstra_path_length(
            graph, source, cutoff=cutoff, weight=lambda u, v, data: data["length"]
        )

    def matrix(previous, current, gps_step):
        scores = np.full((len(previous), len(current)), -float(penalty))
        for i, (old_arc, old_length, old_fraction) in enumerate(previous):
            for j, (new_arc, new_length, new_fraction) in enumerate(current):
                if old_arc == new_arc and new_fraction >= old_fraction:
                    distance = (new_fraction - old_fraction) * old_length
                elif old_arc in endpoints and new_arc in endpoints:
                    old_end = endpoints[old_arc][1]
                    new_start = endpoints[new_arc][0]
                    middle = distances(old_end).get(new_start, np.inf)
                    distance = (1 - old_fraction) * old_length + middle + new_fraction * new_length
                else:
                    continue
                score = max(-penalty, -abs(distance - gps_step) / beta)
                # Entering the opposite-direction twin of the previous arc is a U-turn: it costs
                # no network distance, so without this term the decoder flips direction freely.
                if old_arc != new_arc and endpoints[old_arc] == endpoints[new_arc][::-1]:
                    score -= uturn
                scores[i, j] = score
        return scores

    return matrix


def decode(cand_df, probs, graph, cfg):
    """Return obs_idx and arc_id for each observation with candidates.

    Pass one positive-class probability per row, or None for distance-only
    emissions. Use a directed road graph with edge lengths in metres.
    Missing observations and long time gaps start independent segments.
    """
    if cand_df.empty:
        return cand_df[["obs_idx", "arc_id"]].copy()
    if "trip_id" in cand_df and cand_df["trip_id"].nunique() != 1:
        raise ValueError("decode expects a single trip")
    max_gap = cfg.get("max_gap_s", 60.0)
    if max_gap <= 0:
        raise ValueError("max_gap_s must be positive")

    frame = cand_df.assign(emission=_emissions(cand_df, probs, cfg))
    frame = frame.sort_values(["obs_idx", "arc_id"])
    transitions = _transitions(graph, cfg)
    selected = []
    backtrack = []
    previous = None

    def finish_segment():
        segment = []
        index = int(scores.argmax())
        for arcs, pointers in reversed(backtrack):
            segment.append(arcs[index])
            if pointers is not None:
                index = pointers[index]
        selected.extend(reversed(segment))
        backtrack.clear()

    for obs_idx, group in frame.groupby("obs_idx", sort=False):
        first = group.iloc[0]
        arcs = group["arc_id"].to_numpy()
        states = list(group[["arc_id", "arc_len", "frac"]].itertuples(index=False, name=None))
        emissions = group["emission"].to_numpy()
        gps = (first["lon"], first["lat"])
        if previous is None or obs_idx != last_idx + 1 or first["t"] - last_t > max_gap:
            if previous is not None:
                finish_segment()
            scores, pointers = emissions, None
        else:
            options = scores[:, None] + transitions(previous, states, _gps_step(last_gps, gps))
            pointers = options.argmax(axis=0)
            scores = emissions + options[pointers, np.arange(len(states))]
        backtrack.append((arcs, pointers))
        previous, last_idx, last_t, last_gps = states, obs_idx, first["t"], gps

    finish_segment()
    result = frame[["obs_idx", "arc_id"]].drop_duplicates("obs_idx").reset_index(drop=True)
    result["arc_id"] = selected
    return result

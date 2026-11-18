"""Candidate generation: GPS fix -> nearby road arcs, via a shapely STRtree spatial index."""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

R_DEFAULT = 100.0   # search radius, metres
K_DEFAULT = 10      # max candidates per fix (nearest first)


def seg_proj(P, A, B):
    """Project points P onto segments AB. Returns (dist, s_unclamped_m, length, unit_dir)."""
    AB = B - A
    L = np.hypot(AB[:, 0], AB[:, 1])
    u = AB / np.maximum(L, 1e-6)[:, None]
    s = ((P - A) * u).sum(1)
    C = A + u * np.clip(s, 0, L)[:, None]
    return np.hypot(*(P - C).T), s, L, u


def build_index(trip):
    A, B = trip["nxy"][trip["arcs"][:, 0]], trip["nxy"][trip["arcs"][:, 1]]
    return STRtree(shapely.linestrings(np.stack([A, B], axis=1)))


def raw_pairs(trip, R, tree=None, fallback=True):
    """(obs_idx, arc_id, dist) for every arc within R metres; with fallback, a fix with no arc
    within R gets its single nearest arc (up to 2 km) so every fix has >= 1 candidate."""
    tree = tree or build_index(trip)
    pts = shapely.points(trip["xy"])
    q = tree.query(pts, predicate="dwithin", distance=R)
    obs, arc = q[0], q[1]
    if fallback:
        miss = np.setdiff1d(np.arange(len(pts)), obs)
        if len(miss):
            qn = tree.query_nearest(pts[miss], max_distance=2000.0, all_matches=False)
            obs, arc = np.r_[obs, miss[qn[0]]], np.r_[arc, qn[1]]
    A, B = trip["nxy"][trip["arcs"][:, 0]], trip["nxy"][trip["arcs"][:, 1]]
    d, *_ = seg_proj(trip["xy"][obs], A[arc], B[arc])
    return obs, arc, d


def generate_candidates(trip, R=R_DEFAULT, K=K_DEFAULT, tree=None):
    """DataFrame[obs_idx, arc_id, d, rank] - K nearest arcs within R of each fix (labels never used)."""
    obs, arc, d = raw_pairs(trip, R, tree)
    df = pd.DataFrame({"obs_idx": obs, "arc_id": arc, "d": d}).sort_values(["obs_idx", "d"], kind="stable")
    df["rank"] = df.groupby("obs_idx").cumcount()
    return df[df["rank"] < K].reset_index(drop=True)

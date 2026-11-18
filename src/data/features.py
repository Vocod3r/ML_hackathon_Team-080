"""Candidate-level feature engineering (contract 2). build_candidates() uses NO reference/label data,
so it runs unchanged on an unseen evaluator trip inside the API.

One row per (GPS fix, candidate arc). Columns: trip_id, obs_idx, t, lon, lat, arc_id, <FEATURE_COLS>.

Feature rationale
 d, rank, d_minus_min, d_ratio   closeness of the fix to the arc, absolute and relative to the other
                                  candidates (a 20 m arc is plausible if all others are 80 m away)
 cos_w1/cos_w3, speed_w1/w3       cosine between the vehicle's motion vector and the arc direction. Most roads
                                  are two arcs in opposite directions, so this is what separates them
                                  (set to 0 when the vehicle is ~stationary: heading is noise then)
 frac, arc_len                    where on the arc the fix projects; short arcs near junctions are ambiguous
 d_k-3..d_k3, d_ctx_*             distance of the 3 previous/next fixes to the SAME arc: a true arc is
                                  consistently close over a window, a parallel road drifts away
 adv_-1/adv_1 (+_ratio)           advance along the arc between consecutive fixes vs the actual step length:
                                  consistent with driving along that arc in the direction of time
 out_deg_head, in_deg_tail,       network topology (junction vs. mid-block, one-way vs two-way road)
 has_reverse
 n_cand, dt_prev, dt_next         candidate-set size and local sampling interval (gap awareness)"""
import numpy as np
import pandas as pd
from .candidates import generate_candidates, seg_proj, build_index, R_DEFAULT, K_DEFAULT

META_COLS = ["trip_id", "obs_idx", "t", "lon", "lat", "arc_id"]


def _w(n, i, k):
    return np.clip(i + k, 0, n - 1)


def build_candidates(trip, R=R_DEFAULT, K=K_DEFAULT, tree=None):
    tree = tree or build_index(trip)
    cand = generate_candidates(trip, R, K, tree)
    n = len(trip["xy"])
    i, a = cand["obs_idx"].values, cand["arc_id"].values
    xy, t = trip["xy"], trip["track"][:, 2]
    A, B = trip["nxy"][trip["arcs"][a, 0]], trip["nxy"][trip["arcs"][a, 1]]
    d, s, L, u = seg_proj(xy[i], A, B)
    f = {"d": d, "rank": cand["rank"].values, "arc_len": L, "frac": np.clip(s, 0, L) / np.maximum(L, 1e-6),
         "n_cand": cand.groupby("obs_idx")["arc_id"].transform("size").values}
    dmin = cand.groupby("obs_idx")["d"].transform("min").values
    f["d_minus_min"], f["d_ratio"] = d - dmin, d / (dmin + 5.0)
    for w in (1, 3):
        i0, i1 = _w(n, i, -w), _w(n, i, w)
        mv = xy[i1] - xy[i0]
        sp = np.hypot(mv[:, 0], mv[:, 1])
        f[f"speed_w{w}"] = sp / np.maximum(t[i1] - t[i0], 1e-3)
        f[f"cos_w{w}"] = np.where(sp > 3.0, (mv / np.maximum(sp, 1e-6)[:, None] * u).sum(1), 0.0)
    ds = []
    for k in (-3, -2, -1, 1, 2, 3):
        j = _w(n, i, k)
        dk, sk, _, _ = seg_proj(xy[j], A, B)
        dk = np.minimum(dk, 200.0)
        f[f"d_k{k}"] = dk
        ds.append(dk)
        if abs(k) == 1:
            adv = (sk - s) * np.sign(k)
            f[f"adv_{k}"] = np.clip(adv, -200, 200)
            f[f"adv_ratio_{k}"] = np.clip(adv / np.maximum(np.hypot(*(xy[j] - xy[i]).T), 1.0), -3, 3)
    ds = np.array(ds)
    f["d_ctx_mean"], f["d_ctx_max"], f["d_ctx_n30"] = ds.mean(0), ds.max(0), (ds < 30).sum(0)
    arcs, N = trip["arcs"], len(trip["nodes"])
    outdeg, indeg = np.bincount(arcs[:, 0], minlength=N), np.bincount(arcs[:, 1], minlength=N)
    f["out_deg_head"], f["in_deg_tail"] = outdeg[arcs[a, 1]], indeg[arcs[a, 0]]
    key, rev = arcs[:, 0].astype(np.int64) * N + arcs[:, 1], arcs[:, 1].astype(np.int64) * N + arcs[:, 0]
    f["has_reverse"] = np.isin(rev[a], key).astype(int)
    f["dt_prev"], f["dt_next"] = t[i] - t[_w(n, i, -1)], t[_w(n, i, 1)] - t[i]
    meta = pd.DataFrame({"trip_id": trip["trip_id"], "obs_idx": i, "t": t[i], "lon": trip["track"][i, 0],
                         "lat": trip["track"][i, 1], "arc_id": a})
    return pd.concat([meta, pd.DataFrame(f).astype(np.float32)], axis=1)


FEATURE_COLS = (["d", "rank", "arc_len", "frac", "n_cand", "d_minus_min", "d_ratio",
                 "speed_w1", "cos_w1", "speed_w3", "cos_w3"]
                + [f"d_k{k}" for k in (-3, -2, -1, 1, 2, 3)]
                + ["adv_-1", "adv_ratio_-1", "adv_1", "adv_ratio_1", "d_ctx_mean", "d_ctx_max", "d_ctx_n30",
                   "out_deg_head", "in_deg_tail", "has_reverse", "dt_prev", "dt_next"])

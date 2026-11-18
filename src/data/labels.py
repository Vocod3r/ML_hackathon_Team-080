"""Reference labels (contract 3). TRAINING / SERVER-SIDE EVALUATION ONLY: the prediction path never
imports this module and never reads .route.

Problem: .route is an ordered list of arcs (59-2270 per trip) with NO timestamps, i.e. not one label per
GPS fix. Method: monotone (order-preserving) alignment. Every fix i is assigned a route position j(i)
with j non-decreasing in time, minimising sum_i min(dist(fix_i, arc_route[j(i)]), cap). This is
solved exactly by dynamic programming in O(n*m) with a running prefix-minimum:
    dp_i[j] = c_i[j] + min_{k<=j} dp_{i-1}[k]
Direction is respected automatically because the two directions of a two-way road are different arc ids
and the route lists the one actually travelled. The reference arc of fix i is route[j(i)].

Limitations (reported in the EDA): the route is a reference, not verified truth. Where the track and
route disagree (aligned distance > TRUST_DIST) the label is unreliable: such fixes are flagged
`trusted=False`, are excluded from training and reported separately in evaluation."""
import numpy as np
import pandas as pd
from .candidates import seg_proj

TRUST_DIST = 50.0   # m: aligned reference arc farther than this -> label not trusted
CAP = 150.0         # m: distance cap inside the DP so a few outliers cannot dominate


def reference_alignment(trip, cap=CAP, trust=TRUST_DIST):
    route = trip["route"]
    xy, n, m = trip["xy"], len(trip["xy"]), len(route)
    A, B = trip["nxy"][trip["arcs"][route, 0]], trip["nxy"][trip["arcs"][route, 1]]
    C = np.empty((n, m), np.float32)
    for i in range(n):
        d, *_ = seg_proj(np.broadcast_to(xy[i], (m, 2)), A, B)
        C[i] = np.minimum(d, cap)
    dp, back, idx = C[0].copy(), np.zeros((n, m), np.int32), np.arange(m)
    for i in range(1, n):
        cm = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp <= cm, idx, 0))   # argmin over k<=j
        dp = C[i] + cm
    j, pos = int(np.argmin(dp)), np.zeros(n, np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    dist = C[np.arange(n), pos].astype(float)
    return pd.DataFrame({"obs_idx": np.arange(n), "ref_arc": route[pos], "route_pos": pos,
                         "align_dist": dist, "trusted": dist <= trust})


def reference_arc_per_obs(trip):
    """Series obs_idx -> reference arc id (contract 3)."""
    a = reference_alignment(trip)
    return pd.Series(a["ref_arc"].values, index=a["obs_idx"].values, name="ref_arc")

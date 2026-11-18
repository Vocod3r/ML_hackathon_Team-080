"""Dataset exploration (train 80 + student_test 10; evaluator data is never touched).
Run from repo root:  python -m src.data.run_eda      -> results/eda/*"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import folium
from .io import load_trip, list_trips
from .graph import connectivity_stats, arc_lengths
from .geo import haversine_m
from .candidates import raw_pairs, build_index
from .labels import reference_alignment

REPO = Path(__file__).resolve().parents[2]
RAW, OUT = REPO / "data" / "raw", REPO / "results" / "eda"
SWEEP_R = [25, 50, 75, 100, 150, 200]
SWEEP_K = [1, 3, 5, 10, 15, 20]


def inventory():
    rows, sweep = [], []
    for part in ("train", "student_test"):
        for tid in list_trips(RAW / part):
            d = RAW / part / tid
            raw = np.loadtxt(d / f"{tid}.track", ndmin=2)
            trip = load_trip(d)
            t, xy = trip["track"][:, 2], trip["xy"]
            dt, step = np.diff(t), np.hypot(*np.diff(xy, axis=0).T)
            hv = haversine_m(*trip["track"][:-1, :2].T, *trip["track"][1:, :2].T)
            route, arcs = trip["route"], trip["arcs"]
            al = reference_alignment(trip)
            gs = connectivity_stats(trip)
            ra = arcs[route]
            tree = build_index(trip)
            obs, arc, dist = raw_pairs(trip, max(SWEEP_R), tree, fallback=False)
            pr = pd.DataFrame({"o": obs, "a": arc, "d": dist}).sort_values(["o", "d"], kind="stable")
            pr["rk"] = pr.groupby("o").cumcount()
            ref = al["ref_arc"].values[pr["o"].values]
            hit = pr[pr["a"].values == ref]
            best = np.full(len(xy), np.inf)      # (distance, rank) of the reference arc per fix
            rk = np.full(len(xy), 10**6)
            best[hit["o"].values], rk[hit["o"].values] = hit["d"].values, hit["rk"].values
            tr = al["trusted"].values
            for R in SWEEP_R:
                for K in SWEEP_K:
                    sweep.append(dict(trip=tid, part=part, R=R, K=K, n_trusted=int(tr.sum()),
                                      n_miss=int((tr & ~((best <= R) & (rk < K))).sum())))
            rows.append(dict(
                partition=part, trip=tid, n_obs=len(xy), n_nan_rows=len(raw) - len(xy),
                duplicate_ts=int((dt == 0).sum()), non_monotonic_ts=int((np.diff(raw[:, 2]) < 0).sum()),
                pct_dt_eq_1=round(100 * (np.abs(dt - 1) < 1e-6).mean(), 2), max_gap_s=float(dt.max()),
                n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt60s=int((dt > 60).sum()), duration_s=float(t[-1] - t[0]),
                track_len_km=round(step.sum() / 1000, 3), max_step_m=round(step.max(), 1),
                utm_vs_haversine_err_pct=round(100 * abs(step.sum() - hv.sum()) / hv.sum(), 3), epsg=trip["epsg"],
                lon_min=raw[:, 0].min(), lon_max=raw[:, 0].max(), lat_min=raw[:, 1].min(), lat_max=raw[:, 1].max(),
                extent_km_x=round(np.ptp(xy[:, 0]) / 1000, 2), extent_km_y=round(np.ptp(xy[:, 1]) / 1000, 2),
                **gs, mean_arc_len_m=round(arc_lengths(trip).mean(), 1),
                route_arcs=len(route), route_unique=len(np.unique(route)),
                route_breaks=int((ra[:-1, 1] != ra[1:, 0]).sum()),
                route_contiguous_frac=round(float((ra[:-1, 1] == ra[1:, 0]).mean()), 4),
                align_dist_p50=round(float(np.median(al.align_dist)), 1), align_dist_p95=round(float(np.percentile(al.align_dist, 95)), 1),
                untrusted_frac=round(float((~al.trusted).mean()), 4),
                track_kb=round(os.path.getsize(d / f"{tid}.track") / 1024, 1),
                nodes_kb=round(os.path.getsize(d / f"{tid}.nodes") / 1024, 1),
                arcs_kb=round(os.path.getsize(d / f"{tid}.arcs") / 1024, 1),
                route_kb=round(os.path.getsize(d / f"{tid}.route") / 1024, 1)))
            print(part, tid, len(xy), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sweep)


def figures(df, sw):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (c, ttl) in zip(ax.flat, [("n_obs", "GPS fixes per trip"), ("track_len_km", "Track length (km)"),
                                     ("n_arcs", "Road arcs per trip"), ("pct_dt_eq_1", "% of steps exactly 1 s"),
                                     ("untrusted_frac", "Frac. fixes >50 m from aligned reference"),
                                     ("n_gaps_gt60s", "# time gaps > 60 s per trip")]):
        for p, col in [("train", "tab:blue"), ("student_test", "tab:orange")]:
            a.hist(df[df.partition == p][c], bins=20, alpha=.6, label=p, color=col)
        a.set_title(ttl); a.legend()
    plt.tight_layout(); plt.savefig(OUT / "eda_distributions.png", dpi=130); plt.close()

    # missing-value / data-quality analysis
    cols = ["n_nan_rows", "duplicate_ts", "non_monotonic_ts", "n_gaps_gt60s", "isolated_nodes", "route_breaks",
            "untrusted_frac"]
    q = (df.set_index("partition")[cols] > 0).groupby(level=0).sum().T
    ax = q.plot.barh(figsize=(8, 4.5)); ax.set_xlabel("# trips affected"); ax.set_title("Data-quality issues by partition")
    plt.tight_layout(); plt.savefig(OUT / "eda_quality.png", dpi=130); plt.close()

    # candidate miss rate vs R and K (trusted fixes, train partition)
    s = sw[sw.part == "train"].groupby(["R", "K"])[["n_miss", "n_trusted"]].sum()
    m = (100 * s.n_miss / s.n_trusted).unstack("K")
    m.to_csv(OUT / "candidate_miss_sweep.csv")
    sw.groupby(["part", "R", "K"])[["n_miss", "n_trusted"]].sum().assign(
        miss_rate=lambda x: x.n_miss / x.n_trusted).reset_index().to_csv(OUT / "candidate_miss_sweep_long.csv", index=False)
    fig, a = plt.subplots(figsize=(7, 4.5))
    for K in SWEEP_K:
        a.plot(m.index, m[K], marker="o", label=f"k={K}")
    a.set_xlabel("search radius R (m)"); a.set_ylabel("candidate miss rate (%)"); a.set_yscale("log")
    a.set_title("Reference arc missing from candidate set (train, trusted fixes)"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "candidate_miss_sweep.png", dpi=130); plt.close()
    return m


def trajectory_plots():
    """GPS + network + reference route in metric coords (train partition)."""
    sample = [("train", t) for t in ("00000001", "00000003", "00000004", "00000002")]
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for a, (part, tid) in zip(ax.flat, sample):
        trip = load_trip(RAW / part / tid)
        segs = np.stack([trip["nxy"][trip["arcs"][:, 0]], trip["nxy"][trip["arcs"][:, 1]]], 1)
        cx, cy = np.median(trip["xy"], 0)
        pad = max(np.ptp(trip["xy"][:, 0]), np.ptp(trip["xy"][:, 1])) * .6 + 300
        a.add_collection(LineCollection(segs, colors="lightgrey", linewidths=.4))
        a.add_collection(LineCollection(segs[trip["route"]], colors="green", linewidths=1.8, label="reference route"))
        a.scatter(*trip["xy"].T, s=2, c="tab:blue", label="GPS fixes")
        a.set_xlim(cx - pad, cx + pad); a.set_ylim(cy - pad, cy + pad); a.set_aspect("equal")
        a.set_title(f"{part} trip {tid}: {len(trip['xy'])} fixes, {len(trip['arcs'])} arcs"); a.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / "eda_trajectories_network.png", dpi=130); plt.close()

    # anomalous trip: reference vs track disagree
    for tid in ("00000054", "00000033"):
        trip = load_trip(RAW / "train" / tid); al = reference_alignment(trip)
        segs = np.stack([trip["nxy"][trip["arcs"][:, 0]], trip["nxy"][trip["arcs"][:, 1]]], 1)
        fig, a = plt.subplots(figsize=(8, 7)); cx, cy = np.median(trip["xy"], 0)
        pad = max(np.ptp(trip["xy"][:, 0]), np.ptp(trip["xy"][:, 1])) * .6 + 300
        a.add_collection(LineCollection(segs, colors="lightgrey", linewidths=.3))
        a.add_collection(LineCollection(segs[trip["route"]], colors="green", linewidths=2, label="reference route"))
        tr = al.trusted.values
        a.scatter(*trip["xy"][tr].T, s=2, c="tab:blue", label="GPS (matches reference)")
        a.scatter(*trip["xy"][~tr].T, s=2, c="red", label="GPS >50 m from reference")
        a.set_xlim(cx - pad, cx + pad); a.set_ylim(cy - pad, cy + pad); a.set_aspect("equal"); a.legend()
        a.set_title(f"Anomalous train trip {tid}: {100*(~tr).mean():.0f}% of fixes untrusted")
        plt.tight_layout(); plt.savefig(OUT / f"anomaly_trip_{tid}.png", dpi=120); plt.close()


def folium_map(part="student_test", tid="00000007", pad_m=600, window=None):
    """Interactive map: raw GPS, road network (clipped to the bbox), reference route.

    window=(first_fix, last_fix) draws only that stretch of the trip, with every fix, at street
    level so GPS noise against the road and the reference route is visible (used for the report)."""
    trip = load_trip(RAW / part / tid)
    ll, xy = trip["track"][:, :2], trip["xy"]
    if window is not None:
        keep = slice(window[0], window[1] + 1)
        ll, xy, pad_m = ll[keep], xy[keep], 120
    lo, hi = xy.min(0) - pad_m, xy.max(0) + pad_m
    a, b = trip["nxy"][trip["arcs"][:, 0]], trip["nxy"][trip["arcs"][:, 1]]
    inb = ((np.minimum(a, b) >= lo) & (np.maximum(a, b) <= hi)).all(1)
    nd = trip["nodes"]
    m = folium.Map(location=[ll[:, 1].mean(), ll[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (clipped to trip bbox)", show=True)
    for i in np.where(inb)[0]:
        p, q = trip["arcs"][i]
        folium.PolyLine([(nd[p, 1], nd[p, 0]), (nd[q, 1], nd[q, 0])], color="#999999", weight=1, opacity=.7).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route (.route)", show=True)
    for i in trip["route"]:
        p, q = trip["arcs"][i]
        if window is not None and not inb[i]:
            continue
        folium.PolyLine([(nd[p, 1], nd[p, 0]), (nd[q, 1], nd[q, 0])], color="green", weight=5, opacity=.8).add_to(ref)
    ref.add_to(m)
    g = folium.FeatureGroup("Raw GPS fixes", show=True)
    for k in range(0, len(ll), 1 if window is not None else 3):
        folium.CircleMarker((ll[k, 1], ll[k, 0]), radius=2, color="blue", fill=True, weight=1,
                            tooltip=f"fix {k + (window[0] if window else 0)}").add_to(g)
    g.add_to(m)
    folium.Marker((ll[0, 1], ll[0, 0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((ll[-1, 1], ll[-1, 0]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    folium.LayerControl().add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;'
              'border:1px solid grey;font-size:13px"><b>Trip %s (%s)</b><br><span style="color:blue">&#9679;</span> raw GPS fixes'
              '<br><span style="color:green">&#9644;</span> reference route<br><span style="color:#999">&#9644;</span> road network</div>'
              % (tid, part))
    m.get_root().html.add_child(folium.Element(legend))
    m.save(OUT / f"folium_eda_{part}_{tid}{'_zoom' if window is not None else ''}.html")


def zoom_window(part, tid, length=120, min_speed_ms=4.0):
    """Fix range of `length` fixes where the vehicle really travels and the trusted fixes lie farthest
    from their reference arc: the clearest example of GPS noise on a road.

    "Travels" uses NET displacement across the window (>= min_speed_ms on average), not the sum of
    per-second steps, because jitter of a parked car would otherwise look like fast movement."""
    trip = load_trip(RAW / part / tid)
    al = reference_alignment(trip)
    xy, n = trip["xy"], len(al)
    if n <= length:
        return 0, n - 1
    d = al["align_dist"].where(al["trusted"], 0).to_numpy()
    starts = np.arange(n - length)
    net = np.hypot(*(xy[starts + length] - xy[starts]).T) / length            # m/s over the window
    noise = np.convolve(d, np.ones(length) / length, mode="valid")[: n - length]
    score = np.where(net >= min_speed_ms, noise, -1.0)
    first = int(np.argmax(score))
    return first, first + length


def missing_value_analysis():
    """Per-partition file availability and invalid-value counts for every raw file type."""
    rows = []
    for part in ("train", "student_test"):
        trips = list_trips(RAW / part)
        r = dict(partition=part, trips=len(trips), trips_missing_a_file=0, empty_files=0,
                 track_nonfinite_rows=0, track_lon_lat_out_of_range=0, nodes_nonfinite_rows=0,
                 nodes_lon_lat_out_of_range=0, arcs_node_id_out_of_range=0, arcs_self_loops=0,
                 route_arc_id_out_of_range=0, track_rows=0, nodes_rows=0, arcs_rows=0, route_rows=0)
        for tid in trips:
            d = RAW / part / tid
            files = {k: d / f"{tid}.{k}" for k in ("track", "nodes", "arcs", "route")}
            r["trips_missing_a_file"] += int(not all(f.exists() for f in files.values()))
            r["empty_files"] += sum(f.exists() and f.stat().st_size == 0 for f in files.values())
            track, nodes = np.loadtxt(files["track"], ndmin=2), np.loadtxt(files["nodes"], ndmin=2)
            arcs, route = np.loadtxt(files["arcs"], dtype=np.int64, ndmin=2), np.loadtxt(files["route"], dtype=np.int64, ndmin=1)
            r["track_rows"] += len(track); r["nodes_rows"] += len(nodes); r["arcs_rows"] += len(arcs); r["route_rows"] += route.size
            r["track_nonfinite_rows"] += int((~np.isfinite(track)).any(1).sum())
            r["nodes_nonfinite_rows"] += int((~np.isfinite(nodes)).any(1).sum())
            r["track_lon_lat_out_of_range"] += int(((np.abs(track[:, 0]) > 180) | (np.abs(track[:, 1]) > 90)).sum())
            r["nodes_lon_lat_out_of_range"] += int(((np.abs(nodes[:, 0]) > 180) | (np.abs(nodes[:, 1]) > 90)).sum())
            r["arcs_node_id_out_of_range"] += int(((arcs < 0) | (arcs >= len(nodes))).any(1).sum())
            r["arcs_self_loops"] += int((arcs[:, 0] == arcs[:, 1]).sum())
            r["route_arc_id_out_of_range"] += int(((route < 0) | (route >= len(arcs))).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "missing_value_analysis.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, sw = inventory()
    df.to_csv(OUT / "eda_inventory.csv", index=False)
    m = figures(df, sw)
    trajectory_plots()
    folium_map("student_test", "00000007")
    folium_map("train", "00000003")
    for part, tid in (("student_test", "00000007"), ("train", "00000003")):
        folium_map(part, tid, window=zoom_window(part, tid))
    print(missing_value_analysis().T.to_string())
    agg = df.groupby("partition").agg(trips=("trip", "count"), obs_total=("n_obs", "sum"), obs_mean=("n_obs", "mean"),
                                      obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), nodes_median=("n_nodes", "median"),
                                      arcs_median=("n_arcs", "median"), route_arcs_mean=("route_arcs", "mean"),
                                      untrusted_mean=("untrusted_frac", "mean"))
    agg.to_csv(OUT / "partition_summary.csv")
    with open(OUT / "eda_summary.txt", "w") as fh:
        fh.write(agg.round(3).to_string() + "\n\n")
        fh.write(f"NaN rows: {df.n_nan_rows.sum()}; duplicate ts: {df.duplicate_ts.sum()}; non-monotonic ts: {df.non_monotonic_ts.sum()}\n")
        fh.write(f"Isolated nodes (trips affected): {(df.isolated_nodes > 0).sum()}; trips with >1 weak component: {(df.n_components > 1).sum()} / {len(df)}\n")
        fh.write(f"Largest-component node share (min over trips): {df.largest_comp_frac.min():.3f}\n")
        fh.write(f"Trips with route breaks: {(df.route_breaks > 0).sum()}; total breaks {df.route_breaks.sum()}; "
                 f"min contiguity {df.route_contiguous_frac.min():.4f}\n")
        fh.write(f"Two-way arc share (mean): {df.two_way_frac.mean():.3f}\n")
        fh.write(f"Share of steps exactly 1 s (mean over trips): {df.pct_dt_eq_1.mean():.2f}%\n")
        fh.write(f"UTM vs haversine track length error: max {df.utm_vs_haversine_err_pct.max():.3f}%\n")
        fh.write("Trips with >5% untrusted fixes:\n" + df[df.untrusted_frac > .05][["partition", "trip", "n_obs", "untrusted_frac"]].to_string() + "\n")
        fh.write("Trips with a gap > 600 s:\n" + df[df.max_gap_s > 600][["partition", "trip", "max_gap_s", "n_gaps_gt60s"]].to_string() + "\n")
        fh.write("Candidate miss rate (%) by radius R (rows) and k (cols), train trusted fixes:\n" + m.round(3).to_string() + "\n")
    print(open(OUT / "eda_summary.txt").read())


if __name__ == "__main__":
    main()

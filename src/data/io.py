"""Raw trip loading (contract 1).

Layout: <root>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
 .track  lon lat t   (whitespace, no header)      .nodes lon lat  (row index = node id)
 .arcs   from to     (row index = arc_id)         .route arc_id per row (reference, ordered)
"""
from pathlib import Path
import numpy as np
from .geo import utm_epsg, to_local_metres


def load_trip(trip_dir, load_route=True):
    """Return dict(trip_id, track, nodes, arcs, route, epsg, xy, nxy).

    track (n,3) lon,lat,t sorted by t; nodes (N,2); arcs (M,2) int; route (R,) int or None.
    xy / nxy are the GPS fixes / nodes projected to local metres (per-trip UTM).
    The prediction pipeline calls this with load_route=False so no reference data is read."""
    d = Path(trip_dir)
    tid = d.name
    track = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    track = track[np.isfinite(track).all(1)]
    track = track[np.argsort(track[:, 2], kind="stable")]
    check_coordinates(track, nodes, arcs, tid)
    trip = dict(trip_id=tid, track=track, nodes=nodes, arcs=arcs, route=None)
    trip["epsg"] = utm_epsg(track[:, 0].mean(), track[:, 1].mean())
    trip["xy"] = to_local_metres(track[:, 0], track[:, 1], trip)
    trip["nxy"] = to_local_metres(nodes[:, 0], nodes[:, 1], trip)
    if load_route:
        trip["route"] = np.loadtxt(d / f"{tid}.route", dtype=np.int64, ndmin=1).ravel()
    return trip


def check_coordinates(track, nodes, arcs, tid=""):
    """Column-order / plausibility checks (lon first, lat second)."""
    assert np.abs(track[:, 0]).max() <= 180 and np.abs(track[:, 1]).max() <= 90, f"{tid}: track not lon,lat"
    assert np.abs(nodes[:, 0]).max() <= 180 and np.abs(nodes[:, 1]).max() <= 90, f"{tid}: nodes not lon,lat"
    assert arcs.min() >= 0 and arcs.max() < len(nodes), f"{tid}: arc references unknown node"


def list_trips(root):
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())

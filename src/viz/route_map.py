"""Folium maps for inspecting GPS trips and map-matching routes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import folium
import numpy as np
from folium.plugins import Fullscreen


def _arc_coords(trip: dict, arc_ids: Iterable[int], bounds: tuple[float, float, float, float],
                max_arcs: int) -> list[list[list[float]]]:
    """Return in-bounds arc polylines, capped for large per-trip networks."""
    arcs = trip["arcs"]
    nodes = trip["nodes"]
    ids = np.asarray(list(arc_ids), dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(arcs))]
    if len(ids) > max_arcs:
        # Deterministic sampling while retaining route order as much as possible.
        ids = ids[np.linspace(0, len(ids) - 1, max_arcs, dtype=int)]
    lon_min, lat_min, lon_max, lat_max = bounds
    result = []
    for arc_id in ids:
        a, b = arcs[arc_id]
        p, q = nodes[a], nodes[b]
        if (lon_min <= p[0] <= lon_max and lat_min <= p[1] <= lat_max) or \
           (lon_min <= q[0] <= lon_max and lat_min <= q[1] <= lat_max):
            result.append([[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]])
    return result


def create_route_map(trip: dict, *, reference_arcs=None, classifier_arcs=None,
                     predicted_arcs=None, max_network_arcs: int = 15000,
                     padding_degrees: float = 0.01) -> folium.Map:
    """Create a layer-controlled GPS/network/reference/prediction map.

    Arc sequences may contain repeated IDs (one per observation); consecutive duplicates
    are removed for display. ``predicted_arcs`` represents the decoded/HMM result.
    """
    track = np.asarray(trip["track"])
    if not len(track):
        raise ValueError("Cannot map a trip with no GPS observations")
    lon, lat = track[:, 0], track[:, 1]
    bounds = (float(lon.min() - padding_degrees), float(lat.min() - padding_degrees),
              float(lon.max() + padding_degrees), float(lat.max() + padding_degrees))
    center = [float(lat.mean()), float(lon.mean())]
    m = folium.Map(location=center, tiles="CartoDB positron", control_scale=True)
    title = folium.Element(f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
                           f'background:white;padding:6px 10px;border:1px solid #aaa">'
                           f'Trip {trip.get("trip_id", "")}</div>')
    m.get_root().html.add_child(title)

    gps = [[float(y), float(x)] for x, y in zip(lon, lat)]
    gps_group = folium.FeatureGroup(name="GPS track", show=True)
    folium.PolyLine(gps, color="#2878b5", weight=3, opacity=0.8).add_to(gps_group)
    folium.CircleMarker(gps[0], radius=4, color="green", fill=True, tooltip="Start").add_to(gps_group)
    folium.CircleMarker(gps[-1], radius=4, color="red", fill=True, tooltip="End").add_to(gps_group)
    gps_group.add_to(m)

    road_group = folium.FeatureGroup(name="Road network (clipped)", show=False)
    network_bounds = (float(lon.min() - padding_degrees), float(lat.min() - padding_degrees),
                      float(lon.max() + padding_degrees), float(lat.max() + padding_degrees))
    for line in _arc_coords(trip, range(len(trip["arcs"])), network_bounds, max_network_arcs):
        folium.PolyLine(line, color="#777777", weight=1, opacity=0.35).add_to(road_group)
    road_group.add_to(m)

    def add_route(name, values, color, show):
        if values is None:
            return
        # Collapse repeated per-observation arc IDs for a concise route layer.
        seq = np.asarray(list(values), dtype=np.int64)
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
        group = folium.FeatureGroup(name=name, show=show)
        for line in _arc_coords(trip, seq, network_bounds, max_network_arcs):
            folium.PolyLine(line, color=color, weight=5, opacity=0.9).add_to(group)
        group.add_to(m)

    add_route("Reference route", reference_arcs, "#2ca02c", True)
    add_route("Classifier route", classifier_arcs, "#ff7f0e", False)
    add_route("Predicted route (HMM)", predicted_arcs, "#d62728", True)
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position="topright").add_to(m)
    m.fit_bounds([[float(lat.min() - padding_degrees), float(lon.min() - padding_degrees)],
                  [float(lat.max() + padding_degrees), float(lon.max() + padding_degrees)]])
    return m


def save_route_map(trip: dict, output_path, **kwargs) -> Path:
    """Create and save a trip map, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    create_route_map(trip, **kwargs).save(str(path))
    return path

"""Geographic helpers (contract 1).

Why per-trip UTM: trips are spread over several continents (lon -100..137, lat -35..62), so no single
projected CRS is valid. Each trip spans < ~50 km, where one UTM zone has <0.1 % scale distortion, so
Euclidean distance in the trip's UTM metres is accurate for point-to-segment geometry and route
distances. Degrees are never used for distances."""
from functools import lru_cache
import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=128)
def _tf(epsg):
    return Transformer.from_crs(4326, epsg, always_xy=True)


def to_local_metres(lon, lat, trip):
    """lon/lat arrays -> (n,2) metres in the trip's UTM zone (trip['epsg'])."""
    x, y = _tf(trip["epsg"]).transform(np.asarray(lon), np.asarray(lat))
    return np.c_[x, y]


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to cross-check the projection)."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

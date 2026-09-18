"""From stations to addresses.

The router answers "how long from each station". People do not live in
stations, so this module lays a grid over the city and gives every cell the
best option among all nearby stations:

    commute(cell) = min over stations of  walk(cell -> station) + time from station

plus the option of walking the whole way.

Walking is modelled as straight-line distance stretched by a detour factor,
because streets are a grid and you cannot cut through buildings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gtfs import Timetable
from .router import UNREACHABLE

WALK_SPEED = 1.3             # metres per second, about 4.7 km/h
DETOUR = 1.3                 # street distance / straight-line distance
METRES_PER_DEGREE = 111_320
UNREACHABLE_MINUTES = 255    # fits the whole answer in one byte per cell

NYC = (40.49, -74.27, 40.92, -73.68)   # south, west, north, east


def walk_seconds(metres: float) -> float:
    return metres * DETOUR / WALK_SPEED


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Flat-earth distance. Accurate to well under 1% across a single city."""
    dy = (lat2 - lat1) * METRES_PER_DEGREE
    dx = (lon2 - lon1) * METRES_PER_DEGREE * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def stations_near(tt: Timetable, lat: float, lon: float, max_walk_min: float) -> dict[int, int]:
    """Stations within walking range of a point -> walking seconds."""
    found = {}
    for i, (slat, slon) in enumerate(zip(tt.station_lats, tt.station_lons)):
        seconds = walk_seconds(distance_m(lat, lon, slat, slon))
        if seconds <= max_walk_min * 60:
            found[i] = int(round(seconds))
    return found


@dataclass
class Grid:
    south: float
    west: float
    north: float
    east: float
    width: int = 360

    def __post_init__(self):
        # Rows are evenly spaced in Web Mercator, not in latitude, so the
        # image lines up exactly when a web map stretches it over the bounds.
        self.lons = np.linspace(self.west, self.east, self.width)
        y_north, y_south = _mercator(self.north), _mercator(self.south)
        lon_span = math.radians(self.east - self.west)
        self.height = max(1, round(self.width * (y_north - y_south) / lon_span))
        ys = np.linspace(y_north, y_south, self.height)
        self.lats = np.degrees(2 * np.arctan(np.exp(ys)) - math.pi / 2)   # north to south

    def cell(self, lat: float, lon: float) -> tuple[int, int] | None:
        if not (self.south <= lat <= self.north and self.west <= lon <= self.east):
            return None
        return int(np.abs(self.lats - lat).argmin()), int(np.abs(self.lons - lon).argmin())


def commute_minutes(
    grid: Grid,
    tt: Timetable,
    latest: list[int],
    deadline: int,
    destination: tuple[float, float],
    max_walk_min: float = 15,
    max_direct_walk_min: float = 40,
) -> np.ndarray:
    """Door-to-door minutes for every grid cell, as uint8 (255 = not reachable)."""
    best = np.full((grid.height, grid.width), np.inf)

    for s, latest_time in enumerate(latest):
        if latest_time != UNREACHABLE:
            _paint(best, grid, tt.station_lats[s], tt.station_lons[s],
                   base_seconds=deadline - latest_time, max_walk_min=max_walk_min)
    _paint(best, grid, destination[0], destination[1],
           base_seconds=0, max_walk_min=max_direct_walk_min)

    minutes = np.ceil(best / 60)
    minutes[~np.isfinite(minutes) | (minutes >= UNREACHABLE_MINUTES)] = UNREACHABLE_MINUTES
    return minutes.astype(np.uint8)


def _paint(best: np.ndarray, grid: Grid, lat: float, lon: float,
           base_seconds: float, max_walk_min: float) -> None:
    """Lower `best` around one point. Only touches cells within walking range,
    which is what keeps a whole-city grid fast (a few hundred cells per
    station instead of a hundred thousand)."""
    reach_m = max_walk_min * 60 * WALK_SPEED / DETOUR            # straight-line radius
    metres_per_lon = METRES_PER_DEGREE * math.cos(math.radians(lat))
    rows = np.nonzero(np.abs(grid.lats - lat) * METRES_PER_DEGREE <= reach_m)[0]
    cols = np.nonzero(np.abs(grid.lons - lon) * metres_per_lon <= reach_m)[0]
    if rows.size == 0 or cols.size == 0:
        return
    r, c = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    dy = (grid.lats[r] - lat) * METRES_PER_DEGREE
    dx = (grid.lons[c] - lon) * metres_per_lon
    straight = np.hypot(dy[:, None], dx[None, :])
    total = base_seconds + straight * DETOUR / WALK_SPEED
    total[straight > reach_m] = np.inf
    np.minimum(best[r, c], total, out=best[r, c])


def _mercator(lat: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

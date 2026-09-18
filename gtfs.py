"""Load a GTFS feed into the compact timetable the router needs.

GTFS is the standard format transit agencies publish schedules in: a zip of
CSV files. The ones that matter here:

    stops.txt        platforms and the stations they belong to
    trips.txt        one row per train run, tagged with a service (Weekday, ...)
    stop_times.txt   when each trip calls at each platform
    transfers.txt    passages between stations and minimum change times
    calendar*.txt    which services run on which dates

The router does not want any of that structure. It wants one flat list of
"connections" (a train leaving station A at time t1 and reaching the next
station B at time t2) sorted by departure time. Building that list is
this module's whole job.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_CHANGE_SECONDS = 120          # used when transfers.txt has no entry for a station
WEEKDAY_NUMBER = {"weekday": 2, "saturday": 5, "sunday": 6}    # Wednesday stands in for weekdays
CALENDAR_COLUMNS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# (departure, arrival, from_station, to_station, trip) with times in seconds since midnight
Connection = tuple[int, int, int, int, int]


@dataclass
class Timetable:
    station_ids: list[str]
    station_names: list[str]
    station_lats: list[float]
    station_lons: list[float]
    connections: list[Connection]                 # sorted by departure time
    departures: list[int]                         # departure column alone, for binary search
    footpaths_from: list[list[tuple[int, int]]]   # station -> [(other station, seconds)]
    footpaths_into: list[list[tuple[int, int]]]   # the same edges, indexed by where they end
    n_trips: int
    service_date: date
    day: str

    def station_index(self, station_id: str) -> int:
        return self.station_ids.index(station_id)


def parse_time(text: str) -> int:
    """'08:30:00' -> 30600. Hours can exceed 23: '25:10:00' is 1:10am on the
    same service day, which is how GTFS writes trips that run past midnight."""
    hours, minutes, seconds = text.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def format_time(seconds: int) -> str:
    seconds = int(seconds) % 86400
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}"


def load_timetable(feed_dir: str | Path, day: str = "weekday", today: date | None = None) -> Timetable:
    feed_dir = Path(feed_dir)
    if day not in WEEKDAY_NUMBER:
        raise ValueError(f"day must be one of {sorted(WEEKDAY_NUMBER)}, got {day!r}")
    if not (feed_dir / "stop_times.txt").exists():
        raise FileNotFoundError(
            f"no GTFS feed found in {feed_dir}. Run `python -m reach.download` first."
        )

    # ---- stations: platforms like 101N / 101S collapse into their parent 101
    station_of: dict[str, str] = {}
    info: dict[str, tuple[str, float, float]] = {}
    for row in _rows(feed_dir / "stops.txt"):
        stop_id = row["stop_id"]
        station_of[stop_id] = (row.get("parent_station") or "").strip() or stop_id
        info[stop_id] = (row["stop_name"], float(row["stop_lat"]), float(row["stop_lon"]))
    station_ids = sorted(set(station_of.values()))
    index = {sid: i for i, sid in enumerate(station_ids)}
    n = len(station_ids)

    # ---- which trips run on the chosen kind of day
    service_date, services = _active_services(feed_dir, day, today or date.today())
    trip_index: dict[str, int] = {}
    for row in _rows(feed_dir / "trips.txt"):
        if row["service_id"] in services:
            trip_index[row["trip_id"]] = len(trip_index)

    # ---- stop_times -> per-trip list of calls
    calls: list[list[tuple[int, int, int, int]]] = [[] for _ in trip_index]
    with open(feed_dir / "stop_times.txt", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        c_trip, c_stop = header.index("trip_id"), header.index("stop_id")
        c_arr, c_dep = header.index("arrival_time"), header.index("departure_time")
        c_seq = header.index("stop_sequence")
        for row in reader:
            trip = trip_index.get(row[c_trip])
            if trip is None or not row[c_arr] or not row[c_dep]:
                continue
            calls[trip].append((
                int(row[c_seq]),
                index[station_of[row[c_stop]]],
                parse_time(row[c_arr]),
                parse_time(row[c_dep]),
            ))

    # ---- consecutive calls of a trip become connections
    raw = []
    for trip, trip_calls in enumerate(calls):
        trip_calls.sort()
        for (seq, a, _, a_dep), (_, b, b_arr, _) in zip(trip_calls, trip_calls[1:]):
            if a != b and b_arr >= a_dep:
                raw.append((a_dep, trip, seq, b_arr, a, b))
    # Sorting by (departure, trip, sequence) keeps a trip's own hops in order
    # even if two of them share a departure time.
    raw.sort()
    connections = [(dep, arr, a, b, trip) for dep, trip, _, arr, a, b in raw]

    # ---- transfers: passages between stations, and change time within one
    change = [DEFAULT_CHANGE_SECONDS] * n
    passages: dict[tuple[int, int], int] = {}
    transfers_file = feed_dir / "transfers.txt"
    if transfers_file.exists():
        for row in _rows(transfers_file):
            a = index.get(station_of.get(row["from_stop_id"], ""))
            b = index.get(station_of.get(row["to_stop_id"], ""))
            if a is None or b is None:
                continue
            seconds = int(row.get("min_transfer_time") or 0) or DEFAULT_CHANGE_SECONDS
            if a == b:
                change[a] = seconds
            else:
                passages[(a, b)] = min(seconds, passages.get((a, b), seconds))

    footpaths_from: list[list[tuple[int, int]]] = [[(s, change[s])] for s in range(n)]
    footpaths_into: list[list[tuple[int, int]]] = [[(s, change[s])] for s in range(n)]
    for (a, b), seconds in passages.items():
        footpaths_from[a].append((b, seconds))
        footpaths_into[b].append((a, seconds))

    return Timetable(
        station_ids=station_ids,
        station_names=[info[s][0] for s in station_ids],
        station_lats=[info[s][1] for s in station_ids],
        station_lons=[info[s][2] for s in station_ids],
        connections=connections,
        departures=[c[0] for c in connections],
        footpaths_from=footpaths_from,
        footpaths_into=footpaths_into,
        n_trips=len(trip_index),
        service_date=service_date,
        day=day,
    )


# --------------------------------------------------------------------------
# Calendar handling
# --------------------------------------------------------------------------

def _active_services(feed_dir: Path, day: str, today: date) -> tuple[date, set[str]]:
    """Pick a representative date for `day` and return the services running on it.

    Preference order: the next matching date from today, then the most recent
    matching date the feed covers (so an out-of-date feed still works). Dates
    with holiday exceptions are skipped when an ordinary date is available.
    """
    calendar = list(_rows(feed_dir / "calendar.txt")) if (feed_dir / "calendar.txt").exists() else []
    exceptions: dict[date, list[tuple[str, str]]] = {}
    if (feed_dir / "calendar_dates.txt").exists():
        for row in _rows(feed_dir / "calendar_dates.txt"):
            exceptions.setdefault(_date(row["date"]), []).append((row["service_id"], row["exception_type"]))

    def services_on(d: date) -> set[str]:
        column = CALENDAR_COLUMNS[d.weekday()]
        active = {
            row["service_id"] for row in calendar
            if row[column] == "1" and _date(row["start_date"]) <= d <= _date(row["end_date"])
        }
        for service, kind in exceptions.get(d, []):
            if kind == "1":
                active.add(service)
            else:
                active.discard(service)
        return active

    wanted = WEEKDAY_NUMBER[day]
    first = today + timedelta(days=(wanted - today.weekday()) % 7)
    candidates = [first + timedelta(weeks=k) for k in range(8)]
    ends = [_date(row["end_date"]) for row in calendar] or list(exceptions) or [today]
    last = max(ends)
    last -= timedelta(days=(last.weekday() - wanted) % 7)
    candidates += [last - timedelta(weeks=k) for k in range(12)]

    running = [(d, services_on(d)) for d in candidates]
    running = [(d, s) for d, s in running if s]
    if not running:
        raise ValueError(f"the feed in {feed_dir} has no {day} service on any date I tried")
    ordinary = [(d, s) for d, s in running if d not in exceptions]
    return (ordinary or running)[0]


def _date(text: str) -> date:
    return datetime.strptime(text.strip(), "%Y%m%d").date()


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)

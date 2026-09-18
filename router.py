"""Routing over the timetable with the Connection Scan Algorithm (CSA).

CSA (Dibbelt, Pajor, Strasser, Wagner, 2013) is the simplest correct way to
route on a timetable. There is no graph and no priority queue. You keep one
label per station and sweep once over the connections in time order:

    "If I could already be on this train, or could be standing at its
     departure station in time, then I can also be wherever it goes next."

The question this project asks is a commute question: "I must be at campus by
9:00. From each station, what is the latest I could walk in and still make
it?" That is CSA run backwards in time from the deadline, and it is
`latest_arrivals` below. `earliest_arrivals` is the ordinary forward version,
used for "leaving at 17:30, how far can I get" and to cross-check the reverse
search in the tests.

Meaning of the transfer times, used identically in both directions:

    Walking into a station, or stepping off a train there, at time t lets you
    board any train leaving that station at t + change_time or later, or any
    train leaving a connected station at t + passage_time or later.
    Staying on the same train costs nothing.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from .gtfs import Timetable

UNREACHABLE = -1            # label for "cannot make the deadline from here"
NEVER = 10**9               # label for "cannot get here"
DEFAULT_HORIZON = 150 * 60  # ignore journeys longer than 2.5 hours


def latest_arrivals(
    tt: Timetable,
    targets: dict[int, int],
    deadline: int,
    horizon: int = DEFAULT_HORIZON,
) -> list[int]:
    """For every station, the latest time you can walk into it and still reach
    the destination by `deadline`. UNREACHABLE (-1) if you cannot.

    `targets` maps station index -> seconds needed to walk from that station
    to the actual destination (a building, not a station).
    """
    latest = [UNREACHABLE] * len(tt.station_ids)
    for station, walk in targets.items():
        latest[station] = max(latest[station], deadline - walk)
    for station, walk in targets.items():                     # passages into a target station
        for other, seconds in tt.footpaths_into[station]:
            if other != station:
                latest[other] = max(latest[other], deadline - walk - seconds)

    connections = tt.connections
    into = tt.footpaths_into
    works = bytearray(tt.n_trips)       # works[trip] = 1 once some later hop of the trip makes the deadline
    first = bisect_left(tt.departures, deadline - horizon)
    last = bisect_right(tt.departures, deadline)

    for i in range(last - 1, first - 1, -1):                   # latest departures first
        dep, arr, frm, to, trip = connections[i]
        if works[trip] or latest[to] >= arr:
            works[trip] = 1
            # Boarding here at `dep` works. So does reaching any connected
            # station early enough to walk over and board.
            for other, seconds in into[frm]:
                if dep - seconds > latest[other]:
                    latest[other] = dep - seconds
    return latest


def earliest_arrivals(
    tt: Timetable,
    sources: dict[int, int],
    depart: int,
    horizon: int = DEFAULT_HORIZON,
) -> list[int]:
    """For every station, the earliest time you can be there if you set off at
    `depart`. NEVER if you cannot get there within the horizon.

    `sources` maps station index -> seconds needed to walk to that station
    from the actual starting point.
    """
    n = len(tt.station_ids)
    arrival = [NEVER] * n       # earliest moment you are at the station
    ready = [NEVER] * n         # earliest moment you can board a train there
    by_train = [NEVER] * n      # earliest arrival by train (decides when to re-relax passages)
    out = tt.footpaths_from

    def arrive(station: int, t: int) -> None:
        if t < arrival[station]:
            arrival[station] = t
        for other, seconds in out[station]:
            if t + seconds < ready[other]:
                ready[other] = t + seconds
            if other != station and t + seconds < arrival[other]:
                arrival[other] = t + seconds

    for station, walk in sources.items():
        arrive(station, depart + walk)

    connections = tt.connections
    on_board = bytearray(tt.n_trips)
    first = bisect_left(tt.departures, depart)
    last = bisect_right(tt.departures, depart + horizon)

    for i in range(first, last):                               # earliest departures first
        dep, arr, frm, to, trip = connections[i]
        if on_board[trip] or ready[frm] <= dep:
            on_board[trip] = 1
            if arr < by_train[to]:
                by_train[to] = arr
                arrive(to, arr)
    return arrival

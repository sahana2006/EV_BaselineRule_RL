from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import traci


@dataclass(frozen=True)
class RouteSignal:
    tl_id: str
    route_index: int
    incoming_edges: tuple[str, ...]
    route_lanes: tuple[str, ...]


@dataclass(frozen=True)
class RouteProgress:
    passed: tuple[RouteSignal, ...]
    upcoming: tuple[RouteSignal, ...]

    @property
    def next_signal(self) -> RouteSignal | None:
        return self.upcoming[0] if self.upcoming else None

    @property
    def remaining_signals(self) -> tuple[RouteSignal, ...]:
        return self.upcoming


def route_lanes_for_tl(route_edges: Sequence[str], tl_id: str) -> list[str]:
    route_set = set(route_edges)
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    return [lane for lane in lanes if lane.rsplit("_", 1)[0] in route_set]


def ordered_route_signals(route_edges: Sequence[str]) -> list[RouteSignal]:
    route_index_by_edge = {edge_id: index for index, edge_id in enumerate(route_edges)}
    route_signals: list[RouteSignal] = []

    for tl_id in traci.trafficlight.getIDList():
        route_lanes = route_lanes_for_tl(route_edges, tl_id)
        if not route_lanes:
            continue

        incoming_edges = sorted({lane.rsplit("_", 1)[0] for lane in route_lanes})
        first_route_index = min(route_index_by_edge[edge_id] for edge_id in incoming_edges if edge_id in route_index_by_edge)
        route_signals.append(
            RouteSignal(
                tl_id=tl_id,
                route_index=first_route_index,
                incoming_edges=tuple(incoming_edges),
                route_lanes=tuple(route_lanes),
            )
        )

    route_signals.sort(key=lambda signal: (signal.route_index, signal.tl_id))
    return route_signals


def ev_has_passed_signal(ev_road: str, signal: RouteSignal) -> bool:
    if ev_road.startswith(":"):
        return False
    return bool(signal.incoming_edges) and ev_road not in signal.incoming_edges


def route_progress_for_vehicle(ev_id: str, route_signals: Sequence[RouteSignal]) -> RouteProgress:
    if ev_id not in traci.vehicle.getIDList():
        return RouteProgress(passed=tuple(), upcoming=tuple(route_signals))

    ev_road = traci.vehicle.getRoadID(ev_id)
    route_index = traci.vehicle.getRouteIndex(ev_id)
    passed: list[RouteSignal] = []
    upcoming: list[RouteSignal] = []

    for signal in route_signals:
        if signal.route_index < route_index:
            passed.append(signal)
            continue
        if signal.route_index > route_index:
            upcoming.append(signal)
            continue

        if ev_road in signal.incoming_edges:
            upcoming.append(signal)
            continue
        if ev_road.startswith(f":{signal.tl_id}"):
            passed.append(signal)
            continue
        if ev_has_passed_signal(ev_road, signal):
            passed.append(signal)
        else:
            upcoming.append(signal)

    return RouteProgress(passed=tuple(passed), upcoming=tuple(upcoming))


def active_emergency_vehicle_ids(candidate_ids: Iterable[str] | None = None) -> list[str]:
    vehicle_ids = candidate_ids if candidate_ids is not None else traci.vehicle.getIDList()
    return [v_id for v_id in vehicle_ids if traci.vehicle.getVehicleClass(v_id) == "emergency"]

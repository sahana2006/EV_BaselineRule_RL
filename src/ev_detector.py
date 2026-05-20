from __future__ import annotations

from typing import Tuple

import traci

from .config import LARGE_ARRIVAL_TIME
from .route_utils import RouteSignal


def ev_to_tl_distance(ev_id: str, tl_lanes: list[str] | tuple[str, ...]) -> float:
    ev_x, ev_y = traci.vehicle.getPosition(ev_id)
    if not tl_lanes:
        return 1e6
    points = [traci.lane.getShape(lane)[-1] for lane in tl_lanes if traci.lane.getShape(lane)]
    if not points:
        return 1e6
    return min(((ev_x - x) ** 2 + (ev_y - y) ** 2) ** 0.5 for x, y in points)


def arrival_time(now: float, distance: float, speed: float) -> float:
    if speed > 0:
        return now + (distance / speed)
    return LARGE_ARRIVAL_TIME


def has_ev_passed_tl(ev_road: str, incoming_lanes: list[str] | tuple[str, ...]) -> bool:
    if ev_road.startswith(":"):
        return False
    incoming_edges = {lane.rsplit("_", 1)[0] for lane in incoming_lanes}
    return bool(incoming_edges) and (ev_road not in incoming_edges)


def signal_distance(ev_id: str, signal: RouteSignal) -> float:
    return ev_to_tl_distance(ev_id, signal.route_lanes)


def colorize_vehicles(
    ev_id: str,
    ev_color: Tuple[int, int, int, int],
    normal_color: Tuple[int, int, int, int],
) -> None:
    for v_id in traci.vehicle.getIDList():
        vehicle_class = traci.vehicle.getVehicleClass(v_id)
        is_emergency = vehicle_class == "emergency" or v_id == ev_id
        traci.vehicle.setColor(v_id, ev_color if is_emergency else normal_color)

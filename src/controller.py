from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import traci

from .config import INTRUSIVE_DISTANCE_THRESHOLD, SATURATION_DISTANCE_THRESHOLD
from .ev_detector import (
    arrival_time,
    signal_distance,
)
from .route_utils import RouteProgress, RouteSignal, ordered_route_signals, route_progress_for_vehicle
from .signal_manager import (
    apply_green_wave,
    infer_ev_green_phase,
    stage1_saturation_reduction,
    stage2_intrusive,
    stage2_non_intrusive,
    stage3_recovery,
)


@dataclass
class TLState:
    default_program: str
    recovery_steps_left: int = 0
    ev_passed: bool = False
    forced_intrusive: bool = False
    last_stage: str = "DEFAULT"


@dataclass
class TrafficSignalController:
    ev_id: str
    mode: str  # fixed_time | intrusive_only | full_model
    route_edges: List[str] = field(default_factory=list)
    route_signals: List[RouteSignal] = field(default_factory=list)
    tl_state: Dict[str, TLState] = field(default_factory=dict)
    _debug_printed_empty: bool = False

    def initialize(self) -> None:
        self.route_edges = traci.vehicle.getRoute(self.ev_id)
        self.route_signals = ordered_route_signals(self.route_edges)
        for signal in self.route_signals:
            tl_id = signal.tl_id
            self.tl_state[tl_id] = TLState(default_program=traci.trafficlight.getProgram(tl_id))
        route_text = " -> ".join(self.route_edges)
        signal_text = " -> ".join(signal.tl_id for signal in self.route_signals) or "(none)"
        print(f"[ROUTE] EV={self.ev_id} edges={route_text}")
        print(f"[ROUTE] EV={self.ev_id} signals={signal_text}")

    def _route_progress(self) -> RouteProgress:
        return route_progress_for_vehicle(self.ev_id, self.route_signals)

    def _log_progress(self, progress: RouteProgress) -> None:
        upcoming_ids = [signal.tl_id for signal in progress.upcoming]
        next_signal = progress.next_signal.tl_id if progress.next_signal else "none"
        if not upcoming_ids and self._debug_printed_empty:
            return
        print(f"[ROUTE] EV={self.ev_id} next_signal={next_signal} upcoming={upcoming_ids}")
        self._debug_printed_empty = not upcoming_ids

    def apply_control(self, now: float) -> Dict[str, float]:
        if self.mode == "fixed_time":
            return {}

        ev_speed = traci.vehicle.getSpeed(self.ev_id)
        progress = self._route_progress()
        self._log_progress(progress)
        arrivals: Dict[str, float] = {}

        for signal in progress.remaining_signals:
            tl_id = signal.tl_id
            state = self.tl_state[tl_id]
            distance = signal_distance(self.ev_id, signal)
            arrivals[tl_id] = arrival_time(now, distance, ev_speed)
            ev_phase = infer_ev_green_phase(tl_id, list(signal.route_lanes))

            if self.mode == "intrusive_only":
                if distance <= INTRUSIVE_DISTANCE_THRESHOLD:
                    stage2_intrusive(tl_id, ev_phase, distance)
                    state.forced_intrusive = True
                elif state.forced_intrusive:
                    state.recovery_steps_left = stage3_recovery(
                        tl_id, state.recovery_steps_left, state.default_program
                    )
                    if state.recovery_steps_left == 0:
                        state.forced_intrusive = False
                continue

            if distance > SATURATION_DISTANCE_THRESHOLD:
                stage1_saturation_reduction(tl_id, traci.vehicle.getRoadID(self.ev_id), distance)
                state.last_stage = "SATURATION"
            elif INTRUSIVE_DISTANCE_THRESHOLD < distance <= SATURATION_DISTANCE_THRESHOLD:
                stage2_non_intrusive(tl_id, ev_phase, distance)
                state.last_stage = "NON_INTRUSIVE"
            elif distance <= INTRUSIVE_DISTANCE_THRESHOLD:
                stage2_intrusive(tl_id, ev_phase, distance)
                state.last_stage = "INTRUSIVE"
                state.forced_intrusive = True

        passed_ids = {signal.tl_id for signal in progress.passed}
        for tl_id, state in self.tl_state.items():
            if tl_id not in passed_ids:
                continue
            if state.ev_passed and state.recovery_steps_left == 0:
                continue
            state.recovery_steps_left = stage3_recovery(
                tl_id, state.recovery_steps_left, state.default_program
            )
            if state.recovery_steps_left == 0:
                state.forced_intrusive = False
                state.ev_passed = True

        if self.mode == "full_model":
            self._apply_green_wave(arrivals, progress)
        return arrivals

    def _apply_green_wave(self, arrivals: Dict[str, float], progress: RouteProgress) -> None:
        if not arrivals or not progress.upcoming:
            return
        for signal in progress.upcoming:
            tl_id = signal.tl_id
            arrival = arrivals.get(tl_id)
            if arrival is None:
                continue
            time_to_arrival = arrival - traci.simulation.getTime()
            if 0 < time_to_arrival <= 12:
                ev_phase = infer_ev_green_phase(tl_id, list(signal.route_lanes))
                apply_green_wave(tl_id, ev_phase, time_to_arrival, arrival)

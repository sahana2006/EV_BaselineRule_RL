from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import traci
from traci import exceptions as traci_exceptions

from .config import STOP_SPEED_THRESHOLD, SimulationConfig
from .ev_detector import (
    signal_distance,
)
from .route_utils import RouteSignal, ordered_route_signals, route_progress_for_vehicle
from .signal_manager import (
    apply_discrete_rl_action,
    infer_ev_green_phase,
)

# Discrete action semantics (global controller: nearest upcoming intersection).
ACTIONS: Dict[int, str] = {
    0: "keep_phase",
    1: "switch_to_ev_green",
    2: "extend_green",
}

ACTION_DIM = len(ACTIONS)


def compute_reward(metrics: Dict[str, Any], queue_weight: float = 0.5, stop_weight: float = 2.0) -> float:
    """Shaped reward from per-step metrics (see TrafficEnv.step for metric keys)."""
    return (
        -float(metrics["ev_waiting_time"])
        - queue_weight * float(metrics["queue_length"])
        - stop_weight * float(metrics["ev_stops"])
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_sumocfg_path(sumocfg: str) -> str:
    if os.path.isabs(sumocfg):
        return sumocfg
    return os.path.join(str(_project_root()), sumocfg)


def _get_sumo_binary(use_gui: bool) -> str:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set. Please set SUMO_HOME before running.")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    from sumolib import checkBinary  # type: ignore

    return checkBinary("sumo-gui" if use_gui else "sumo")


def _approach_direction(lane_id: str, junction_pos: Tuple[float, float]) -> str:
    shape = traci.lane.getShape(lane_id)
    if not shape:
        return "north"

    start_x, start_y = shape[0]
    junction_x, junction_y = junction_pos
    dx = start_x - junction_x
    dy = start_y - junction_y

    # Map each incoming lane to the side from which vehicles approach the junction.
    if abs(dx) >= abs(dy):
        return "west" if dx < 0 else "east"
    return "south" if dy < 0 else "north"


class TrafficEnv:
    """Single global RL agent: actions apply to the nearest traffic light ahead of the EV."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        headless: bool = True,
        max_episode_steps: Optional[int] = None,
        intersection_clear_bonus: float = 2.0,
    ) -> None:
        self.config = config
        self.headless = headless
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else config.max_steps
        self.intersection_clear_bonus = intersection_clear_bonus
        self.ev_id = config.ev_id
        self.step_length = config.step_length

        self._route_signals: List[RouteSignal] = []
        self._neighbor_map: Dict[str, List[str]] = {}
        self._episode_steps = 0
        self._prev_ev_wait = 0.0
        self._prev_ev_stopped = False
        self._focus_tl: Optional[str] = None
        self._started = False
        self._state_debug_counter = 0

    def _close(self) -> None:
        if not self._started:
            return
        try:
            traci.close()
        except traci_exceptions.FatalTraCIError:
            pass
        self._started = False

    def close(self) -> None:
        """Close TraCI connection (safe to call multiple times)."""
        self._close()

    def _start_sumo(self) -> None:
        self._close()
        sumo_binary = _get_sumo_binary(use_gui=not self.headless)
        sumocfg_path = _resolve_sumocfg_path(self.config.sumo_config)
        if not os.path.exists(sumocfg_path):
            raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")
        start_cmd = [sumo_binary, "-c", sumocfg_path, "--step-length", str(self.step_length)]
        if not self.headless:
            start_cmd.append("--start")
        traci.start(start_cmd)
        self._started = True

    def _wait_for_ev(self, max_wait_steps: int = 4000) -> None:
        for _ in range(max_wait_steps):
            try:
                traci.simulationStep()
            except traci_exceptions.FatalTraCIError as err:
                raise RuntimeError(f"SUMO closed while waiting for EV: {err}") from err
            if self.ev_id in traci.vehicle.getIDList():
                return
        raise RuntimeError(f"Vehicle {self.ev_id} did not enter the network within {max_wait_steps} steps.")

    def _cache_route_tls(self) -> None:
        route_edges = traci.vehicle.getRoute(self.ev_id)
        self._route_signals = ordered_route_signals(route_edges)
        self._neighbor_map = self._build_neighbor_map()

    def _build_neighbor_map(self) -> Dict[str, List[str]]:
        tls_ids = list(traci.trafficlight.getIDList())
        positions = {tl_id: traci.junction.getPosition(tl_id) for tl_id in tls_ids}
        positive_axis_distances: list[float] = []

        for tl_id, (x1, y1) in positions.items():
            for other_id, (x2, y2) in positions.items():
                if tl_id == other_id:
                    continue
                if abs(y1 - y2) < 1e-6 and abs(x1 - x2) > 1e-6:
                    positive_axis_distances.append(abs(x1 - x2))
                elif abs(x1 - x2) < 1e-6 and abs(y1 - y2) > 1e-6:
                    positive_axis_distances.append(abs(y1 - y2))

        spacing = min(positive_axis_distances) if positive_axis_distances else 200.0
        tolerance = max(5.0, spacing * 0.15)
        neighbor_map: Dict[str, List[str]] = {}

        for tl_id, (x1, y1) in positions.items():
            neighbors: list[str] = []
            for other_id, (x2, y2) in positions.items():
                if tl_id == other_id:
                    continue
                same_row = abs(y1 - y2) <= tolerance and abs(abs(x1 - x2) - spacing) <= tolerance
                same_col = abs(x1 - x2) <= tolerance and abs(abs(y1 - y2) - spacing) <= tolerance
                if same_row or same_col:
                    neighbors.append(other_id)
            neighbor_map[tl_id] = sorted(neighbors)
        return neighbor_map

    def _nearest_signal_ahead(self) -> Optional[RouteSignal]:
        if self.ev_id not in traci.vehicle.getIDList():
            return None
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals)
        return progress.next_signal

    def _queue_length_network(self) -> int:
        veh_ids = traci.vehicle.getIDList()
        return sum(1 for v_id in veh_ids if traci.vehicle.getSpeed(v_id) < STOP_SPEED_THRESHOLD)

    def _directional_queues(self, tl_id: str) -> Dict[str, int]:
        junction_pos = traci.junction.getPosition(tl_id)
        queues = {"north": 0, "south": 0, "east": 0, "west": 0}
        for lane_id in set(traci.trafficlight.getControlledLanes(tl_id)):
            direction = _approach_direction(lane_id, junction_pos)
            queues[direction] += traci.lane.getLastStepHaltingNumber(lane_id)
        return queues

    def _nearby_vehicle_count(self, tl_id: str, radius: float = 120.0) -> int:
        junction_x, junction_y = traci.junction.getPosition(tl_id)
        nearby_count = 0
        for vehicle_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vehicle_id)
            if ((x - junction_x) ** 2 + (y - junction_y) ** 2) ** 0.5 <= radius:
                nearby_count += 1
        return nearby_count

    def _average_neighbor_queue(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0

        totals = []
        for neighbor_id in neighbors:
            queues = self._directional_queues(neighbor_id)
            totals.append(float(sum(queues.values())))
        return float(sum(totals) / len(totals)) if totals else 0.0

    def _phase_features(self, tl_id: str) -> Tuple[float, float]:
        logic = traci.trafficlight.getAllProgramLogics(tl_id)
        n_phases = max(1, len(logic[0].phases)) if logic else 1
        phase_idx = traci.trafficlight.getPhase(tl_id)
        phase_norm = float(phase_idx) / float(n_phases)
        t = traci.simulation.getTime()
        try:
            next_sw = traci.trafficlight.getNextSwitch(tl_id)
            remaining = max(0.0, float(next_sw) - t)
        except traci_exceptions.TraCIException:
            remaining = 0.0
        remaining_norm = min(1.0, remaining / 120.0)
        return phase_norm, remaining_norm

    def _build_state(self, signal: RouteSignal | None) -> np.ndarray:
        speed = traci.vehicle.getSpeed(self.ev_id)
        speed_norm = float(np.clip(speed / 25.0, 0.0, 1.0))

        if signal is None:
            dist_norm = 0.0
            phase_norm, rem_norm = 0.0, 0.0
            queues = {"north": 0, "south": 0, "east": 0, "west": 0}
            nearby_count = 0
            neighbor_queue_avg = 0.0
            tl_label = "none"
        else:
            # Distance is clipped to the local control horizon around the next signal.
            dist = signal_distance(self.ev_id, signal)
            dist_norm = float(np.clip(dist / 300.0, 0.0, 1.0))
            phase_norm, rem_norm = self._phase_features(signal.tl_id)

            # Queue features are approach-specific halting counts at the controlled intersection.
            queues = self._directional_queues(signal.tl_id)

            # Local density counts vehicles near the active junction only, not the whole city.
            nearby_count = self._nearby_vehicle_count(signal.tl_id)

            # Neighbor awareness is the mean queued vehicles at adjacent traffic lights.
            neighbor_queue_avg = self._average_neighbor_queue(signal.tl_id)
            tl_label = signal.tl_id

        # Normalization keeps all features roughly in [0, 1] for stable DQN training.
        queue_north = float(np.clip(queues["north"] / 20.0, 0.0, 1.0))
        queue_south = float(np.clip(queues["south"] / 20.0, 0.0, 1.0))
        queue_east = float(np.clip(queues["east"] / 20.0, 0.0, 1.0))
        queue_west = float(np.clip(queues["west"] / 20.0, 0.0, 1.0))
        nearby_norm = float(np.clip(nearby_count / 40.0, 0.0, 1.0))
        neighbor_norm = float(np.clip(neighbor_queue_avg / 20.0, 0.0, 1.0))

        state = np.array(
            [
                dist_norm,
                speed_norm,
                queue_north,
                queue_south,
                queue_east,
                queue_west,
                phase_norm,
                rem_norm,
                nearby_norm,
                neighbor_norm,
            ],
            dtype=np.float32,
        )

        self._state_debug_counter += 1
        if self._state_debug_counter <= 5 or self._state_debug_counter % 25 == 0:
            print(
                f"[RL_STATE] tl={tl_label} state={state.tolist()} "
                f"queues={queues} neighbor_avg={neighbor_queue_avg:.2f} nearby={nearby_count}"
            )
        return state

    def get_state(self) -> np.ndarray:
        """Return a compact, normalized local traffic state around the next controlled signal."""
        if self.ev_id not in traci.vehicle.getIDList():
            return np.zeros(self.state_dim, dtype=np.float32)

        signal = self._nearest_signal_ahead()
        return self._build_state(signal)

    @property
    def state_dim(self) -> int:
        # Features:
        # 1. EV distance to next signal
        # 2. EV speed
        # 3-6. directional queues (N/S/E/W)
        # 7. current phase index
        # 8. remaining phase time
        # 9. nearby vehicle density
        # 10. average neighboring-intersection queue
        return 10

    def reset(self) -> np.ndarray:
        self._start_sumo()
        self._wait_for_ev()
        self._cache_route_tls()
        self._episode_steps = 0
        self._prev_ev_wait = traci.vehicle.getWaitingTime(self.ev_id)
        self._prev_ev_stopped = traci.vehicle.getSpeed(self.ev_id) < STOP_SPEED_THRESHOLD
        focus_signal = self._nearest_signal_ahead()
        self._focus_tl = focus_signal.tl_id if focus_signal else None
        self._state_debug_counter = 0
        state = self.get_state()
        print(f"[RL_STATE] state_dim={self.state_dim}")
        print(f"[RL_STATE] example_state={state.tolist()}")
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        done = False
        info: Dict[str, Any] = {}
        clear_bonus = 0.0

        if self.ev_id in traci.vehicle.getIDList():
            focus_signal = self._nearest_signal_ahead()
            self._focus_tl = focus_signal.tl_id if focus_signal else None
            if focus_signal is not None:
                dist = signal_distance(self.ev_id, focus_signal)
                ev_phase = infer_ev_green_phase(focus_signal.tl_id, list(focus_signal.route_lanes))
                apply_discrete_rl_action(focus_signal.tl_id, int(action), ev_phase, dist)

        try:
            traci.simulationStep()
        except traci_exceptions.FatalTraCIError:
            done = True
            return self.get_state(), 0.0, True, {"error": "traci_closed"}

        self._episode_steps += 1

        ev_present = self.ev_id in traci.vehicle.getIDList()
        if self._focus_tl is not None and ev_present:
            progress = route_progress_for_vehicle(self.ev_id, self._route_signals)
            if self._focus_tl in {signal.tl_id for signal in progress.passed}:
                clear_bonus = self.intersection_clear_bonus
                info["cleared_tl"] = self._focus_tl

        wait = traci.vehicle.getWaitingTime(self.ev_id) if ev_present else self._prev_ev_wait
        delta_wait = max(0.0, wait - self._prev_ev_wait) if ev_present else 0.0
        self._prev_ev_wait = wait

        speed = traci.vehicle.getSpeed(self.ev_id) if ev_present else 0.0
        ev_stopped = speed < STOP_SPEED_THRESHOLD
        stop_event = 1.0 if (ev_stopped and not self._prev_ev_stopped and ev_present) else 0.0
        self._prev_ev_stopped = ev_stopped if ev_present else False

        queue_len = self._queue_length_network()
        metrics = {
            "ev_waiting_time": delta_wait,
            "queue_length": float(queue_len),
            "ev_stops": stop_event,
        }
        reward = compute_reward(metrics) + clear_bonus

        if not ev_present:
            done = True
        if self._episode_steps >= self.max_episode_steps:
            done = True
        if traci.simulation.getMinExpectedNumber() == 0 and not ev_present:
            done = True

        return self.get_state(), float(reward), done, info

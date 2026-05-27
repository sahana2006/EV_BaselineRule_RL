from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import traci
from traci import exceptions as traci_exceptions

from .config import STOP_SPEED_THRESHOLD, SimulationConfig
from .ev_detector import arrival_time, signal_distance
from .route_utils import RouteSignal, ordered_route_signals, route_progress_for_vehicle
from .signal_manager import apply_discrete_rl_action, infer_ev_green_phase

ACTIONS: Dict[int, str] = {
    0: "keep_phase",
    1: "switch_to_ev_green",
    2: "extend_green",
}

ACTION_DIM = len(ACTIONS)

CONTROLLER_SINGLE_AGENT = "single_agent"
CONTROLLER_INDEPENDENT_MARL = "independent_marl"
CONTROLLER_COORDINATED_MARL = "coordinated_marl"
CONTROLLER_MULTI_AGENT = CONTROLLER_INDEPENDENT_MARL

BASE_STATE_DIM = 10
COORDINATED_STATE_DIM = 15

REWARD_WEIGHTS: Dict[str, float] = {
    "ev_delay": 2.4,
    "ev_stop": 9.0,
    "low_speed_near_signal": 3.2,
    "queue": 0.14,
    "queue_growth": 0.22,
    "throughput": 0.85,
    "switch": 0.35,
    "intersection_clear": 2.0,
    "neighbor_congestion": 0.10,
    "network_congestion": 0.16,
    "corridor_flow": 0.40,
    "downstream_blockage": 0.18,
    "traffic_stability": 0.16,
}

LOW_SPEED_NEAR_SIGNAL_THRESHOLD = 3.0
NEAR_SIGNAL_DISTANCE_THRESHOLD = 60.0
MAX_EV_FEATURE_DISTANCE = 300.0
DOWNSTREAM_CONGESTION_QUEUE = 12.0


def compute_reward(metrics: Dict[str, Any], weights: Dict[str, float] | None = None) -> tuple[float, Dict[str, float]]:
    """Return total reward plus a readable per-component breakdown."""
    w = REWARD_WEIGHTS if weights is None else weights
    components = {
        "ev_delay_penalty": -w["ev_delay"] * float(metrics["ev_waiting_time"]),
        "ev_stop_penalty": -w["ev_stop"] * float(metrics["ev_stops"]),
        "low_speed_penalty": -w["low_speed_near_signal"] * float(metrics["low_speed_near_signal"]),
        "queue_penalty": -w["queue"] * float(metrics["queue_length"]),
        "queue_growth_penalty": -w["queue_growth"] * float(metrics["queue_growth"]),
        "throughput_reward": w["throughput"] * float(metrics["throughput"]),
        "switch_penalty": -w["switch"] * float(metrics["signal_switch"]),
        "intersection_clear_reward": w["intersection_clear"] * float(metrics["intersection_clear"]),
        "neighbor_congestion_penalty": -w["neighbor_congestion"] * float(metrics["neighbor_congestion"]),
        "network_congestion_penalty": -w["network_congestion"] * float(metrics["network_congestion"]),
        "corridor_flow_reward": w["corridor_flow"] * float(metrics["corridor_flow"]),
        "downstream_blockage_penalty": -w["downstream_blockage"] * float(metrics["downstream_blockage"]),
        "traffic_stability_penalty": -w["traffic_stability"] * float(metrics["traffic_stability"]),
    }
    total = float(sum(components.values()))
    return total, components


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

    if abs(dx) >= abs(dy):
        return "west" if dx < 0 else "east"
    return "south" if dy < 0 else "north"


@dataclass
class AgentContext:
    tl_id: str
    route_lanes: tuple[str, ...]
    ev_phase: int
    ev_distance: float
    ev_distance_norm: float
    phase_norm: float
    remaining_norm: float
    phase_idx: int
    queues: Dict[str, int]
    local_density: int
    local_queue: float
    neighbor_queue_avg: float
    downstream_congestion: float
    neighbor_phase_avg: float
    incoming_traffic_estimate: float
    ev_neighbor_eta: float
    approaching_density: float
    ev_relevant: float
    on_ev_route: bool


class TrafficEnv:
    """
    Practical shared-policy RL environment with three controller layouts:

    - `single_agent`: legacy DQN acting on the next EV intersection only.
    - `independent_marl`: one shared policy, local per-intersection states.
    - `coordinated_marl`: same shared policy, plus neighbor-aware coordination features.

    Coordinated MARL stays scalable because agents exchange only lightweight neighbor
    summaries instead of a giant centralized city state vector.
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        headless: bool = True,
        max_episode_steps: Optional[int] = None,
        intersection_clear_bonus: float = 2.0,
        controller_type: str = CONTROLLER_SINGLE_AGENT,
        traffic_scale: float = 1.0,
    ) -> None:
        self.config = config
        self.headless = headless
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else config.max_steps
        self.intersection_clear_bonus = intersection_clear_bonus
        self.controller_type = controller_type
        self.traffic_scale = traffic_scale
        self.ev_id = config.ev_id
        self.step_length = config.step_length

        self._route_signals: List[RouteSignal] = []
        self._route_signal_by_id: Dict[str, RouteSignal] = {}
        self._neighbor_map: Dict[str, List[str]] = {}
        self._episode_steps = 0
        self._prev_ev_wait = 0.0
        self._prev_ev_stopped = False
        self._prev_network_queue = 0.0
        self._prev_phase_idx: Optional[int] = None
        self._focus_tl: Optional[str] = None
        self._started = False
        self._state_debug_counter = 0
        self._reward_debug_counter = 0
        self._route_log_counter = 0
        self._agent_ids: List[str] = []
        self._prev_agent_queue_totals: Dict[str, float] = {}
        self._prev_agent_phases: Dict[str, int] = {}
        self._last_reward_breakdowns: Dict[str, Dict[str, float]] = {}
        self._last_coordination_terms: Dict[str, Dict[str, float]] = {}

    def _close(self) -> None:
        if not self._started:
            return
        try:
            traci.close()
        except traci_exceptions.FatalTraCIError:
            pass
        self._started = False

    def close(self) -> None:
        self._close()

    def _start_sumo(self) -> None:
        self._close()
        sumo_binary = _get_sumo_binary(use_gui=not self.headless)
        sumocfg_path = _resolve_sumocfg_path(self.config.sumo_config)
        if not os.path.exists(sumocfg_path):
            raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")
        start_cmd = [sumo_binary, "-c", sumocfg_path, "--step-length", str(self.step_length)]
        if self.traffic_scale != 1.0:
            start_cmd.extend(["--scale", str(self.traffic_scale)])
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
        self._route_signal_by_id = {signal.tl_id: signal for signal in self._route_signals}
        self._neighbor_map = self._build_neighbor_map()
        self._agent_ids = sorted(traci.trafficlight.getIDList())

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

    def get_agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    @property
    def is_multi_agent(self) -> bool:
        return self.controller_type in {CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL}

    @property
    def coordination_enabled(self) -> bool:
        return self.controller_type == CONTROLLER_COORDINATED_MARL

    @property
    def shared_policy_enabled(self) -> bool:
        return self.is_multi_agent

    @property
    def state_dim(self) -> int:
        return COORDINATED_STATE_DIM if self.coordination_enabled else BASE_STATE_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def get_neighbor_snapshot(self, tl_id: str) -> Dict[str, float]:
        """
        Coordination hook for future message-passing experiments.

        Right now we expose compact neighbor summaries only, which keeps the design
        scalable for larger grids.
        """
        ctx = self._build_agent_context(tl_id)
        return {
            "neighbor_queue_avg": ctx.neighbor_queue_avg,
            "downstream_congestion": ctx.downstream_congestion,
            "incoming_traffic_estimate": ctx.incoming_traffic_estimate,
            "neighbor_phase_avg": ctx.neighbor_phase_avg,
            "ev_neighbor_eta": ctx.ev_neighbor_eta,
        }

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
        totals = [float(sum(self._directional_queues(neighbor_id).values())) for neighbor_id in neighbors]
        return float(sum(totals) / len(totals)) if totals else 0.0

    def _phase_features(self, tl_id: str) -> Tuple[float, float, int]:
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
        return phase_norm, remaining_norm, phase_idx

    def _distance_to_tl(self, tl_id: str) -> float:
        if self.ev_id not in traci.vehicle.getIDList():
            return MAX_EV_FEATURE_DISTANCE

        route_signal = self._route_signal_by_id.get(tl_id)
        if route_signal is not None:
            try:
                return float(max(0.0, signal_distance(self.ev_id, route_signal)))
            except traci_exceptions.TraCIException:
                pass

        ev_x, ev_y = traci.vehicle.getPosition(self.ev_id)
        tl_x, tl_y = traci.junction.getPosition(tl_id)
        return float(((ev_x - tl_x) ** 2 + (ev_y - tl_y) ** 2) ** 0.5)

    def _ev_relevance(self, distance: float) -> float:
        if distance >= MAX_EV_FEATURE_DISTANCE:
            return 0.0
        return float(1.0 - np.clip(distance / MAX_EV_FEATURE_DISTANCE, 0.0, 1.0))

    def _outgoing_neighbor_pressure(self, tl_id: str) -> float:
        """
        Downstream pressure estimate.

        We approximate spillback risk using neighboring queue totals and local neighbor
        density, which is cheap to compute and scales well across large grids.
        """
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0
        pressure_terms = []
        for neighbor_id in neighbors:
            neighbor_queue = float(sum(self._directional_queues(neighbor_id).values()))
            neighbor_density = float(self._nearby_vehicle_count(neighbor_id))
            pressure_terms.append((neighbor_queue / 20.0) + (neighbor_density / 40.0))
        return float(sum(pressure_terms) / len(pressure_terms))

    def _neighbor_phase_average(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0
        norms = [self._phase_features(neighbor_id)[0] for neighbor_id in neighbors]
        return float(sum(norms) / len(norms)) if norms else 0.0

    def _incoming_traffic_estimate(self, tl_id: str) -> float:
        """
        Lightweight incoming demand prediction.

        We combine upstream queue, neighboring outgoing flow, and approaching density
        instead of adding a separate forecasting model.
        """
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0

        demand_terms: list[float] = []
        for neighbor_id in neighbors:
            neighbor_queue = float(sum(self._directional_queues(neighbor_id).values()))
            neighbor_density = float(self._nearby_vehicle_count(neighbor_id))
            outgoing_flow = 0.0
            for lane_id in set(traci.trafficlight.getControlledLanes(neighbor_id)):
                try:
                    outgoing_flow += float(traci.lane.getLastStepVehicleNumber(lane_id))
                except traci_exceptions.TraCIException:
                    continue
            demand_terms.append((neighbor_queue / 20.0) * 0.45 + (outgoing_flow / 25.0) * 0.35 + (neighbor_density / 40.0) * 0.20)
        return float(sum(demand_terms) / len(demand_terms))

    def _ev_neighbor_eta(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors or self.ev_id not in traci.vehicle.getIDList():
            return 1.0

        ev_speed = traci.vehicle.getSpeed(self.ev_id)
        now = traci.simulation.getTime()
        eta_values: list[float] = []
        for neighbor_id in neighbors:
            distance = self._distance_to_tl(neighbor_id)
            eta = arrival_time(now, distance, max(ev_speed, 0.1)) - now
            eta_values.append(float(np.clip(eta / 120.0, 0.0, 1.0)))
        return float(sum(eta_values) / len(eta_values)) if eta_values else 1.0

    def _build_agent_context(self, tl_id: str) -> AgentContext:
        dist = self._distance_to_tl(tl_id)
        phase_norm, rem_norm, phase_idx = self._phase_features(tl_id)
        queues = self._directional_queues(tl_id)
        local_density = self._nearby_vehicle_count(tl_id)
        local_queue = float(sum(queues.values()))
        neighbor_queue_avg = self._average_neighbor_queue(tl_id)
        downstream_congestion = self._outgoing_neighbor_pressure(tl_id)
        neighbor_phase_avg = self._neighbor_phase_average(tl_id)
        incoming_traffic_estimate = self._incoming_traffic_estimate(tl_id)
        ev_neighbor_eta = self._ev_neighbor_eta(tl_id)
        route_signal = self._route_signal_by_id.get(tl_id)
        route_lanes = route_signal.route_lanes if route_signal is not None else tuple()
        ev_phase = infer_ev_green_phase(tl_id, list(route_lanes))

        return AgentContext(
            tl_id=tl_id,
            route_lanes=route_lanes,
            ev_phase=ev_phase,
            ev_distance=dist,
            ev_distance_norm=float(np.clip(dist / MAX_EV_FEATURE_DISTANCE, 0.0, 1.0)),
            phase_norm=phase_norm,
            remaining_norm=rem_norm,
            phase_idx=phase_idx,
            queues=queues,
            local_density=local_density,
            local_queue=local_queue,
            neighbor_queue_avg=neighbor_queue_avg,
            downstream_congestion=downstream_congestion,
            neighbor_phase_avg=neighbor_phase_avg,
            incoming_traffic_estimate=incoming_traffic_estimate,
            ev_neighbor_eta=ev_neighbor_eta,
            approaching_density=float(local_density) / 40.0,
            ev_relevant=self._ev_relevance(dist),
            on_ev_route=route_signal is not None,
        )

    def _normalize_base_state(self, ctx: AgentContext) -> list[float]:
        speed = traci.vehicle.getSpeed(self.ev_id) if self.ev_id in traci.vehicle.getIDList() else 0.0
        speed_norm = float(np.clip(speed / 25.0, 0.0, 1.0))
        return [
            ctx.ev_distance_norm,
            speed_norm,
            float(np.clip(ctx.queues["north"] / 20.0, 0.0, 1.0)),
            float(np.clip(ctx.queues["south"] / 20.0, 0.0, 1.0)),
            float(np.clip(ctx.queues["east"] / 20.0, 0.0, 1.0)),
            float(np.clip(ctx.queues["west"] / 20.0, 0.0, 1.0)),
            ctx.phase_norm,
            ctx.remaining_norm,
            float(np.clip(ctx.local_density / 40.0, 0.0, 1.0)),
            float(np.clip(ctx.neighbor_queue_avg / 20.0, 0.0, 1.0)),
        ]

    def _state_from_context(self, ctx: AgentContext) -> np.ndarray:
        state_values = self._normalize_base_state(ctx)
        if self.coordination_enabled:
            # Coordination features expose only compact neighbor summaries so each
            # intersection can reason about corridor health without a centralized state.
            state_values.extend(
                [
                    float(np.clip(ctx.downstream_congestion / 2.0, 0.0, 1.0)),
                    float(np.clip(ctx.neighbor_phase_avg, 0.0, 1.0)),
                    float(np.clip(ctx.incoming_traffic_estimate, 0.0, 1.0)),
                    float(np.clip(ctx.ev_neighbor_eta, 0.0, 1.0)),
                    float(np.clip(ctx.approaching_density, 0.0, 1.0)),
                ]
            )
        return np.asarray(state_values, dtype=np.float32)

    def _log_state(self, tl_id: str, state: np.ndarray, ctx: AgentContext) -> None:
        self._state_debug_counter += 1
        if self._state_debug_counter <= 10 or self._state_debug_counter % 60 == 0:
            print(
                f"[RL_STATE] agent={tl_id} state={state.tolist()} local_queue={ctx.local_queue:.1f} "
                f"neighbor_queue={ctx.neighbor_queue_avg:.2f} downstream={ctx.downstream_congestion:.2f} "
                f"incoming={ctx.incoming_traffic_estimate:.2f} neighbor_phase={ctx.neighbor_phase_avg:.2f} "
                f"ev_eta_neighbor={ctx.ev_neighbor_eta:.2f}"
            )

    def _route_debug(self) -> None:
        if self.ev_id not in traci.vehicle.getIDList():
            return
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals)
        self._route_log_counter += 1
        if self._route_log_counter <= 5 or self._route_log_counter % 20 == 0:
            print(
                f"[EV_ROUTE] ev={self.ev_id} next={progress.next_signal.tl_id if progress.next_signal else 'none'} "
                f"upcoming={[signal.tl_id for signal in progress.upcoming]} "
                f"passed={[signal.tl_id for signal in progress.passed]}"
            )

    def _build_all_states(self) -> Dict[str, np.ndarray]:
        states: Dict[str, np.ndarray] = {}
        for tl_id in self._agent_ids:
            ctx = self._build_agent_context(tl_id)
            state = self._state_from_context(ctx)
            states[tl_id] = state
            self._log_state(tl_id, state, ctx)
        return states

    def get_state(self) -> np.ndarray | Dict[str, np.ndarray]:
        if self.is_multi_agent:
            if self.ev_id not in traci.vehicle.getIDList() and not self._started:
                return {tl_id: np.zeros(self.state_dim, dtype=np.float32) for tl_id in self._agent_ids}
            return self._build_all_states()

        if self.ev_id not in traci.vehicle.getIDList():
            return np.zeros(self.state_dim, dtype=np.float32)

        signal = self._nearest_signal_ahead()
        if signal is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        ctx = self._build_agent_context(signal.tl_id)
        state = self._state_from_context(ctx)
        self._log_state(signal.tl_id, state, ctx)
        return state

    def reset(self) -> np.ndarray | Dict[str, np.ndarray]:
        self._start_sumo()
        self._wait_for_ev()
        self._cache_route_tls()
        self._episode_steps = 0
        self._prev_ev_wait = traci.vehicle.getWaitingTime(self.ev_id)
        self._prev_ev_stopped = traci.vehicle.getSpeed(self.ev_id) < STOP_SPEED_THRESHOLD
        self._prev_network_queue = float(self._queue_length_network())
        focus_signal = self._nearest_signal_ahead()
        self._focus_tl = focus_signal.tl_id if focus_signal else None
        self._prev_phase_idx = traci.trafficlight.getPhase(self._focus_tl) if self._focus_tl else None
        self._state_debug_counter = 0
        self._reward_debug_counter = 0
        self._route_log_counter = 0
        self._prev_agent_queue_totals = {
            tl_id: float(sum(self._directional_queues(tl_id).values())) for tl_id in self._agent_ids
        }
        self._prev_agent_phases = {tl_id: traci.trafficlight.getPhase(tl_id) for tl_id in self._agent_ids}
        self._last_reward_breakdowns = {}
        self._last_coordination_terms = {}
        state = self.get_state()

        print(
            f"[RL_INIT] controller_type={self.controller_type} agents={len(self._agent_ids)} "
            f"shared_policy={self.shared_policy_enabled} state_dim={self.state_dim} action_dim={self.action_dim} "
            f"coordination={self.coordination_enabled} traffic_scale={self.traffic_scale:.2f}"
        )
        print(f"[RL_INIT] active_intersections={self._agent_ids}")
        print(f"[RL_REWARD] weights={REWARD_WEIGHTS}")
        self._route_debug()
        return state

    def _network_congestion_ratio(self) -> float:
        vehicle_count = max(len(traci.vehicle.getIDList()), 1)
        return float(self._queue_length_network()) / float(vehicle_count)

    def _corridor_flow_score(self, ctx: AgentContext) -> float:
        queue_relief = max(0.0, 1.0 - np.clip(ctx.local_queue / 20.0, 0.0, 1.0))
        downstream_safe = max(0.0, 1.0 - np.clip(ctx.downstream_congestion / 2.0, 0.0, 1.0))
        incoming_balanced = max(0.0, 1.0 - np.clip(abs(ctx.incoming_traffic_estimate - ctx.approaching_density), 0.0, 1.0))
        return float((queue_relief * 0.45) + (downstream_safe * 0.35) + (incoming_balanced * 0.20))

    def _coordination_terms(self, ctx: AgentContext, switch_event: float) -> Dict[str, float]:
        network_congestion = self._network_congestion_ratio()
        downstream_blockage = float(np.clip(ctx.downstream_congestion / 2.0, 0.0, 1.0))
        traffic_stability = float(np.clip(abs(ctx.incoming_traffic_estimate - ctx.approaching_density), 0.0, 1.0))
        if ctx.downstream_congestion >= 1.0:
            traffic_stability = min(1.0, traffic_stability + (0.25 * switch_event))
        corridor_flow = self._corridor_flow_score(ctx)
        return {
            "network_congestion": network_congestion,
            "corridor_flow": corridor_flow,
            "downstream_blockage": downstream_blockage,
            "traffic_stability": traffic_stability,
        }

    def _compute_agent_rewards(
        self,
        actions: Dict[str, int],
        previous_passed: set[str],
    ) -> Dict[str, float]:
        ev_present = self.ev_id in traci.vehicle.getIDList()
        wait = traci.vehicle.getWaitingTime(self.ev_id) if ev_present else self._prev_ev_wait
        delta_wait = max(0.0, wait - self._prev_ev_wait) if ev_present else 0.0
        self._prev_ev_wait = wait

        speed = traci.vehicle.getSpeed(self.ev_id) if ev_present else 0.0
        ev_stopped = speed < STOP_SPEED_THRESHOLD
        stop_event = 1.0 if (ev_stopped and not self._prev_ev_stopped and ev_present) else 0.0
        self._prev_ev_stopped = ev_stopped if ev_present else False

        queue_len = float(self._queue_length_network())
        throughput = float(traci.simulation.getArrivedNumber())
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if ev_present else None
        passed_now = {signal.tl_id for signal in progress.passed} if progress is not None else set()

        rewards: Dict[str, float] = {}
        self._last_reward_breakdowns = {}
        self._last_coordination_terms = {}

        for tl_id in self._agent_ids:
            ctx = self._build_agent_context(tl_id)
            local_queue = ctx.local_queue
            prev_local_queue = self._prev_agent_queue_totals.get(tl_id, local_queue)
            queue_growth = max(0.0, local_queue - prev_local_queue)
            self._prev_agent_queue_totals[tl_id] = local_queue

            phase_idx = traci.trafficlight.getPhase(tl_id)
            prev_phase = self._prev_agent_phases.get(tl_id, phase_idx)
            switch_event = 1.0 if phase_idx != prev_phase else 0.0
            self._prev_agent_phases[tl_id] = phase_idx

            low_speed_near_signal = 0.0
            if ev_present and ctx.ev_distance <= NEAR_SIGNAL_DISTANCE_THRESHOLD and speed < LOW_SPEED_NEAR_SIGNAL_THRESHOLD:
                low_speed_near_signal = ((LOW_SPEED_NEAR_SIGNAL_THRESHOLD - speed) / LOW_SPEED_NEAR_SIGNAL_THRESHOLD) * ctx.ev_relevant

            clear_bonus = self.intersection_clear_bonus if tl_id in (passed_now - previous_passed) else 0.0
            throughput_share = throughput / max(len(self._agent_ids), 1)

            metrics = {
                "ev_waiting_time": delta_wait * max(ctx.ev_relevant, 0.25 if ctx.on_ev_route else 0.0),
                "queue_length": local_queue,
                "queue_growth": queue_growth,
                "ev_stops": stop_event * ctx.ev_relevant,
                "low_speed_near_signal": low_speed_near_signal,
                "throughput": throughput_share,
                "signal_switch": switch_event,
                "intersection_clear": clear_bonus / max(self.intersection_clear_bonus, 1e-6),
                "neighbor_congestion": ctx.neighbor_queue_avg,
                "network_congestion": queue_len / max(len(traci.vehicle.getIDList()), 1),
                "corridor_flow": 0.0,
                "downstream_blockage": 0.0,
                "traffic_stability": 0.0,
            }
            coordination_terms = self._coordination_terms(ctx, switch_event) if self.coordination_enabled else {
                "network_congestion": 0.0,
                "corridor_flow": 0.0,
                "downstream_blockage": 0.0,
                "traffic_stability": 0.0,
            }
            metrics.update(coordination_terms)

            reward, breakdown = compute_reward(metrics)
            rewards[tl_id] = reward
            self._last_reward_breakdowns[tl_id] = breakdown
            self._last_coordination_terms[tl_id] = coordination_terms

            self._reward_debug_counter += 1
            if self._reward_debug_counter <= 16 or self._reward_debug_counter % 60 == 0:
                print(
                    f"[RL_REWARD] agent={tl_id} action={actions.get(tl_id, 0)} local_queue={local_queue:.1f} "
                    f"neighbor_queue={ctx.neighbor_queue_avg:.2f} downstream={ctx.downstream_congestion:.2f} "
                    f"incoming={ctx.incoming_traffic_estimate:.2f} reward={reward:.3f} "
                    f"coord={coordination_terms} breakdown={breakdown}"
                )

        self._prev_network_queue = queue_len
        return rewards

    def _done(self) -> bool:
        ev_present = self.ev_id in traci.vehicle.getIDList()
        if not ev_present:
            return True
        if self._episode_steps >= self.max_episode_steps:
            return True
        if traci.simulation.getMinExpectedNumber() == 0 and not ev_present:
            return True
        return False

    def _step_multi(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        self._route_debug()
        previous_progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if self.ev_id in traci.vehicle.getIDList() else None
        previous_passed = {signal.tl_id for signal in previous_progress.passed} if previous_progress is not None else set()

        action_log: Dict[str, Dict[str, float]] = {}
        for tl_id in self._agent_ids:
            ctx = self._build_agent_context(tl_id)
            action = int(actions.get(tl_id, 0))

            # Coordinated MARL becomes conservative when the downstream corridor is
            # already saturated, which helps reduce spillback and oscillations.
            if self.coordination_enabled and ctx.downstream_congestion >= 1.0 and action == 2:
                action = 0
            apply_discrete_rl_action(tl_id, action, ctx.ev_phase, ctx.ev_distance)
            action_log[tl_id] = {
                "action": float(action),
                "local_queue": ctx.local_queue,
                "neighbor_queue": ctx.neighbor_queue_avg,
                "downstream_congestion": ctx.downstream_congestion,
                "incoming_traffic_estimate": ctx.incoming_traffic_estimate,
            }
            actions[tl_id] = action

        print(f"[RL_ACTION] active_agents={len(self._agent_ids)} actions={action_log}")

        try:
            traci.simulationStep()
        except traci_exceptions.FatalTraCIError:
            zero_states = {tl_id: np.zeros(self.state_dim, dtype=np.float32) for tl_id in self._agent_ids}
            zero_rewards = {tl_id: 0.0 for tl_id in self._agent_ids}
            return zero_states, zero_rewards, True, {"error": "traci_closed"}

        self._episode_steps += 1
        rewards = self._compute_agent_rewards(actions, previous_passed)
        next_states = self.get_state()
        done = self._done()
        info = {
            "active_intersections": list(self._agent_ids),
            "reward_breakdowns": self._last_reward_breakdowns,
            "coordination_terms": self._last_coordination_terms,
        }
        return next_states, rewards, done, info

    def _step_single(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        done = False
        info: Dict[str, Any] = {}
        clear_bonus = 0.0
        switch_event = 0.0

        if self.ev_id in traci.vehicle.getIDList():
            self._route_debug()
            focus_signal = self._nearest_signal_ahead()
            self._focus_tl = focus_signal.tl_id if focus_signal else None
            if focus_signal is not None:
                current_phase = traci.trafficlight.getPhase(focus_signal.tl_id)
                if self._prev_phase_idx is not None and current_phase != self._prev_phase_idx:
                    switch_event = 1.0
                dist = signal_distance(self.ev_id, focus_signal)
                ev_phase = infer_ev_green_phase(focus_signal.tl_id, list(focus_signal.route_lanes))
                apply_discrete_rl_action(focus_signal.tl_id, int(action), ev_phase, dist)
                self._prev_phase_idx = traci.trafficlight.getPhase(focus_signal.tl_id)
                print(f"[RL_ACTION] agent={focus_signal.tl_id} action={action}")
            else:
                self._prev_phase_idx = None

        previous_progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if self.ev_id in traci.vehicle.getIDList() else None
        previous_passed = {signal.tl_id for signal in previous_progress.passed} if previous_progress is not None else set()

        try:
            traci.simulationStep()
        except traci_exceptions.FatalTraCIError:
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {"error": "traci_closed"}

        self._episode_steps += 1
        ev_present = self.ev_id in traci.vehicle.getIDList()
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if ev_present else None
        passed_now = {signal.tl_id for signal in progress.passed} if progress is not None else set()

        if self._focus_tl is not None and self._focus_tl in (passed_now - previous_passed):
            clear_bonus = self.intersection_clear_bonus
            info["cleared_tl"] = self._focus_tl

        wait = traci.vehicle.getWaitingTime(self.ev_id) if ev_present else self._prev_ev_wait
        delta_wait = max(0.0, wait - self._prev_ev_wait) if ev_present else 0.0
        self._prev_ev_wait = wait

        speed = traci.vehicle.getSpeed(self.ev_id) if ev_present else 0.0
        ev_stopped = speed < STOP_SPEED_THRESHOLD
        stop_event = 1.0 if (ev_stopped and not self._prev_ev_stopped and ev_present) else 0.0
        self._prev_ev_stopped = ev_stopped if ev_present else False

        queue_len = float(self._queue_length_network())
        queue_growth = max(0.0, queue_len - self._prev_network_queue)
        self._prev_network_queue = queue_len
        throughput = float(traci.simulation.getArrivedNumber())

        low_speed_near_signal = 0.0
        neighbor_congestion = 0.0
        if ev_present and self._focus_tl is not None:
            dist = self._distance_to_tl(self._focus_tl)
            neighbor_congestion = self._average_neighbor_queue(self._focus_tl)
            if dist <= NEAR_SIGNAL_DISTANCE_THRESHOLD and speed < LOW_SPEED_NEAR_SIGNAL_THRESHOLD:
                low_speed_near_signal = (LOW_SPEED_NEAR_SIGNAL_THRESHOLD - speed) / LOW_SPEED_NEAR_SIGNAL_THRESHOLD

        metrics = {
            "ev_waiting_time": delta_wait,
            "queue_length": queue_len,
            "queue_growth": queue_growth,
            "ev_stops": stop_event,
            "low_speed_near_signal": low_speed_near_signal,
            "throughput": throughput,
            "signal_switch": switch_event,
            "intersection_clear": clear_bonus / max(self.intersection_clear_bonus, 1e-6),
            "neighbor_congestion": neighbor_congestion,
            "network_congestion": queue_len / max(len(traci.vehicle.getIDList()), 1),
            "corridor_flow": 0.0,
            "downstream_blockage": 0.0,
            "traffic_stability": 0.0,
        }
        reward, breakdown = compute_reward(metrics)

        self._reward_debug_counter += 1
        if self._reward_debug_counter <= 8 or self._reward_debug_counter % 25 == 0:
            print(f"[RL_REWARD] agent={self._focus_tl or 'none'} action={action} reward={reward:.3f} breakdown={breakdown}")

        if not ev_present:
            done = True
        if self._episode_steps >= self.max_episode_steps:
            done = True
        if traci.simulation.getMinExpectedNumber() == 0 and not ev_present:
            done = True

        next_state = self.get_state()
        return next_state, float(reward), done, info  # type: ignore[return-value]

    def step(
        self,
        action: int | Dict[str, int],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]] | Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        if self.is_multi_agent:
            if not isinstance(action, dict):
                raise TypeError("Multi-agent environment expects a dictionary of actions keyed by traffic light id.")
            return self._step_multi(action)
        if isinstance(action, dict):
            raise TypeError("Single-agent environment expects a scalar action.")
        return self._step_single(action)

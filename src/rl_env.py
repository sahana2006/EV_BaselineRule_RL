from __future__ import annotations

import os
import sys
import time
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
CONTROLLER_GLOBAL_PPO = "global_ppo"
CONTROLLER_COORDINATED_PPO = "coordinated_ppo"
CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO = "adaptive_reward_coordinated_ppo"
CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO = "congestion_aware_coordinated_ppo"
CONTROLLER_MULTI_LEVEL_COORDINATED_PPO = "multi_level_coordinated_ppo"
CONTROLLER_MULTI_LEVEL_COORDINATED_DQN = "multi_level_coordinated_dqn"
CONTROLLER_INDEPENDENT_MARL = "independent_marl"
CONTROLLER_COORDINATED_MARL = "coordinated_marl"
CONTROLLER_MULTI_AGENT = CONTROLLER_INDEPENDENT_MARL

BASE_STATE_DIM = 10
COORDINATED_STATE_DIM = 15
MULTI_LEVEL_COORDINATED_STATE_DIM = 29
CONGESTION_AWARE_COORDINATED_STATE_DIM = 26

REWARD_NORMALIZATION_SCALES: Dict[str, float] = {
    "ev_waiting_time": 30.0,
    "ev_stops": 1.0,
    "low_speed_near_signal": 1.0,
    "queue_length": 20.0,
    "queue_growth": 10.0,
    "throughput": 10.0,
    "signal_switch": 1.0,
    "intersection_clear": 1.0,
    "neighbor_congestion": 20.0,
    "network_congestion": 1.0,
    "corridor_flow": 1.0,
    "downstream_blockage": 1.0,
    "traffic_stability": 1.0,
    "congestion_reduction": 1.0,
    "queue_stabilization": 1.0,
    "spillback_prevention": 1.0,
    "congestion_transfer": 1.0,
    "corridor_congestion": 1.0,
    "congestion_imbalance": 1.0,
    "congestion_trend": 1.0,
}

REWARD_WEIGHTS: Dict[str, float] = {
    "ev_delay": 1.92,
    "ev_stop": 7.2,
    "low_speed_near_signal": 2.56,
    "queue": 0.28,
    "queue_growth": 0.44,
    "throughput": 1.275,
    "switch": 0.35,
    "intersection_clear": 3.0,
    "neighbor_congestion": 0.20,
    "network_congestion": 0.32,
    "corridor_flow": 0.40,
    "downstream_blockage": 0.36,
    "traffic_stability": 0.24,
    "congestion_reduction": 0.90,
    "queue_stabilization": 0.70,
    "spillback_prevention": 1.00,
    "congestion_transfer": 0.55,
    "corridor_congestion": 0.60,
    "congestion_imbalance": 0.35,
    "congestion_trend": 0.45,
}

LOW_SPEED_NEAR_SIGNAL_THRESHOLD = 3.0
NEAR_SIGNAL_DISTANCE_THRESHOLD = 60.0
MAX_EV_FEATURE_DISTANCE = 300.0
DOWNSTREAM_CONGESTION_QUEUE = 12.0

EV_REWARD_COMPONENT_KEYS = frozenset(
    {
        "ev_delay_penalty",
        "ev_stop_penalty",
        "low_speed_penalty",
        "intersection_clear_reward",
    }
)

ADAPTIVE_CONGESTION_LOW_THRESHOLD = 0.33
ADAPTIVE_CONGESTION_HIGH_THRESHOLD = 0.66
ADAPTIVE_REWARD_WEIGHTS_BY_LEVEL: Dict[str, tuple[float, float]] = {
    "low": (0.7, 0.3),
    "medium": (0.5, 0.5),
    "high": (0.3, 0.7),
}


def classify_congestion_level(congestion_index: float) -> str:
    if congestion_index < ADAPTIVE_CONGESTION_LOW_THRESHOLD:
        return "low"
    if congestion_index < ADAPTIVE_CONGESTION_HIGH_THRESHOLD:
        return "medium"
    return "high"


def adaptive_scales_for_level(level: str) -> tuple[float, float]:
    return ADAPTIVE_REWARD_WEIGHTS_BY_LEVEL.get(level, ADAPTIVE_REWARD_WEIGHTS_BY_LEVEL["medium"])


def _normalized_reward_metric(metrics: Dict[str, Any], key: str) -> float:
    scale = REWARD_NORMALIZATION_SCALES.get(key, 1.0)
    if scale <= 0:
        return float(metrics[key])
    return float(np.clip(float(metrics[key]) / scale, 0.0, 1.0))


def compute_reward(
    metrics: Dict[str, Any],
    weights: Dict[str, float] | None = None,
    *,
    adaptive_scales: tuple[float, float] | None = None,
) -> tuple[float, Dict[str, float]]:
    """Return total reward plus a readable per-component breakdown."""
    w = REWARD_WEIGHTS if weights is None else weights

    ev_delay = _normalized_reward_metric(metrics, "ev_waiting_time")
    ev_stop = _normalized_reward_metric(metrics, "ev_stops")
    low_speed = _normalized_reward_metric(metrics, "low_speed_near_signal")
    queue_length = _normalized_reward_metric(metrics, "queue_length")
    queue_growth = _normalized_reward_metric(metrics, "queue_growth")
    throughput = _normalized_reward_metric(metrics, "throughput")
    signal_switch = _normalized_reward_metric(metrics, "signal_switch")
    intersection_clear = _normalized_reward_metric(metrics, "intersection_clear")
    neighbor_congestion = _normalized_reward_metric(metrics, "neighbor_congestion")
    network_congestion = _normalized_reward_metric(metrics, "network_congestion")
    corridor_flow = _normalized_reward_metric(metrics, "corridor_flow")
    downstream_blockage = _normalized_reward_metric(metrics, "downstream_blockage")
    traffic_stability = _normalized_reward_metric(metrics, "traffic_stability")
    congestion_reduction = _normalized_reward_metric(metrics, "congestion_reduction")
    queue_stabilization = _normalized_reward_metric(metrics, "queue_stabilization")
    spillback_prevention = _normalized_reward_metric(metrics, "spillback_prevention")
    congestion_transfer = _normalized_reward_metric(metrics, "congestion_transfer")
    corridor_congestion = _normalized_reward_metric(metrics, "corridor_congestion")
    congestion_imbalance = _normalized_reward_metric(metrics, "congestion_imbalance")
    congestion_trend = _normalized_reward_metric(metrics, "congestion_trend")

    components = {
        "ev_delay_penalty": -w["ev_delay"] * ev_delay,
        "ev_stop_penalty": -w["ev_stop"] * ev_stop,
        "low_speed_penalty": -w["low_speed_near_signal"] * low_speed,
        "queue_penalty": -w["queue"] * queue_length,
        "queue_growth_penalty": -w["queue_growth"] * queue_growth,
        "throughput_reward": w["throughput"] * throughput,
        "switch_penalty": -w["switch"] * signal_switch,
        "intersection_clear_reward": w["intersection_clear"] * intersection_clear,
        "neighbor_congestion_penalty": -w["neighbor_congestion"] * neighbor_congestion,
        "network_congestion_penalty": -w["network_congestion"] * network_congestion,
        "corridor_flow_reward": w["corridor_flow"] * corridor_flow,
        "downstream_blockage_penalty": -w["downstream_blockage"] * downstream_blockage,
        "traffic_stability_penalty": -w["traffic_stability"] * traffic_stability,
        "congestion_reduction_reward": w["congestion_reduction"] * congestion_reduction,
        "queue_stabilization_reward": w["queue_stabilization"] * queue_stabilization,
        "spillback_prevention_reward": w["spillback_prevention"] * spillback_prevention,
        "congestion_transfer_penalty": -w["congestion_transfer"] * congestion_transfer,
        "corridor_congestion_penalty": -w["corridor_congestion"] * corridor_congestion,
        "congestion_imbalance_penalty": -w["congestion_imbalance"] * congestion_imbalance,
        "congestion_trend_penalty": -w["congestion_trend"] * congestion_trend,
    }
    anti_gridlock_penalty = 0.0
    if float(metrics["network_congestion"]) > 0.30:
        anti_gridlock_penalty -= min(1.0, (float(metrics["network_congestion"]) - 0.30) * 2.5)
    if float(metrics["queue_length"]) > 20.0:
        anti_gridlock_penalty -= min(1.0, (float(metrics["queue_length"]) - 20.0) / 20.0)
    components["anti_gridlock_penalty"] = anti_gridlock_penalty

    ev_sum = float(sum(components[key] for key in EV_REWARD_COMPONENT_KEYS))
    network_sum = float(sum(value for key, value in components.items() if key not in EV_REWARD_COMPONENT_KEYS))

    if adaptive_scales is not None:
        ev_weight, network_weight = adaptive_scales
        total = float(ev_weight * ev_sum + network_weight * network_sum)
        components["_ev_reward_component"] = ev_sum
        components["_network_reward_component"] = network_sum
        components["_adaptive_ev_weight"] = float(ev_weight)
        components["_adaptive_network_weight"] = float(network_weight)
    else:
        total = float(ev_sum + network_sum)

    return total, components


def complete_reward_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Fill any missing reward inputs with 0.0 so controller-specific paths can
    safely call compute_reward() before every metric is available.
    """
    complete: Dict[str, float] = {
        "ev_waiting_time": 0.0,
        "queue_length": 0.0,
        "queue_growth": 0.0,
        "ev_stops": 0.0,
        "low_speed_near_signal": 0.0,
        "throughput": 0.0,
        "signal_switch": 0.0,
        "intersection_clear": 0.0,
        "neighbor_congestion": 0.0,
        "network_congestion": 0.0,
        "corridor_flow": 0.0,
        "downstream_blockage": 0.0,
        "traffic_stability": 0.0,
        "congestion_reduction": 0.0,
        "queue_stabilization": 0.0,
        "spillback_prevention": 0.0,
        "congestion_transfer": 0.0,
        "corridor_congestion": 0.0,
        "congestion_imbalance": 0.0,
        "congestion_trend": 0.0,
    }
    for key, value in metrics.items():
        if key in complete:
            complete[key] = float(value)
    return complete


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
    neighbor_queue_std: float
    downstream_congestion: float
    neighbor_phase_avg: float
    incoming_traffic_estimate: float
    ev_neighbor_eta: float
    approaching_density: float
    neighbor_waiting_time: float
    neighbor_throughput_estimate: float
    downstream_spillback_indicator: float
    neighbor_emergency_vehicle_present: float
    ev_relevant: float
    on_ev_route: bool
    local_congestion_index: float
    neighbor_congestion_index: float
    downstream_blockage_ratio: float
    queue_growth_rate: float
    corridor_congestion_pressure: float
    network_congestion_score: float
    congestion_imbalance_ns: float
    congestion_imbalance_ew: float
    congestion_trend_local: float
    congestion_trend_network: float
    congestion_transfer_risk: float


@dataclass
class StepCache:
    sim_time: float
    vehicle_ids: tuple[str, ...]
    vehicle_set: set[str]
    vehicle_positions: Dict[str, tuple[float, float]]
    vehicle_speeds: Dict[str, float]
    vehicle_waiting_times: Dict[str, float]
    queue_length_network: int
    throughput: float
    ev_present: bool
    ev_position: tuple[float, float] | None
    ev_speed: float
    ev_wait: float
    ev_road: str
    ev_route_index: int
    route_progress: RouteProgress | None
    progress_upcoming_ids: set[str]
    progress_passed_ids: set[str]
    trafficlight_phases: Dict[str, int]
    trafficlight_next_switch: Dict[str, float]
    lane_halting_numbers: Dict[str, int]
    lane_vehicle_numbers: Dict[str, int]
    lane_waiting_times: Dict[str, float]


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
        debug_logs: bool = False,
        profile_runtime: bool = False,
    ) -> None:
        self.config = config
        self.headless = headless
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else config.max_steps
        self.intersection_clear_bonus = intersection_clear_bonus
        self.debug_logs = debug_logs
        self.profile_runtime = profile_runtime
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
        self._controlled_lanes_by_tl: Dict[str, tuple[str, ...]] = {}
        self._junction_positions: Dict[str, tuple[float, float]] = {}
        self._lane_directions_by_tl: Dict[str, Dict[str, str]] = {}
        self._phase_count_by_tl: Dict[str, int] = {}
        self._lane_endpoints: Dict[str, tuple[float, float]] = {}
        self._ev_phase_by_tl: Dict[str, int] = {}
        self._step_cache: StepCache | None = None
        self._profile_totals: Dict[str, float] = {
            'action_application': 0.0,
            'reward_computation': 0.0,
            'state_construction': 0.0,
            'step_total': 0.0,
        }
        self._profile_steps = 0
        self._prev_agent_queue_totals: Dict[str, float] = {}
        self._prev_agent_phases: Dict[str, int] = {}
        self._last_reward_breakdowns: Dict[str, Dict[str, float]] = {}
        self._last_coordination_terms: Dict[str, Dict[str, float]] = {}
        self._last_congestion_features: Dict[str, Dict[str, float]] = {}
        self._last_multi_level_diagnostics: Dict[str, Dict[str, float]] = {}
        self._reward_weights_logged = False
        self.debug_congestion_features = self.debug_logs
        self._congestion_feature_seen_nonzero = False
        self._congestion_feature_warning_logged = False
        self._adaptive_mode_steps: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        self._adaptive_ev_weight_sum = 0.0
        self._adaptive_network_weight_sum = 0.0
        self._adaptive_step_count = 0
        self._adaptive_ev_component_sum = 0.0
        self._adaptive_network_component_sum = 0.0
        self._adaptive_congestion_index_sum = 0.0
        self._last_adaptive_congestion_level = "medium"
        self._last_adaptive_congestion_index = 0.0
        self._last_adaptive_reward_info: Dict[str, float | str] = {}
        self._red_signal_probe_samples: Dict[int, int] = {1: 0, 2: 0}
        self._red_signal_probe_reward_sums: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._red_signal_probe_move_sums: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._red_signal_probe_wait_delta_sums: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._red_signal_probe_speed_delta_sums: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._red_signal_probe_print_count = 0

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
        self._invalidate_step_cache()
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

    def _invalidate_step_cache(self) -> None:
        self._step_cache = None

    def _cache_trafficlight_geometry(self) -> None:
        self._controlled_lanes_by_tl = {}
        self._junction_positions = {}
        self._lane_directions_by_tl = {}
        self._phase_count_by_tl = {}
        self._lane_endpoints = {}
        for tl_id in self._agent_ids:
            self._junction_positions[tl_id] = tuple(traci.junction.getPosition(tl_id))
            controlled_lanes = tuple(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
            self._controlled_lanes_by_tl[tl_id] = controlled_lanes
            self._phase_count_by_tl[tl_id] = max(1, len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)) if traci.trafficlight.getAllProgramLogics(tl_id) else 1
            lane_directions: Dict[str, str] = {}
            junction_pos = self._junction_positions[tl_id]
            for lane_id in controlled_lanes:
                lane_directions[lane_id] = _approach_direction(lane_id, junction_pos)
                if lane_id not in self._lane_endpoints:
                    shape = traci.lane.getShape(lane_id)
                    if shape:
                        self._lane_endpoints[lane_id] = tuple(shape[-1])
            self._lane_directions_by_tl[tl_id] = lane_directions

    def _current_step_cache(self) -> StepCache:
        sim_time = float(traci.simulation.getTime())
        if self._step_cache is not None and self._step_cache.sim_time == sim_time:
            return self._step_cache

        vehicle_ids = tuple(traci.vehicle.getIDList())
        vehicle_set = set(vehicle_ids)
        vehicle_positions: Dict[str, tuple[float, float]] = {}
        vehicle_speeds: Dict[str, float] = {}
        vehicle_waiting_times: Dict[str, float] = {}
        queue_length_network = 0
        for vehicle_id in vehicle_ids:
            try:
                vehicle_positions[vehicle_id] = tuple(traci.vehicle.getPosition(vehicle_id))
            except traci_exceptions.TraCIException:
                continue
            try:
                speed = float(traci.vehicle.getSpeed(vehicle_id))
            except traci_exceptions.TraCIException:
                speed = 0.0
            vehicle_speeds[vehicle_id] = speed
            if speed < STOP_SPEED_THRESHOLD:
                queue_length_network += 1
            try:
                vehicle_waiting_times[vehicle_id] = float(traci.vehicle.getWaitingTime(vehicle_id))
            except traci_exceptions.TraCIException:
                vehicle_waiting_times[vehicle_id] = 0.0

        lane_halting_numbers: Dict[str, int] = {}
        lane_vehicle_numbers: Dict[str, int] = {}
        lane_waiting_times: Dict[str, float] = {}
        for lane_ids in self._controlled_lanes_by_tl.values():
            for lane_id in lane_ids:
                if lane_id in lane_halting_numbers:
                    continue
                try:
                    lane_halting_numbers[lane_id] = int(traci.lane.getLastStepHaltingNumber(lane_id))
                except traci_exceptions.TraCIException:
                    lane_halting_numbers[lane_id] = 0
                try:
                    lane_vehicle_numbers[lane_id] = int(traci.lane.getLastStepVehicleNumber(lane_id))
                except traci_exceptions.TraCIException:
                    lane_vehicle_numbers[lane_id] = 0
                try:
                    lane_waiting_times[lane_id] = float(traci.lane.getWaitingTime(lane_id))
                except traci_exceptions.TraCIException:
                    lane_waiting_times[lane_id] = 0.0

        ev_present = self.ev_id in vehicle_set
        ev_position: tuple[float, float] | None = vehicle_positions.get(self.ev_id) if ev_present else None
        ev_speed = vehicle_speeds.get(self.ev_id, 0.0) if ev_present else 0.0
        ev_wait = vehicle_waiting_times.get(self.ev_id, 0.0) if ev_present else 0.0
        ev_road = traci.vehicle.getRoadID(self.ev_id) if ev_present else ''
        ev_route_index = traci.vehicle.getRouteIndex(self.ev_id) if ev_present else -1
        route_progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if ev_present and self._route_signals else None
        progress_upcoming_ids = {signal.tl_id for signal in route_progress.upcoming} if route_progress is not None else set()
        progress_passed_ids = {signal.tl_id for signal in route_progress.passed} if route_progress is not None else set()

        cache = StepCache(
            sim_time=sim_time,
            vehicle_ids=vehicle_ids,
            vehicle_set=vehicle_set,
            vehicle_positions=vehicle_positions,
            vehicle_speeds=vehicle_speeds,
            vehicle_waiting_times=vehicle_waiting_times,
            queue_length_network=queue_length_network,
            throughput=float(traci.simulation.getArrivedNumber()),
            ev_present=ev_present,
            ev_position=ev_position,
            ev_speed=ev_speed,
            ev_wait=ev_wait,
            ev_road=ev_road,
            ev_route_index=ev_route_index,
            route_progress=route_progress,
            progress_upcoming_ids=progress_upcoming_ids,
            progress_passed_ids=progress_passed_ids,
            trafficlight_phases={tl_id: int(traci.trafficlight.getPhase(tl_id)) for tl_id in self._agent_ids},
            trafficlight_next_switch={
                tl_id: float(traci.trafficlight.getNextSwitch(tl_id)) if tl_id in self._agent_ids else sim_time
                for tl_id in self._agent_ids
            },
            lane_halting_numbers=lane_halting_numbers,
            lane_vehicle_numbers=lane_vehicle_numbers,
            lane_waiting_times=lane_waiting_times,
        )
        self._step_cache = cache
        return cache

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
        self._cache_trafficlight_geometry()
        self._ev_phase_by_tl = {}
        for tl_id in self._agent_ids:
            route_signal = self._route_signal_by_id.get(tl_id)
            route_lanes = list(route_signal.route_lanes) if route_signal is not None else []
            self._ev_phase_by_tl[tl_id] = infer_ev_green_phase(tl_id, route_lanes)

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
        return self.controller_type in {
            CONTROLLER_INDEPENDENT_MARL,
            CONTROLLER_COORDINATED_MARL,
            CONTROLLER_COORDINATED_PPO,
            CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
            CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO,
            CONTROLLER_MULTI_LEVEL_COORDINATED_PPO,
            CONTROLLER_MULTI_LEVEL_COORDINATED_DQN,
        }

    @property
    def coordination_enabled(self) -> bool:
        return self.controller_type in {
            CONTROLLER_COORDINATED_MARL,
            CONTROLLER_COORDINATED_PPO,
            CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
            CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO,
            CONTROLLER_MULTI_LEVEL_COORDINATED_PPO,
            CONTROLLER_MULTI_LEVEL_COORDINATED_DQN,
        }

    @property
    def adaptive_reward_enabled(self) -> bool:
        return self.controller_type == CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO

    @property
    def shared_policy_enabled(self) -> bool:
        return self.is_multi_agent

    @property
    def state_dim(self) -> int:
        if self.controller_type in {CONTROLLER_MULTI_LEVEL_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
            return MULTI_LEVEL_COORDINATED_STATE_DIM
        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO:
            return CONGESTION_AWARE_COORDINATED_STATE_DIM
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
            "local_congestion_index": ctx.local_congestion_index,
            "neighbor_congestion_index": ctx.neighbor_congestion_index,
            "downstream_blockage_ratio": ctx.downstream_blockage_ratio,
            "queue_growth_rate": ctx.queue_growth_rate,
            "corridor_congestion_pressure": ctx.corridor_congestion_pressure,
            "network_congestion_score": ctx.network_congestion_score,
            "congestion_transfer_risk": ctx.congestion_transfer_risk,
        }

    def _nearest_signal_ahead(self) -> Optional[RouteSignal]:
        if self.ev_id not in traci.vehicle.getIDList():
            return None
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals)
        return progress.next_signal

    def _queue_length_network(self) -> int:
        cache = self._step_cache or self._current_step_cache()
        return cache.queue_length_network

    def _directional_queues(self, tl_id: str) -> Dict[str, int]:
        queues = {"north": 0, "south": 0, "east": 0, "west": 0}
        lane_directions = self._lane_directions_by_tl.get(tl_id)
        if lane_directions is None:
            junction_pos = self._junction_positions.get(tl_id, tuple(traci.junction.getPosition(tl_id)))
            lane_directions = {}
            for lane_id in self._controlled_lanes_by_tl.get(tl_id, tuple(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))):
                lane_directions[lane_id] = _approach_direction(lane_id, junction_pos)
        cache = self._step_cache or self._current_step_cache()
        for lane_id, direction in lane_directions.items():
            queues[direction] += int(cache.lane_halting_numbers.get(lane_id, 0))
        return queues

    def _nearby_vehicle_count(self, tl_id: str, radius: float = 120.0) -> int:
        cache = self._step_cache or self._current_step_cache()
        junction_x, junction_y = self._junction_positions.get(tl_id, tuple(traci.junction.getPosition(tl_id)))
        nearby_count = 0
        radius_sq = radius * radius
        for x, y in cache.vehicle_positions.values():
            if ((x - junction_x) ** 2 + (y - junction_y) ** 2) <= radius_sq:
                nearby_count += 1
        return nearby_count

    def _average_neighbor_queue(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0
        totals = [float(sum(self._directional_queues(neighbor_id).values())) for neighbor_id in neighbors]
        return float(sum(totals) / len(totals)) if totals else 0.0

    def _neighbor_queue_std(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if len(neighbors) < 2:
            return 0.0
        totals = [float(sum(self._directional_queues(neighbor_id).values())) for neighbor_id in neighbors]
        return float(np.std(np.asarray(totals, dtype=np.float32)))

    def _average_neighbor_waiting_time(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0
        cache = self._step_cache or self._current_step_cache()
        wait_totals: list[float] = []
        for neighbor_id in neighbors:
            lane_waits = [cache.lane_waiting_times.get(lane_id, 0.0) for lane_id in self._controlled_lanes_by_tl.get(neighbor_id, tuple())]
            wait_totals.append(float(sum(lane_waits) / len(lane_waits)) if lane_waits else 0.0)
        return float(sum(wait_totals) / len(wait_totals)) if wait_totals else 0.0

    def _average_neighbor_throughput(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        if not neighbors:
            return 0.0
        cache = self._step_cache or self._current_step_cache()
        throughput_totals: list[float] = []
        for neighbor_id in neighbors:
            lane_flows = [cache.lane_vehicle_numbers.get(lane_id, 0) for lane_id in self._controlled_lanes_by_tl.get(neighbor_id, tuple())]
            throughput_totals.append(float(sum(lane_flows) / len(lane_flows)) if lane_flows else 0.0)
        return float(sum(throughput_totals) / len(throughput_totals)) if throughput_totals else 0.0

    def _neighbor_emergency_vehicle_present(self, tl_id: str) -> float:
        neighbors = self._neighbor_map.get(tl_id, [])
        cache = self._step_cache or self._current_step_cache()
        if not neighbors or not cache.ev_present or not cache.route_progress:
            return 0.0
        upcoming_ids = cache.progress_upcoming_ids
        return 1.0 if any(neighbor_id in upcoming_ids for neighbor_id in neighbors) else 0.0

    def _phase_features(self, tl_id: str) -> Tuple[float, float, int]:
        cache = self._step_cache or self._current_step_cache()
        phase_idx = int(cache.trafficlight_phases.get(tl_id, traci.trafficlight.getPhase(tl_id)))
        n_phases = self._phase_count_by_tl.get(tl_id, 1)
        phase_norm = float(phase_idx) / float(max(n_phases, 1))
        next_sw = cache.trafficlight_next_switch.get(tl_id, float(traci.trafficlight.getNextSwitch(tl_id)))
        remaining = max(0.0, float(next_sw) - cache.sim_time)
        remaining_norm = min(1.0, remaining / 120.0)
        return phase_norm, remaining_norm, phase_idx

    def _distance_to_tl(self, tl_id: str) -> float:
        cache = self._step_cache or self._current_step_cache()
        if not cache.ev_present or cache.ev_position is None:
            return MAX_EV_FEATURE_DISTANCE

        route_signal = self._route_signal_by_id.get(tl_id)
        ev_x, ev_y = cache.ev_position
        if route_signal is not None:
            endpoints = [self._lane_endpoints.get(lane_id) for lane_id in route_signal.route_lanes]
            route_points = [point for point in endpoints if point is not None]
            if route_points:
                return float(min(((ev_x - x) ** 2 + (ev_y - y) ** 2) ** 0.5 for x, y in route_points))
            try:
                return float(max(0.0, signal_distance(self.ev_id, route_signal)))
            except traci_exceptions.TraCIException:
                pass

        tl_x, tl_y = self._junction_positions.get(tl_id, tuple(traci.junction.getPosition(tl_id)))
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

        cache = self._step_cache or self._current_step_cache()
        demand_terms: list[float] = []
        for neighbor_id in neighbors:
            neighbor_queue = float(sum(self._directional_queues(neighbor_id).values()))
            neighbor_density = float(self._nearby_vehicle_count(neighbor_id))
            outgoing_flow = 0.0
            for lane_id in self._controlled_lanes_by_tl.get(neighbor_id, tuple()):
                outgoing_flow += float(cache.lane_vehicle_numbers.get(lane_id, 0))
            demand_terms.append((neighbor_queue / 20.0) * 0.45 + (outgoing_flow / 25.0) * 0.35 + (neighbor_density / 40.0) * 0.20)
        return float(sum(demand_terms) / len(demand_terms))

    def _global_traffic_features(self) -> Dict[str, float]:
        cache = self._step_cache or self._current_step_cache()
        vehicle_count = max(len(cache.vehicle_ids), 1)
        agent_count = max(len(self._agent_ids), 1)
        queues = [float(sum(self._directional_queues(tl_id).values())) for tl_id in self._agent_ids]
        waits = list(cache.vehicle_waiting_times.values())
        avg_queue_length_raw = float(sum(queues) / len(queues)) if queues else 0.0
        avg_waiting_time_raw = float(sum(waits) / len(waits)) if waits else 0.0
        total_network_congestion_raw = self._network_congestion_ratio()
        throughput_raw = float(cache.throughput)
        pct_congested_intersections_raw = float(
            sum(1 for tl_id in self._agent_ids if float(np.clip(sum(self._directional_queues(tl_id).values()) / 20.0, 0.0, 1.0)) >= 0.5)
            / max(len(self._agent_ids), 1)
        ) if self._agent_ids else 0.0
        blocked_approaches_raw = 0.0
        total_approaches = 0
        for tl_id in self._agent_ids:
            for lane_id in self._controlled_lanes_by_tl.get(tl_id, tuple()):
                total_approaches += 1
                if float(cache.lane_halting_numbers.get(lane_id, 0)) >= 3.0:
                    blocked_approaches_raw += 1.0
        global_traffic_density_raw = float(vehicle_count / max(agent_count * 12.0, 1.0))
        ev_network_progress_raw = 0.0
        if cache.ev_present and cache.route_progress is not None:
            ev_network_progress_raw = float(len(cache.progress_passed_ids) / max(len(self._route_signals), 1))

        return {
            "avg_queue_length": float(np.clip(avg_queue_length_raw / 20.0, 0.0, 1.0)),
            "avg_waiting_time": float(np.clip(avg_waiting_time_raw / 60.0, 0.0, 1.0)),
            "total_network_congestion": float(np.clip(total_network_congestion_raw, 0.0, 1.0)),
            "throughput": float(np.clip(throughput_raw / max(agent_count * 5.0, 1.0), 0.0, 1.0)),
            "pct_congested_intersections": float(np.clip(pct_congested_intersections_raw, 0.0, 1.0)),
            "blocked_approaches": float(np.clip(blocked_approaches_raw / max(total_approaches, 1), 0.0, 1.0)),
            "global_traffic_density": float(np.clip(global_traffic_density_raw, 0.0, 1.0)),
            "ev_network_progress": float(np.clip(ev_network_progress_raw, 0.0, 1.0)),
        }

    def _multi_level_reward_contributions(self, breakdown: Dict[str, float]) -> Dict[str, float]:
        local = float(
            breakdown.get("ev_delay_penalty", 0.0)
            + breakdown.get("ev_stop_penalty", 0.0)
            + breakdown.get("low_speed_penalty", 0.0)
            + breakdown.get("queue_penalty", 0.0)
            + breakdown.get("queue_growth_penalty", 0.0)
            + breakdown.get("throughput_reward", 0.0)
            + breakdown.get("switch_penalty", 0.0)
            + breakdown.get("intersection_clear_reward", 0.0)
            + breakdown.get("anti_gridlock_penalty", 0.0)
        )
        neighbor = float(
            breakdown.get("neighbor_congestion_penalty", 0.0)
            + breakdown.get("downstream_blockage_penalty", 0.0)
            + breakdown.get("traffic_stability_penalty", 0.0)
            + breakdown.get("corridor_flow_reward", 0.0)
            + breakdown.get("congestion_transfer_penalty", 0.0)
            + breakdown.get("corridor_congestion_penalty", 0.0)
            + breakdown.get("congestion_imbalance_penalty", 0.0)
            + breakdown.get("congestion_trend_penalty", 0.0)
        )
        global_ = float(
            breakdown.get("network_congestion_penalty", 0.0)
            + breakdown.get("congestion_reduction_reward", 0.0)
            + breakdown.get("queue_stabilization_reward", 0.0)
            + breakdown.get("spillback_prevention_reward", 0.0)
        )
        return {
            "local_reward_contribution": local,
            "neighbor_coordination_contribution": neighbor,
            "global_optimization_contribution": global_,
        }

    def _controlled_lane_count(self, tl_id: str) -> int:
        return max(len(self._controlled_lanes_by_tl.get(tl_id, tuple())), 1)

    def _network_congestion_score(self) -> float:
        cache = self._step_cache or self._current_step_cache()
        vehicle_count = max(len(cache.vehicle_ids), 1)
        return float(np.clip(cache.queue_length_network / vehicle_count, 0.0, 1.0))

    def _congestion_feature_snapshot(self, tl_id: str, ctx: AgentContext) -> Dict[str, float]:
        controlled_lane_count = self._controlled_lane_count(tl_id)
        prev_local_queue = self._prev_agent_queue_totals.get(tl_id, ctx.local_queue)
        prev_network_queue = self._prev_network_queue
        cache = self._step_cache or self._current_step_cache()
        network_queue = float(cache.queue_length_network)
        local_queue_delta = ctx.local_queue - prev_local_queue
        network_queue_delta = network_queue - prev_network_queue
        local_congestion_index = float(np.clip(ctx.local_queue / max(controlled_lane_count * 15.0, 1.0), 0.0, 1.0))
        neighbor_congestion_index = float(np.clip(ctx.neighbor_queue_avg / 60.0, 0.0, 1.0))
        downstream_blockage_ratio = float(np.clip(ctx.downstream_congestion / 4.0, 0.0, 1.0))
        queue_growth_rate = float(np.clip(local_queue_delta / 20.0, -1.0, 1.0))
        corridor_congestion_pressure = float(
            np.clip((local_congestion_index * 0.4) + (neighbor_congestion_index * 0.25) + (downstream_blockage_ratio * 0.25) + (queue_growth_rate * 0.10), 0.0, 1.0)
        )
        network_congestion_score = self._network_congestion_score()
        congestion_imbalance_ns = float(
            np.clip(abs(ctx.queues["north"] - ctx.queues["south"]) / max(controlled_lane_count * 6.0, 1.0), 0.0, 1.0)
        )
        congestion_imbalance_ew = float(
            np.clip(abs(ctx.queues["east"] - ctx.queues["west"]) / max(controlled_lane_count * 6.0, 1.0), 0.0, 1.0)
        )
        congestion_trend_local = float(np.clip(local_queue_delta / 20.0, -1.0, 1.0))
        congestion_trend_network = float(np.clip(network_queue_delta / max(len(self._agent_ids) * 10.0, 1.0), -1.0, 1.0))
        congestion_transfer_risk = float(
            np.clip((neighbor_congestion_index * 0.35) + (downstream_blockage_ratio * 0.45) + (network_congestion_score * 0.20), 0.0, 1.0)
        )
        return {
            "local_congestion_index": local_congestion_index,
            "neighbor_congestion_index": neighbor_congestion_index,
            "downstream_blockage_ratio": downstream_blockage_ratio,
            "queue_growth_rate": queue_growth_rate,
            "corridor_congestion_pressure": corridor_congestion_pressure,
            "network_congestion_score": network_congestion_score,
            "congestion_imbalance_ns": congestion_imbalance_ns,
            "congestion_imbalance_ew": congestion_imbalance_ew,
            "congestion_trend_local": congestion_trend_local,
            "congestion_trend_network": congestion_trend_network,
            "congestion_transfer_risk": congestion_transfer_risk,
        }

    def _log_congestion_snapshot(self, step: int, tl_id: str, features: Dict[str, float]) -> None:
        if not self.debug_logs or not self.debug_congestion_features:
            return
        if step <= 0 or step % 100 != 0:
            return
        print("[CONGESTION_DEBUG]")
        print(f"step={step} tl_id={tl_id}")
        for key in [
            "local_congestion_index",
            "neighbor_congestion_index",
            "downstream_blockage_ratio",
            "queue_growth_rate",
            "corridor_congestion_pressure",
            "network_congestion_score",
            "congestion_imbalance_ns",
            "congestion_imbalance_ew",
            "congestion_trend_local",
            "congestion_trend_network",
            "congestion_transfer_risk",
        ]:
            print(f"{key}={features[key]:.4f}")

    def _maybe_warn_inactive_congestion_features(self, done: bool) -> None:
        if not done or self.controller_type != CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO:
            return
        if self._congestion_feature_warning_logged:
            return
        if not self._congestion_feature_seen_nonzero:
            print("[WARNING]")
            print("Congestion-aware features appear inactive.")
        self._congestion_feature_warning_logged = True

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
        route_signal = self._route_signal_by_id.get(tl_id)
        route_lanes = route_signal.route_lanes if route_signal is not None else tuple()
        ev_phase = self._ev_phase_by_tl.get(
            tl_id,
            infer_ev_green_phase(tl_id, list(route_lanes)),
        )
        ev_neighbor_eta = 1.0
        if dist <= min(MAX_EV_FEATURE_DISTANCE, 300.0):
            ev_neighbor_eta = self._ev_neighbor_eta(tl_id)

        controlled_lane_count = self._controlled_lane_count(tl_id)
        prev_local_queue = self._prev_agent_queue_totals.get(tl_id, local_queue)
        prev_network_queue = self._prev_network_queue
        network_queue = float(self._queue_length_network())
        local_queue_delta = local_queue - prev_local_queue
        network_queue_delta = network_queue - prev_network_queue
        local_congestion_index = float(np.clip(local_queue / max(controlled_lane_count * 15.0, 1.0), 0.0, 1.0))
        neighbor_congestion_index = float(np.clip(neighbor_queue_avg / 60.0, 0.0, 1.0))
        downstream_blockage_ratio = float(np.clip(downstream_congestion / 4.0, 0.0, 1.0))
        queue_growth_rate = float(np.clip(local_queue_delta / 20.0, -1.0, 1.0))
        corridor_congestion_pressure = float(
            np.clip((local_congestion_index * 0.4) + (neighbor_congestion_index * 0.25) + (downstream_blockage_ratio * 0.25) + (queue_growth_rate * 0.10), 0.0, 1.0)
        )
        network_congestion_score = self._network_congestion_score()
        congestion_imbalance_ns = float(np.clip(abs(queues["north"] - queues["south"]) / max(controlled_lane_count * 6.0, 1.0), 0.0, 1.0))
        congestion_imbalance_ew = float(np.clip(abs(queues["east"] - queues["west"]) / max(controlled_lane_count * 6.0, 1.0), 0.0, 1.0))
        congestion_trend_local = float(np.clip(local_queue_delta / 20.0, -1.0, 1.0))
        congestion_trend_network = float(np.clip(network_queue_delta / max(len(self._agent_ids) * 10.0, 1.0), -1.0, 1.0))
        congestion_transfer_risk = float(
            np.clip((neighbor_congestion_index * 0.35) + (downstream_blockage_ratio * 0.45) + (network_congestion_score * 0.20), 0.0, 1.0)
        )

        ctx = AgentContext(
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
            neighbor_queue_std=self._neighbor_queue_std(tl_id),
            downstream_congestion=downstream_congestion,
            neighbor_phase_avg=neighbor_phase_avg,
            incoming_traffic_estimate=incoming_traffic_estimate,
            ev_neighbor_eta=ev_neighbor_eta,
            approaching_density=float(local_density) / 40.0,
            neighbor_waiting_time=self._average_neighbor_waiting_time(tl_id),
            neighbor_throughput_estimate=self._average_neighbor_throughput(tl_id),
            downstream_spillback_indicator=float(np.clip(downstream_congestion / 2.0, 0.0, 1.0)),
            neighbor_emergency_vehicle_present=(
                self._neighbor_emergency_vehicle_present(tl_id) if dist <= min(MAX_EV_FEATURE_DISTANCE, 300.0) else 0.0
            ),
            ev_relevant=self._ev_relevance(dist),
            on_ev_route=route_signal is not None,
            local_congestion_index=local_congestion_index,
            neighbor_congestion_index=neighbor_congestion_index,
            downstream_blockage_ratio=downstream_blockage_ratio,
            queue_growth_rate=queue_growth_rate,
            corridor_congestion_pressure=corridor_congestion_pressure,
            network_congestion_score=network_congestion_score,
            congestion_imbalance_ns=congestion_imbalance_ns,
            congestion_imbalance_ew=congestion_imbalance_ew,
            congestion_trend_local=congestion_trend_local,
            congestion_trend_network=congestion_trend_network,
            congestion_transfer_risk=congestion_transfer_risk,
        )
        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO:
            features = [
                ctx.local_congestion_index,
                ctx.neighbor_congestion_index,
                ctx.downstream_blockage_ratio,
                ctx.queue_growth_rate,
                ctx.corridor_congestion_pressure,
                ctx.network_congestion_score,
                ctx.congestion_imbalance_ns,
                ctx.congestion_imbalance_ew,
                ctx.congestion_trend_local,
                ctx.congestion_trend_network,
                ctx.congestion_transfer_risk,
            ]
            if any(abs(float(value)) > 1e-9 for value in features):
                self._congestion_feature_seen_nonzero = True
        return ctx

    def _normalize_base_state(self, ctx: AgentContext) -> list[float]:
        cache = self._step_cache or self._current_step_cache()
        speed = cache.vehicle_speeds.get(self.ev_id, 0.0) if cache.ev_present else 0.0
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

    def _state_from_context(self, ctx: AgentContext, global_features: Dict[str, float] | None = None) -> np.ndarray:
        state_values = self._normalize_base_state(ctx)
        if global_features is None:
            global_features = self._global_traffic_features()
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
        if self.controller_type in {CONTROLLER_MULTI_LEVEL_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
            state_values.extend(
                [
                    float(np.clip(ctx.neighbor_queue_std / 20.0, 0.0, 1.0)),
                    float(np.clip(ctx.neighbor_congestion_index, 0.0, 1.0)),
                    float(np.clip(ctx.neighbor_waiting_time / 60.0, 0.0, 1.0)),
                    float(np.clip(ctx.downstream_spillback_indicator, 0.0, 1.0)),
                    float(np.clip(ctx.neighbor_throughput_estimate / 20.0, 0.0, 1.0)),
                    float(np.clip(ctx.neighbor_emergency_vehicle_present, 0.0, 1.0)),
                    global_features["avg_queue_length"],
                    global_features["avg_waiting_time"],
                    global_features["total_network_congestion"],
                    global_features["throughput"],
                    global_features["pct_congested_intersections"],
                    global_features["blocked_approaches"],
                    global_features["global_traffic_density"],
                    global_features["ev_network_progress"],
                ]
            )
        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO:
            state_values.extend(
                [
                    ctx.local_congestion_index,
                    ctx.neighbor_congestion_index,
                    ctx.downstream_blockage_ratio,
                    ctx.queue_growth_rate,
                    ctx.corridor_congestion_pressure,
                    ctx.network_congestion_score,
                    ctx.congestion_imbalance_ns,
                    ctx.congestion_imbalance_ew,
                    ctx.congestion_trend_local,
                    ctx.congestion_trend_network,
                    ctx.congestion_transfer_risk,
                ]
            )
        return np.asarray(state_values, dtype=np.float32)

    def _log_state(self, tl_id: str, state: np.ndarray, ctx: AgentContext) -> None:
        if not self.debug_logs:
            return
        self._state_debug_counter += 1
        if self._state_debug_counter <= 10 or self._state_debug_counter % 60 == 0:
            print(
                f"[RL_STATE] agent={tl_id} state={state.tolist()} local_queue={ctx.local_queue:.1f} "
                f"neighbor_queue={ctx.neighbor_queue_avg:.2f} downstream={ctx.downstream_congestion:.2f} "
                f"incoming={ctx.incoming_traffic_estimate:.2f} neighbor_phase={ctx.neighbor_phase_avg:.2f} "
                f"ev_eta_neighbor={ctx.ev_neighbor_eta:.2f}"
            )

    def _route_debug(self) -> None:
        if not self.debug_logs or self.ev_id not in traci.vehicle.getIDList():
            return
        progress = route_progress_for_vehicle(self.ev_id, self._route_signals)
        self._route_log_counter += 1
        if self._route_log_counter <= 5 or self._route_log_counter % 20 == 0:
            print(
                f"[EV_ROUTE] ev={self.ev_id} next={progress.next_signal.tl_id if progress.next_signal else 'none'} "
                f"upcoming={[signal.tl_id for signal in progress.upcoming]} "
                f"passed={[signal.tl_id for signal in progress.passed]}"
            )

    def _build_all_states(
        self,
        contexts: Dict[str, AgentContext] | None = None,
        global_features: Dict[str, float] | None = None,
    ) -> Dict[str, np.ndarray]:
        states: Dict[str, np.ndarray] = {}
        if contexts is None:
            contexts = {tl_id: self._build_agent_context(tl_id) for tl_id in self._agent_ids}
        if global_features is None:
            global_features = self._global_traffic_features()
        for tl_id in self._agent_ids:
            ctx = contexts[tl_id]
            state = self._state_from_context(ctx, global_features)
            states[tl_id] = state
            self._log_state(tl_id, state, ctx)
        return states

    def get_state(self) -> np.ndarray | Dict[str, np.ndarray]:
        if self.is_multi_agent:
            cache = self._step_cache or self._current_step_cache()
            if not cache.ev_present and not self._started:
                return {tl_id: np.zeros(self.state_dim, dtype=np.float32) for tl_id in self._agent_ids}
            return self._build_all_states()

        if self.ev_id not in traci.vehicle.getIDList():
            return np.zeros(self.state_dim, dtype=np.float32)

        signal = self._nearest_signal_ahead()
        if signal is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        ctx = self._build_agent_context(signal.tl_id)
        global_features = self._global_traffic_features()
        state = self._state_from_context(ctx, global_features)
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
        self._prev_agent_phases = {tl_id: traci.trafficlight.getPhase(tl_id) for tl_id in self._agent_ids}
        self._last_reward_breakdowns = {}
        self._last_coordination_terms = {}
        self._last_multi_level_diagnostics = {}
        self._reset_red_signal_probe_stats()
        self._reset_adaptive_episode_stats()
        if self.is_multi_agent:
            self._current_step_cache()
            initial_contexts = {tl_id: self._build_agent_context(tl_id) for tl_id in self._agent_ids}
            self._prev_agent_queue_totals = {tl_id: ctx.local_queue for tl_id, ctx in initial_contexts.items()}
            state = self._build_all_states(contexts=initial_contexts)
        else:
            self._prev_agent_queue_totals = {
                tl_id: float(sum(self._directional_queues(tl_id).values())) for tl_id in self._agent_ids
            }
            state = self.get_state()

        print(
            f"[RL_INIT] controller_type={self.controller_type} agents={len(self._agent_ids)} "
            f"shared_policy={self.shared_policy_enabled} state_dim={self.state_dim} action_dim={self.action_dim} "
            f"coordination={self.coordination_enabled} traffic_scale={self.traffic_scale:.2f}"
        )
        print(f"[RL_INIT] active_intersections={self._agent_ids}")
        if not self._reward_weights_logged:
            print("[REWARD_WEIGHTS]")
            print("ev_delay -> EV waiting-time penalty")
            print("ev_stop -> EV stop penalty")
            print("low_speed_near_signal -> slow-EV penalty near signals")
            print("queue -> local queue penalty")
            print("queue_growth -> queue buildup penalty")
            print("throughput -> throughput reward")
            print("intersection_clear -> reward for clearing intersections")
            print("neighbor_congestion -> neighbor queue penalty")
            print("network_congestion -> network-wide congestion penalty")
            print("downstream_blockage -> spillback penalty")
            print("traffic_stability -> queue instability penalty")
            print("congestion_reduction -> reward for reducing congestion")
            print("queue_stabilization -> reward for stabilizing queues")
            print("spillback_prevention -> reward for preventing downstream spillback")
            print("congestion_transfer -> penalty for pushing congestion to neighbors")
            print("corridor_congestion -> corridor congestion penalty")
            print("congestion_imbalance -> penalty for imbalance between approaches")
            print("congestion_trend -> penalty for worsening congestion trend")
            print("anti_gridlock_penalty -> extra penalty when network congestion or queue length crosses threshold")
            print({key: float(value) for key, value in REWARD_WEIGHTS.items()})
            self._reward_weights_logged = True
        if self.controller_type in {CONTROLLER_MULTI_LEVEL_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
            print("[MULTI_LEVEL_COORDINATED_PPO_INIT]")
            print("level_1=local optimization")
            print("level_2=neighbor coordination")
            print("level_3=global optimization")
            print(f"state_dim={self.state_dim}")
            print(f"action_space={self.action_dim}")
        self._route_debug()
        return state

    def _network_congestion_ratio(self) -> float:
        cache = self._step_cache or self._current_step_cache()
        vehicle_count = max(len(cache.vehicle_ids), 1)
        return float(cache.queue_length_network) / float(vehicle_count)

    def _reset_adaptive_episode_stats(self) -> None:
        self._adaptive_mode_steps = {"low": 0, "medium": 0, "high": 0}
        self._adaptive_ev_weight_sum = 0.0
        self._adaptive_network_weight_sum = 0.0
        self._adaptive_step_count = 0
        self._adaptive_ev_component_sum = 0.0
        self._adaptive_network_component_sum = 0.0
        self._adaptive_congestion_index_sum = 0.0
        self._last_adaptive_congestion_level = "medium"
        self._last_adaptive_congestion_index = 0.0

    def _compute_adaptive_congestion_context(self) -> tuple[float, str, float, float]:
        veh_ids = traci.vehicle.getIDList()
        veh_count = max(len(veh_ids), 1)
        queue_len = float(self._queue_length_network())
        waits = [traci.vehicle.getWaitingTime(v_id) for v_id in veh_ids]
        avg_wait = float(sum(waits) / len(waits)) if waits else 0.0
        occupancy = queue_len / float(veh_count)

        queue_norm = float(np.clip(queue_len / 40.0, 0.0, 1.0))
        wait_norm = float(np.clip(avg_wait / 30.0, 0.0, 1.0))
        occupancy_norm = float(np.clip(occupancy, 0.0, 1.0))
        congestion_index = float((0.35 * queue_norm) + (0.35 * wait_norm) + (0.30 * occupancy_norm))

        level = classify_congestion_level(congestion_index)
        ev_weight, network_weight = adaptive_scales_for_level(level)
        return congestion_index, level, ev_weight, network_weight

    def _record_adaptive_step(self, congestion_index: float, level: str, ev_weight: float, network_weight: float) -> None:
        self._adaptive_mode_steps[level] = self._adaptive_mode_steps.get(level, 0) + 1
        self._adaptive_ev_weight_sum += float(ev_weight)
        self._adaptive_network_weight_sum += float(network_weight)
        self._adaptive_congestion_index_sum += float(congestion_index)
        self._adaptive_step_count += 1
        self._last_adaptive_congestion_level = level
        self._last_adaptive_congestion_index = float(congestion_index)

    def _record_adaptive_reward_components(self, breakdown: Dict[str, float]) -> None:
        self._adaptive_ev_component_sum += float(breakdown.get("_ev_reward_component", 0.0))
        self._adaptive_network_component_sum += float(breakdown.get("_network_reward_component", 0.0))

    def _reset_red_signal_probe_stats(self) -> None:
        self._red_signal_probe_samples = {1: 0, 2: 0}
        self._red_signal_probe_reward_sums = {1: 0.0, 2: 0.0}
        self._red_signal_probe_move_sums = {1: 0.0, 2: 0.0}
        self._red_signal_probe_wait_delta_sums = {1: 0.0, 2: 0.0}
        self._red_signal_probe_speed_delta_sums = {1: 0.0, 2: 0.0}
        self._red_signal_probe_print_count = 0

    def _is_waiting_at_red(self, distance_before: float, speed_before: float, wait_before: float) -> bool:
        if distance_before > NEAR_SIGNAL_DISTANCE_THRESHOLD:
            return False
        if speed_before >= LOW_SPEED_NEAR_SIGNAL_THRESHOLD:
            return False
        if wait_before <= 0.0:
            return False
        return True

    def _record_red_signal_probe(
        self,
        *,
        tl_id: str,
        ctx: AgentContext,
        current_phase: int,
        requested_action: int,
        applied_action: int,
        speed_before: float,
        speed_after: float,
        wait_before: float,
        wait_after: float,
        distance_before: float,
        distance_after: float,
        rewards: Dict[str, float],
        breakdown: Dict[str, float],
    ) -> Dict[str, float | int | str] | None:
        if requested_action not in {1, 2}:
            return None
        if current_phase == ctx.ev_phase:
            return None
        if not self._is_waiting_at_red(distance_before, speed_before, wait_before):
            return None

        action = int(requested_action)
        moved_m = max(0.0, float(distance_before) - float(distance_after))
        wait_delta = float(wait_after) - float(wait_before)
        speed_delta = float(speed_after) - float(speed_before)
        reward_total = float(rewards.get(tl_id, 0.0))
        local_reward = float(
            breakdown.get("ev_delay_penalty", 0.0)
            + breakdown.get("ev_stop_penalty", 0.0)
            + breakdown.get("low_speed_penalty", 0.0)
            + breakdown.get("queue_penalty", 0.0)
            + breakdown.get("queue_growth_penalty", 0.0)
            + breakdown.get("throughput_reward", 0.0)
            + breakdown.get("switch_penalty", 0.0)
            + breakdown.get("intersection_clear_reward", 0.0)
            + breakdown.get("anti_gridlock_penalty", 0.0)
        )
        neighbor_reward = float(
            breakdown.get("neighbor_congestion_penalty", 0.0)
            + breakdown.get("downstream_blockage_penalty", 0.0)
            + breakdown.get("traffic_stability_penalty", 0.0)
            + breakdown.get("corridor_flow_reward", 0.0)
            + breakdown.get("congestion_transfer_penalty", 0.0)
            + breakdown.get("corridor_congestion_penalty", 0.0)
            + breakdown.get("congestion_imbalance_penalty", 0.0)
            + breakdown.get("congestion_trend_penalty", 0.0)
        )
        global_reward = float(
            breakdown.get("network_congestion_penalty", 0.0)
            + breakdown.get("congestion_reduction_reward", 0.0)
            + breakdown.get("queue_stabilization_reward", 0.0)
            + breakdown.get("spillback_prevention_reward", 0.0)
        )

        self._red_signal_probe_samples[action] = self._red_signal_probe_samples.get(action, 0) + 1
        self._red_signal_probe_reward_sums[action] = self._red_signal_probe_reward_sums.get(action, 0.0) + reward_total
        self._red_signal_probe_move_sums[action] = self._red_signal_probe_move_sums.get(action, 0.0) + moved_m
        self._red_signal_probe_wait_delta_sums[action] = self._red_signal_probe_wait_delta_sums.get(action, 0.0) + wait_delta
        self._red_signal_probe_speed_delta_sums[action] = self._red_signal_probe_speed_delta_sums.get(action, 0.0) + speed_delta

        probe = {
            "tl_id": tl_id,
            "current_phase": int(current_phase),
            "required_ev_phase": int(ctx.ev_phase),
            "requested_action": int(requested_action),
            "applied_action": int(applied_action),
            "action_label": "action_1_switch_to_ev_green" if action == 1 else "action_2_extend_green",
            "ev_distance_before": float(distance_before),
            "ev_distance_after": float(distance_after),
            "ev_speed_before": float(speed_before),
            "ev_speed_after": float(speed_after),
            "ev_wait_before": float(wait_before),
            "ev_wait_after": float(wait_after),
            "ev_move_m": float(moved_m),
            "ev_wait_delta": float(wait_delta),
            "ev_speed_delta": float(speed_delta),
            "step_index": int(self._episode_steps),
            "reward_total": reward_total,
            "local_reward": local_reward,
            "neighbor_reward": neighbor_reward,
            "global_reward": global_reward,
            "switch_penalty": float(breakdown.get("switch_penalty", 0.0)),
            "intersection_clear_reward": float(breakdown.get("intersection_clear_reward", 0.0)),
            "queue_penalty": float(breakdown.get("queue_penalty", 0.0)),
            "network_congestion_penalty": float(breakdown.get("network_congestion_penalty", 0.0)),
            "downstream_blockage_penalty": float(breakdown.get("downstream_blockage_penalty", 0.0)),
        }

        if self.debug_logs and (self._red_signal_probe_print_count < 40 or self._red_signal_probe_print_count % 50 == 0):
            print(
                "[RED_SIGNAL_PROBE] "
                f"agent={tl_id} phase={current_phase} ev_phase={ctx.ev_phase} "
                f"req={requested_action} appl={applied_action} "
                f"speed_before={speed_before:.2f} speed_after={speed_after:.2f} "
                f"wait_before={wait_before:.1f} wait_after={wait_after:.1f} "
                f"move_m={moved_m:.2f} reward={reward_total:.3f} "
                f"local={local_reward:.3f} neighbor={neighbor_reward:.3f} global={global_reward:.3f}"
            )
        self._red_signal_probe_print_count += 1
        return probe

    def get_red_signal_probe_diagnostics(self) -> Dict[str, Dict[str, float]]:
        diagnostics: Dict[str, Dict[str, float]] = {}
        for action in (1, 2):
            samples = max(self._red_signal_probe_samples.get(action, 0), 1)
            diagnostics[str(action)] = {
                "samples": float(self._red_signal_probe_samples.get(action, 0)),
                "avg_reward": float(self._red_signal_probe_reward_sums.get(action, 0.0) / samples),
                "avg_move_m": float(self._red_signal_probe_move_sums.get(action, 0.0) / samples),
                "avg_wait_delta": float(self._red_signal_probe_wait_delta_sums.get(action, 0.0) / samples),
                "avg_speed_delta": float(self._red_signal_probe_speed_delta_sums.get(action, 0.0) / samples),
            }
        return diagnostics

    def get_adaptive_episode_diagnostics(self) -> Dict[str, float | str]:
        steps = max(self._adaptive_step_count, 1)
        total_mode_steps = sum(self._adaptive_mode_steps.values()) or 1
        dominant_level = max(self._adaptive_mode_steps, key=self._adaptive_mode_steps.get)
        return {
            "congestion_level": dominant_level,
            "last_congestion_level": self._last_adaptive_congestion_level,
            "avg_congestion_index": float(self._adaptive_congestion_index_sum / steps),
            "adaptive_ev_weight": float(self._adaptive_ev_weight_sum / steps),
            "adaptive_network_weight": float(self._adaptive_network_weight_sum / steps),
            "ev_reward_component": float(self._adaptive_ev_component_sum),
            "network_reward_component": float(self._adaptive_network_component_sum),
            "pct_low_congestion": float(100.0 * self._adaptive_mode_steps["low"] / total_mode_steps),
            "pct_medium_congestion": float(100.0 * self._adaptive_mode_steps["medium"] / total_mode_steps),
            "pct_high_congestion": float(100.0 * self._adaptive_mode_steps["high"] / total_mode_steps),
        }

    def get_multi_level_episode_diagnostics(self) -> Dict[str, float]:
        if not self._last_multi_level_diagnostics:
            return {}
        steps = max(len(self._last_multi_level_diagnostics), 1)
        totals: Dict[str, float] = {
            "local_reward_contribution": 0.0,
            "neighbor_coordination_contribution": 0.0,
            "global_optimization_contribution": 0.0,
            "average_neighbor_congestion": 0.0,
            "average_global_congestion": 0.0,
            "average_neighbor_waiting_time": 0.0,
            "state_dim": float(self.state_dim),
        }
        for diagnostics in self._last_multi_level_diagnostics.values():
            for key in totals:
                if key == "state_dim":
                    continue
                totals[key] += float(diagnostics.get(key, 0.0))
        for key in totals:
            if key != "state_dim":
                totals[key] /= steps
        return totals


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
        contexts: Dict[str, AgentContext] | None = None,
        global_features: Dict[str, float] | None = None,
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
        self._last_congestion_features = {}
        contexts = contexts or {}
        if global_features is None:
            global_features = self._global_traffic_features()

        adaptive_scales: tuple[float, float] | None = None
        adaptive_level = "medium"
        adaptive_index = 0.0
        if self.adaptive_reward_enabled:
            adaptive_index, adaptive_level, ev_weight, network_weight = self._compute_adaptive_congestion_context()
            adaptive_scales = (ev_weight, network_weight)
            self._record_adaptive_step(adaptive_index, adaptive_level, ev_weight, network_weight)
            self._last_adaptive_reward_info = {
                "congestion_index": adaptive_index,
                "congestion_level": adaptive_level,
                "adaptive_ev_weight": ev_weight,
                "adaptive_network_weight": network_weight,
            }
        else:
            self._last_adaptive_reward_info = {}

        for tl_id in self._agent_ids:
            ctx = contexts.get(tl_id) if contexts else None
            if ctx is None:
                ctx = self._build_agent_context(tl_id)
            congestion_features = self._congestion_feature_snapshot(tl_id, ctx)
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
            congestion_reduction = float(
                np.clip(
                    (max(0.0, prev_local_queue - local_queue) / max(prev_local_queue, 1.0))
                    + (max(0.0, self._prev_network_queue - queue_len) / max(self._prev_network_queue, 1.0)),
                    0.0,
                    1.0,
                )
            )
            queue_stabilization = float(np.clip(1.0 - abs(congestion_features["queue_growth_rate"]), 0.0, 1.0))
            spillback_prevention = float(np.clip(1.0 - congestion_features["downstream_blockage_ratio"], 0.0, 1.0))
            congestion_transfer = congestion_features["congestion_transfer_risk"]
            corridor_congestion = congestion_features["corridor_congestion_pressure"]
            congestion_imbalance = float(np.clip((congestion_features["congestion_imbalance_ns"] + congestion_features["congestion_imbalance_ew"]) / 2.0, 0.0, 1.0))
            congestion_trend = float(np.clip(max(0.0, (congestion_features["congestion_trend_local"] + congestion_features["congestion_trend_network"]) / 2.0), 0.0, 1.0))

            metrics = complete_reward_metrics(
                {
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
                    "downstream_blockage": congestion_features["downstream_blockage_ratio"],
                    "traffic_stability": congestion_features["queue_growth_rate"],
                    "congestion_reduction": congestion_reduction,
                    "queue_stabilization": queue_stabilization,
                    "spillback_prevention": spillback_prevention,
                    "congestion_transfer": congestion_transfer,
                    "corridor_congestion": corridor_congestion,
                    "congestion_imbalance": congestion_imbalance,
                    "congestion_trend": congestion_trend,
                }
            )
            coordination_terms = self._coordination_terms(ctx, switch_event) if self.coordination_enabled else {
                "network_congestion": 0.0,
                "corridor_flow": 0.0,
                "downstream_blockage": 0.0,
                "traffic_stability": 0.0,
            }
            metrics.update(coordination_terms)

            reward, breakdown = compute_reward(metrics, adaptive_scales=adaptive_scales)
            if self.adaptive_reward_enabled:
                self._record_adaptive_reward_components(breakdown)
            rewards[tl_id] = reward
            self._last_reward_breakdowns[tl_id] = breakdown
            self._last_coordination_terms[tl_id] = coordination_terms
            self._last_congestion_features[tl_id] = congestion_features
            if self.controller_type in {CONTROLLER_MULTI_LEVEL_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
                multi_level = self._multi_level_reward_contributions(breakdown)
                assert global_features is not None
                multi_level.update(
                    {
                        "average_neighbor_congestion": float(ctx.neighbor_queue_avg / 20.0),
                        "average_global_congestion": float(global_features["total_network_congestion"]),
                        "average_neighbor_waiting_time": float(ctx.neighbor_waiting_time / 60.0),
                        "state_dim": float(self.state_dim),
                    }
                )
                self._last_multi_level_diagnostics[tl_id] = multi_level

            if self.debug_logs:
                self._reward_debug_counter += 1
                if self._reward_debug_counter <= 16 or self._reward_debug_counter % 60 == 0:
                    print(
                        f"[RL_REWARD] agent={tl_id} action={actions.get(tl_id, 0)} local_queue={local_queue:.1f} "
                        f"neighbor_queue={ctx.neighbor_queue_avg:.2f} downstream={ctx.downstream_congestion:.2f} "
                        f"incoming={ctx.incoming_traffic_estimate:.2f} reward={reward:.3f} "
                        f"congestion={congestion_features} coord={coordination_terms} breakdown={breakdown}"
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
        ev_present_before = self.ev_id in traci.vehicle.getIDList()
        ev_speed_before = traci.vehicle.getSpeed(self.ev_id) if ev_present_before else 0.0
        ev_wait_before = traci.vehicle.getWaitingTime(self.ev_id) if ev_present_before else 0.0
        previous_progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if ev_present_before else None
        previous_passed = {signal.tl_id for signal in previous_progress.passed} if previous_progress is not None else set()

        pre_cache = self._current_step_cache()
        pre_contexts = {tl_id: self._build_agent_context(tl_id) for tl_id in self._agent_ids}
        action_log: Dict[str, Dict[str, float]] = {}
        action_trace: Dict[str, Dict[str, float]] = {}
        red_signal_probes: list[Dict[str, float | int | str]] = []
        ev_distance_before_map: Dict[str, float] = {}
        phase_before_map = {tl_id: float(pre_cache.trafficlight_phases.get(tl_id, 0)) for tl_id in self._agent_ids}
        for tl_id in self._agent_ids:
            ctx = pre_contexts[tl_id]
            ev_distance_before_map[tl_id] = float(ctx.ev_distance)
            requested_action = int(actions.get(tl_id, 0))
            applied_action = requested_action

            # Coordinated MARL becomes conservative when the downstream corridor is
            # already saturated, which helps reduce spillback and oscillations.
            if self.coordination_enabled and ctx.downstream_congestion >= 1.0 and applied_action == 2:
                applied_action = 0
            apply_discrete_rl_action(tl_id, applied_action, ctx.ev_phase, ctx.ev_distance)
            action_log[tl_id] = {
                "action": float(applied_action),
                "requested_action": float(requested_action),
                "local_queue": ctx.local_queue,
                "neighbor_queue": ctx.neighbor_queue_avg,
                "downstream_congestion": ctx.downstream_congestion,
                "incoming_traffic_estimate": ctx.incoming_traffic_estimate,
            }
            action_trace[tl_id] = {
                "requested_action": float(requested_action),
                "applied_action": float(applied_action),
                "phase_before": phase_before_map.get(tl_id, float("nan")),
                "phase_after": float("nan"),
            }
            actions[tl_id] = applied_action

        if self.debug_logs:
            print(f"[RL_ACTION] active_agents={len(self._agent_ids)} actions={action_log}")

        try:
            traci.simulationStep()
        except traci_exceptions.FatalTraCIError:
            zero_states = {tl_id: np.zeros(self.state_dim, dtype=np.float32) for tl_id in self._agent_ids}
            zero_rewards = {tl_id: 0.0 for tl_id in self._agent_ids}
            return zero_states, zero_rewards, True, {"error": "traci_closed"}

        self._invalidate_step_cache()
        self._episode_steps += 1
        post_cache = self._current_step_cache()
        post_contexts = {tl_id: self._build_agent_context(tl_id) for tl_id in self._agent_ids}
        global_features = self._global_traffic_features()
        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO and self._agent_ids:
            sample_tl = self._agent_ids[0]
            sample_ctx = post_contexts[sample_tl]
            self._log_congestion_snapshot(self._episode_steps, sample_tl, self._congestion_feature_snapshot(sample_tl, sample_ctx))
        rewards = self._compute_agent_rewards(
            actions,
            previous_passed,
            contexts=post_contexts,
            global_features=global_features,
        )
        next_states = self._build_all_states(contexts=post_contexts, global_features=global_features)
        done = self._done()
        self._maybe_warn_inactive_congestion_features(done)
        ev_present_after = self.ev_id in traci.vehicle.getIDList()
        ev_speed_after = traci.vehicle.getSpeed(self.ev_id) if ev_present_after else ev_speed_before
        ev_wait_after = traci.vehicle.getWaitingTime(self.ev_id) if ev_present_after else ev_wait_before
        ev_distance_after_map = {
            tl_id: float(post_contexts[tl_id].ev_distance) if ev_present_after else ev_distance_before_map.get(tl_id, 0.0)
            for tl_id in self._agent_ids
        }
        for tl_id in self._agent_ids:
            if tl_id in action_trace:
                action_trace[tl_id]["phase_after"] = float(post_cache.trafficlight_phases.get(tl_id, traci.trafficlight.getPhase(tl_id)))
            probe = self._record_red_signal_probe(
                tl_id=tl_id,
                ctx=post_contexts[tl_id],
                current_phase=int(action_trace[tl_id]["phase_before"]) if tl_id in action_trace else int(traci.trafficlight.getPhase(tl_id)),
                requested_action=int(action_trace[tl_id]["requested_action"]) if tl_id in action_trace else 0,
                applied_action=int(action_trace[tl_id]["applied_action"]) if tl_id in action_trace else 0,
                speed_before=ev_speed_before,
                speed_after=ev_speed_after,
                wait_before=ev_wait_before,
                wait_after=ev_wait_after,
                distance_before=ev_distance_before_map.get(tl_id, 0.0),
                distance_after=ev_distance_after_map.get(tl_id, ev_distance_before_map.get(tl_id, 0.0)),
                rewards=rewards,
                breakdown=self._last_reward_breakdowns.get(tl_id, {}),
            )
            if probe is not None:
                red_signal_probes.append(probe)
        info = {
            "active_intersections": list(self._agent_ids),
            "reward_breakdowns": self._last_reward_breakdowns,
            "coordination_terms": self._last_coordination_terms,
            "congestion_metrics": self._last_congestion_features,
            "action_trace": action_trace,
            "red_signal_probes": red_signal_probes,
        }
        if self.controller_type in {CONTROLLER_MULTI_LEVEL_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
            info["multi_level_diagnostics"] = dict(self._last_multi_level_diagnostics)
        if self.adaptive_reward_enabled:
            info["adaptive_reward"] = dict(self._last_adaptive_reward_info)
        return next_states, rewards, done, info

    def _step_single(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        done = False
        info: Dict[str, Any] = {}
        clear_bonus = 0.0
        switch_event = 0.0
        action_trace: Dict[str, Dict[str, float]] = {}

        if self.ev_id in traci.vehicle.getIDList():
            self._route_debug()
            focus_signal = self._nearest_signal_ahead()
            self._focus_tl = focus_signal.tl_id if focus_signal else None
            if focus_signal is not None:
                requested_action = int(action)
                current_phase = traci.trafficlight.getPhase(focus_signal.tl_id)
                if self._prev_phase_idx is not None and current_phase != self._prev_phase_idx:
                    switch_event = 1.0
                dist = signal_distance(self.ev_id, focus_signal)
                ev_phase = infer_ev_green_phase(focus_signal.tl_id, list(focus_signal.route_lanes))
                applied_action = requested_action
                apply_discrete_rl_action(focus_signal.tl_id, applied_action, ev_phase, dist)
                self._prev_phase_idx = traci.trafficlight.getPhase(focus_signal.tl_id)
                action_trace[focus_signal.tl_id] = {
                    "requested_action": float(requested_action),
                    "applied_action": float(applied_action),
                    "phase_before": float(current_phase),
                    "phase_after": float(self._prev_phase_idx),
                }
                if self.debug_logs:
                    print(f"[RL_ACTION] agent={focus_signal.tl_id} action={applied_action}")
            else:
                self._prev_phase_idx = None

        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO and self._focus_tl is not None:
            info["congestion_metrics"] = self._congestion_feature_snapshot(self._focus_tl, self._build_agent_context(self._focus_tl))

        previous_progress = route_progress_for_vehicle(self.ev_id, self._route_signals) if self.ev_id in traci.vehicle.getIDList() else None
        previous_passed = {signal.tl_id for signal in previous_progress.passed} if previous_progress is not None else set()

        try:
            traci.simulationStep()
        except traci_exceptions.FatalTraCIError:
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {"error": "traci_closed"}

        self._episode_steps += 1
        if self.controller_type == CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO and self._agent_ids:
            sample_tl = self._agent_ids[0]
            sample_ctx = self._build_agent_context(sample_tl)
            self._log_congestion_snapshot(self._episode_steps, sample_tl, self._congestion_feature_snapshot(sample_tl, sample_ctx))
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

        metrics = complete_reward_metrics(
            {
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
        )
        reward, breakdown = compute_reward(metrics)

        if self.debug_logs:
            self._reward_debug_counter += 1
            if self._reward_debug_counter <= 8 or self._reward_debug_counter % 25 == 0:
                print(f"[RL_REWARD] agent={self._focus_tl or 'none'} action={action} reward={reward:.3f} breakdown={breakdown}")

        if not ev_present:
            done = True
        if self._episode_steps >= self.max_episode_steps:
            done = True
        if traci.simulation.getMinExpectedNumber() == 0 and not ev_present:
            done = True

        self._maybe_warn_inactive_congestion_features(done)
        next_state = self.get_state()
        if action_trace:
            info["action_trace"] = action_trace
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


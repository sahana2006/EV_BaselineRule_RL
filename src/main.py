from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import traci
from traci import exceptions as traci_exceptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.controller import TrafficSignalController
from src.coordinated_ppo_agent import CoordinatedPPOAgent
from src.coord_dueling_dqn_agent import CoordinatedDuelingDQNAgent
from src.dqn_agent import DQNAgent
from src.ev_detector import colorize_vehicles
from src.global_ppo_agent import GlobalPPOAgent
from src.metrics import TimeSeriesMetrics
from src.plotting import plot_comparison, plot_timeseries
from src.rl_env import (
    ACTION_DIM,
    CONTROLLER_GLOBAL_PPO,
    CONTROLLER_COORDINATED_PPO,
    CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
    CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO,
    CONTROLLER_MULTI_LEVEL_COORDINATED_PPO,
    CONTROLLER_MULTI_LEVEL_COORDINATED_DQN,
    CONTROLLER_COORDINATED_MARL,
    CONTROLLER_INDEPENDENT_MARL,
    CONTROLLER_SINGLE_AGENT,
    TrafficEnv,
)
from src.route_utils import active_emergency_vehicle_ids

CONTROLLER_RULE_BASED = "rule_based"
CONTROLLER_REGULAR_RL = "regular_rl"
CONTROLLER_INDEPENDENT_MARL_BENCH = "independent_marl"
CONTROLLER_COORDINATED_MARL_BENCH = "coordinated_marl"

COMPARISON_CSV_DIR = PROJECT_ROOT / "outputs" / "csv" / "comparisons"
COMPARISON_PLOT_DIR = PROJECT_ROOT / "outputs" / "plots" / "comparisons"

COORDINATED_PPO_CONTROLLER_METADATA: Dict[str, tuple[str, str, str]] = {
    CONTROLLER_COORDINATED_PPO: ("coordinated_ppo", "Coordinated PPO", "COORDINATED_PPO_INIT"),
    CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO: (
        "congestion_aware_coordinated_ppo",
        "Congestion-Aware Coordinated PPO",
        "CONGESTION_AWARE_COORDINATED_PPO_INIT",
    ),
    CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO: (
        "adaptive_reward_coordinated_ppo",
        "Adaptive Reward Coordinated PPO",
        "ADAPTIVE_REWARD_COORDINATED_PPO_INIT",
    ),
    CONTROLLER_MULTI_LEVEL_COORDINATED_PPO: (
        "multi_level_coordinated_ppo",
        "Multi-Level Coordinated PPO",
        "MULTI_LEVEL_COORDINATED_PPO_INIT",
    ),
}


def resolve_sumocfg_path(sumocfg: str) -> str:
    if os.path.isabs(sumocfg):
        return sumocfg
    return os.path.join(str(PROJECT_ROOT), sumocfg)


def get_sumo_binary(use_gui: bool) -> str:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set. Please set SUMO_HOME before running.")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    from sumolib import checkBinary  # type: ignore

    return checkBinary("sumo-gui" if use_gui else "sumo")


def fit_gui_to_network(view_id: str = "View #0", padding: float = 40.0) -> None:
    junction_ids = [j_id for j_id in traci.junction.getIDList() if not j_id.startswith(":")]
    if not junction_ids:
        return

    xs: list[float] = []
    ys: list[float] = []
    for junction_id in junction_ids:
        x, y = traci.junction.getPosition(junction_id)
        xs.append(x)
        ys.append(y)

    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding
    traci.gui.setBoundary(view_id, min_x, min_y, max_x, max_y)


def ensure_output_dirs(config: SimulationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    config.csv_dir.mkdir(parents=True, exist_ok=True)
    COMPARISON_CSV_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_metrics(metrics: TimeSeriesMetrics) -> Dict[str, float]:
    ev_travel_time = metrics.time_s[-1] - metrics.time_s[0] if len(metrics.time_s) >= 2 else 0.0
    ev_waiting_time = metrics.ev_waiting_time[-1] if metrics.ev_waiting_time else 0.0
    avg_network_waiting_time = float(np.mean(metrics.avg_waiting_time)) if metrics.avg_waiting_time else 0.0
    avg_queue_length = float(np.mean(metrics.queue_length)) if metrics.queue_length else 0.0
    total_throughput = float(np.sum(metrics.throughput)) if metrics.throughput else 0.0
    network_congestion = float(np.mean(metrics.queue_length) / max(np.mean(metrics.throughput) + 1.0, 1.0)) if metrics.queue_length else 0.0
    return {
        "ev_travel_time": float(ev_travel_time),
        "ev_waiting_time": float(ev_waiting_time),
        "ev_stop_count": float(metrics.ev_stops),
        "avg_network_waiting_time": avg_network_waiting_time,
        "avg_queue_length": avg_queue_length,
        "throughput": total_throughput,
        "network_congestion": network_congestion,
        "avg_waiting_time": metrics.avg_waiting_time[-1] if metrics.avg_waiting_time else 0.0,
        "ev_stops": float(metrics.ev_stops),
    }


def _strip_run_artifacts(summary: Dict[str, Any]) -> Dict[str, float]:
    summary.pop("_metrics", None)
    summary.pop("_reward_trend", None)
    summary.pop("_adaptive_diagnostics", None)
    summary.pop("_multi_level_diagnostics", None)
    return {key: float(value) for key, value in summary.items() if not key.startswith("_")}


def _float_list(values: Any) -> List[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]


def _snapshot_coordinated_policy(agent: CoordinatedPPOAgent, states: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    import torch

    if not states:
        return {}

    agent_ids = sorted(states.keys())
    batch = np.stack([states[agent_id] for agent_id in agent_ids])
    state_t = torch.as_tensor(batch, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        logits = agent.actor(state_t)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        values = agent.critic(state_t)
        entropy = torch.distributions.Categorical(logits=logits).entropy()

    snapshot: Dict[str, Dict[str, Any]] = {}
    for index, agent_id in enumerate(agent_ids):
        selected_action = int(actions[index].item())
        snapshot[agent_id] = {
            "observation": _float_list(states[agent_id]),
            "logits": _float_list(logits[index]),
            "probabilities": _float_list(probs[index]),
            "selected_action": selected_action,
            "selected_prob": float(probs[index, selected_action].item()),
            "max_prob": float(probs[index].max().item()),
            "value": float(values[index].item()),
            "entropy": float(entropy[index].item()),
        }
    return snapshot


def _snapshot_trace_row(
    *,
    controller_type: str,
    controller_label: str,
    step_index: int,
    tl_id: str,
    state_dim: int,
    policy_snapshot: Dict[str, Any],
    action_trace: Dict[str, Any] | None,
    reward: float,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "controller_type": controller_type,
        "controller_label": controller_label,
        "step_index": int(step_index),
        "tl_id": tl_id,
        "state_dim": int(state_dim),
        "observation": json.dumps(policy_snapshot.get("observation", [])),
        "logits": json.dumps(policy_snapshot.get("logits", [])),
        "probabilities": json.dumps(policy_snapshot.get("probabilities", [])),
        "selected_action": int(policy_snapshot.get("selected_action", -1)),
        "selected_prob": float(policy_snapshot.get("selected_prob", 0.0)),
        "max_prob": float(policy_snapshot.get("max_prob", 0.0)),
        "value": float(policy_snapshot.get("value", 0.0)),
        "entropy": float(policy_snapshot.get("entropy", 0.0)),
        "requested_action": int((action_trace or {}).get("requested_action", policy_snapshot.get("selected_action", -1))),
        "applied_action": int((action_trace or {}).get("applied_action", policy_snapshot.get("selected_action", -1))),
        "phase_before": float((action_trace or {}).get("phase_before", float("nan"))),
        "phase_after": float((action_trace or {}).get("phase_after", float("nan"))),
        "reward": float(reward),
    }
    if extra:
        row.update(extra)
    return row


def _write_trace_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compare_trace_rows(trace_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    controller_names = list(trace_map.keys())
    if not controller_names:
        return {"first_divergence": None, "common_steps": 0}

    keyed_rows: Dict[str, Dict[tuple[int, str], Dict[str, Any]]] = {}
    key_sets: List[set[tuple[int, str]]] = []
    for controller_name, rows in trace_map.items():
        row_map: Dict[tuple[int, str], Dict[str, Any]] = {}
        for row in rows:
            key = (int(row["step_index"]), str(row["tl_id"]))
            row_map[key] = row
        keyed_rows[controller_name] = row_map
        key_sets.append(set(row_map.keys()))

    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    for key in common_keys:
        rows = [keyed_rows[controller_name][key] for controller_name in controller_names]
        observations = [json.loads(str(row["observation"])) for row in rows]
        logits = [json.loads(str(row["logits"])) for row in rows]
        probabilities = [json.loads(str(row["probabilities"])) for row in rows]
        selected_actions = [int(row["selected_action"]) for row in rows]
        applied_actions = [int(row["applied_action"]) for row in rows]
        phases = [float(row["phase_after"]) for row in rows]

        def _all_close(vectors: List[List[float]]) -> bool:
            first = np.asarray(vectors[0], dtype=np.float32)
            return all(np.allclose(first, np.asarray(vector, dtype=np.float32), atol=1e-6, rtol=1e-6) for vector in vectors[1:])

        obs_equal = _all_close(observations)
        logits_equal = _all_close(logits)
        probs_equal = _all_close(probabilities)
        selected_equal = len(set(selected_actions)) == 1
        applied_equal = len(set(applied_actions)) == 1
        phase_equal = len(set(phases)) == 1

        if not obs_equal or not logits_equal or not probs_equal or not selected_equal or not applied_equal or not phase_equal:
            if not obs_equal:
                stage = "observation_vectors"
            elif not logits_equal or not probs_equal:
                stage = "policy_outputs"
            elif not selected_equal:
                stage = "argmax_actions"
            elif not applied_equal:
                stage = "safety_override"
            else:
                stage = "applied_phases"
            return {
                "first_divergence": {
                    "step_index": key[0],
                    "tl_id": key[1],
                    "stage": stage,
                    "observations_equal": obs_equal,
                    "logits_equal": logits_equal,
                    "probabilities_equal": probs_equal,
                    "selected_actions_equal": selected_equal,
                    "applied_actions_equal": applied_equal,
                    "phases_equal": phase_equal,
                    "rows": {controller_name: keyed_rows[controller_name][key] for controller_name in controller_names},
                },
                "common_steps": len(common_keys),
            }

    return {"first_divergence": None, "common_steps": len(common_keys)}


def print_ppo_evaluation_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    column_order = ["Rule Based", "PPO", "Coordinated PPO", "Adaptive Reward Coordinated PPO"]
    metric_rows = [
        ("ev_travel_time", "EV travel time (s)"),
        ("ev_waiting_time", "EV waiting time (s)"),
        ("ev_stops", "EV stops"),
        ("throughput", "Throughput"),
        ("avg_queue_length", "Average queue length"),
        ("network_congestion", "Network congestion"),
        ("avg_waiting_time", "Average waiting time (s)"),
    ]
    widths = [34, 14, 14, 20, 32]
    header = f"{'Metric':<{widths[0]}}" + "".join(f"{name:>{width}}" for name, width in zip(column_order, widths[1:]))
    print("\n=== PPO Controller Evaluation Comparison ===")
    print(header)
    print("-" * len(header))
    for metric_key, label in metric_rows:
        row = f"{label:<{widths[0]}}"
        for col_name, width in zip(column_order, widths[1:]):
            value = results.get(col_name, {}).get(metric_key, 0.0)
            row += f"{value:>{width}.2f}"
        print(row)
    print("=" * len(header))


def run_compare_ppo_eval(
    config: SimulationConfig,
    *,
    ppo_model_path: str,
    coordinated_ppo_model_path: str,
    adaptive_reward_coordinated_ppo_model_path: str,
    multi_level_coordinated_ppo_model_path: str,
    multi_level_coordinated_dqn_model_path: str,
    traffic_scale: float,
) -> None:
    ensure_output_dirs(config)
    print(f"[COMPARE_PPO_EVAL] traffic_scale={traffic_scale:.2f}")

    rule_summary = _strip_run_artifacts(
        run_scenario(
            config,
            mode="full_model",
            csv_name="compare_rule_based_ppo_eval.csv",
            traffic_scale=traffic_scale,
        )
    )
    ppo_summary = _strip_run_artifacts(
        run_global_ppo_scenario(
            config,
            model_path=ppo_model_path,
            csv_name="compare_global_ppo_eval.csv",
            controller_type=CONTROLLER_GLOBAL_PPO,
            traffic_scale=traffic_scale,
        )
    )
    coordinated_summary = _strip_run_artifacts(
        run_coordinated_ppo_scenario(
            config,
            model_path=coordinated_ppo_model_path,
            csv_name="compare_coordinated_ppo_eval.csv",
            controller_type=CONTROLLER_COORDINATED_PPO,
            traffic_scale=traffic_scale,
        )
    )
    adaptive_summary = _strip_run_artifacts(
        run_coordinated_ppo_scenario(
            config,
            model_path=adaptive_reward_coordinated_ppo_model_path,
            csv_name="compare_adaptive_reward_coordinated_ppo_eval.csv",
            controller_type=CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
            traffic_scale=traffic_scale,
            diagnostics=True,
        )
    )

    print_ppo_evaluation_comparison_table(
        {
            "Rule Based": rule_summary,
            "PPO": ppo_summary,
            "Coordinated PPO": coordinated_summary,
            "Adaptive Reward Coordinated PPO": adaptive_summary,
        }
    )


def run_scenario(config: SimulationConfig, mode: str, csv_name: str, *, traffic_scale: float = 1.0) -> Dict[str, Any]:
    sumo_binary = get_sumo_binary(config.use_gui)
    sumocfg_path = resolve_sumocfg_path(config.sumo_config)
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")

    start_cmd = [sumo_binary, "-c", sumocfg_path, "--step-length", str(config.step_length)]
    if traffic_scale != 1.0:
        start_cmd.extend(["--scale", str(traffic_scale)])
    if config.use_gui:
        start_cmd.append("--start")
    traci.start(start_cmd)
    metrics = TimeSeriesMetrics()

    try:
        controller = None
        initialized = False
        post_ev_steps_left: int | None = None
        for _ in range(config.max_steps):
            try:
                traci.simulationStep()
            except traci_exceptions.FatalTraCIError as err:
                print(f"[WARN] SUMO closed connection: {err}")
                break

            active_evs = active_emergency_vehicle_ids()
            if traci.simulation.getMinExpectedNumber() == 0 and config.ev_id not in traci.vehicle.getIDList():
                print("[WARN] No vehicles are running (minExpected=0). Ending scenario safely.")
                break

            if config.ev_id not in active_evs:
                if initialized:
                    if post_ev_steps_left is None:
                        post_ev_steps_left = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
                        print(f"[INFO] EV left network. Keeping simulation for {config.post_ev_buffer_seconds}s more.")
                    if post_ev_steps_left <= 0:
                        break
                    post_ev_steps_left -= 1
                continue

            if not initialized:
                controller = TrafficSignalController(ev_id=config.ev_id, mode=mode)
                controller.initialize()
                if config.use_gui:
                    fit_gui_to_network()
                initialized = True

            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            if controller:
                controller.apply_control(traci.simulation.getTime())
            metrics.capture(config.ev_id)
    finally:
        try:
            traci.close()
        except traci_exceptions.FatalTraCIError:
            pass

    csv_path = config.csv_dir / csv_name
    metrics.save_csv(csv_path)
    summary = summarize_metrics(metrics)
    summary["_metrics"] = metrics
    summary["_reward_trend"] = []
    return summary


def print_rl_startup(env: TrafficEnv) -> None:
    print(
        f"[RL_STARTUP] agents={len(env.get_agent_ids())} shared_policy={env.shared_policy_enabled} "
        f"state_dim={env.state_dim} action_space={env.action_dim} coordination={env.coordination_enabled}"
    )


def run_rl_scenario(
    config: SimulationConfig,
    model_path: str,
    csv_name: str,
    *,
    controller_type: str,
    traffic_scale: float = 1.0,
) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"RL checkpoint not found: {path}. Train first: python src/train_rl.py --model-out {path}")

    env = TrafficEnv(
        config,
        headless=not config.use_gui,
        max_episode_steps=config.max_steps,
        controller_type=controller_type,
        traffic_scale=traffic_scale,
    )
    agent = DQNAgent(state_dim=env.state_dim, action_dim=ACTION_DIM)
    agent.load(path)
    agent.epsilon = agent.epsilon_min
    metrics = TimeSeriesMetrics()
    reward_trend: List[float] = []
    diagnostics_action_histogram: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    diagnostics_selected_prob_sum = 0.0
    diagnostics_max_prob_sum = 0.0
    diagnostics_entropy_sum = 0.0
    diagnostics_action_samples = 0
    diagnostics_congestion_feature_sums: Dict[str, float] = {
        "local_congestion_index": 0.0,
        "neighbor_congestion_index": 0.0,
        "downstream_blockage_ratio": 0.0,
        "queue_growth_rate": 0.0,
        "corridor_congestion_pressure": 0.0,
        "network_congestion_score": 0.0,
        "congestion_imbalance_ns": 0.0,
        "congestion_imbalance_ew": 0.0,
        "congestion_trend_local": 0.0,
        "congestion_trend_network": 0.0,
        "congestion_transfer_risk": 0.0,
    }
    diagnostics_congestion_steps = 0
    diagnostics_multi_level_sums: Dict[str, float] = {
        "local_reward_contribution": 0.0,
        "neighbor_coordination_contribution": 0.0,
        "global_optimization_contribution": 0.0,
        "average_neighbor_congestion": 0.0,
        "average_global_congestion": 0.0,
        "average_neighbor_waiting_time": 0.0,
    }
    diagnostics_multi_level_steps = 0

    try:
        state = env.reset()
        print_rl_startup(env)
        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_DQN:
            print(
                "[MULTI_LEVEL_COORDINATED_DQN_INIT]\n"
                f"state_dim={env.state_dim}\n"
                f"action_dim={env.action_dim}\n"
                "shared_policy=True\n"
                "multi_level=True"
            )

        done = False
        while not done:
            if controller_type in {CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL, CONTROLLER_MULTI_LEVEL_COORDINATED_DQN}:
                if not isinstance(state, dict):
                    raise TypeError("Expected multi-agent state dictionary.")
                actions = agent.choose_actions(state, greedy=True)
                next_state, rewards, done, _info = env.step(actions)
                if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                    raise TypeError("Expected multi-agent outputs from environment.")
                reward_trend.append(float(sum(rewards.values())))
                if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_DQN and isinstance(_info, dict):
                    multi_level_info = _info.get("multi_level_diagnostics", {})
                    if isinstance(multi_level_info, dict) and multi_level_info:
                        agent_rows = [row for row in multi_level_info.values() if isinstance(row, dict)]
                        if agent_rows:
                            diagnostics_multi_level_steps += 1
                            for feature_name in diagnostics_multi_level_sums:
                                feature_values = [float(row.get(feature_name, 0.0)) for row in agent_rows]
                                diagnostics_multi_level_sums[feature_name] += float(sum(feature_values) / len(feature_values))
                state = next_state
            else:
                if isinstance(state, dict):
                    raise TypeError("Expected single-agent state vector.")
                action = agent.predict(state)
                next_state, reward, done, _info = env.step(action)
                if isinstance(next_state, dict) or isinstance(reward, dict):
                    raise TypeError("Expected single-agent outputs from environment.")
                reward_trend.append(float(reward))
                state = next_state

            if config.ev_id in traci.vehicle.getIDList():
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                metrics.capture(config.ev_id)
        if config.ev_id not in traci.vehicle.getIDList():
            post_ev_steps = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
            print(f"[INFO] EV left network in RL mode. Keeping simulation for {config.post_ev_buffer_seconds}s more.")
            for _ in range(post_ev_steps):
                try:
                    traci.simulationStep()
                except traci_exceptions.FatalTraCIError as err:
                    print(f"[WARN] SUMO closed connection during RL post-EV buffer: {err}")
                    break
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                if traci.simulation.getMinExpectedNumber() == 0:
                    print("[WARN] No vehicles are running (minExpected=0). Ending RL scenario safely.")
                    break
    finally:
        env.close()

    csv_path = config.csv_dir / csv_name
    metrics.save_csv(csv_path)
    summary = summarize_metrics(metrics)
    summary["_metrics"] = metrics
    summary["_reward_trend"] = reward_trend
    if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_DQN:
        summary["_multi_level_diagnostics"] = {
            key: value / max(diagnostics_multi_level_steps, 1) for key, value in diagnostics_multi_level_sums.items()
        }
    return summary


def run_coord_dueling_dqn_scenario(
    config: SimulationConfig,
    model_path: str,
    csv_name: str,
    *,
    controller_type: str,
    traffic_scale: float = 1.0,
) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Coord dueling DQN checkpoint not found: {path}. Train first: python src/train_rl.py --controller-type coordinated_dueling_dqn --model-out {path}"
        )

    env = TrafficEnv(
        config,
        headless=not config.use_gui,
        max_episode_steps=config.max_steps,
        controller_type=CONTROLLER_COORDINATED_MARL,
        traffic_scale=traffic_scale,
    )
    agent = CoordinatedDuelingDQNAgent(state_dim=env.state_dim, action_dim=ACTION_DIM)
    agent.load(path)
    agent.epsilon = agent.epsilon_min
    metrics = TimeSeriesMetrics()
    reward_trend: List[float] = []
    diagnostics_action_histogram: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    diagnostics_selected_prob_sum = 0.0
    diagnostics_max_prob_sum = 0.0
    diagnostics_entropy_sum = 0.0
    diagnostics_action_samples = 0
    diagnostics_congestion_feature_sums: Dict[str, float] = {
        "local_congestion_index": 0.0,
        "neighbor_congestion_index": 0.0,
        "downstream_blockage_ratio": 0.0,
        "queue_growth_rate": 0.0,
        "corridor_congestion_pressure": 0.0,
        "network_congestion_score": 0.0,
        "congestion_imbalance_ns": 0.0,
        "congestion_imbalance_ew": 0.0,
        "congestion_trend_local": 0.0,
        "congestion_trend_network": 0.0,
        "congestion_transfer_risk": 0.0,
    }
    diagnostics_congestion_steps = 0

    try:
        state = env.reset()
        print(
            "[COORD_DUELING_DQN_INIT]\n"
            f"agents={len(env.get_agent_ids())}\n"
            "shared_policy=True\n"
            "double_dqn=True\n"
            "dueling_network=True\n"
            "prioritized_replay=True\n"
            f"state_dim={env.state_dim}\n"
            f"action_dim={env.action_dim}"
        )
        if not isinstance(state, dict):
            raise TypeError("Coordinated dueling DQN evaluation expected a multi-agent state dictionary.")
        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        done = False
        while not done:
            actions = agent.predict_actions(state)
            next_state, rewards, done, _info = env.step(actions)
            if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                raise TypeError("Coordinated dueling DQN evaluation returned unexpected single-agent outputs.")
            reward_trend.append(float(sum(rewards.values())))
            state = next_state

            if config.ev_id in traci.vehicle.getIDList():
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                metrics.capture(config.ev_id)


        if config.ev_id not in traci.vehicle.getIDList():
            post_ev_steps = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
            mode_name = "coordinated dueling DQN"
            print(f"[INFO] EV left network in {mode_name} mode. Keeping simulation for {config.post_ev_buffer_seconds}s more.")
            for _ in range(post_ev_steps):
                try:
                    traci.simulationStep()
                except traci_exceptions.FatalTraCIError as err:
                    print(f"[WARN] SUMO closed connection during {mode_name} post-EV buffer: {err}")
                    break
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                if traci.simulation.getMinExpectedNumber() == 0:
                    print(f"[WARN] No vehicles are running (minExpected=0). Ending {mode_name} scenario safely.")
                    break
    finally:
        env.close()

    csv_path = config.csv_dir / csv_name
    metrics.save_csv(csv_path)
    summary = summarize_metrics(metrics)
    summary["_metrics"] = metrics
    summary["_reward_trend"] = reward_trend
    return summary


def run_global_ppo_scenario(
    config: SimulationConfig,
    model_path: str,
    csv_name: str,
    *,
    controller_type: str = CONTROLLER_GLOBAL_PPO,
    traffic_scale: float = 1.0,
) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Global PPO checkpoint not found: {path}. Train first: python src/train_rl.py --controller-type global_ppo --model-out {path}")

    env = TrafficEnv(
        config,
        headless=not config.use_gui,
        max_episode_steps=config.max_steps,
        controller_type=controller_type,
        traffic_scale=traffic_scale,
    )
    metrics = TimeSeriesMetrics()
    reward_trend: List[float] = []
    diagnostics_action_histogram: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    diagnostics_selected_prob_sum = 0.0
    diagnostics_max_prob_sum = 0.0
    diagnostics_entropy_sum = 0.0
    diagnostics_action_samples = 0
    diagnostics_congestion_feature_sums: Dict[str, float] = {
        "local_congestion_index": 0.0,
        "neighbor_congestion_index": 0.0,
        "downstream_blockage_ratio": 0.0,
        "queue_growth_rate": 0.0,
        "corridor_congestion_pressure": 0.0,
        "network_congestion_score": 0.0,
        "congestion_imbalance_ns": 0.0,
        "congestion_imbalance_ew": 0.0,
        "congestion_trend_local": 0.0,
        "congestion_trend_network": 0.0,
        "congestion_transfer_risk": 0.0,
    }
    diagnostics_congestion_steps = 0

    try:
        state = env.reset()
        if isinstance(state, dict):
            raise TypeError("Global PPO evaluation expected a single state vector.")
        agent = GlobalPPOAgent(
            state_dim=env.state_dim,
            action_dim=ACTION_DIM,
            learning_rate=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.001,
            value_coef=0.5,
            ppo_epochs=4,
            batch_size=256,
        )
        agent.load(path)
        print(
            "[GLOBAL_PPO_INIT]\n"
            f"state_dim={env.state_dim}\n"
            f"action_dim={env.action_dim}\n"
            "lr=1e-3\n"
            "entropy_coef=0.001\n"
            "reward_normalization=True"
        )

        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        done = False
        while not done:
            action = agent.predict(state)
            next_state, reward, done, _info = env.step(action)
            if isinstance(next_state, dict) or isinstance(reward, dict):
                raise TypeError("Global PPO evaluation returned unexpected multi-agent outputs.")
            reward_trend.append(float(reward))
            state = next_state

            if config.ev_id in traci.vehicle.getIDList():
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                metrics.capture(config.ev_id)


        if config.ev_id not in traci.vehicle.getIDList():
            post_ev_steps = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
            mode_name = "global PPO"
            print(f"[INFO] EV left network in {mode_name} mode. Keeping simulation for {config.post_ev_buffer_seconds}s more.")
            for _ in range(post_ev_steps):
                try:
                    traci.simulationStep()
                except traci_exceptions.FatalTraCIError as err:
                    print(f"[WARN] SUMO closed connection during {mode_name} post-EV buffer: {err}")
                    break
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                if traci.simulation.getMinExpectedNumber() == 0:
                    print(f"[WARN] No vehicles are running (minExpected=0). Ending {mode_name} scenario safely.")
                    break
    finally:
        env.close()

    csv_path = config.csv_dir / csv_name
    metrics.save_csv(csv_path)
    summary = summarize_metrics(metrics)
    summary["_metrics"] = metrics
    summary["_reward_trend"] = reward_trend
    return summary


def run_coordinated_ppo_scenario(
    config: SimulationConfig,
    model_path: str,
    csv_name: str,
    *,
    controller_type: str = CONTROLLER_COORDINATED_PPO,
    traffic_scale: float = 1.0,
    diagnostics: bool = False,
    trace_rows: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    path = Path(model_path)
    if controller_type not in COORDINATED_PPO_CONTROLLER_METADATA:
        raise ValueError(f"Unsupported coordinated PPO controller type: {controller_type}")
    controller_name, controller_label, init_tag = COORDINATED_PPO_CONTROLLER_METADATA[controller_type]
    if not path.is_file():
        raise FileNotFoundError(
            f"{controller_label} checkpoint not found: {path}. Train first: python src/train_rl.py --controller-type {controller_name} --model-out {path}"
        )

    env = TrafficEnv(
        config,
        headless=not config.use_gui,
        max_episode_steps=config.max_steps,
        controller_type=controller_type,
        traffic_scale=traffic_scale,
    )
    metrics = TimeSeriesMetrics()
    reward_trend: List[float] = []
    diagnostics_action_histogram: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    diagnostics_selected_prob_sum = 0.0
    diagnostics_max_prob_sum = 0.0
    diagnostics_entropy_sum = 0.0
    diagnostics_action_samples = 0
    diagnostics_congestion_feature_sums: Dict[str, float] = {
        "local_congestion_index": 0.0,
        "neighbor_congestion_index": 0.0,
        "downstream_blockage_ratio": 0.0,
        "queue_growth_rate": 0.0,
        "corridor_congestion_pressure": 0.0,
        "network_congestion_score": 0.0,
        "congestion_imbalance_ns": 0.0,
        "congestion_imbalance_ew": 0.0,
        "congestion_trend_local": 0.0,
        "congestion_trend_network": 0.0,
        "congestion_transfer_risk": 0.0,
    }
    diagnostics_congestion_steps = 0
    diagnostics_adaptive_mode_steps: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    diagnostics_adaptive_weight_sum = 0.0
    diagnostics_adaptive_network_weight_sum = 0.0
    diagnostics_adaptive_steps = 0
    diagnostics_multi_level_sums: Dict[str, float] = {
        "local_reward_contribution": 0.0,
        "neighbor_coordination_contribution": 0.0,
        "global_optimization_contribution": 0.0,
        "average_neighbor_congestion": 0.0,
        "average_global_congestion": 0.0,
        "average_neighbor_waiting_time": 0.0,
    }
    diagnostics_multi_level_steps = 0
    trace_step_index = 0
    enable_diagnostics = diagnostics or controller_type in {CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO, CONTROLLER_MULTI_LEVEL_COORDINATED_PPO}

    try:
        state = env.reset()
        if not isinstance(state, dict):
            raise TypeError("Coordinated PPO evaluation expected a multi-agent state dictionary.")
        agent_ids = env.get_agent_ids() or sorted(state.keys())
        agent = CoordinatedPPOAgent(
            state_dim=env.state_dim,
            action_dim=ACTION_DIM,
            learning_rate=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.001,
            value_coef=0.5,
            ppo_epochs=4,
            batch_size=256,
        )
        agent.load(path)
        try:
            import torch

            checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
        except TypeError:
            import torch

            checkpoint = torch.load(path, map_location=agent.device)
        except Exception:
            checkpoint = None

        if checkpoint is not None:
            ckpt_state_dim = checkpoint.get("state_dim")
            ckpt_action_dim = checkpoint.get("action_dim")
            ckpt_controller = checkpoint.get("controller_type")
            actor_fp = "unknown"
            try:
                import hashlib

                digest = hashlib.sha256()
                actor_state = checkpoint.get("actor", {})
                if isinstance(actor_state, dict):
                    for key in sorted(actor_state.keys()):
                        digest.update(actor_state[key].cpu().numpy().tobytes())
                    actor_fp = digest.hexdigest()[:16]
            except Exception:
                pass
            ckpt_mtime = path.stat().st_mtime
            print(
                "[MODEL_LOAD]\n"
                f"checkpoint={path.resolve()}\n"
                f"checkpoint_mtime={ckpt_mtime:.3f}\n"
                f"checkpoint_size_bytes={path.stat().st_size}\n"
                f"checkpoint_controller_type={ckpt_controller}\n"
                f"expected_controller_type={controller_type}\n"
                f"checkpoint_state_dim={ckpt_state_dim}\n"
                f"checkpoint_action_dim={ckpt_action_dim}\n"
                f"checkpoint_actor_fingerprint={actor_fp}\n"
                f"env_state_dim={env.state_dim}\n"
                f"env_action_dim={env.action_dim}"
            )
            if ckpt_controller is not None and ckpt_controller != controller_type:
                raise ValueError(
                    f"Checkpoint controller_type={ckpt_controller!r} does not match expected {controller_type!r}. "
                    f"Refusing to load {path} for {controller_label} evaluation."
                )
            if ckpt_state_dim is not None and int(ckpt_state_dim) != int(env.state_dim):
                print("[WARNING] checkpoint state_dim does not match environment state_dim.")
            if ckpt_action_dim is not None and int(ckpt_action_dim) != int(env.action_dim):
                print("[WARNING] checkpoint action_dim does not match environment action_dim.")

        print(
            f"[{init_tag}]\n"
            f"agents={len(agent_ids)}\n"
            "shared_policy=True\n"
            f"state_dim={env.state_dim}\n"
            f"action_dim={env.action_dim}\n"
            "coordination=True"
        )
        if controller_type == CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO:
            print("adaptive_reward=True (dynamic EV/network reward weighting only)")
        if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_PPO:
            print("multi_level=True (local + neighbor + global observations only)")

        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        done = False
        while not done:
            policy_snapshot = _snapshot_coordinated_policy(agent, state) if trace_rows is not None else None
            if enable_diagnostics:
                actions, _, _, entropy_map, selected_prob_map, max_prob_map = agent.act(state, deterministic=True, include_probs=True)
                diagnostics_action_histogram[0] += sum(1 for action in actions.values() if action == 0)
                diagnostics_action_histogram[1] += sum(1 for action in actions.values() if action == 1)
                diagnostics_action_histogram[2] += sum(1 for action in actions.values() if action == 2)
                diagnostics_selected_prob_sum += float(sum(selected_prob_map.values()))
                diagnostics_max_prob_sum += float(sum(max_prob_map.values()))
                diagnostics_entropy_sum += float(sum(entropy_map.values()))
                diagnostics_action_samples += max(len(selected_prob_map), 1)
            else:
                actions = agent.predict_actions(state)
            next_state, rewards, done, info = env.step(actions)
            if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                raise TypeError("Coordinated PPO evaluation returned unexpected single-agent outputs.")
            reward_trend.append(float(sum(rewards.values())))
            if enable_diagnostics and isinstance(info, dict):
                if controller_type == CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO:
                    adaptive_info = info.get("adaptive_reward", {})
                    if isinstance(adaptive_info, dict) and adaptive_info:
                        level = str(adaptive_info.get("congestion_level", "medium"))
                        diagnostics_adaptive_mode_steps[level] = diagnostics_adaptive_mode_steps.get(level, 0) + 1
                        diagnostics_adaptive_weight_sum += float(adaptive_info.get("adaptive_ev_weight", 0.0))
                        diagnostics_adaptive_network_weight_sum += float(adaptive_info.get("adaptive_network_weight", 0.0))
                        diagnostics_adaptive_steps += 1
                if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_PPO:
                    multi_level_info = info.get("multi_level_diagnostics", {})
                    if isinstance(multi_level_info, dict) and multi_level_info:
                        agent_rows = [row for row in multi_level_info.values() if isinstance(row, dict)]
                        if agent_rows:
                            diagnostics_multi_level_steps += 1
                            for feature_name in diagnostics_multi_level_sums:
                                feature_values = [float(row.get(feature_name, 0.0)) for row in agent_rows]
                                diagnostics_multi_level_sums[feature_name] += float(sum(feature_values) / len(feature_values))
                if diagnostics:
                    congestion_metrics = info.get("congestion_metrics", {})
                    if isinstance(congestion_metrics, dict) and congestion_metrics:
                        agent_rows = [row for row in congestion_metrics.values() if isinstance(row, dict)]
                        if agent_rows:
                            for feature_name in diagnostics_congestion_feature_sums:
                                feature_values = [float(row.get(feature_name, 0.0)) for row in agent_rows]
                                diagnostics_congestion_feature_sums[feature_name] += float(sum(feature_values) / len(feature_values))
                            diagnostics_congestion_steps += 1
            if trace_rows is not None and policy_snapshot is not None and isinstance(info, dict):
                action_trace = info.get("action_trace", {}) if isinstance(info.get("action_trace", {}), dict) else {}
                multi_level_diag = info.get("multi_level_diagnostics", {}) if isinstance(info.get("multi_level_diagnostics", {}), dict) else {}
                for tl_id in sorted(state.keys()):
                    extra: Dict[str, Any] = {}
                    if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_PPO:
                        diag_row = multi_level_diag.get(tl_id, {}) if isinstance(multi_level_diag, dict) else {}
                        if isinstance(diag_row, dict):
                            extra = {
                                "local_reward_contribution": float(diag_row.get("local_reward_contribution", 0.0)),
                                "neighbor_coordination_contribution": float(diag_row.get("neighbor_coordination_contribution", 0.0)),
                                "global_optimization_contribution": float(diag_row.get("global_optimization_contribution", 0.0)),
                                "average_neighbor_congestion": float(diag_row.get("average_neighbor_congestion", 0.0)),
                                "average_global_congestion": float(diag_row.get("average_global_congestion", 0.0)),
                                "average_neighbor_waiting_time": float(diag_row.get("average_neighbor_waiting_time", 0.0)),
                            }
                    trace_rows.append(
                        _snapshot_trace_row(
                            controller_type=controller_type,
                            controller_label=controller_label,
                            step_index=trace_step_index,
                            tl_id=tl_id,
                            state_dim=env.state_dim,
                            policy_snapshot=policy_snapshot.get(tl_id, {}),
                            action_trace=action_trace.get(tl_id, {}) if isinstance(action_trace, dict) else None,
                            reward=float(rewards.get(tl_id, 0.0)),
                            extra=extra,
                        )
                    )
                trace_step_index += 1

            state = next_state

            if config.ev_id in traci.vehicle.getIDList():
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                metrics.capture(config.ev_id)


        if enable_diagnostics:
            total_actions = sum(diagnostics_action_histogram.values()) or 1
            avg_selected_prob = diagnostics_selected_prob_sum / max(diagnostics_action_samples, 1)
            avg_max_prob = diagnostics_max_prob_sum / max(diagnostics_action_samples, 1)
            avg_entropy = diagnostics_entropy_sum / max(diagnostics_action_samples, 1)
            print(
                f"[EVAL_DIAGNOSTICS] action_0={diagnostics_action_histogram[0]} ({diagnostics_action_histogram[0] / total_actions:.2%}) "
                f"action_1={diagnostics_action_histogram[1]} ({diagnostics_action_histogram[1] / total_actions:.2%}) "
                f"action_2={diagnostics_action_histogram[2]} ({diagnostics_action_histogram[2] / total_actions:.2%})"
            )
            print(
                f"[EVAL_PPO_DIAGNOSTICS] selected_prob={avg_selected_prob:.4f} max_prob={avg_max_prob:.4f} entropy={avg_entropy:.4f}"
            )
            if controller_type == CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO:
                adaptive_steps = max(diagnostics_adaptive_steps, 1)
                mode_total = sum(diagnostics_adaptive_mode_steps.values()) or 1
                adaptive_diag = env.get_adaptive_episode_diagnostics()
                print(
                    f"[EVAL_ADAPTIVE_REWARD] avg_ev_weight={diagnostics_adaptive_weight_sum / adaptive_steps:.3f} "
                    f"avg_network_weight={diagnostics_adaptive_network_weight_sum / adaptive_steps:.3f} "
                    f"pct_low={100.0 * diagnostics_adaptive_mode_steps.get('low', 0) / mode_total:.1f}% "
                    f"pct_medium={100.0 * diagnostics_adaptive_mode_steps.get('medium', 0) / mode_total:.1f}% "
                    f"pct_high={100.0 * diagnostics_adaptive_mode_steps.get('high', 0) / mode_total:.1f}% "
                    f"episode_ev_component={adaptive_diag.get('ev_reward_component', 0.0):.3f} "
                    f"episode_network_component={adaptive_diag.get('network_reward_component', 0.0):.3f}"
                )
            if diagnostics:
                avg_congestion_features = {
                    key: value / max(diagnostics_congestion_steps, 1)
                    for key, value in diagnostics_congestion_feature_sums.items()
                }
                print(
                    f"[EVAL_CONGESTION] local_index={avg_congestion_features['local_congestion_index']:.3f} "
                    f"neighbor_index={avg_congestion_features['neighbor_congestion_index']:.3f} "
                    f"downstream_blockage={avg_congestion_features['downstream_blockage_ratio']:.3f} "
                    f"queue_growth={avg_congestion_features['queue_growth_rate']:.3f} "
                    f"corridor_pressure={avg_congestion_features['corridor_congestion_pressure']:.3f} "
                    f"network_score={avg_congestion_features['network_congestion_score']:.3f}"
                )
            if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_PPO:
                avg_multi_level_features = {
                    key: value / max(diagnostics_multi_level_steps, 1)
                    for key, value in diagnostics_multi_level_sums.items()
                }
                print(
                    f"[EVAL_MULTI_LEVEL] controller_type={controller_type} state_dim={env.state_dim} "
                    f"local_reward={avg_multi_level_features['local_reward_contribution']:.3f} "
                    f"neighbor_coordination={avg_multi_level_features['neighbor_coordination_contribution']:.3f} "
                    f"global_optimization={avg_multi_level_features['global_optimization_contribution']:.3f} "
                    f"avg_neighbor_congestion={avg_multi_level_features['average_neighbor_congestion']:.3f} "
                    f"avg_global_congestion={avg_multi_level_features['average_global_congestion']:.3f} "
                    f"avg_neighbor_waiting_time={avg_multi_level_features['average_neighbor_waiting_time']:.3f} "
                    f"action_0={diagnostics_action_histogram[0]} ({diagnostics_action_histogram[0] / total_actions:.2%}) "
                    f"action_1={diagnostics_action_histogram[1]} ({diagnostics_action_histogram[1] / total_actions:.2%}) "
                    f"action_2={diagnostics_action_histogram[2]} ({diagnostics_action_histogram[2] / total_actions:.2%})"
                )

        if config.ev_id not in traci.vehicle.getIDList():
            post_ev_steps = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
            mode_name = controller_label.lower()
            print(f"[INFO] EV left network in {mode_name} mode. Keeping simulation for {config.post_ev_buffer_seconds}s more.")
            for _ in range(post_ev_steps):
                try:
                    traci.simulationStep()
                except traci_exceptions.FatalTraCIError as err:
                    print(f"[WARN] SUMO closed connection during {mode_name} post-EV buffer: {err}")
                    break
                colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
                if traci.simulation.getMinExpectedNumber() == 0:
                    print(f"[WARN] No vehicles are running (minExpected=0). Ending {mode_name} scenario safely.")
                    break
    finally:
        env.close()

    csv_path = config.csv_dir / csv_name
    metrics.save_csv(csv_path)
    summary = summarize_metrics(metrics)
    summary["_metrics"] = metrics
    summary["_reward_trend"] = reward_trend
    if controller_type == CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO:
        summary["_adaptive_diagnostics"] = env.get_adaptive_episode_diagnostics()
    if controller_type == CONTROLLER_MULTI_LEVEL_COORDINATED_PPO:
        summary["_multi_level_diagnostics"] = {
            key: value / max(diagnostics_multi_level_steps, 1) for key, value in diagnostics_multi_level_sums.items()
        }
    return summary


def run_compare_coordinated_ppo_pipeline(
    config: SimulationConfig,
    *,
    coordinated_ppo_model_path: str,
    adaptive_reward_coordinated_ppo_model_path: str,
    congestion_aware_coordinated_ppo_model_path: str,
    multi_level_coordinated_ppo_model_path: str,
    multi_level_coordinated_dqn_model_path: str | None = None,
    traffic_scale: float,
) -> None:
    ensure_output_dirs(config)
    trace_dir = config.log_dir / "ppo_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    controller_specs = [
        (
            CONTROLLER_COORDINATED_PPO,
            coordinated_ppo_model_path,
            "Standard PPO",
            config.coordinated_ppo_model_csv_name,
            False,
        ),
        (
            CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
            adaptive_reward_coordinated_ppo_model_path,
            "Adaptive Reward PPO",
            config.adaptive_reward_coordinated_ppo_model_csv_name,
            True,
        ),
        (
            CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO,
            congestion_aware_coordinated_ppo_model_path,
            "Congestion-Aware PPO",
            config.congestion_aware_coordinated_ppo_model_csv_name,
            True,
        ),
        (
            CONTROLLER_MULTI_LEVEL_COORDINATED_PPO,
            multi_level_coordinated_ppo_model_path,
            "Multi-Level PPO",
            config.multi_level_coordinated_ppo_model_csv_name,
            True,
        ),
    ]

    summaries: Dict[str, Dict[str, float]] = {}
    trace_map: Dict[str, List[Dict[str, Any]]] = {}
    trace_paths: Dict[str, Path] = {}

    for controller_type, model_path, label, csv_name, diagnostics in controller_specs:
        rows: List[Dict[str, Any]] = []
        summary = run_coordinated_ppo_scenario(
            config,
            model_path=model_path,
            csv_name=csv_name,
            controller_type=controller_type,
            traffic_scale=traffic_scale,
            diagnostics=diagnostics,
            trace_rows=rows,
        )
        summaries[label] = _strip_run_artifacts(summary)
        trace_map[label] = rows
        trace_path = trace_dir / f"{controller_type}_inference_trace.csv"
        trace_paths[label] = trace_path
        _write_trace_csv(trace_path, rows)
        print(f"[TRACE_CAPTURE] controller={label} rows={len(rows)} trace_csv={trace_path}")

    metric_rows = [
        ("ev_travel_time", "EV travel time (s)"),
        ("ev_waiting_time", "EV waiting time (s)"),
        ("ev_stops", "EV stops"),
        ("avg_network_waiting_time", "Average network waiting time (s)"),
        ("avg_queue_length", "Average queue length"),
        ("throughput", "Throughput"),
        ("network_congestion", "Network congestion"),
        ("avg_waiting_time", "Average waiting time (s)"),
    ]
    column_order = [label for _, _, label, _, _ in controller_specs]
    widths = [34] + [24] * len(column_order)
    header = f"{'Metric':<{widths[0]}}" + "".join(f"{name:>{width}}" for name, width in zip(column_order, widths[1:]))
    print("\n=== Coordinated PPO Pipeline Comparison ===")
    print(header)
    print("-" * len(header))
    for metric_key, label in metric_rows:
        row = f"{label:<{widths[0]}}"
        for controller_label, width in zip(column_order, widths[1:]):
            value = summaries.get(controller_label, {}).get(metric_key, 0.0)
            row += f"{value:>{width}.2f}"
        print(row)
    print("=" * len(header))

    comparison = _compare_trace_rows(trace_map)
    first_divergence = comparison.get("first_divergence")
    if first_divergence is None:
        print("[TRACE_COMPARISON] all four controllers stayed aligned for every comparable timestep.")
        print("[TRACE_COMPARISON] This means the pipeline did not diverge in observations, policy outputs, argmax actions, overrides, or applied phases.")
    else:
        step_index = first_divergence["step_index"]
        tl_id = first_divergence["tl_id"]
        stage = first_divergence["stage"]
        print(f"[TRACE_COMPARISON] first_divergence_step={step_index} tl_id={tl_id} stage={stage}")
        print(
            f"[TRACE_COMPARISON] observations_equal={first_divergence['observations_equal']} "
            f"logits_equal={first_divergence['logits_equal']} probabilities_equal={first_divergence['probabilities_equal']} "
            f"selected_actions_equal={first_divergence['selected_actions_equal']} applied_actions_equal={first_divergence['applied_actions_equal']} "
            f"phases_equal={first_divergence['phases_equal']}"
        )
        for controller_label in column_order:
            row = first_divergence["rows"][controller_label]
            obs_preview = json.loads(row["observation"])
            logits_preview = json.loads(row["logits"])
            probs_preview = json.loads(row["probabilities"])
            print(
                f"[TRACE_ROW] controller={controller_label} tl_id={tl_id} step={step_index} "
                f"observation={obs_preview} logits={logits_preview} probabilities={probs_preview} "
                f"selected_action={row['selected_action']} requested_action={row['requested_action']} "
                f"applied_action={row['applied_action']} phase_before={row['phase_before']} phase_after={row['phase_after']}"
            )

    print("[TRACE_FILES]")
    for controller_label, trace_path in trace_paths.items():
        print(f"{controller_label}: {trace_path}")


def execute_single(
    config: SimulationConfig,
    mode: str = "full_model",
    model_path: str | None = None,
    coord_dueling_model_path: str | None = None,
    ppo_model_path: str | None = None,
    coordinated_ppo_model_path: str | None = None,
    congestion_aware_coordinated_ppo_model_path: str | None = None,
    adaptive_reward_coordinated_ppo_model_path: str | None = None,
    multi_level_coordinated_ppo_model_path: str | None = None,
    multi_level_coordinated_dqn_model_path: str | None = None,
    independent_marl_model_path: str | None = None,
    coordinated_marl_model_path: str | None = None,
    traffic_scale: float = 1.0,
) -> None:
    ensure_output_dirs(config)
    print(f"Running single scenario mode={mode} traffic_scale={traffic_scale:.2f}")
    if mode == "rl_model":
        if not model_path:
            raise ValueError("rl_model requires --model-path")
        summary = run_rl_scenario(
            config,
            model_path=model_path,
            csv_name=config.rl_model_csv_name,
            controller_type=CONTROLLER_SINGLE_AGENT,
            traffic_scale=traffic_scale,
        )
    elif mode == "coordinated_ppo":
        if not coordinated_ppo_model_path:
            raise ValueError("coordinated_ppo requires --coordinated-ppo-model-path")
        summary = run_coordinated_ppo_scenario(
            config,
            model_path=coordinated_ppo_model_path,
            csv_name=config.coordinated_ppo_model_csv_name,
            controller_type=CONTROLLER_COORDINATED_PPO,
            traffic_scale=traffic_scale,
        )
    elif mode == "congestion_aware_coordinated_ppo":
        if not congestion_aware_coordinated_ppo_model_path:
            raise ValueError("congestion_aware_coordinated_ppo requires --congestion-aware-coordinated-ppo-model-path")
        summary = run_coordinated_ppo_scenario(
            config,
            model_path=congestion_aware_coordinated_ppo_model_path,
            csv_name=config.congestion_aware_coordinated_ppo_model_csv_name,
            controller_type=CONTROLLER_CONGESTION_AWARE_COORDINATED_PPO,
            traffic_scale=traffic_scale,
            diagnostics=True,
        )
    elif mode == "adaptive_reward_coordinated_ppo":
        if not adaptive_reward_coordinated_ppo_model_path:
            raise ValueError("adaptive_reward_coordinated_ppo requires --adaptive-reward-coordinated-ppo-model-path")
        summary = run_coordinated_ppo_scenario(
            config,
            model_path=adaptive_reward_coordinated_ppo_model_path,
            csv_name=config.adaptive_reward_coordinated_ppo_model_csv_name,
            controller_type=CONTROLLER_ADAPTIVE_REWARD_COORDINATED_PPO,
            traffic_scale=traffic_scale,
            diagnostics=False,
        )
    elif mode == "multi_level_coordinated_ppo":
        if not multi_level_coordinated_ppo_model_path:
            raise ValueError("multi_level_coordinated_ppo requires --multi-level-coordinated-ppo-model-path")
        summary = run_coordinated_ppo_scenario(
            config,
            model_path=multi_level_coordinated_ppo_model_path,
            csv_name=config.multi_level_coordinated_ppo_model_csv_name,
            controller_type=CONTROLLER_MULTI_LEVEL_COORDINATED_PPO,
            traffic_scale=traffic_scale,
            diagnostics=True,
        )
    elif mode == "multi_level_coordinated_dqn":
        if not multi_level_coordinated_dqn_model_path:
            raise ValueError("multi_level_coordinated_dqn requires --multi-level-coordinated-dqn-model-path")
        summary = run_rl_scenario(
            config,
            model_path=multi_level_coordinated_dqn_model_path,
            csv_name=config.multi_level_coordinated_dqn_model_csv_name,
            controller_type=CONTROLLER_MULTI_LEVEL_COORDINATED_DQN,
            traffic_scale=traffic_scale,
        )
    elif mode == "coordinated_dueling_dqn":
        if not coord_dueling_model_path:
            raise ValueError("coordinated_dueling_dqn requires --coord-dueling-model-path")
        summary = run_coord_dueling_dqn_scenario(
            config,
            model_path=coord_dueling_model_path,
            csv_name=config.coord_dueling_dqn_model_csv_name,
            controller_type=CONTROLLER_COORDINATED_MARL,
            traffic_scale=traffic_scale,
        )
    elif mode == "global_ppo":
        if not ppo_model_path:
            raise ValueError("global_ppo requires --ppo-model-path")
        summary = run_global_ppo_scenario(
            config,
            model_path=ppo_model_path,
            csv_name=config.global_ppo_model_csv_name,
            controller_type=CONTROLLER_GLOBAL_PPO,
            traffic_scale=traffic_scale,
        )
    elif mode == "independent_marl_model":
        if not independent_marl_model_path:
            raise ValueError("independent_marl_model requires --independent-marl-model-path")
        summary = run_rl_scenario(
            config,
            model_path=independent_marl_model_path,
            csv_name="independent_marl_model_metrics.csv",
            controller_type=CONTROLLER_INDEPENDENT_MARL,
            traffic_scale=traffic_scale,
        )
    elif mode in {"coordinated_marl_model", "marl_model"}:
        if not coordinated_marl_model_path:
            raise ValueError("coordinated_marl_model requires --coordinated-marl-model-path")
        summary = run_rl_scenario(
            config,
            model_path=coordinated_marl_model_path,
            csv_name="coordinated_marl_model_metrics.csv",
            controller_type=CONTROLLER_COORDINATED_MARL,
            traffic_scale=traffic_scale,
        )
    else:
        summary = run_scenario(config, mode=mode, csv_name=config.csv_name, traffic_scale=traffic_scale)

    metrics = summary.pop("_metrics", None)
    summary.pop("_reward_trend", None)
    adaptive_diag = summary.pop("_adaptive_diagnostics", None)
    multi_level_diag = summary.pop("_multi_level_diagnostics", None)
    if metrics is None:
        raise RuntimeError("Metrics were not captured.")
    if mode == "coordinated_dueling_dqn":
        plot_dir = config.plot_dir / "coord_dueling_dqn"
    else:
        plot_dir = config.plot_dir
    plot_timeseries(metrics, plot_dir, prefix=mode)
    print(f"Scenario summary: {summary}")
    if multi_level_diag:
        print(
            f"[MULTI_LEVEL_EVAL] controller_type={mode} ev_travel_time={summary.get('ev_travel_time', 0.0):.2f} "
            f"ev_waiting_time={summary.get('ev_waiting_time', 0.0):.2f} ev_stops={summary.get('ev_stops', 0.0):.0f} "
            f"avg_network_waiting_time={summary.get('avg_network_waiting_time', 0.0):.2f} "
            f"avg_queue_length={summary.get('avg_queue_length', 0.0):.2f} throughput={summary.get('throughput', 0.0):.2f} "
            f"network_congestion={summary.get('network_congestion', 0.0):.3f} avg_waiting_time={summary.get('avg_waiting_time', 0.0):.2f} "
            f"local_reward={multi_level_diag.get('local_reward_contribution', 0.0):.3f} "
            f"neighbor_coordination={multi_level_diag.get('neighbor_coordination_contribution', 0.0):.3f} "
            f"global_optimization={multi_level_diag.get('global_optimization_contribution', 0.0):.3f} "
            f"avg_neighbor_congestion={multi_level_diag.get('average_neighbor_congestion', 0.0):.3f} "
            f"avg_global_congestion={multi_level_diag.get('average_global_congestion', 0.0):.3f} "
            f"avg_neighbor_waiting_time={multi_level_diag.get('average_neighbor_waiting_time', 0.0):.3f}"
        )
    if adaptive_diag:
        print(f"Adaptive reward diagnostics: {adaptive_diag}")


def execute_all(config: SimulationConfig, rl_summary: Dict[str, float] | None = None, *, traffic_scale: float = 1.0) -> None:
    ensure_output_dirs(config)

    print("Running baseline: fixed-time")
    fixed_summary = run_scenario(config, mode="fixed_time", csv_name=config.fixed_csv_name, traffic_scale=traffic_scale)
    print("Running baseline: intrusive-only")
    intrusive_summary = run_scenario(config, mode="intrusive_only", csv_name=config.intrusive_csv_name, traffic_scale=traffic_scale)
    print("Running full model: DRRS + green-wave + recovery")
    full_summary = run_scenario(config, mode="full_model", csv_name=config.csv_name, traffic_scale=traffic_scale)

    fixed_metrics = fixed_summary.pop("_metrics", None)
    intrusive_metrics = intrusive_summary.pop("_metrics", None)
    full_metrics = full_summary.pop("_metrics", None)
    fixed_summary.pop("_reward_trend", None)
    intrusive_summary.pop("_reward_trend", None)
    full_summary.pop("_reward_trend", None)

    if full_metrics is None or fixed_metrics is None or intrusive_metrics is None:
        raise RuntimeError("Scenario metrics were not captured.")

    plot_timeseries(full_metrics, config.plot_dir, prefix="full_model")
    plot_comparison(
        {"ev_travel_time": fixed_summary["ev_travel_time"], "avg_waiting_time": fixed_summary["avg_waiting_time"], "ev_stops": fixed_summary["ev_stops"]},
        {"ev_travel_time": intrusive_summary["ev_travel_time"], "avg_waiting_time": intrusive_summary["avg_waiting_time"], "ev_stops": intrusive_summary["ev_stops"]},
        {"ev_travel_time": full_summary["ev_travel_time"], "avg_waiting_time": full_summary["avg_waiting_time"], "ev_stops": full_summary["ev_stops"]},
        config.plot_dir,
        rl_model=rl_summary,
    )

    print("\n=== Comparison ===")
    print(f"Fixed-time: {fixed_summary}")
    print(f"Intrusive only: {intrusive_summary}")
    print(f"Full model: {full_summary}")
    if rl_summary is not None:
        print(f"RL model: {rl_summary}")


def _controller_csv_path(controller_key: str) -> Path:
    mapping = {
        CONTROLLER_RULE_BASED: COMPARISON_CSV_DIR / "rule_based_results.csv",
        CONTROLLER_REGULAR_RL: COMPARISON_CSV_DIR / "regular_rl_results.csv",
        CONTROLLER_INDEPENDENT_MARL_BENCH: COMPARISON_CSV_DIR / "independent_marl_results.csv",
        CONTROLLER_COORDINATED_MARL_BENCH: COMPARISON_CSV_DIR / "coordinated_marl_results.csv",
    }
    return mapping[controller_key]


def write_controller_results(controller_key: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = _controller_csv_path(controller_key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def average_trial_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    averaged: Dict[str, float] = {}
    metric_keys = [
        "ev_travel_time",
        "ev_waiting_time",
        "ev_stop_count",
        "avg_network_waiting_time",
        "avg_queue_length",
        "throughput",
        "network_congestion",
        "episode_reward",
    ]
    for key in metric_keys:
        values = [float(row[key]) for row in rows if key in row]
        averaged[key] = float(np.mean(values)) if values else 0.0
    return averaged


def plot_benchmark_results(averages: Dict[str, Dict[str, Any]]) -> None:
    COMPARISON_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Rule-Based", "Single-Agent RL", "Independent MARL", "Coordinated MARL"]
    keys = [
        CONTROLLER_RULE_BASED,
        CONTROLLER_REGULAR_RL,
        CONTROLLER_INDEPENDENT_MARL_BENCH,
        CONTROLLER_COORDINATED_MARL_BENCH,
    ]
    metric_specs = [
        ("ev_travel_time", "EV Travel Time (s)", "comparison_ev_travel_time.png"),
        ("avg_network_waiting_time", "Average Waiting Time (s)", "comparison_avg_waiting_time.png"),
        ("avg_queue_length", "Average Queue Length (veh)", "comparison_queue_length.png"),
        ("throughput", "Throughput", "comparison_throughput.png"),
        ("ev_stop_count", "EV Stop Count", "comparison_ev_stops.png"),
        ("network_congestion", "Network Congestion", "comparison_network_congestion.png"),
    ]
    colors = ["#5B6C5D", "#4472C4", "#8E44AD", "#E67E22"]
    for metric_key, title, filename in metric_specs:
        values = [averages[key][metric_key] for key in keys]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, values, color=colors)
        plt.title(title)
        plt.ylabel(title)
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(COMPARISON_PLOT_DIR / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(11, 5))
    for key, label, color in [
        (CONTROLLER_REGULAR_RL, "Single-Agent RL", "#4472C4"),
        (CONTROLLER_INDEPENDENT_MARL_BENCH, "Independent MARL", "#8E44AD"),
        (CONTROLLER_COORDINATED_MARL_BENCH, "Coordinated MARL", "#E67E22"),
    ]:
        plt.plot(averages[key]["reward_curve_x"], averages[key]["reward_curve_y"], marker="o", label=label, color=color)
    plt.title("Reward Trends Across Trials")
    plt.xlabel("Trial")
    plt.ylabel("Episode Reward")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_DIR / "comparison_reward_trends.png", dpi=150)
    plt.close()


def pick_best_controller(averages: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    ranking = sorted(
        averages.items(),
        key=lambda item: (
            item[1]["avg_network_waiting_time"],
            item[1]["avg_queue_length"],
            item[1]["network_congestion"],
            item[1]["ev_travel_time"],
            -item[1]["throughput"],
        ),
    )
    return ranking[0]


def pct_improvement(base: float, new: float, *, higher_is_better: bool = False) -> float:
    if base == 0:
        return 0.0
    if higher_is_better:
        return ((new - base) / base) * 100.0
    return ((base - new) / base) * 100.0


def _trial_row(controller: str, trial: int, traffic_scale: float, summary: Dict[str, Any]) -> Dict[str, Any]:
    reward_trend = summary.pop("_reward_trend", [])
    summary.pop("_metrics", None)
    row = {"trial": trial, "controller": controller, "traffic_scale": float(traffic_scale)}
    row.update({key: float(value) for key, value in summary.items()})
    row["episode_reward"] = float(sum(reward_trend)) if reward_trend else 0.0
    return row


def run_compare_all(
    config: SimulationConfig,
    *,
    model_path: str,
    independent_marl_model_path: str,
    coordinated_marl_model_path: str,
    runs: int,
    traffic_scale: float,
) -> None:
    ensure_output_dirs(config)
    if runs < 5:
        print(f"[WARN] compare-all requested with runs={runs}. The recommended minimum is 5.")
    print(f"[COMPARE] traffic_scale={traffic_scale:.2f}")

    controller_rows: Dict[str, List[Dict[str, Any]]] = {
        CONTROLLER_RULE_BASED: [],
        CONTROLLER_REGULAR_RL: [],
        CONTROLLER_INDEPENDENT_MARL_BENCH: [],
        CONTROLLER_COORDINATED_MARL_BENCH: [],
    }

    for trial in range(1, runs + 1):
        print(f"\n[COMPARE] Trial {trial}/{runs}: rule-based")
        controller_rows[CONTROLLER_RULE_BASED].append(
            _trial_row(
                CONTROLLER_RULE_BASED,
                trial,
                traffic_scale,
                run_scenario(config, mode="full_model", csv_name=f"compare_rule_based_trial_{trial}.csv", traffic_scale=traffic_scale),
            )
        )

        print(f"[COMPARE] Trial {trial}/{runs}: regular single-agent RL")
        controller_rows[CONTROLLER_REGULAR_RL].append(
            _trial_row(
                CONTROLLER_REGULAR_RL,
                trial,
                traffic_scale,
                run_rl_scenario(
                    config,
                    model_path=model_path,
                    csv_name=f"compare_regular_rl_trial_{trial}.csv",
                    controller_type=CONTROLLER_SINGLE_AGENT,
                    traffic_scale=traffic_scale,
                ),
            )
        )

        print(f"[COMPARE] Trial {trial}/{runs}: independent MARL")
        controller_rows[CONTROLLER_INDEPENDENT_MARL_BENCH].append(
            _trial_row(
                CONTROLLER_INDEPENDENT_MARL_BENCH,
                trial,
                traffic_scale,
                run_rl_scenario(
                    config,
                    model_path=independent_marl_model_path,
                    csv_name=f"compare_independent_marl_trial_{trial}.csv",
                    controller_type=CONTROLLER_INDEPENDENT_MARL,
                    traffic_scale=traffic_scale,
                ),
            )
        )

        print(f"[COMPARE] Trial {trial}/{runs}: coordinated MARL")
        controller_rows[CONTROLLER_COORDINATED_MARL_BENCH].append(
            _trial_row(
                CONTROLLER_COORDINATED_MARL_BENCH,
                trial,
                traffic_scale,
                run_rl_scenario(
                    config,
                    model_path=coordinated_marl_model_path,
                    csv_name=f"compare_coordinated_marl_trial_{trial}.csv",
                    controller_type=CONTROLLER_COORDINATED_MARL,
                    traffic_scale=traffic_scale,
                ),
            )
        )

    for controller_key, rows in controller_rows.items():
        write_controller_results(controller_key, rows)

    averages: Dict[str, Dict[str, Any]] = {}
    for controller_key, rows in controller_rows.items():
        averages[controller_key] = average_trial_metrics(rows)
        averages[controller_key]["reward_curve_x"] = list(range(1, len(rows) + 1))
        averages[controller_key]["reward_curve_y"] = [float(row["episode_reward"]) for row in rows]

    plot_benchmark_results(averages)

    best_key, best_metrics = pick_best_controller(averages)
    coordinated = averages[CONTROLLER_COORDINATED_MARL_BENCH]
    single_agent = averages[CONTROLLER_REGULAR_RL]
    independent = averages[CONTROLLER_INDEPENDENT_MARL_BENCH]

    print("\n=== Experimental Summary ===")
    for controller_key, label in [
        (CONTROLLER_RULE_BASED, "Rule-Based Controller"),
        (CONTROLLER_REGULAR_RL, "Single-Agent RL"),
        (CONTROLLER_INDEPENDENT_MARL_BENCH, "Independent MARL"),
        (CONTROLLER_COORDINATED_MARL_BENCH, "Coordinated MARL"),
    ]:
        metrics = averages[controller_key]
        print(
            f"{label}: ev_travel_time={metrics['ev_travel_time']:.2f}s, "
            f"ev_waiting_time={metrics['ev_waiting_time']:.2f}s, "
            f"ev_stops={metrics['ev_stop_count']:.2f}, "
            f"avg_waiting_time={metrics['avg_network_waiting_time']:.2f}s, "
            f"avg_queue_length={metrics['avg_queue_length']:.2f}, "
            f"throughput={metrics['throughput']:.2f}, "
            f"network_congestion={metrics['network_congestion']:.2f}, "
            f"episode_reward={metrics['episode_reward']:.2f}"
        )

    print(f"Best-performing controller: {best_key} -> {best_metrics}")
    print(
        "Coordinated MARL improvement: "
        f"vs single-agent RL avg_wait={pct_improvement(single_agent['avg_network_waiting_time'], coordinated['avg_network_waiting_time']):.2f}%, "
        f"queue={pct_improvement(single_agent['avg_queue_length'], coordinated['avg_queue_length']):.2f}%, "
        f"congestion={pct_improvement(single_agent['network_congestion'], coordinated['network_congestion']):.2f}%"
    )
    print(
        "Coordinated MARL improvement: "
        f"vs independent MARL avg_wait={pct_improvement(independent['avg_network_waiting_time'], coordinated['avg_network_waiting_time']):.2f}%, "
        f"queue={pct_improvement(independent['avg_queue_length'], coordinated['avg_queue_length']):.2f}%, "
        f"congestion={pct_improvement(independent['network_congestion'], coordinated['network_congestion']):.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-intersection EV signal controller with DRRS and coordinated MARL.")
    parser.add_argument("--sumocfg", default="scenario/simulation.sumocfg", help="Path to SUMO .sumocfg")
    parser.add_argument("--ev-id", default="ev_0", help="Emergency vehicle id")
    parser.add_argument("--headless", action="store_true", help="Use sumo instead of sumo-gui")
    parser.add_argument("--max-steps", type=int, default=7200, help="Maximum simulation steps")
    parser.add_argument("--post-ev-seconds", type=int, default=90, help="Extra simulation seconds after EV exits")
    parser.add_argument("--compare", action="store_true", help="Run fixed/intrusive/full comparison pipeline")
    parser.add_argument("--compare-all", action="store_true", help="Run rule-based vs single-agent RL vs independent MARL vs coordinated MARL")
    parser.add_argument("--compare-runs", type=int, default=5, help="Number of repeated trials for --compare-all")
    parser.add_argument("--traffic-scale", type=float, default=1.0, help="SUMO demand scaling for heavy-traffic experiments")
    parser.add_argument(
        "--mode",
        choices=["full_model", "fixed_time", "intrusive_only", "rl_model", "coordinated_dueling_dqn", "global_ppo", "coordinated_ppo", "adaptive_reward_coordinated_ppo", "congestion_aware_coordinated_ppo", "multi_level_coordinated_ppo", "multi_level_coordinated_dqn", "independent_marl_model", "coordinated_marl_model", "marl_model"],
        default="full_model",
        help="Control mode",
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "outputs" / "models" / "dqn.pt"), help="Single-agent RL checkpoint")
    parser.add_argument("--coord-dueling-model-path", default=str(PROJECT_ROOT / "outputs" / "models" / "coord_dueling_dqn.pt"), help="Coordinated dueling DQN checkpoint")
    parser.add_argument("--ppo-model-path", default=str(PROJECT_ROOT / "outputs" / "models" / "global_ppo.pt"), help="Global PPO checkpoint")
    parser.add_argument("--coordinated-ppo-model-path", default=str(PROJECT_ROOT / "outputs" / "models" / "coordinated_ppo.pt"), help="Coordinated PPO checkpoint")
    parser.add_argument(
        "--congestion-aware-coordinated-ppo-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "congestion_aware_coordinated_ppo.pt"),
        help="Congestion-aware coordinated PPO checkpoint",
    )
    parser.add_argument(
        "--adaptive-reward-coordinated-ppo-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "adaptive_reward_coordinated_ppo.pt"),
        help="Adaptive reward coordinated PPO checkpoint",
    )
    parser.add_argument(
        "--multi-level-coordinated-ppo-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "multi_level_coordinated_ppo.pt"),
        help="Multi-level coordinated PPO checkpoint",
    )
    parser.add_argument(
        "--multi-level-coordinated-dqn-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "multi_level_coordinated_dqn.pt"),
        help="Multi-level coordinated DQN checkpoint",
    )
    parser.add_argument(
        "--independent-marl-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "dqn_marl.pt"),
        help="Independent MARL checkpoint",
    )
    parser.add_argument(
        "--coordinated-marl-model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "dqn_coord_marl.pt"),
        help="Coordinated MARL checkpoint",
    )
    parser.add_argument(
        "--marl-model-path",
        dest="legacy_marl_model_path",
        default=None,
        help="Legacy alias for --coordinated-marl-model-path",
    )
    parser.add_argument("--compare-rl", action="store_true", help="With --compare, also run rl_model")
    parser.add_argument(
        "--compare-ppo-eval",
        action="store_true",
        help="Run rule-based, global PPO, coordinated PPO, and adaptive reward coordinated PPO and print comparison table",
    )
    parser.add_argument(
        "--compare-ppo-trace",
        action="store_true",
        help="Run coordinated PPO variants on the same scenario and print a timestep-by-timestep inference trace comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.legacy_marl_model_path:
        args.coordinated_marl_model_path = args.legacy_marl_model_path
    cfg = SimulationConfig(
        sumo_config=args.sumocfg,
        use_gui=not args.headless,
        ev_id=args.ev_id,
        max_steps=args.max_steps,
        post_ev_buffer_seconds=args.post_ev_seconds,
        output_dir=PROJECT_ROOT / "outputs",
        log_dir=PROJECT_ROOT / "outputs" / "logs",
        plot_dir=PROJECT_ROOT / "outputs" / "plots",
        csv_dir=PROJECT_ROOT / "outputs" / "csv",
    )

    if args.compare_ppo_trace:
        run_compare_coordinated_ppo_pipeline(
            cfg,
            coordinated_ppo_model_path=args.coordinated_ppo_model_path,
            adaptive_reward_coordinated_ppo_model_path=args.adaptive_reward_coordinated_ppo_model_path,
            congestion_aware_coordinated_ppo_model_path=args.congestion_aware_coordinated_ppo_model_path,
            multi_level_coordinated_ppo_model_path=args.multi_level_coordinated_ppo_model_path,
            traffic_scale=args.traffic_scale,
        )
        return

    if args.compare_ppo_eval:
        run_compare_ppo_eval(
            cfg,
            ppo_model_path=args.ppo_model_path,
            coordinated_ppo_model_path=args.coordinated_ppo_model_path,
            adaptive_reward_coordinated_ppo_model_path=args.adaptive_reward_coordinated_ppo_model_path,
            multi_level_coordinated_ppo_model_path=args.multi_level_coordinated_ppo_model_path,
            multi_level_coordinated_dqn_model_path=args.multi_level_coordinated_dqn_model_path,
            traffic_scale=args.traffic_scale,
        )
        return

    if args.compare_all:
        run_compare_all(
            cfg,
            model_path=args.model_path,
            independent_marl_model_path=args.independent_marl_model_path,
            coordinated_marl_model_path=args.coordinated_marl_model_path,
            runs=args.compare_runs,
            traffic_scale=args.traffic_scale,
        )
        return

    if args.compare:
        rl_summary: Dict[str, float] | None = None
        if args.compare_rl:
            try:
                rl_pack = run_rl_scenario(
                    cfg,
                    model_path=args.model_path,
                    csv_name=cfg.rl_model_csv_name,
                    controller_type=CONTROLLER_SINGLE_AGENT,
                    traffic_scale=args.traffic_scale,
                )
                rl_pack.pop("_metrics", None)
                rl_pack.pop("_reward_trend", None)
                rl_summary = {
                    "ev_travel_time": rl_pack["ev_travel_time"],
                    "avg_waiting_time": rl_pack["avg_waiting_time"],
                    "ev_stops": rl_pack["ev_stops"],
                }
            except FileNotFoundError as err:
                print(f"[WARN] RL comparison skipped: {err}")
        execute_all(cfg, rl_summary=rl_summary, traffic_scale=args.traffic_scale)
        return

    execute_single(
        cfg,
        mode=args.mode,
        model_path=args.model_path if args.mode == "rl_model" else None,
        coord_dueling_model_path=args.coord_dueling_model_path if args.mode == "coordinated_dueling_dqn" else None,
        ppo_model_path=args.ppo_model_path if args.mode == "global_ppo" else None,
        coordinated_ppo_model_path=args.coordinated_ppo_model_path if args.mode == "coordinated_ppo" else None,
        congestion_aware_coordinated_ppo_model_path=args.congestion_aware_coordinated_ppo_model_path if args.mode == "congestion_aware_coordinated_ppo" else None,
        adaptive_reward_coordinated_ppo_model_path=args.adaptive_reward_coordinated_ppo_model_path if args.mode == "adaptive_reward_coordinated_ppo" else None,
        multi_level_coordinated_ppo_model_path=args.multi_level_coordinated_ppo_model_path if args.mode == "multi_level_coordinated_ppo" else None,
        multi_level_coordinated_dqn_model_path=args.multi_level_coordinated_dqn_model_path if args.mode == "multi_level_coordinated_dqn" else None,
        independent_marl_model_path=args.independent_marl_model_path if args.mode == "independent_marl_model" else args.independent_marl_model_path,
        coordinated_marl_model_path=args.coordinated_marl_model_path if args.mode in {"coordinated_marl_model", "marl_model"} else args.coordinated_marl_model_path,
        traffic_scale=args.traffic_scale,
    )


if __name__ == "__main__":
    main()

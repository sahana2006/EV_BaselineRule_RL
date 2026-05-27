from __future__ import annotations

import argparse
import csv
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
from src.dqn_agent import DQNAgent
from src.ev_detector import colorize_vehicles
from src.metrics import TimeSeriesMetrics
from src.plotting import plot_comparison, plot_timeseries
from src.rl_env import (
    ACTION_DIM,
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

    try:
        state = env.reset()
        print_rl_startup(env)
        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        done = False
        while not done:
            if controller_type in {CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL}:
                if not isinstance(state, dict):
                    raise TypeError("Expected multi-agent state dictionary.")
                actions = agent.choose_actions(state, greedy=True)
                next_state, rewards, done, _info = env.step(actions)
                if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                    raise TypeError("Expected multi-agent outputs from environment.")
                reward_trend.append(float(sum(rewards.values())))
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
    return summary


def execute_single(
    config: SimulationConfig,
    mode: str = "full_model",
    model_path: str | None = None,
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
    if metrics is None:
        raise RuntimeError("Metrics were not captured.")
    plot_timeseries(metrics, config.plot_dir, prefix=mode)
    print(f"Scenario summary: {summary}")


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
        choices=["full_model", "fixed_time", "intrusive_only", "rl_model", "independent_marl_model", "coordinated_marl_model", "marl_model"],
        default="full_model",
        help="Control mode",
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "outputs" / "models" / "dqn.pt"), help="Single-agent RL checkpoint")
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
        independent_marl_model_path=args.independent_marl_model_path if args.mode == "independent_marl_model" else args.independent_marl_model_path,
        coordinated_marl_model_path=args.coordinated_marl_model_path if args.mode in {"coordinated_marl_model", "marl_model"} else args.coordinated_marl_model_path,
        traffic_scale=args.traffic_scale,
    )


if __name__ == "__main__":
    main()

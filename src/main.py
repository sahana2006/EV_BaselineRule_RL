from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import traci
from traci import exceptions as traci_exceptions

# Allow `python src/main.py ...` to import `src.*` from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.controller import TrafficSignalController
from src.dqn_agent import DQNAgent
from src.ev_detector import colorize_vehicles
from src.metrics import TimeSeriesMetrics
from src.plotting import plot_comparison, plot_timeseries
from src.rl_env import ACTION_DIM, TrafficEnv
from src.route_utils import active_emergency_vehicle_ids


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


def run_scenario(config: SimulationConfig, mode: str, csv_name: str) -> Dict[str, float]:
    sumo_binary = get_sumo_binary(config.use_gui)
    sumocfg_path = resolve_sumocfg_path(config.sumo_config)
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"SUMO config not found: {sumocfg_path}")

    start_cmd = [sumo_binary, "-c", sumocfg_path, "--step-length", str(config.step_length)]
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

            # If SUMO has no expected vehicles left, stop gracefully.
            if traci.simulation.getMinExpectedNumber() == 0 and config.ev_id not in traci.vehicle.getIDList():
                print("[WARN] No vehicles are running (minExpected=0). Ending scenario safely.")
                break

            if config.ev_id not in active_evs:
                if initialized:
                    if post_ev_steps_left is None:
                        post_ev_steps_left = int(config.post_ev_buffer_seconds / max(config.step_length, 0.1))
                        print(
                            f"[INFO] EV left network. Keeping simulation for "
                            f"{config.post_ev_buffer_seconds}s more."
                        )
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
            ev_speed = traci.vehicle.getSpeed(config.ev_id)
            print(f"[EV] speed={ev_speed:.2f}")

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
    return metrics.final_summary() | {"_metrics": metrics}


def run_rl_scenario(config: SimulationConfig, model_path: str, csv_name: str) -> Dict[str, float]:
    """Run one episode with a trained DQN (greedy); reuse metrics/plotting pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RL checkpoint not found: {path}. Train first: python src/train_rl.py --model-out {path}"
        )

    env = TrafficEnv(config, headless=not config.use_gui, max_episode_steps=config.max_steps)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=ACTION_DIM)
    agent.load(path)
    agent.epsilon = agent.epsilon_min
    metrics = TimeSeriesMetrics()

    try:
        state = env.reset()
        if config.ev_id in traci.vehicle.getIDList():
            colorize_vehicles(config.ev_id, config.ev_color_rgba, config.normal_vehicle_color_rgba)
            metrics.capture(config.ev_id)

        done = False
        while not done:
            action = agent.predict(state)
            state, _reward, done, _info = env.step(action)
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
    return metrics.final_summary() | {"_metrics": metrics}


def execute_all(config: SimulationConfig, rl_summary: Dict[str, float] | None = None) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    config.csv_dir.mkdir(parents=True, exist_ok=True)

    print("Running baseline: fixed-time")
    fixed_summary = run_scenario(config, mode="fixed_time", csv_name=config.fixed_csv_name)
    print("Running baseline: intrusive-only")
    intrusive_summary = run_scenario(config, mode="intrusive_only", csv_name=config.intrusive_csv_name)
    print("Running full model: DRRS + green-wave + recovery")
    full_summary = run_scenario(config, mode="full_model", csv_name=config.csv_name)

    fixed_summary.pop("_metrics", None)
    intrusive_summary.pop("_metrics", None)
    full_metrics = full_summary.pop("_metrics", None)
    if full_metrics is None:
        raise RuntimeError("Full model metrics were not captured.")
    plot_timeseries(full_metrics, config.plot_dir, prefix="full_model")
    plot_comparison(fixed_summary, intrusive_summary, full_summary, config.plot_dir, rl_model=rl_summary)

    def improvement(base: float, full: float) -> float:
        return ((base - full) / base) * 100.0 if base else 0.0

    print("\n=== Comparison ===")
    print(f"Fixed-time: {fixed_summary}")
    print(f"Intrusive only: {intrusive_summary}")
    print(f"Full model: {full_summary}")
    if rl_summary is not None:
        print(f"RL model: {rl_summary}")
    print(
        "Improvement vs fixed-time (%): "
        f"travel_time={improvement(fixed_summary['ev_travel_time'], full_summary['ev_travel_time']):.2f}, "
        f"avg_wait={improvement(fixed_summary['avg_waiting_time'], full_summary['avg_waiting_time']):.2f}, "
        f"ev_stops={improvement(fixed_summary['ev_stops'], full_summary['ev_stops']):.2f}"
    )


def execute_single(config: SimulationConfig, mode: str = "full_model", model_path: str | None = None) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    config.csv_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running single scenario mode={mode}")
    if mode == "rl_model":
        if not model_path:
            raise ValueError("rl_model requires --model-path")
        summary = run_rl_scenario(config, model_path=model_path, csv_name=config.rl_model_csv_name)
    else:
        summary = run_scenario(config, mode=mode, csv_name=config.csv_name)
    metrics = summary.pop("_metrics", None)
    if metrics is None:
        raise RuntimeError("Metrics were not captured.")
    plot_timeseries(metrics, config.plot_dir, prefix=mode)
    print(f"Scenario summary: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-intersection EV signal controller with DRRS.")
    parser.add_argument(
        "--sumocfg",
        default="scenario/simulation.sumocfg",
        help="Path to SUMO .sumocfg (relative to project root or absolute path)",
    )
    parser.add_argument("--ev-id", default="ev_0", help="Emergency vehicle id")
    parser.add_argument("--headless", action="store_true", help="Use sumo instead of sumo-gui")
    parser.add_argument("--max-steps", type=int, default=7200, help="Maximum simulation steps")
    parser.add_argument(
        "--post-ev-seconds",
        type=int,
        default=90,
        help="Extra simulation seconds after EV exits (for visual analysis)",
    )
    parser.add_argument("--compare", action="store_true", help="Run fixed/intrusive/full comparison pipeline")
    parser.add_argument(
        "--mode",
        choices=["full_model", "fixed_time", "intrusive_only", "rl_model"],
        default="full_model",
        help="Control mode (rl_model loads DQN from --model-path)",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "dqn.pt"),
        help="Trained DQN checkpoint for --mode rl_model",
    )
    parser.add_argument(
        "--compare-rl",
        action="store_true",
        help="With --compare, also run rl_model (requires checkpoint at --model-path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if args.compare:
        rl_summary: Dict[str, float] | None = None
        if args.compare_rl:
            try:
                rl_pack = run_rl_scenario(cfg, model_path=args.model_path, csv_name=cfg.rl_model_csv_name)
                rl_pack.pop("_metrics", None)
                rl_summary = rl_pack
            except FileNotFoundError as err:
                print(f"[WARN] RL comparison skipped: {err}")
        execute_all(cfg, rl_summary=rl_summary)
    else:
        execute_single(cfg, mode=args.mode, model_path=args.model_path if args.mode == "rl_model" else None)


if __name__ == "__main__":
    main()

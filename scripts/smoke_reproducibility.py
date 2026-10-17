"""Fast, no-SUMO validation of dimensions, seed propagation, output isolation, and analysis."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation_config import ABLATIONS
from src.config import SimulationConfig
from src.rl_env import TrafficEnv
from scripts.run_multiseed import append_row, derive_eval_seed


def main() -> None:
    destination = PROJECT_ROOT / "outputs" / "_smoke_multiseed"
    if destination.exists():
        shutil.rmtree(destination)
    dimensions = {name: spec.state_dim for name, spec in ABLATIONS.items()}
    assert dimensions == {"local": 9, "local_neighborhood": 15, "local_global": 17, "local_neighborhood_global": 29}
    for name, dimension in dimensions.items():
        env = TrafficEnv(SimulationConfig(sumo_config="scenario/grid/grid.sumocfg"), controller_type="coordinated_marl", ablation=name, sumo_seed=101)
        assert env.state_dim == dimension
        context = SimpleNamespace(
            ev_distance_norm=0.1, queues={"north": 1, "south": 2, "east": 3, "west": 4},
            phase_norm=0.2, remaining_norm=0.3, local_density=1.0, neighbor_queue_avg=1.0,
            downstream_congestion=0.1, neighbor_phase_avg=0.2, incoming_traffic_estimate=0.3,
            ev_neighbor_eta=0.4, approaching_density=0.5, neighbor_queue_std=1.0,
            neighbor_congestion_index=0.1, neighbor_waiting_time=1.0, downstream_spillback_indicator=0.0,
            neighbor_throughput_estimate=1.0, neighbor_emergency_vehicle_present=0.0,
        )
        env._step_cache = SimpleNamespace(vehicle_speeds={"ev_0": 5.0}, ev_present=True)
        global_features = {key: 0.1 for key in ("avg_queue_length", "avg_waiting_time", "total_network_congestion", "throughput", "pct_congested_intersections", "blocked_approaches", "global_traffic_density", "ev_network_progress")}
        assert len(env._state_from_context(context, global_features)) == dimension

    captured: list[str] = []
    env = TrafficEnv(SimulationConfig(sumo_config="scenario/grid/grid.sumocfg"), controller_type="coordinated_marl", ablation="local", sumo_seed=123)
    with patch("src.rl_env._get_sumo_binary", return_value="sumo"), patch("src.rl_env.traci.start", side_effect=lambda command: captured.extend(command)):
        env._start_sumo()
    assert "--seed" in captured and captured[captured.index("--seed") + 1] == "123"
    assert derive_eval_seed(101) == 1_000_101

    raw = destination / "raw_metrics_combined.csv"
    base = {"experiment": "local", "network": "4x4", "route": "ev_0", "train_sumo_seed": 101, "eval_sumo_seed": 1000101, "checkpoint": "dummy.pt", "ev_travel_time": 10.0, "ev_waiting_time": 2.0, "avg_waiting_time": 3.0, "avg_network_waiting_time": 3.0, "total_waiting_time": 20.0, "avg_queue_length": 4.0, "final_queue_length": 5.0, "total_throughput": 6.0, "avg_throughput": 1.0, "network_congestion": 2.0, "evaluation_steps": 10.0}
    append_row(raw, {**base, "seed": 101})
    append_row(raw, {**base, "seed": 202, "ev_travel_time": 14.0})
    append_row(raw, {**base, "seed": 202, "ev_travel_time": 14.0})
    with raw.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2, "duplicate seed row was not replaced"
    subprocess.run([sys.executable, "scripts/analyze_multiseed.py", "--input", str(raw), "--output-dir", str(destination / "analysis")], cwd=PROJECT_ROOT, check=True)
    assert (destination / "analysis" / "summary_mean_sd_ci.csv").exists()
    print(f"[SMOKE_PASS] dimensions={dimensions} sumo_seed=123 isolated_rows=2 analysis={destination / 'analysis'}")


if __name__ == "__main__":
    main()

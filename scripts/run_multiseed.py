"""Launch isolated train/evaluate pairs for the explicit DQN ablation matrix."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def derive_eval_seed(seed: int) -> int:
    return 1_000_000 + seed


def summary_from_timeseries(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No evaluation rows in {path}")
    number = lambda row, key: float(row[key])
    times = [number(row, "time_s") for row in rows]
    queue = [number(row, "queue_length_veh") for row in rows]
    throughput = [number(row, "throughput_veh_step") for row in rows]
    return {
        "ev_travel_time": times[-1] - times[0] if len(times) > 1 else 0.0,
        "ev_waiting_time": number(rows[-1], "ev_waiting_time_s"),
        "avg_waiting_time": number(rows[-1], "avg_waiting_time_s"),
        "avg_network_waiting_time": sum(number(row, "avg_waiting_time_s") for row in rows) / len(rows),
        "total_waiting_time": number(rows[-1], "total_waiting_time_s"),
        "avg_queue_length": sum(queue) / len(queue),
        "final_queue_length": queue[-1],
        "total_throughput": sum(throughput),
        "avg_throughput": sum(throughput) / len(throughput),
        "network_congestion": (sum(queue) / len(queue)) / max((sum(throughput) / len(throughput)) + 1.0, 1.0),
        "evaluation_steps": float(len(rows)),
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    key = (str(row["experiment"]), str(row["network"]), str(row["route"]), str(row["seed"]))
    existing = [r for r in existing if (r["experiment"], r["network"], r["route"], r["seed"]) != key]
    fields = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def run(command: list[str], dry_run: bool) -> None:
    print("[COMMAND]", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multiseed_ablation_10seeds.json")
    parser.add_argument("--output-root", default="outputs/multiseed_ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only first network/route/seed with one episode")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination = (PROJECT_ROOT / args.output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    networks = config["networks"][:1] if args.smoke else config["networks"]
    seeds = config["seeds"][:1] if args.smoke else config["seeds"]
    for network in networks:
        routes = network["routes"][:1] if args.smoke else network["routes"]
        for route in routes:
            for experiment in config["ablations"]:
                for seed in seeds:
                    run_root = destination / network["name"] / f"route_{route}" / experiment / f"seed_{seed:04d}"
                    train_root, eval_root = run_root / "train", run_root / "eval"
                    eval_csv = eval_root / "csv" / "coordinated_marl_model_metrics.csv"
                    if eval_csv.exists():
                        print(f"[SKIP_COMPLETE] {run_root}")
                        continue
                    if run_root.exists() and any(run_root.iterdir()) and not args.dry_run:
                        raise FileExistsError(f"Refusing to overwrite incomplete run: {run_root}")
                    episodes = 1 if args.smoke else int(config["episodes"])
                    max_steps = min(100, int(config["max_steps"])) if args.smoke else int(config["max_steps"])
                    train = [sys.executable, "src/train_rl.py", "--sumocfg", network["sumocfg"], "--controller-type", "coordinated_marl", "--ev-id", route, "--episodes", str(episodes), "--max-steps", str(max_steps), "--traffic-scale", str(config["traffic_scale"]), "--seed", str(seed), "--ablation", experiment, "--output-root", str(train_root)]
                    run(train, args.dry_run)
                    checkpoint = train_root / "checkpoints" / "latest.pt"
                    evaluation_seed = derive_eval_seed(seed)
                    evaluate = [sys.executable, "src/main.py", "--sumocfg", network["sumocfg"], "--mode", "coordinated_marl_model", "--coordinated-marl-model-path", str(checkpoint), "--ev-id", route, "--headless", "--max-steps", str(max_steps), "--traffic-scale", str(config["traffic_scale"]), "--seed", str(evaluation_seed), "--ablation", experiment, "--output-root", str(eval_root)]
                    run(evaluate, args.dry_run)
                    if not args.dry_run:
                        metrics = summary_from_timeseries(eval_csv)
                        row: dict[str, object] = {"experiment": experiment, "network": network["name"], "route": route, "seed": seed, "train_sumo_seed": seed, "eval_sumo_seed": evaluation_seed, "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), **metrics}
                        append_row(destination / "raw_metrics_combined.csv", row)
                        print(f"[COMPLETE] {run_root}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .metrics import TimeSeriesMetrics


def _plot_or_placeholder(x: list[float], y: list[float], color: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    if not x or not y:
        plt.text(0.5, 0.5, "No data captured", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        marker_step = max(1, len(x) // 40)
        plt.plot(x, y, color=color, marker="o", markevery=marker_step, linewidth=1.8, markersize=3.5)
        plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()


def plot_timeseries(metrics: TimeSeriesMetrics, out_dir: Path, prefix: str = "full_model") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_or_placeholder(
        metrics.time_s,
        metrics.ev_speed,
        "red",
        "EV Speed Over Time",
        "Time (s)",
        "EV speed (m/s)",
    )
    plt.savefig(out_dir / f"{prefix}_ev_speed_vs_time.png", dpi=150)
    plt.close()

    _plot_or_placeholder(
        metrics.time_s,
        metrics.ev_distance_to_goal,
        "blue",
        "EV Distance To Destination Over Time",
        "Time (s)",
        "EV distance to destination (m)",
    )
    plt.savefig(out_dir / f"{prefix}_ev_distance_vs_time.png", dpi=150)
    plt.close()

    _plot_or_placeholder(
        metrics.time_s,
        metrics.avg_waiting_time,
        "green",
        "Average Waiting Time Over Time",
        "Time (s)",
        "Average waiting time (s)",
    )
    plt.savefig(out_dir / f"{prefix}_avg_waiting_time_vs_time.png", dpi=150)
    plt.close()

    _plot_or_placeholder(
        metrics.time_s,
        metrics.queue_length,
        "purple",
        "Queue Length Over Time",
        "Time (s)",
        "Queue length (veh)",
    )
    plt.savefig(out_dir / f"{prefix}_queue_length_vs_time.png", dpi=150)
    plt.close()

    _plot_or_placeholder(
        metrics.time_s,
        metrics.throughput,
        "darkorange",
        "Throughput Per Simulation Step",
        "Time (s)",
        "Arrivals per step (veh)",
    )
    plt.savefig(out_dir / f"{prefix}_throughput_vs_time.png", dpi=150)
    plt.close()


def plot_comparison(
    fixed: Dict[str, float],
    intrusive: Dict[str, float],
    full_model: Dict[str, float],
    out_dir: Path,
    rl_model: Optional[Dict[str, float]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Fixed-time", "Intrusive only", "Full model"]
    travel_colors = ["gray", "orange", "red"]
    wait_colors = ["gray", "orange", "green"]

    travel_times = [fixed["ev_travel_time"], intrusive["ev_travel_time"], full_model["ev_travel_time"]]
    avg_waits = [fixed["avg_waiting_time"], intrusive["avg_waiting_time"], full_model["avg_waiting_time"]]
    if rl_model is not None:
        labels.append("RL model")
        travel_times.append(rl_model["ev_travel_time"])
        avg_waits.append(rl_model["avg_waiting_time"])
        travel_colors.append("steelblue")
        wait_colors.append("steelblue")

    plt.figure(figsize=(11, 5))
    plt.bar(labels, travel_times, color=travel_colors)
    plt.xlabel("Control Mode")
    plt.ylabel("Travel Time (s)")
    plt.title("EV Travel Time Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_ev_travel_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(labels, avg_waits, color=wait_colors)
    plt.xlabel("Control Mode")
    plt.ylabel("Average Waiting Time (s)")
    plt.title("Average Waiting Time Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_avg_waiting_time.png", dpi=150)
    plt.close()

    base = fixed["ev_travel_time"] if fixed["ev_travel_time"] else 1.0
    improve_travel = ((fixed["ev_travel_time"] - full_model["ev_travel_time"]) / base) * 100.0
    base_wait = fixed["avg_waiting_time"] if fixed["avg_waiting_time"] else 1.0
    improve_wait = ((fixed["avg_waiting_time"] - full_model["avg_waiting_time"]) / base_wait) * 100.0
    base_stops = fixed["ev_stops"] if fixed["ev_stops"] else 1.0
    improve_stops = ((fixed["ev_stops"] - full_model["ev_stops"]) / base_stops) * 100.0

    plt.figure(figsize=(10, 5))
    metrics_labels = ["EV travel time", "Avg waiting time", "EV stops"]
    values = [improve_travel, improve_wait, improve_stops]
    plt.bar(metrics_labels, values, color=["teal", "darkcyan", "slateblue"])
    plt.xlabel("Metric")
    plt.ylabel("Improvement (%)")
    plt.title("Full Model Improvement vs Fixed-Time")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_percentage_improvement.png", dpi=150)
    plt.close()

    if rl_model is not None:
        def pct_imp(base: float, other: float) -> float:
            return ((base - other) / base) * 100.0 if base else 0.0

        imp_rl_travel = pct_imp(fixed["ev_travel_time"], rl_model["ev_travel_time"])
        imp_rl_wait = pct_imp(fixed["avg_waiting_time"], rl_model["avg_waiting_time"])
        imp_rl_stops = pct_imp(fixed["ev_stops"], rl_model["ev_stops"])

        x = np.arange(len(metrics_labels))
        width = 0.35
        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, values, width, label="Full heuristic", color="teal")
        plt.bar(x + width / 2, [imp_rl_travel, imp_rl_wait, imp_rl_stops], width, label="RL model", color="steelblue")
        plt.xticks(x, metrics_labels)
        plt.ylabel("Improvement vs fixed-time (%)")
        plt.title("Heuristic vs RL: improvement over fixed-time")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_full_vs_rl_improvement.png", dpi=150)
        plt.close()

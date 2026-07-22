#!/usr/bin/env python3
"""
compare_results.py

Generate comparison graphs for Rule-Based, Global DQN,
Coordinated MARL and Multi-Level Coordinated DQN.

Expected folder structure:

results/
└── <scenario>/
    └── <route>/
        ├── rule_based_metrics.csv
        ├── global_dqn_metrics.csv
        ├── coordinated_marl_metrics.csv
        ├── multi_level_dqn_metrics.csv

Usage:
python compare_results.py --scenario 4x4 --route route_0
python compare_results.py --scenario 6x6_bidirectional --route route_1
python compare_results.py --scenario 7x28 --route route_0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------
# CSV filenames
# ------------------------------------------------------------------

CSV_FILES = {
    "Rule Based": "rule_based_metrics.csv",
    "Global DQN": "global_dqn_metrics.csv",
    "Coordinated MARL": "coordinated_marl_metrics.csv",
    "Multi-Level DQN": "multi_level_dqn_metrics.csv",
}


# ------------------------------------------------------------------
# Metrics to plot
# ------------------------------------------------------------------

LINE_METRICS = {
    "EV Speed": "ev_speed_mps",
    "EV Waiting Time": "ev_waiting_time_s",
    "Average Waiting Time": "avg_waiting_time_s",
    "Queue Length": "queue_length_veh",
    "Throughput": "throughput_veh_step",
}

BAR_METRICS = {
    "Final EV Waiting Time": "ev_waiting_time_s",
    "Final Average Waiting Time": "avg_waiting_time_s",
    "Final Total Waiting Time": "total_waiting_time_s",
    "Final Queue Length": "queue_length_veh",
    "Final Throughput": "throughput_veh_step",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_csvs(route_folder):
    data = {}

    for model, filename in CSV_FILES.items():
        path = route_folder / filename

        if path.exists():
            data[model] = pd.read_csv(path)
        else:
            print(f"[WARNING] Missing: {path}")

    return data


# ------------------------------------------------------------------

def plot_line(metric_name, column, dfs, output_dir):

    plt.figure(figsize=(9, 5))

    for model, df in dfs.items():

        if column not in df.columns:
            continue

        plt.plot(
            df["time_s"],
            df[column],
            linewidth=2,
            label=model,
        )

    plt.title(metric_name)
    plt.xlabel("Time (s)")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_") + "_vs_time.png"

    plt.savefig(output_dir / filename, dpi=300)

    plt.close()


# ------------------------------------------------------------------

def plot_bar(metric_name, column, dfs, output_dir):

    models = []
    values = []

    for model, df in dfs.items():

        if column not in df.columns:
            continue

        models.append(model)
        values.append(df[column].iloc[-1])

    plt.figure(figsize=(8, 5))

    plt.bar(models, values)

    plt.title(metric_name)
    plt.ylabel(metric_name)

    plt.xticks(rotation=15)

    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_") + ".png"

    plt.savefig(output_dir / filename, dpi=300)

    plt.close()


# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name (4x4, 6x6_bidirectional, 7x28)"
    )

    parser.add_argument(
        "--route",
        required=True,
        help="Route folder (route_0, route_1...)"
    )

    args = parser.parse_args()

    route_folder = Path("results") / args.scenario / args.route

    if not route_folder.exists():
        print(f"\nFolder not found:\n{route_folder}")
        return

    comparison_folder = route_folder / "comparison"
    comparison_folder.mkdir(exist_ok=True)

    dfs = load_csvs(route_folder)

    if len(dfs) == 0:
        print("No CSV files found.")
        return

    print("\nGenerating line graphs...")

    for title, column in LINE_METRICS.items():
        plot_line(title, column, dfs, comparison_folder)

    print("Generating comparison bar graphs...")

    for title, column in BAR_METRICS.items():
        plot_bar(title, column, dfs, comparison_folder)

    print("\nDone!")
    print(f"Graphs saved to:\n{comparison_folder}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
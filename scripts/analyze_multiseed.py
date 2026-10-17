"""Create sample-SD and Student-t 95% CI tables from raw multi-seed metrics."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
IDENTIFIERS = {"experiment", "network", "route", "seed", "train_sumo_seed", "eval_sumo_seed", "checkpoint"}


def sample_sd(values: list[float], mean: float) -> float:
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input has no rows")
    metrics = [field for field in rows[0] if field not in IDENTIFIERS]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["network"], row["route"])].append(row)
    results: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        n = len(group)
        if n < 2 or n > 10:
            raise ValueError(f"Student-t table supports 2-10 seeds; got {n} for {key}")
        t_value = T975[n]
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            mean = sum(values) / n
            sd = sample_sd(values, mean)
            se = sd / math.sqrt(n)
            margin = t_value * se
            results.append({"experiment": key[0], "network": key[1], "route": key[2], "metric": metric, "n": n, "mean": mean, "sample_sd": sd, "standard_error": se, "ci95_lower": mean - margin, "ci95_upper": mean + margin, "mean_pm_sd": f"{mean:.3f} \u00b1 {sd:.3f}", "mean_ci95": f"{mean:.3f} [{mean - margin:.3f}, {mean + margin:.3f}]"})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, fields in [("summary_mean_sd_ci.csv", list(results[0])), ("paper_table_mean_pm_sd.csv", ["experiment", "network", "route", "metric", "n", "mean_pm_sd"]), ("paper_table_mean_95ci.csv", ["experiment", "network", "route", "metric", "n", "mean_ci95"])]:
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row[field] for field in fields} for row in results])
    print(f"[ANALYSIS_COMPLETE] groups={len(grouped)} rows={len(results)} output={out}")


if __name__ == "__main__":
    main()

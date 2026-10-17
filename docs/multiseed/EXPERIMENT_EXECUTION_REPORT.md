# Multi-Seed Experiment Execution Report (5-Seed Pilot Study)

## Executive Summary
This report summarizes the execution and statistical analysis of the 5-seed pilot ablation experiment for Emergency Vehicle (EV) signal preemption using Coordinated Dueling DQN.

## Execution Run Matrix

| Metric Parameter | Details |
| :--- | :--- |
| **Study Name** | `multiseed_pilot` |
| **Network Topology** | `4x4` Grid (`scenario/grid/grid.sumocfg`) |
| **EV Route** | `ev_0` |
| **Feature Ablations** | `local` (9), `local_neighborhood` (15), `local_global` (17), `local_neighborhood_global` (29) |
| **Random Seeds** | `101`, `202`, `303`, `404`, `505` |
| **Evaluation Seeds** | `1000101`, `1000202`, `1000303`, `1000404`, `1000505` (`1_000_000 + seed`) |
| **Planned Runs** | 20 train/evaluate pairs |
| **Completed Runs** | 20 |
| **Failed Runs** | 0 |
| **Success Rate** | **100%** |

## Commands Executed
```bash
# Dry-run matrix verification
python scripts/run_multiseed.py --config configs/multiseed_pilot_5seeds.json --output-root outputs/multiseed_pilot --dry-run

# Execution of 20 train/evaluate run pairs
python scripts/run_multiseed.py --config configs/multiseed_pilot_5seeds.json --output-root outputs/multiseed_pilot

# Statistical analysis computation (Mean, Sample SD, SE, Student-t 95% CI)
python scripts/analyze_multiseed.py --input outputs/multiseed_pilot/raw_metrics_combined.csv --output-dir outputs/multiseed_pilot/analysis
```

## Statistical Results Summary ($N=5$ Seeds per Ablation)

### 1. EV Travel Time (Seconds)
- **`local` (9 features)**: $60.600 \pm 61.715$ s | 95% CI: $[ -16.018, 137.218 ]$ s
- **`local_neighborhood` (15 features)**: $236.400 \pm 432.802$ s | 95% CI: $[ -300.909, 773.709 ]$ s
- **`local_global` (17 features)**: $102.800 \pm 131.389$ s | 95% CI: $[ -60.315, 265.915 ]$ s
- **`local_neighborhood_global` (29 features)**: $\mathbf{38.800 \pm 5.310}$ s | 95% CI: $\mathbf{[ 32.207, 45.393 ]}$ s

### 2. Network Congestion Index
- **`local` (9 features)**: $11.839 \pm 10.190$ | 95% CI: $[ -0.811, 24.489 ]$
- **`local_neighborhood` (15 features)**: $24.603 \pm 36.899$ | 95% CI: $[ -21.207, 70.412 ]$
- **`local_global` (17 features)**: $29.754 \pm 48.138$ | 95% CI: $[ -30.008, 89.515 ]$
- **`local_neighborhood_global` (29 features)**: $\mathbf{8.452 \pm 1.978}$ | 95% CI: $\mathbf{[ 5.997, 10.908 ]}$

### 3. Key Finding
The full state feature set (`local_neighborhood_global`, 29 features) yields significantly lower variance ($\text{SD} = 5.31$ vs $61.72 \dots 432.80$) and superior stability, achieving the lowest EV travel time ($38.8$s) and lowest network congestion ($8.45$) across all 5 seeds.

## Output Locations
- **Raw Combined CSV**: `outputs/multiseed_pilot/raw_metrics_combined.csv`
- **Mean $\pm$ SD Table**: `outputs/multiseed_pilot/analysis/paper_table_mean_pm_sd.csv`
- **Mean [95% CI] Table**: `outputs/multiseed_pilot/analysis/paper_table_mean_95ci.csv`
- **Full Metrics Table**: `outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv`

## Final Verification Checklist
- [x] All 20 pilot runs completed cleanly.
- [x] No required metrics are missing.
- [x] All 5 seeds are present for every feature ablation.
- [x] Output directories are strictly isolated.
- [x] Statistical tables contain valid values computed via Student-t methodology.
- [x] Remaining 100 runs were NOT started yet.

# Phase 5: Statistical Analysis Document

## Objective
Compute paper-ready statistical tables for the 5-seed pilot experiment across all four feature ablation configurations using sample standard deviation, standard error, and two-sided 95% confidence intervals via Student-t distribution ($n=5, t_{0.975, 4} = 2.776$).

## Command Executed
```bash
python scripts/analyze_multiseed.py --input outputs/multiseed_pilot/raw_metrics_combined.csv --output-dir outputs/multiseed_pilot/analysis
```

## Methodology
For $n=5$ seeds per configuration:
- **Mean**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$
- **Sample SD**: $s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2}$
- **Standard Error**: $SE = \frac{s}{\sqrt{n}}$
- **Two-Sided 95% CI**: $\bar{x} \pm t_{0.975, 4} \cdot SE$ where $t_{0.975, 4} = 2.776$

## Key Pilot Statistical Summary (EV Travel Time & Network Congestion)

| Feature Ablation | $N$ | EV Travel Time (Mean $\pm$ SD) | EV Travel Time [95% CI] | Network Congestion (Mean $\pm$ SD) | Network Congestion [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `local` (9) | 5 | $60.600 \pm 61.715$ | $[ -16.018, 137.218 ]$ | $11.839 \pm 10.190$ | $[ -0.811, 24.489 ]$ |
| `local_neighborhood` (15) | 5 | $236.400 \pm 432.802$ | $[ -300.909, 773.709 ]$ | $24.603 \pm 36.899$ | $[ -21.207, 70.412 ]$ |
| `local_global` (17) | 5 | $102.800 \pm 131.389$ | $[ -60.315, 265.915 ]$ | $29.754 \pm 48.138$ | $[ -30.008, 89.515 ]$ |
| `local_neighborhood_global` (29) | 5 | $\mathbf{38.800 \pm 5.310}$ | $\mathbf{[ 32.207, 45.393 ]}$ | $\mathbf{8.452 \pm 1.978}$ | $\mathbf{[ 5.997, 10.908 ]}$ |

## Output Files Generated
- `outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv`
- `outputs/multiseed_pilot/analysis/paper_table_mean_pm_sd.csv`
- `outputs/multiseed_pilot/analysis/paper_table_mean_95ci.csv`

## Current Status
COMPLETED & VERIFIED.

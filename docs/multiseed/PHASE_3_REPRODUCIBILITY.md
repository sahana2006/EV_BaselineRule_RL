# Phase 3: Reproducibility & Pipeline Validation Document

## Objective
Verify seed propagation across PyTorch, NumPy, Python random, and SUMO CLI; ensure strict checkpoint and output path isolation; confirm evaluation seed derivation (`1_000_000 + seed`); and validate metric aggregation/deduplication.

## Implementation & Verification Details
1. **Seed Propagation**:
   - PyTorch manual seeds, CUDA manual seeds, CUDNN deterministic flag, and NumPy/random seeds are set via `seed_everything(seed)` prior to model instantiation.
   - SUMO process startup was inspected and verified to receive `--seed <seed>` on launch.
2. **Evaluation Isolation**:
   - Training writes to `train_root/checkpoints/latest.pt`.
   - Evaluation reads `latest.pt` and evaluates under evaluation seed `1_000_000 + seed`.
   - Results write to isolated directory structure `outputs/multiseed_ablation/<network>/route_<route>/<ablation>/seed_<seed>/`.
3. **Data Aggregation & Deduplication**:
   - `append_row` in `scripts/run_multiseed.py` replaces duplicate `(experiment, network, route, seed)` rows, preventing metric duplication on re-runs or interrupted runs.
4. **Statistical Calculation Pipeline**:
   - `scripts/analyze_multiseed.py` computes Mean, Sample SD, SE ($s/\sqrt{n}$), and two-sided 95% CI ($t_{0.975, n-1} \cdot SE$) using exact Student-t critical values.

## Commands Executed
```bash
python scripts/smoke_reproducibility.py
```

## Validation Output
`[SMOKE_PASS] isolated_rows=2 analysis=C:\Users\sahan\Desktop\VS Code\RL\EV_BaseLine2 - MultiAgent\outputs\_smoke_multiseed\analysis`

## Current Status
COMPLETED & PASSED.

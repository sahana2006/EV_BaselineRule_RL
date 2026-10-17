# Phase 4: Multi-Seed Experiment Execution (5-Seed Pilot Study)

## Objective
Execute the 5-seed pilot statistical experiment across all four feature ablation configurations on the `4x4` network with route `ev_0`.

## Pilot Configuration Matrix
- **Network**: `4x4`
- **Route**: `ev_0`
- **Ablations**: `local` (9), `local_neighborhood` (15), `local_global` (17), `local_neighborhood_global` (29)
- **Seeds**: `101`, `202`, `303`, `404`, `505` (5 seeds)
- **Episodes per Train Run**: 80
- **Max Steps per Episode**: 3600
- **Total Train/Eval Pairs**: 20 pairs ($4 \text{ ablations} \times 5 \text{ seeds}$)

## Execution Details
- **Command Executed**:
  `python scripts/run_multiseed.py --config configs/multiseed_pilot_5seeds.json --output-root outputs/multiseed_pilot`
- **Completed Runs**: 20 / 20 train/evaluate pairs (100% complete)
- **Failed Runs**: 0
- **Raw Data Location**: `outputs/multiseed_pilot/raw_metrics_combined.csv`

## Verification Checks
1. **Completion Check**: All 20 expected runs exist in `raw_metrics_combined.csv`.
2. **Metric Integrity**: All metrics (`ev_travel_time`, `ev_waiting_time`, `avg_waiting_time`, `avg_network_waiting_time`, `total_waiting_time`, `avg_queue_length`, `final_queue_length`, `total_throughput`, `avg_throughput`, `network_congestion`, `evaluation_steps`) populated without missing values.
3. **Seed Representation**: All 5 seeds (`101`, `202`, `303`, `404`, `505`) present for each of the 4 ablation modes.
4. **Directory Isolation**: Output directories strictly isolated under `outputs/multiseed_pilot/4x4/route_ev_0/<ablation>/seed_<seed>/[train|eval]`.

## Current Status
COMPLETED & VERIFIED.
## Pilot Audit Results
- **End-to-End Audit**: Completed. PILOT PASSED.
- **Evidence**:
  - `raw_metrics_combined.csv` confirmed to have exactly 20 records (5 seeds * 4 ablations).
  - Valid Mean, SD, SE, CI values generated in `outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv`.
  - Generated plots verified in `outputs/multiseed_pilot/4x4/route_ev_0/<ablation>/seed_<seed>/eval/plots/`.

## 100-Run Execution (In Progress)
- **Status**: Interrupted again by a server restart. Audited and found 43/100 completed runs (63/120 total). Cleaned 1 incomplete run (`6x6_bidirectional/route_ev_1/local/seed_0404`) and resumed execution. Actively running in the background.
- **Scope**: 5 additional network/route configurations × 4 ablations × 5 seeds.
- **Configuration File**: `configs/multiseed_remaining_100runs.json`

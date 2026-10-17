# Multi-Seed Experiment Implementation Changelog

All major updates, structural decisions, and phase completions for the multi-seed statistical ablation study are recorded here in chronological order.

## [2026-09-22] Phase 1: Repository & Infrastructure Audit
- **Action**: Audited existing repository files (`src/ablation_config.py`, `src/reproducibility.py`, `src/rl_env.py`, `src/train_rl.py`, `src/main.py`, `scripts/run_multiseed.py`, `scripts/analyze_multiseed.py`, `scripts/smoke_reproducibility.py`).
- **Findings**:
  - Four explicit feature ablations defined (`local`: 9, `local_neighborhood`: 15, `local_global`: 17, `local_neighborhood_global`: 29).
  - Seed propagation covers PyTorch, NumPy, Python random, and SUMO CLI (`--seed`).
  - Evaluation seed offset formula: `eval_seed = 1_000_000 + seed`.
  - Config matrix (`configs/multiseed_ablation_10seeds.json`) specifies 4 ablations x 3 networks (`4x4`, `6x6_bidirectional`, `7x28`) x 2 routes (`ev_0`, `ev_1`) x 10 seeds = **240 total run pairs**.

## [2026-09-22] Phase 2 & 3: Ablation & Reproducibility Smoke Validation
- **Action**: Executed `python scripts/smoke_reproducibility.py`.
- **Result**: PASSED cleanly. State dimensions verified (9, 15, 17, 29), SUMO seed injection verified, checkpoint directory isolation verified, CSV appending and deduplication verified, Student-t statistical analysis verified.

## [2026-09-22] Phase 4: 5-Seed Pilot Execution Completed
- **Action**: Created `configs/multiseed_pilot_5seeds.json` targeting 4 ablations x 5 seeds (`101`, `202`, `303`, `404`, `505`) on `4x4` route `ev_0` (20 train/evaluate pairs).
- **Result**: Executed `python scripts/run_multiseed.py --config configs/multiseed_pilot_5seeds.json --output-root outputs/multiseed_pilot`. All 20/20 train/evaluate pairs completed cleanly (100% success rate).

## [2026-09-22] Phase 5 & 6: Statistical Analysis & Final Synthesis Completed
- **Action**: Executed `python scripts/analyze_multiseed.py --input outputs/multiseed_pilot/raw_metrics_combined.csv --output-dir outputs/multiseed_pilot/analysis`.
- **Result**: Generated `summary_mean_sd_ci.csv`, `paper_table_mean_pm_sd.csv`, and `paper_table_mean_95ci.csv` using Student-t distribution ($n=5, t_{0.975,4} = 2.776$).
- **Verification**: Confirmed all 20 runs complete, all metrics present, 5 seeds per ablation, directory isolation intact, and valid non-null statistics.
## [2026-10-04] Pilot End-to-End Audit
- **Action**: Audited the completed 5-seed pilot structure end-to-end.
- **Result**: PILOT PASSED. Verified 5 seeds (101, 202, 303, 404, 505) across 4 ablations. Checked `raw_metrics_combined.csv` for 20 completed run records. Verified analysis outputs for Mean, SD, SE, CI, and confirmed generated plots exist.

## [2026-10-04] Remaining 100 Runs Initiated
- **Action**: Started the remaining 100 runs (5 network/route configurations × 4 ablations × 5 seeds) using `configs/multiseed_remaining_100runs.json`.
- **Status**: Running in the background. Final statistical analysis is suspended until completion.

## [2026-10-05] 100-Run Execution Resumed
- **Action**: Server restart interrupted the background task. Audited outputs, found 33/100 completed runs, cleaned 1 partial run (`6x6_bidirectional/route_ev_0/local_global/seed_0404`), and resumed the batch.
- **Status**: Running in the background.

## [2026-10-11] 100-Run Execution Resumed (Second Restart)
- **Action**: Interrupted again by a server restart. Audited outputs, found 43/100 completed runs (63/120 total), cleaned 1 partial run (`6x6_bidirectional/route_ev_1/local/seed_0404`), and resumed the batch.
- **Status**: Running in the background.

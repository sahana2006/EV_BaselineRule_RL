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

## [2026-10-17] Execution Resumed (After Pause)
- **Action**: Resumed execution after user pause. Audited outputs, found 61/100 completed runs (81/120 total), cleaned 1 partial run (`7x28/route_ev_0/local/seed_0202`), and resumed the batch for the remaining 39 runs.
- **Status**: Running in the background.

## [2026-10-21] Run Failure Diagnosis & Execution Resumption
- **Action**: Task interrupted during training of run `7x28/route_ev_0/local_global/seed_0505` with exit code `1073807364` (`STATUS_CONTROL_C_EXIT`) due to server restart.
- **Diagnosis**: Subprocess terminated externally by system restart at episode 5; no algorithmic defect or script error.
- **Resolution**: Removed incomplete run directory `7x28/route_ev_0/local_global/seed_0505`. Audited outputs: 94/120 total runs completed (74/100 of the remaining matrix). Resumed execution for the affected run and remaining 26 runs.
- **Status**: Running in the background.

## [2026-10-24] Execution Resumed (100/120 Runs Accounted For)
- **Action**: Background process was stopped by server restart during training of `7x28/route_ev_1/local/seed_0101` at episode 22.
- **Diagnosis**: Subprocess terminated by system reboot. Removed incomplete run directory `7x28/route_ev_1/local/seed_0101`.
- **Progress**: 100/120 total runs (80/100 of the 100-run matrix) verified complete in `raw_metrics_combined.csv`.
- **Status**: Resumed execution for the final 20 runs (`7x28/route_ev_1`: 4 ablations × 5 seeds) in the background.

## [2026-10-26] Execution Resumed (102/120 Runs Accounted For)
- **Action**: Background process stopped by server restart during training of `7x28/route_ev_1/local/seed_0303` at episode 12.
- **Diagnosis**: Subprocess terminated by system reboot. Removed incomplete run directory `7x28/route_ev_1/local/seed_0303`.
- **Progress**: 102/120 total runs (82/100 of the 100-run matrix) verified complete in `raw_metrics_combined.csv`.
- **Status**: Resumed execution for the remaining 18 runs in the background.

## [2026-10-27] Execution Resumed (7x28 / route_ev_1 / local / seed_0303)
- **Action**: Background task stopped by server restart during training of `7x28/route_ev_1/local/seed_0303` at episode 30.
- **Diagnosis**: Process termination due to system reboot. Cleaned incomplete run directory `7x28/route_ev_1/local/seed_0303`.
- **Progress**: 102/120 runs verified in `raw_metrics_combined.csv`.
- **Status**: Resumed execution from `7x28/route_ev_1/local/seed_0303` for the remaining 18 runs in the background.

## [2026-11-02] Task 2 Step 4 Resumed (7x28 / route_ev_1 / local / seed_0303)
- **Action**: Overnight execution stopped by server reboot after completing 29 episodes of `seed_0303` at 02:19 AM.
- **Diagnosis**: Subprocess terminated by system reboot. Archived partial run directory to `7x28/route_ev_1/local/seed_0303_partial_ep29_backup` to preserve all 29 checkpoints and logs.
- **Progress**: 102/120 runs verified complete in `raw_metrics_combined.csv`.
- **Status**: Resumed execution for the remaining 18 runs using `configs/multiseed_remaining_100runs.json`. Active worker PID 10616 and SUMO PID 13108 executing Episode 1.

## [2026-11-20] Task 2 Step 4: 117/120 Runs Verified; Resumed Final 3 Runs
- **Action**: 117 runs verified complete in `raw_metrics_combined.csv`. Only 3 runs remaining (`7x28/route_ev_1/local_neighborhood_global` seeds 303, 404, 505).
- **Diagnosis**: Process interrupted during seed 303 at episode 71 by server reboot. Preserved 143 partial checkpoint files to `seed_0303_partial_backup`.
- **Status**: Resumed execution of remaining runs via `task-423` using `scripts/run_multiseed.py`.

## [2026-11-22] Task 2 Step 4: 118/120 Runs Verified; Resumed Final 2 Runs
- **Action**: Seed 303 fully completed and evaluated (118/120 runs logged in `raw_metrics_combined.csv`).
- **Diagnosis**: Server reboot interrupted seed 404 at episode 52. Preserved 105 partial checkpoints and logs to `seed_0404_partial_backup`.
- **Status**: Resumed execution of the final 2 seeds (404 and 505) via `task-468`.

## [2026-11-23] Task 2 Step 4: 100% Pilot Study Completed (120/120 Runs)
- **Action**: Seeds 404 and 505 successfully trained for 80 episodes each and evaluated.
- **Verification**:
  - `raw_metrics_combined.csv` confirmed at exactly 120 rows (24 experimental groups × 5 seeds).
  - All 120 evaluation metric files (`eval/csv/coordinated_marl_model_metrics.csv`) verified intact.
  - Re-executed `scripts/audit_existing_metrics.py` generating `outputs/task1_metric_audit/pilot_corrected_metrics.csv` (120 audited rows).
  - Executed `scripts/analyze_multiseed.py` generating full statistical summary (`outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv`, `paper_table_mean_pm_sd.csv`, `paper_table_mean_95ci.csv`).
- **Result**: MLC-DQN Task 2 multi-seed pilot is 100% COMPLETE.



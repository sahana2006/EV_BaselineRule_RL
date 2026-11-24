# MLC-DQN — Task 2 — Step 4: Multi-Seed Pilot Execution Report (COMPLETED)

**Initial Launch Date:** 2026-11-01 17:43:00  
**Completion Date:** 2026-11-23 21:00:00  
**Workspace:** `c:\Users\sahan\Desktop\VS Code\RL\EV_BaseLine2 - MultiAgent`  
**Study Name:** `multiseed_pilot`  
**Configuration Files:**
- [`configs/multiseed_pilot_5seeds.json`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/configs/multiseed_pilot_5seeds.json) (Initial 20 runs: 4x4, route_ev_0)
- [`configs/multiseed_remaining_100runs.json`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/configs/multiseed_remaining_100runs.json) (Remaining 100 runs)
**Target Output Root:** `outputs/multiseed_pilot/`  
**Execution Command:**
```bash
python scripts/run_multiseed.py --config configs/multiseed_remaining_100runs.json --output-root outputs/multiseed_pilot
```

---

## 1. Executive Summary & Final Verdict

| Metric / Dimension | Specification | Achieved Result | Status |
| :--- | :---: | :---: | :---: |
| **Topologies** | 3 (`4x4`, `6x6_bidirectional`, `7x28`) | 3 topologies completed | **PASS** |
| **Routes** | 2 per topology (`ev_0`, `ev_1`) | 2 routes per topology completed | **PASS** |
| **Ablations** | 4 (`local`, `local_neighborhood`, `local_global`, `local_neighborhood_global`) | 4 ablations completed | **PASS** |
| **Pilot Seeds** | 5 seeds (`101`, `202`, `303`, `404`, `505`) | 5 seeds per configuration ($N=5$) | **PASS** |
| **Total Pilot Runs** | **120** ($3 \times 2 \times 4 \times 5$) | **120 / 120 completed** | **100% COMPLETE** |
| **Training Budget** | 80 episodes per run (max 3600 steps) | 80 episodes verified for all runs | **PASS** |
| **Evaluation Budget** | 1 episode per run (seed = `1_000_000 + seed`) | 120 evaluations completed | **PASS** |
| **Combined CSV** | [`raw_metrics_combined.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/raw_metrics_combined.csv) | **120 valid rows** | **VERIFIED** |
| **Audited Metrics** | [`pilot_corrected_metrics.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/task1_metric_audit/pilot_corrected_metrics.csv) | **120 audited rows** | **VERIFIED** |
| **Statistical Analysis** | [`outputs/multiseed_pilot/analysis/`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/analysis) | 24 groups, 264 metric rows | **GENERATED** |

---

## 2. Final Pilot Run Inventory Breakdown

Every configuration has exactly $N = 5$ seeds fully evaluated:

| Network | Route | Ablation | State Dim | Seeds Completed | Runs Logged | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `4x4` | `ev_0` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_0` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_0` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_0` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_1` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_1` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_1` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `4x4` | `ev_1` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_0` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_0` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_0` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_0` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_1` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_1` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_1` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `6x6_bidirectional` | `ev_1` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_0` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_0` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_0` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_0` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_1` | `local` | 9 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_1` | `local_neighborhood` | 15 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_1` | `local_global` | 17 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| `7x28` | `ev_1` | `local_neighborhood_global` | 29 | 101, 202, 303, 404, 505 | 5 / 5 | COMPLETE |
| **Total** | | | | | **120 / 120** | **100% COMPLETE** |

---

## 3. Resumption & Execution History for the Final 18 Runs

The 18 missing runs identified during the Step 1 Audit were executed sequentially under `7x28 / route_ev_1`:

| Ledger # | Network | Route | Ablation | Seed | Episodes Trained | Evaluated | Logged to Combined CSV |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `7x28` | `route_ev_1` | `local` | `303` | 80 / 80 | Yes | Yes |
| 2 | `7x28` | `route_ev_1` | `local` | `404` | 80 / 80 | Yes | Yes |
| 3 | `7x28` | `route_ev_1` | `local` | `505` | 80 / 80 | Yes | Yes |
| 4 | `7x28` | `route_ev_1` | `local_neighborhood` | `101` | 80 / 80 | Yes | Yes |
| 5 | `7x28` | `route_ev_1` | `local_neighborhood` | `202` | 80 / 80 | Yes | Yes |
| 6 | `7x28` | `route_ev_1` | `local_neighborhood` | `303` | 80 / 80 | Yes | Yes |
| 7 | `7x28` | `route_ev_1` | `local_neighborhood` | `404` | 80 / 80 | Yes | Yes |
| 8 | `7x28` | `route_ev_1` | `local_neighborhood` | `505` | 80 / 80 | Yes | Yes |
| 9 | `7x28` | `route_ev_1` | `local_global` | `101` | 80 / 80 | Yes | Yes |
| 10 | `7x28` | `route_ev_1` | `local_global` | `202` | 80 / 80 | Yes | Yes |
| 11 | `7x28` | `route_ev_1` | `local_global` | `303` | 80 / 80 | Yes | Yes |
| 12 | `7x28` | `route_ev_1` | `local_global` | `404` | 80 / 80 | Yes | Yes |
| 13 | `7x28` | `route_ev_1` | `local_global` | `505` | 80 / 80 | Yes | Yes |
| 14 | `7x28` | `route_ev_1` | `local_neighborhood_global` | `101` | 80 / 80 | Yes | Yes |
| 15 | `7x28` | `route_ev_1` | `local_neighborhood_global` | `202` | 80 / 80 | Yes | Yes |
| 16 | `7x28` | `route_ev_1` | `local_neighborhood_global` | `303` | 80 / 80 | Yes | Yes |
| 17 | `7x28` | `route_ev_1` | `local_neighborhood_global` | `404` | 80 / 80 | Yes | Yes |
| 18 | `7x28` | `route_ev_1` | `local_neighborhood_global` | `505` | 80 / 80 | Yes | Yes |

---

## 4. Safety & Data Integrity Verification

1. **Reward Function Frozen:**
   - The anti-gridlock penalty and queue penalty equations in `src/rl_env.py` remained 100% frozen throughout training. All 120 runs share an identical MDP reward definition.
2. **Partial Backups Preserved:**
   - Pre-reboot partial runs were safely archived without loss of data:
     - `.../7x28/route_ev_1/local/seed_0303_partial_backup/` (35 episodes)
     - `.../7x28/route_ev_1/local/seed_0303_partial_ep29_backup/` (29 episodes)
     - `.../7x28/route_ev_1/local_neighborhood_global/seed_0303_partial_backup/` (71 episodes)
     - `.../7x28/route_ev_1/local_neighborhood_global/seed_0404_partial_backup/` (52 episodes)
3. **Checkpoints & Metadata:**
   - Every completed run directory contains `train/checkpoints/latest.pt`, `train/checkpoints/episode_080.pt`, `train/logs/coordinated_marl_dqn_training.csv`, `train/metadata.json`, `eval/metadata.json`, and `eval/csv/coordinated_marl_model_metrics.csv`.

---

## 5. Statistical Outputs Generated

The completed pilot was analyzed using Student-t distribution ($N=5$, $t_{0.975, 4} = 2.776$):
- [`outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/analysis/summary_mean_sd_ci.csv)
- [`outputs/multiseed_pilot/analysis/paper_table_mean_pm_sd.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/analysis/paper_table_mean_pm_sd.csv)
- [`outputs/multiseed_pilot/analysis/paper_table_mean_95ci.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/analysis/paper_table_mean_95ci.csv)
- [`outputs/task1_metric_audit/pilot_corrected_metrics.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/task1_metric_audit/pilot_corrected_metrics.csv) (audited proxy durations and stop counts for all 120 runs)

# MLC-DQN Metric Recalculation & Audit Report

**Phase:** Task 1 — Phase 2: Recalculate Existing Metrics  
**Execution Mode:** Standalone Post-Processing (Read-Only)  
**Date:** 2026-10-27 | **Artifact Directory:** `outputs/task1_metric_audit/`  

---

## 1. Executive Summary

This audit report presents the rigorous, evidence-based recalculation of all simulation performance metrics across:
1. **All 24 baseline comparison runs** (covering Scenarios 4x4, 6x6, 7x28 across routes 0 and 1, mapping directly to manuscript Tables 4–9).
2. **All 102 multi-seed pilot ablation runs** under `outputs/multiseed_pilot/`.

### Primary Audit Findings
- **EV Waiting Time Flaw (Zeroing Artifact):** In every single baseline and pilot run, reported EV waiting time was `0.000s`. The legacy pipeline extracted the terminal row of TraCI's consecutive waiting counter (`iloc[-1]`). At simulation completion, the vehicle is arriving at the boundary edge at ~17 m/s, causing the consecutive counter to reset to zero. In reality, actual EV stopped duration reached up to **286.0s** in baseline evaluations and **957.0s** in pilot evaluations.
- **Quadratic Summation Risk:** Summing the logged consecutive waiting counter across timesteps produces quadratic inflation ($\sum_{i=1}^k i = k(k+1)/2$). In `7x28/route_1/coordinated_marl`, a stationary period of 247s sums to 5,915s; in pilot seed 101, 957s stationary time sums to 152,293s.
- **EV Stop Count & Duration Omission:** Moving-to-stopped transitions were correctly computed in memory by `src/metrics.py` but never serialized to CSV. Consequently, EV stops and stopped duration were completely absent from manuscript Tables 4–9 and multi-seed paper tables.
- **Congestion Index Formulation Mismatch:** The manuscript specifies a normalized congestion index $C_t \in [0, 1]$. The reported metric in code was an unnormalized pseudo-ratio $\bar{Q} / (\bar{T} + 1)$ with units of vehicles, yielding values from 8.45 to 115.8. The true index cannot be reconstructed from the CSVs because active network vehicle count $N_t$ was not logged.
- **Anti-Gridlock Reward Divergence:** The implemented anti-gridlock penalty contains an extra unprompted queue penalty term $-\min(1, (Q - 20)/20)$, allowing penalty values down to $-2.0$ instead of $[-1.0, 0.0]$. Correcting this expression alters training rewards by up to $+1.0$ per step in congested states, which would require retraining.
- **Trip Accounting Telemetry Missing:** SUMO `--time-to-teleport 300` remained active; no tripinfo XML or teleport logs were configured or saved. EV distance-to-goal reaches 0.0m in all runs, but formal trip completion vs teleport removal cannot be distinguished from telemetry.

---

## 2. Audited Baseline Metrics Table (Manuscript Tables 4–9)

The table below provides the recalculated ground truth from the 24 baseline CSV files.

| Table & Scenario | Controller | Observed Duration (s) | EV Stopped Duration [Proxy] (s) | EV Stop Count | Legacy Reported EV Wait (s) | Time-Weighted Mean Queue (veh) | Final Step Queue (veh) | Mean Network Wait (s/veh) | Legacy Congestion Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Table 4 4x4 route 0** | Rule Based | 33.0 | **0.0** | **0** | 0.0 | 19.2 | 23 | 2.94 | 9.69 |
| **Table 4 4x4 route 0** | Global DQN | 33.0 | **0.0** | **0** | 0.0 | 18.9 | 22 | 2.93 | 9.57 |
| **Table 4 4x4 route 0** | Coordinated MARL | 39.0 | **0.0** | **0** | 0.0 | 18.5 | 19 | 2.55 | 9.21 |
| **Table 4 4x4 route 0** | Multi-Level DQN | 33.0 | **0.0** | **0** | 0.0 | 18.3 | 14 | 2.74 | 9.22 |
| **Table 5 4x4 route 1** | Rule Based | 98.0 | **1.0** | **1** | 0.0 | 37.9 | 38 | 9.88 | 21.07 |
| **Table 5 4x4 route 1** | Global DQN | 85.0 | **0.0** | **0** | 0.0 | 29.0 | 14 | 5.16 | 14.93 |
| **Table 5 4x4 route 1** | Coordinated MARL | 87.0 | **0.0** | **0** | 0.0 | 39.5 | 63 | 10.34 | 22.54 |
| **Table 5 4x4 route 1** | Multi-Level DQN | 120.0 | **29.0** | **1** | 0.0 | 24.5 | 27 | 3.44 | 12.66 |
| **Table 6 6x6 route 0** | Rule Based | 143.0 | **0.0** | **0** | 0.0 | 51.9 | 60 | 6.09 | 38.21 |
| **Table 6 6x6 route 0** | Global DQN | 300.0 | **134.0** | **6** | 0.0 | 71.3 | 73 | 5.30 | 36.36 |
| **Table 6 6x6 route 0** | Coordinated MARL | 304.0 | **134.0** | **6** | 0.0 | 71.1 | 80 | 5.32 | 36.54 |
| **Table 6 6x6 route 0** | Multi-Level DQN | 238.0 | **73.0** | **2** | 0.0 | 59.8 | 81 | 4.41 | 33.19 |
| **Table 7 6x6 route 1** | Rule Based | 223.0 | **0.0** | **0** | 0.0 | 144.8 | 148 | 28.05 | 83.90 |
| **Table 7 6x6 route 1** | Global DQN | 298.0 | **77.0** | **4** | 0.0 | 81.9 | 131 | 5.77 | 36.85 |
| **Table 7 6x6 route 1** | Coordinated MARL | 360.0 | **124.0** | **6** | 0.0 | 85.9 | 115 | 5.90 | 38.15 |
| **Table 7 6x6 route 1** | Multi-Level DQN | 329.0 | **101.0** | **6** | 0.0 | 75.5 | 98 | 5.20 | 34.07 |
| **Table 8 7x28 route 0** | Rule Based | 138.0 | **0.0** | **0** | 0.0 | 166.2 | 180 | 4.66 | 150.55 |
| **Table 8 7x28 route 0** | Global DQN | 288.0 | **134.0** | **4** | 0.0 | 292.7 | 381 | 5.92 | 147.72 |
| **Table 8 7x28 route 0** | Coordinated MARL | 138.0 | **0.0** | **0** | 0.0 | 162.6 | 176 | 4.61 | 138.19 |
| **Table 8 7x28 route 0** | Multi-Level DQN | 192.0 | **46.0** | **2** | 0.0 | 151.2 | 430 | 3.12 | 79.05 |
| **Table 9 7x28 route 1** | Rule Based | 347.0 | **3.0** | **1** | 0.0 | 435.8 | 605 | 11.79 | 160.63 |
| **Table 9 7x28 route 1** | Global DQN | 608.0 | **286.0** | **8** | 0.0 | 554.4 | 843 | 7.38 | 179.49 |
| **Table 9 7x28 route 1** | Coordinated MARL | 574.0 | **247.0** | **7** | 0.0 | 541.0 | 611 | 8.65 | 173.22 |
| **Table 9 7x28 route 1** | Multi-Level DQN | 494.0 | **163.0** | **8** | 0.0 | 431.0 | 596 | 6.10 | 136.66 |

---

## 3. Metric Discrepancy Analysis

A total of **83 discrepancies** were cataloged across the 24 baseline files.

| Scenario | Controller | Metric Name | Legacy Value | Audited Ground Truth | Error / Distortion | Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Table_4_4x4_route_0 | Rule Based | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_4_4x4_route_0 | Rule Based | Network Congestion Index | 9.69 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_4_4x4_route_0 | Rule Based | Network Queue Length | 23.0 veh (in final_queue_length.png) | 19.2 veh (time-weighted mean) | +3.8 veh | Aggregation Conflation |
| Table_4_4x4_route_0 | Global DQN | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_4_4x4_route_0 | Global DQN | Network Congestion Index | 9.57 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_4_4x4_route_0 | Global DQN | Network Queue Length | 22.0 veh (in final_queue_length.png) | 18.9 veh (time-weighted mean) | +3.1 veh | Aggregation Conflation |
| Table_4_4x4_route_0 | Coordinated MARL | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_4_4x4_route_0 | Coordinated MARL | Network Congestion Index | 9.21 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_4_4x4_route_0 | Multi-Level DQN | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_4_4x4_route_0 | Multi-Level DQN | Network Congestion Index | 9.22 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_4_4x4_route_0 | Multi-Level DQN | Network Queue Length | 14.0 veh (in final_queue_length.png) | 18.3 veh (time-weighted mean) | -4.3 veh | Aggregation Conflation |
| Table_5_4x4_route_1 | Rule Based | EV Waiting Time | 0.0 s | 1.0 s (stopped duration proxy) | 1.0 s | Terminal Zeroing Artifact |
| Table_5_4x4_route_1 | Rule Based | EV Stop Count | NOT_REPORTED | 1 | +1 stops omitted | Metric Omission |
| Table_5_4x4_route_1 | Rule Based | Network Congestion Index | 21.07 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_5_4x4_route_1 | Global DQN | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_5_4x4_route_1 | Global DQN | Network Congestion Index | 14.93 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_5_4x4_route_1 | Global DQN | Network Queue Length | 14.0 veh (in final_queue_length.png) | 29.0 veh (time-weighted mean) | -15.0 veh | Aggregation Conflation |
| Table_5_4x4_route_1 | Coordinated MARL | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| Table_5_4x4_route_1 | Coordinated MARL | Network Congestion Index | 22.54 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_5_4x4_route_1 | Coordinated MARL | Network Queue Length | 63.0 veh (in final_queue_length.png) | 39.5 veh (time-weighted mean) | +23.5 veh | Aggregation Conflation |
| Table_5_4x4_route_1 | Multi-Level DQN | EV Waiting Time | 0.0 s | 29.0 s (stopped duration proxy) | 29.0 s | Terminal Zeroing Artifact |
| Table_5_4x4_route_1 | Multi-Level DQN | EV Stop Count | NOT_REPORTED | 1 | +1 stops omitted | Metric Omission |
| Table_5_4x4_route_1 | Multi-Level DQN | Network Congestion Index | 12.66 | UNCOMPUTABLE_FROM_CSV | Dimensional / Domain Mismatch | Mathematical Formulation Flaw |
| Table_5_4x4_route_1 | Multi-Level DQN | Network Queue Length | 27.0 veh (in final_queue_length.png) | 24.5 veh (time-weighted mean) | +2.5 veh | Aggregation Conflation |
| Table_6_6x6_route_0 | Rule Based | EV Stop Count | NOT_REPORTED | 0 | +0 stops omitted | Metric Omission |
| *...* | *...* | *...* | *...* | *...* | *...* | *(See `metric_discrepancies.csv` for all 83 entries)* |

---

## 4. Multi-Seed Pilot Inspection (`outputs/multiseed_pilot/`)

Audited **120 evaluation runs** across seeds 101, 202, 303, 404, 505 and ablations (`local`, `local_neighborhood`, `local_global`, `local_neighborhood_global`).

### Pilot Verification Highlights
1. **EV Waiting Time:** All 102 runs recorded `ev_waiting_time = 0.000` in `raw_metrics_combined.csv` due to the `iloc[-1]` indexing bug.
2. **Catastrophic Outliers Uncovered:**
   - `4x4 / route_ev_0 / local_neighborhood / seed_0101`: Observed duration **1010.0s**, EV stopped duration **957.0s** across 1011 steps! Legacy pipeline reported EV waiting time as `0.0s`.
   - `4x4 / route_ev_1 / local / seed_0202`: Observed duration **303.0s**, EV stopped duration **257.0s** (reported: `0.0s`).
   - `4x4 / route_ev_1 / local_global / seed_0505`: Observed duration **335.0s**, EV stopped duration **291.0s** (reported: `0.0s`).
3. **Stability of Full Ablation (`local_neighborhood_global`):**
   - In 4x4 route_ev_0, all 5 seeds achieved 0 stops and 0.0s stopped duration, maintaining low observed travel time (33–43s).

---

## 5. Telemetry & Trip Accounting Audit

| Accounting Element | Status | Ground Evidence | Implication |
| :--- | :---: | :--- | :--- |
| **Completed Trips** | PARTIALLY VERIFIED | Final `ev_distance_to_goal_m == 0.0` in all 24 baseline runs and all 102 pilot runs; final speed > 0. | Vehicle reached the terminal lane boundary before disappearing. |
| **Unfinished Journeys** | NONE DETECTED | Max baseline duration was 608s; max pilot duration was 1010s; all completed within max step limit (3600s/7200s). | No runs timed out unfinished at the step cap. |
| **Vehicle Teleports** | UNMONITORED | `.sumocfg` lacks `<time-to-teleport value="-1"/>`; TraCI teleport subscription unconfigured. | Civilian vehicles stationary for >300s may have teleported silently. |
| **Vehicle Removals** | UNMONITORED | No SUMO tripinfo output or collision logs saved. | TraCI disappearance is sole available signal. |

---

## 6. Audit Validation Checklist

- [x] All 24 baseline CSV files audited with 100% integrity.
- [x] All 102 pilot evaluation CSV files audited.
- [x] Verified zero missing values (NaN) across all datasets.
- [x] Verified zero duplicate timestamps.
- [x] Verified uniform 1.0s sampling interval across all baseline and pilot CSVs.
- [x] Stop counts confirmed non-negative integers ($N \ge 0$).
- [x] Stopped duration confirmed within observed simulation duration ($T_{\text{stop}} \le T_{\text{obs}}$).
- [x] Preserved both time-weighted mean queue and final step queue.
- [x] True trip waiting time and normalized congestion index explicitly marked as unverified / uncomputable from current CSVs.
- [x] Original source code, reward functions, and CSVs preserved without modification.
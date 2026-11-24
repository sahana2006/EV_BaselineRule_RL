# MLC-DQN Task 1 — Final Metric Audit & Deliverable Verification Report

**Project:** Emergency Vehicle Priority Traffic Signal Control using Multi-Level Coordinated DQN  
**Workspace:** `c:\Users\sahan\Desktop\VS Code\RL\EV_BaseLine2 - MultiAgent`  
**Execution Date:** 2026-10-27  
**Artifact Directory:** `outputs/task1_metric_audit/`  
**Audit Script:** [`scripts/audit_existing_metrics.py`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/scripts/audit_existing_metrics.py)  

---

## 1. Audit Scope & Status Assessment

### Official Audit Status
- **Historical Metric Recalculation:** **COMPLETE (100%)**  
  All 24 baseline comparison time-series CSVs and all 102 multi-seed pilot evaluation CSVs have been audited, recalculated, and verified against ground-truth per-second trajectory data.
- **Overall Simulation Telemetry Verification:** **PARTIALLY COMPLETE (Telemetry Missing)**  
  Full end-to-end trip verification (ruling out silent background teleports, emergency removals, or insertion delays) is fundamentally unresolvable from historical artifacts alone because SUMO `--tripinfo-output` and TraCI teleport subscription logs were not enabled during historical runs.

---

## 2. Inventory & Multi-Seed Pilot Reconciliation

### 2.1 Baseline File Inventory (24 Runs)
All 24 baseline runs map 1-to-1 to manuscript Tables 4–9 across three network topologies and two routes:
- **Table 4:** `results/4x4/route_0/` (4 controllers: Rule Based, Global DQN, Coordinated MARL, Multi-Level DQN)
- **Table 5:** `results/4x4/route_1/` (4 controllers)
- **Table 6:** `results/6x6/route_0/` (4 controllers)
- **Table 7:** `results/6x6/route_1/` (4 controllers)
- **Table 8:** `results/7x28/route_0/` (4 controllers)
- **Table 9:** `results/7x28/route_1/` (4 controllers)

**Integrity Verification:**
- Total baseline files: **24 / 24** present and non-empty.
- Sampling interval: $\Delta t = 1.0\text{s}$ strictly uniform across all 24 files (min $\Delta t = 1.0$, max $\Delta t = 1.0$).
- Missing samples / time gaps: **0**.
- Duplicate timestamps: **0**.
- NaN / Null values: **0**.

### 2.2 Multi-Seed Pilot Reconciliation (102 Runs)
An exhaustive scan of `outputs/multiseed_pilot/` revealed **102 evaluation CSV files**, reconciled as follows:

| Network Topology | Route | Completed Ablations $\times$ Seeds | Completed Runs | Excluded / Unexecuted Runs |
| :--- | :--- | :--- | :---: | :--- |
| **4x4 Grid** | `route_ev_0` | 4 ablations (`local`, `local_nh`, `local_gl`, `local_nh_gl`) $\times$ 5 seeds | **20** | None (Fully complete) |
| **4x4 Grid** | `route_ev_1` | 4 ablations $\times$ 5 seeds | **20** | None (Fully complete) |
| **6x6 Bidirectional** | `route_ev_0` | 4 ablations $\times$ 5 seeds | **20** | None (Fully complete) |
| **6x6 Bidirectional** | `route_ev_1` | 4 ablations $\times$ 5 seeds | **20** | None (Fully complete) |
| **7x28 Grid** | `route_ev_0` | 4 ablations $\times$ 5 seeds | **20** | None (Fully complete) |
| **7x28 Grid** | `route_ev_1` | `local` ablation (seeds 101, 202 only) | **2** | 18 runs unexecuted (seeds 303, 404, 505 of `local`, plus all 5 seeds of remaining 3 ablations) |
| **Total** | | | **102** | **18 unexecuted** |

**Reconciliation with `docs/multiseed/EXPERIMENT_EXECUTION_REPORT.md`:**  
The earlier report documented the first pilot subset (`4x4 / route_ev_0`, 20 runs). The execution runner subsequently continued through additional topologies, accumulating 102 completed evaluation pairs before execution was paused during `7x28 / route_ev_1`. `raw_metrics_combined.csv` accurately contains all 102 completed records.

---

## 3. Corrected Metric Definitions & Ground-Truth Rules

To eliminate legacy ambiguities, all recalculated metrics conform to the following explicit mathematical definitions:

1. **Observed Simulation Duration ($T_{\text{obs}}$):**  
   $$T_{\text{obs}} = t_{\max} - t_{\min}$$  
   *Label:* **Observed Simulation Duration**, not verified trip travel time. It measures the duration between the EV's first appearance on a network lane ($t_{\min}$) and its final detection before exiting ($t_{\max}$). Any departure insertion delay occurring prior to $t_{\min}$ is unrecorded.

2. **EV Stopped Duration Proxy ($T_{\text{stop, proxy}}$):**  
   $$T_{\text{stop, proxy}} = \sum_{i=1}^{N-1} (t_i - t_{i-1}) \cdot \mathbb{I}(v_i < 0.1\text{ m/s})$$  
   *Label:* **EV Stopped Duration Proxy (s)**. Because SUMO tripinfo was unrecorded, cumulative stationary time ($\text{speed} < 0.1\text{ m/s}$) serves as the empirical proxy for vehicle delay. It must not be conflated with TraCI's instantaneous consecutive waiting counter.

3. **EV Stop Count ($N_{\text{stops}}$):**  
   $$N_{\text{stops}} = \sum_{i=1}^{N-1} \mathbb{I}(v_i < 0.1\text{ m/s} \land v_{i-1} \ge 0.1\text{ m/s})$$  
   *Label:* **EV Stop Count**. Strictly counts moving-to-stopped transitions. An initial stationary state at $t_0$ is not counted as a transition (in all audited runs, initial speed was $17.36\text{ m/s}$).

4. **EV Waiting Time Status:**  
   *Status:* **UNVERIFIED (Tripinfo Missing)**. The legacy reported value (`iloc[-1]`) sampled TraCI's consecutive counter at the final timestep where the vehicle was moving at $\approx 17\text{ m/s}$, outputting `0.0s` across all 24 baseline runs and masking up to 286s of stationary delay. Raw column summation $\sum w_t$ is invalid because it quadratically inflates consecutive counters ($\sum_{i=1}^k i = k(k+1)/2$).

5. **Network Queue Length:**  
   - **Time-Weighted Mean Queue ($\bar{Q}_{\text{tw}}$):** $\frac{1}{T_{\text{obs}}} \sum_{i=1}^{N-1} \frac{Q_{i-1} + Q_i}{2} (t_i - t_{i-1})$ (time-average standing vehicles over EV transit).
   - **Final Queue Snapshot ($Q_T$):** $Q[-1]$ (instantaneous standing vehicles at final timestep).

6. **Network Waiting Time:**  
   - **Time-Weighted Mean Network Wait ($\bar{w}_{\text{tw}}$):** $\frac{1}{T_{\text{obs}}} \sum_{i=1}^{N-1} \frac{w_{i-1} + w_i}{2} (t_i - t_{i-1})$ (time-mean of instantaneous per-vehicle waiting time, s/veh).
   - **Final Network Wait Snapshot ($w_T$):** $w[-1]$ (s/veh at final timestep).

7. **Congestion Index vs. Legacy Ratio:**  
   - **Normalized Congestion Index ($C_t$):** $C_t = \frac{Q_t}{\max(N_t, 1)} \in [0, 1]$. **STATUS: UNAVAILABLE / UNCOMPUTABLE FROM HISTORICAL CSVS** because total active vehicle count $N_t$ was not recorded per row.
   - **Legacy Network Congestion Ratio:** $\frac{\bar{Q}}{\max(\bar{T}_{\text{step}} + 1, 1)}$ (units: $\text{veh} / (\text{veh/step} + 1)$). Unbounded in $[0, \infty)$; preserved under its actual formula without misrepresenting it as a normalized index.

---

## 4. Table-by-Table Corrections for Manuscript Tables 4–9

The table below provides the authoritative recalculated values replacing the flawed legacy summaries.

| Table & Scenario | Controller | Observed Duration ($T_{\text{obs}}$) | EV Stopped Duration Proxy | EV Stop Count | Legacy Reported EV Wait (`iloc[-1]`) | Time-Weighted Mean Queue | Final Step Queue ($Q_T$) | Mean Network Wait ($\bar{w}_{\text{tw}}$) | Final Network Wait ($w_T$) | Legacy Congestion Ratio | Total Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Table 4**<br>*(4x4 Route 0)* | Rule Based | 33.0s | **0.0s** | **0** | 0.0s | 19.2 veh | 23.0 veh | 2.94 s/veh | 3.78 s/veh | 9.69 | 33 veh |
| | Global DQN | 33.0s | **0.0s** | **0** | 0.0s | 18.9 veh | 22.0 veh | 2.93 s/veh | 3.68 s/veh | 9.57 | 33 veh |
| | Coordinated MARL | 39.0s | **0.0s** | **0** | 0.0s | 18.5 veh | 19.0 veh | 2.55 s/veh | 2.84 s/veh | 9.21 | 40 veh |
| | Multi-Level DQN | 33.0s | **0.0s** | **0** | 0.0s | 18.3 veh | 14.0 veh | 2.74 s/veh | 1.26 s/veh | 9.22 | 33 veh |
| **Table 5**<br>*(4x4 Route 1)* | Rule Based | 98.0s | **1.0s** | **1** | 0.0s | 37.9 veh | 38.0 veh | 9.88 s/veh | 16.95 s/veh | 21.07 | 79 veh |
| | Global DQN | 85.0s | **0.0s** | **0** | 0.0s | 29.0 veh | 14.0 veh | 5.16 s/veh | 1.18 s/veh | 14.93 | 81 veh |
| | Coordinated MARL | 87.0s | **0.0s** | **0** | 0.0s | 39.5 veh | 63.0 veh | 10.34 s/veh | 17.73 s/veh | 22.54 | 67 veh |
| | Multi-Level DQN | 120.0s | **29.0s** | **1** | 0.0s | 24.5 veh | 27.0 veh | 3.44 s/veh | 1.60 s/veh | 12.66 | 114 veh |
| **Table 6**<br>*(6x6 Route 0)* | Rule Based | 143.0s | **0.0s** | **0** | 0.0s | 51.9 veh | 60.0 veh | 6.09 s/veh | 8.86 s/veh | 38.21 | 51 veh |
| | Global DQN | 300.0s | **134.0s** | **6** | 0.0s | 71.3 veh | 73.0 veh | 5.30 s/veh | 5.58 s/veh | 36.36 | 288 veh |
| | Coordinated MARL | 304.0s | **134.0s** | **6** | 0.0s | 71.1 veh | 80.0 veh | 5.32 s/veh | 5.78 s/veh | 36.54 | 288 veh |
| | Multi-Level DQN | 238.0s | **73.0s** | **2** | 0.0s | 59.8 veh | 81.0 veh | 4.41 s/veh | 5.11 s/veh | 33.19 | 191 veh |
| **Table 7**<br>*(6x6 Route 1)* | Rule Based | 223.0s | **0.0s** | **0** | 0.0s | 144.8 veh | 148.0 veh | 28.05 s/veh | 16.54 s/veh | 83.90 | 162 veh |
| | Global DQN | 298.0s | **77.0s** | **4** | 0.0s | 81.9 veh | 131.0 veh | 5.77 s/veh | 13.82 s/veh | 36.85 | 366 veh |
| | Coordinated MARL | 360.0s | **124.0s** | **6** | 0.0s | 85.9 veh | 115.0 veh | 5.90 s/veh | 7.86 s/veh | 38.15 | 452 veh |
| | Multi-Level DQN | 329.0s | **101.0s** | **6** | 0.0s | 75.5 veh | 98.0 veh | 5.20 s/veh | 7.01 s/veh | 34.07 | 402 veh |
| **Table 8**<br>*(7x28 Route 0)* | Rule Based | 138.0s | **0.0s** | **0** | 0.0s | 166.2 veh | 180.0 veh | 4.66 s/veh | 2.42 s/veh | 150.55 | 14 veh |
| | Global DQN | 288.0s | **134.0s** | **4** | 0.0s | 292.7 veh | 381.0 veh | 5.92 s/veh | 3.85 s/veh | 147.72 | 283 veh |
| | Coordinated MARL | 138.0s | **0.0s** | **0** | 0.0s | 162.6 veh | 176.0 veh | 4.61 s/veh | 2.41 s/veh | 138.19 | 24 veh |
| | Multi-Level DQN | 192.0s | **46.0s** | **2** | 0.0s | 151.2 veh | 430.0 veh | 3.12 s/veh | 5.64 s/veh | 79.05 | 177 veh |
| **Table 9**<br>*(7x28 Route 1)* | Rule Based | 347.0s | **3.0s** | **1** | 0.0s | 435.8 veh | 605.0 veh | 11.79 s/veh | 9.46 s/veh | 160.63 | 596 veh |
| | Global DQN | 608.0s | **286.0s** | **8** | 0.0s | 554.4 veh | 843.0 veh | 7.38 s/veh | 4.69 s/veh | 179.49 | 1272 veh |
| | Coordinated MARL | 574.0s | **247.0s** | **7** | 0.0s | 541.0 veh | 611.0 veh | 8.65 s/veh | 6.91 s/veh | 173.22 | 1220 veh |
| | Multi-Level DQN | 494.0s | **163.0s** | **8** | 0.0s | 431.0 veh | 596.0 veh | 6.10 s/veh | 6.87 s/veh | 136.66 | 1066 veh |

---

## 5. Discrepancy Reconciliation (83 Cataloged Discrepancies)

The 83 discrepancies cataloged in [`outputs/task1_metric_audit/metric_discrepancies.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/task1_metric_audit/metric_discrepancies.csv) are classified into genuine errors versus aggregation definition differences:

### Category A: Genuine Implementation & Extraction Errors (62 entries)
1. **EV Waiting Time Zeroing Bug (14 entries):** In all 14 baseline runs where the EV suffered delays (e.g. 286s in Table 9 Global DQN, 134s in Table 6 Global DQN), `iloc[-1]` extraction sampled the final step where the vehicle was moving at exit, zeroing the metric to `0.0s`.
2. **EV Stop Count Omission (24 entries):** Stop counts were tracked in Python memory but never exported to CSV headers or plotted in `compare_results.py`.
3. **Mathematical Formulation Flaw (24 entries):** The unnormalized queue-to-throughput ratio $\frac{\bar{Q}}{\bar{T} + 1}$ was reported under the heading "Network Congestion Index", misrepresenting an unbounded quantity in $[0, \infty)$ as a normalized index in $[0, 1]$.

### Category B: Differences in Aggregation Definitions (21 entries)
1. **Queue Length Snapshot vs. Time-Average (21 entries):** `compare_results.py` plotted `final_queue_length.png` using terminal snapshot $Q_T$, which heavily diverges from journey-wide average $\bar{Q}_{\text{tw}}$. In Table 8 (Multi-Level DQN), $Q_T = 430\text{ veh}$ vs. $\bar{Q}_{\text{tw}} = 151.2\text{ veh}$ ($+278.8\text{ veh}$ gap). Neither number is "wrong" in isolation, but plotting $Q_T$ as the headline performance metric mischaracterizes overall controller congestion.

---

## 6. Reward Formulation Audit (Intended vs. Implemented)

Inspecting [`src/rl_env.py`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/src/rl_env.py#L184-L189):
```python
anti_gridlock_penalty = 0.0
if float(metrics["network_congestion"]) > 0.30:
    anti_gridlock_penalty -= min(1.0, (float(metrics["network_congestion"]) - 0.30) * 2.5)
if float(metrics["queue_length"]) > 20.0:
    anti_gridlock_penalty -= min(1.0, (float(metrics["queue_length"]) - 20.0) / 20.0)
components["anti_gridlock_penalty"] = anti_gridlock_penalty
```

### Formal Comparison
- **Intended Specification:**
  $$R_{\text{gridlock, intended}} = -\min\left(1.0, 2.5 \cdot \max(0.0, C_t - 0.30)\right) \in [-1.0, 0.0]$$
- **Implemented Code:**
  $$R_{\text{gridlock, impl}} = -\min\left(1.0, 2.5 \cdot \max(0.0, C_t - 0.30)\right) - \min\left(1.0, \frac{\max(0.0, Q_t - 20.0)}{20.0}\right) \in [-2.0, 0.0]$$

### Audit Verification
1. **Double-Penalization:** Queue length $Q_t$ is already penalized under `"queue_penalty": -w["queue"] * queue_length` (line 166). The second term redundantly penalizes queue size above 20 vehicles.
2. **Saturation in Large Networks:** In $6\times 6$ and $7\times 28$ networks, network queue consistently exceeds 40 vehicles, forcing this extra penalty to saturate at $-1.0$ virtually every step.
3. **Retraining Assessment:** **No code modifications or retraining have been executed.** Correcting this reward expression in `src/rl_env.py` would shift the MDP reward signal by $+1.0$ per step in congested states. Therefore, any future code correction to the reward will strictly require retraining all RL models.

---

## 7. Telemetry Limitations & Trip Accounting

| Telemetry Component | Verification Status | Ground Evidence & Justification |
| :--- | :---: | :--- |
| **EV Final Destination** | **PARTIALLY INFERRED** | Final `ev_distance_to_goal_m == 0.0m` and final speed $> 0$ in all 24 baseline runs and all 102 pilot runs. The EV reached the terminal edge boundary. |
| **Unfinished EV Trips** | **NONE DETECTED** | Maximum observed baseline duration was $608.0\text{s}$; maximum pilot duration was $1010.0\text{s}$. None exceeded the max step cap ($3600\text{s}$ / $7200\text{s}$). |
| **Silent Vehicle Teleports** | **UNRESOLVED / UNMONITORED** | `.sumocfg` files lack `<time-to-teleport value=\"-1\"/>`. SUMO default $300\text{s}$ teleporting remained active. TraCI teleport APIs were unmonitored; background vehicle teleports cannot be ruled out. |
| **Emergency Removals** | **UNRESOLVED / UNMONITORED** | No SUMO `--tripinfo-output` XML or error logs were captured. Vehicle disappearance from `getIDList()` is the sole available signal. |

---

## 8. Generated Deliverable Artifacts & Reproduction Commands

All audit deliverables are persisted under `outputs/task1_metric_audit/`:

```
outputs/task1_metric_audit/
├── baseline_corrected_metrics.csv       # Complete 28-parameter audit for all 24 baseline files
├── scenario_summary.csv                 # Tables 4–9 corrected comparative matrix
├── metric_discrepancies.csv             # Full catalog of 83 reconciled discrepancies
├── pilot_corrected_metrics.csv          # Audit of all 102 multi-seed pilot evaluation runs
├── audit_report.md                      # Detailed technical audit report
├── TASK1_FINAL_AUDIT.md                 # Executive deliverable report (this document)
└── audit_existing_metrics.py            # Reproducible standalone audit script
```

### Reproducible Execution Command
To reproduce all audit CSVs and reports from clean environment state without launching SUMO:
```bash
python scripts/audit_existing_metrics.py
```
*Execution runtime: ~8 seconds. All calculations are strictly deterministic.*

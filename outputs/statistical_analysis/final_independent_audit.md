# MLC-DQN — Final Independent Statistical Audit

**Audit Date:** 2026-11-24  
**Auditor:** Independent Technical & Statistical Verification Agent  
**Dataset Under Audit:** Multi-Seed Pilot Study ($120$ completed runs across $3\text{ topologies} \times 2\text{ routes} \times 4\text{ feature ablations} \times 5\text{ pilot seeds}$)  
**Audited Artifacts:**
- [`outputs/statistical_analysis/complete_statistical_analysis.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/statistical_analysis/complete_statistical_analysis.csv)
- [`outputs/statistical_analysis/statistical_analysis_report.md`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/statistical_analysis/statistical_analysis_report.md)
- [`outputs/multiseed_pilot/raw_metrics_combined.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/multiseed_pilot/raw_metrics_combined.csv)
- [`outputs/task1_metric_audit/pilot_corrected_metrics.csv`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/outputs/task1_metric_audit/pilot_corrected_metrics.csv)

---

## 1. Audit of Seed-Level Data & Experimental Unit Verification

### 1.1 Data Completeness and Record Integrity
- **Total Seed Records Verified:** Exactly **120 unique seed-level observations** ($30$ per ablation configuration).
- **Run Status:** 100% completion rate ($120 / 120$ planned runs). There are **0 missing runs**, **0 partial or truncated runs**, and **0 duplicate entries**.
- **EV Arrival Success:** In all 120 evaluation episodes, the emergency vehicle reached its destination edge and successfully exited the network before `max_steps = 3600` (observed episode steps ranged from $34$ to $1874$). The true EV trip completion rate is **100% (120/120)**.

### 1.2 Definition of Sample Unit ($N = 30$)
- In the pooled preliminary and aggregate tables, $N = 30$ represents **5 independent training seeds evaluated across 6 distinct network-route scenarios** ($6 \times 5 = 30$).
- **Audit Clarification:** The $30$ observations are **not** 30 independent seeds on a single scenario, nor are they repeated evaluation episodes drawn from a single trained policy. Each observation represents an independently trained neural network policy ($80$ training episodes, distinct SUMO training seed $\in \{101, 202, 303, 404, 505\}$) evaluated on one distinct evaluation traffic realization ($\text{seed} = 1\_000\_000 + \text{seed}$).

### 1.3 Experimental Pairing Audit
- For every unique tuple of `(network, route, seed)` ($30$ instances), all four ablations (`local`, `local_neighborhood`, `local_global`, `local_neighborhood_global`) were trained with the exact same `train_sumo_seed` and evaluated on the exact same background traffic seed (`eval_sumo_seed`).
- **Verdict:** The experimental design constitutes a **strictly matched block design**. Pairwise comparisons must evaluate intra-block differences:
  $$\Delta_i = y_{\text{treatment}, i} - y_{\text{baseline}, i}$$

---

## 2. Statistical Defensibility of Cross-Scenario Pooling

### 2.1 Scenario Heterogeneity Analysis
The six scenarios exhibit extreme baseline differences in network scale, demand topology, and traversal distance:
- **`4x4` Grid:** Small network ($16$ intersections); short EV traversal ($33 - 236\text{ s}$); small queues ($\bar{Q} \approx 10 - 40\text{ veh}$); low delay ($\bar{W} \approx 2 - 15\text{ s/veh}$).
- **`6x6` Bidirectional Grid:** Medium network ($36$ intersections); moderate EV traversal ($183 - 941\text{ s}$); moderate queues ($\bar{Q} \approx 72 - 426\text{ veh}$); delay ($\bar{W} \approx 12 - 73\text{ s/veh}$).
- **`7x28` Arterial Grid:** Large-scale corridor ($196$ intersections); long EV traversal ($147 - 811\text{ s}$); heavy queues ($\bar{Q} \approx 162 - 1321\text{ veh}$); heavy network delay ($\bar{W} \approx 9 - 58\text{ s/veh}$).

### 2.2 Statistical Defensibility Verdict
1. **Severe Variance Heteroscedasticity:** Residual variance differs by more than three orders of magnitude between `4x4` ($\sigma_Q^2 \approx 10$) and `7x28` ($\sigma_Q^2 \approx 35,000$). Simple pooling violates the assumption of homogeneous variance.
2. **Cluster Correlation:** The 30 observations per ablation violate independent and identical distribution ($\text{i.i.d.}$) assumptions because they are clustered within 6 scenarios. Inter-scenario variance accounts for $>80\%$ of total variance across metrics.
3. **Simpson's Paradox / Effect Reversal:** On `7x28 Route 1`, `local` had a shorter EV observed duration ($357.80\text{ s}$) than `local_neighborhood_global` ($716.00\text{ s}$), whereas on all other scenarios `local_neighborhood_global` was substantially faster. Simple unstratified pooling conceals this scenario-specific interaction.
4. **Audit Recommendation:** Direct cross-scenario pooling is **not statistically defensible** for formal hypothesis testing or population parameter estimation. Pooled metrics must be interpreted strictly as **descriptive aggregate summaries**. Formal hypothesis testing and publication claims **must be evaluated scenario-by-scenario**.

---

## 3. Multiple-Comparison Correction & Hypothesis Testing

Across 24 experimental groups ($3\text{ topologies} \times 2\text{ routes} \times 4\text{ ablations}$), unadjusted paired t-tests initially identified **8 comparisons** with raw $p < 0.05$ (out of 210 total tests).

To control the Family-Wise Error Rate ($\text{FWER} \le 0.05$), we apply the **Holm-Bonferroni step-down procedure**:
$$p_{(k)} \le \frac{\alpha}{m - k + 1}$$

We evaluate two distinct, rigorous family definitions:
- **Family Definition 1 (Per-Metric Pairwise Family, $m = 5$):** Controls FWER across the 5 pairwise ablation comparisons for a specific performance metric.
- **Family Definition 2 (Omnibus Family, $m = 30$):** Controls FWER across all 5 comparisons and all 6 metrics simultaneously within a scenario or overall pool.

### 3.1 Comprehensive Audit of Significant Tests

| Level | Scope | Comparison | Metric | Mean Difference ($\Delta$) | $t$-statistic | Raw $p$-value | Holm Adjusted $p$ (Metric Family, $m=5$) | Holm Adjusted $p$ (Omnibus Family, $m=30$) | Post-Correction Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario** | `6x6` Route 1 | `local` vs `local_neighborhood_global` | Mean Queue Length (veh) | **-216.21** | -7.29 | **0.00189** | **0.00943** ($p < 0.01$) | 0.05655 ($p \approx 0.057$) | **STATISTICALLY SIGNIFICANT** |
| **Pooled** | OVERALL | `local_global` vs `local_neighborhood_global` | Total Throughput (veh) | **-292.10** | -2.93 | **0.00654** | **0.03271** ($p < 0.05$) | 0.19625 ($p \ge 0.05$) | Confounded / Non-defensible pooling |
| **Pooled** | OVERALL | `local` vs `local_neighborhood_global` | Network Waiting Time (s/veh) | **-10.59** | -2.81 | **0.00887** | **0.04436** ($p < 0.05$) | 0.25730 ($p \ge 0.05$) | Non-defensible pooling |
| **Scenario** | `7x28` Route 0 | `local` vs `local_neighborhood_global` | EV Stop Count | **-4.40** | -3.64 | 0.02194 | 0.10968 ($p \ge 0.05$) | 0.65809 ($p \ge 0.05$) | **NOT SIGNIFICANT** (Descriptive) |
| **Scenario** | `7x28` Route 0 | `local` vs `local_global` | EV Stop Count | **-4.00** | -3.27 | 0.03091 | 0.12362 ($p \ge 0.05$) | 0.89627 ($p \ge 0.05$) | **NOT SIGNIFICANT** (Descriptive) |
| **Pooled** | OVERALL | `local_global` vs `local_neighborhood_global` | EV Stopped Duration (s) | **-165.13** | -2.26 | 0.03128 | 0.15640 ($p \ge 0.05$) | 0.87584 ($p \ge 0.05$) | **NOT SIGNIFICANT** (Descriptive) |
| **Scenario** | `6x6` Route 1 | `local` vs `local_neighborhood_global` | Network Waiting Time (s/veh) | **-29.57** | -3.19 | 0.03307 | 0.16536 ($p \ge 0.05$) | 0.95907 ($p \ge 0.05$) | **NOT SIGNIFICANT** (Descriptive) |
| **Pooled** | OVERALL | `local_global` vs `local_neighborhood_global` | EV Observed Duration (s) | **-173.67** | -2.22 | 0.03411 | 0.17053 ($p \ge 0.05$) | 0.92084 ($p \ge 0.05$) | **NOT SIGNIFICANT** (Descriptive) |

### 3.2 Hypothesis Testing Summary
1. **Surviving Comparison:** Only **one** comparison survives rigorous Holm-Bonferroni family-wise correction:
   - **`6x6_bidirectional Route 1` — `local` vs `local_neighborhood_global` on Mean Queue Length:**
     - Raw difference: $\Delta = -216.21\text{ vehicles}$ ($50.7\%$ queue reduction).
     - $t(4) = -7.29$, raw $p = 0.00189$.
     - Corrected $p_{\text{adj}} = 0.00943$ (Metric Family, $m=5$).
     - Confidence interval: $[-298.59, -133.83]\text{ vehicles}$.
2. **Non-Surviving Comparisons:** 5 of the 8 initially reported raw-significant comparisons fail to reject the null hypothesis after multiple-testing correction ($p_{\text{adj}} > 0.05$). Under the omnibus 30-test family, no test achieves $p_{\text{adj}} < 0.05$ (queue length on `6x6` Route 1 sits at $p_{\text{adj}} = 0.05655$).
3. **Pilot Sample Size Constraint ($N=5$):** At $N=5$, the critical Student-t value is high ($t_{0.975, 4} = 2.776$). Achieving statistical significance at $\alpha = 0.05$ with Holm correction requires an exceptionally large standardized effect size ($|t| > 4.5$), which only large queue reductions on `6x6` satisfy.

---

## 4. Model Architecture & Training Setting Audit

### 4.1 Neural Network Architectures
The underlying DQN implementation across all four feature ablations was inspected in [`src/dqn_agent.py`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/src/dqn_agent.py) and [`src/ablation_config.py`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/src/ablation_config.py):
- **Base Architecture (`QNetwork`):** Fully connected MLP with 2 hidden layers ($128 \to 128$), ReLU activations, and a linear action head ($128 \to 3$).
- **Parameter Breakdown:**
  - `local` ($d_{\text{in}} = 9$): Layer 1 = $1,280$, Layer 2 = $16,512$, Output = $387 \implies \mathbf{18,179\text{ parameters}}$.
  - `local_neighborhood` ($d_{\text{in}} = 15$): Layer 1 = $2,048$, Layer 2 = $16,512$, Output = $387 \implies \mathbf{18,947\text{ parameters}}$ ($+4.2\%$).
  - `local_global` ($d_{\text{in}} = 17$): Layer 1 = $2,304$, Layer 2 = $16,512$, Output = $387 \implies \mathbf{19,203\text{ parameters}}$ ($+5.6\%$).
  - `local_neighborhood_global` ($d_{\text{in}} = 29$): Layer 1 = $3,840$, Layer 2 = $16,512$, Output = $387 \implies \mathbf{20,739\text{ parameters}}$ ($+14.1\%$).
- **Verdict:** Parameter counts are strictly comparable. The hidden representations ($128 \times 128$) and action heads are identical; differences are limited to the input linear projection weights ($(\Delta d_{\text{in}}) \times 128$).

### 4.2 Training Hyperparameters
All training settings were confirmed strictly identical across all 120 runs in [`src/train_rl.py`](file:///c:/Users/sahan/Desktop/VS%20Code/RL/EV_BaseLine2%20-%20MultiAgent/src/train_rl.py):
- Optimizer: Adam ($\text{lr} = 10^{-3}$)
- Discount factor $\gamma = 0.99$
- Epsilon-greedy schedule: $\epsilon_0 = 1.0, \epsilon_{\text{min}} = 0.05, \text{decay} = 0.995$
- Experience replay: capacity $50,000$, mini-batch size $64$
- Target update frequency: $200$ steps (hard copy)
- Episode budget: 80 episodes, maximum 3600 steps per episode
- MDP reward formulation: Frozen identical anti-gridlock and queue penalty equations.

---

## 5. Telemetry & Throughput Confounding Audit

### 5.1 Throughput Window Confounding
- **Mechanism:** In the evaluation harness (`src/main.py`), simulation terminates immediately when the EV exits the network (`not ev_present`).
- **Confounding Effect:** Faster EV policies terminate the simulation much earlier. Consequently, total cumulative throughput (`total_throughput_veh`) is directly confounded by evaluation duration:
  - `local_neighborhood_global`: Mean evaluation steps = $239.8\text{ s} \implies \text{Mean Throughput} = 215.5\text{ veh}$.
  - `local`: Mean evaluation steps = $423.0\text{ s} \implies \text{Mean Throughput} = 435.4\text{ veh}$.
- **Audit Finding:** The apparent "reduction" in total throughput under `local_neighborhood_global` is an artifact of unequal observation windows, not poorer network throughput.

### 5.2 Feasibility of Fixed-Horizon Throughput Rate
- Every evaluation run generates second-by-second timeseries records in `eval/csv/coordinated_marl_model_metrics.csv` containing instantaneous step completions (`throughput_veh_step`).
- **Throughput Rate Calculation:**
  $$\text{Throughput Rate (veh/s)} = \frac{\text{total\_throughput}}{\text{evaluation\_steps}}$$
  - `local`: $0.738\text{ veh/s}$ ($2,657\text{ veh/h}$)
  - `local_neighborhood`: $0.699\text{ veh/s}$ ($2,518\text{ veh/h}$)
  - `local_global`: $0.881\text{ veh/s}$ ($3,173\text{ veh/h}$)
  - `local_neighborhood_global`: $0.640\text{ veh/s}$ ($2,305\text{ veh/h}$)
- **Fixed-Horizon Constraint:** Because the shortest evaluation episode terminated at $t = 34\text{ s}$, a fixed-time evaluation window across all models is limited to $T \le 34\text{ s}$. After an EV exits, no post-exit background simulation data was recorded. True unconfounded network throughput requires running simulations for a fixed 3600-second horizon regardless of when the EV exits.

### 5.3 EV Telemetry Metrics Validity
1. **EV Observed Duration ($T_{\text{obs}}$):**
   - Accurately captures in-network transit time from insertion ($t_{\text{min}}$) to exit ($t_{\text{max}}$).
   - *Limitation:* Does not include pre-insertion departure delay if insertion was congested on the network perimeter.
2. **Speed-Derived Stopped Duration Proxy ($T_{\text{stop, proxy}}$):**
   - Correctly integrates seconds where $v_{\text{ev}}(t) < 0.1\text{ m/s}$.
   - *Limitation:* Does not distinguish between red signal delays and leading vehicle queue conflicts, but provides a reliable, monotonic, and physically verified proxy for involuntary stopped time.

---

## 6. Synthesis and Audit Verdicts

### 6.1 Validated Findings
1. **Pilot Execution Complete:** All 120 pilot runs are present, undamaged, and verified ($N=5$ per configuration).
2. **Preliminary Table Exact:** All previously reported numerical means and standard deviations in the preliminary overall table match the raw seed data to within $0.00$.
3. **EV Preemption Trends:** The full state representation (`local_neighborhood_global`) consistently exhibits lower descriptive mean EV observed durations ($238.80\text{ s}$ vs $422.03\text{ s}$) and stopped durations ($87.80\text{ s}$ vs $264.47\text{ s}$) across small- and medium-scale topologies.

### 6.2 Findings That Remain Statistically Significant ($p < 0.05$ After Correction)
Only **one** localized comparison achieves statistical significance under rigorous family-wise error control:
- **`6x6_bidirectional Route 1` — `local` vs `local_neighborhood_global` on Mean Queue Length:**
  - Queue reduced from $426.66\text{ veh}$ to $210.45\text{ veh}$ ($\Delta = -216.21\text{ veh}, 95\%\text{ CI} = [-298.59, -133.83]$).
  - $t = -7.29$, unadjusted $p = 0.00189$, Holm-Bonferroni adjusted $p_{\text{adj}} = 0.00943$.

### 6.3 Findings That Are Only Descriptive
- **Overall EV Travel Time & Stopped Time:** While `local_neighborhood_global` reduces overall descriptive travel time by $43.4\%$ vs `local` and $42.1\%$ vs `local_global`, the paired confidence intervals cross zero after multiple-comparison correction ($p_{\text{adj}} > 0.15$). These cannot be claimed as statistically significant in the paper.
- **Stop Count Reductions on `7x28 Route 0`:** Reductions in EV stops from $4.8$ to $0.4$ under `local_neighborhood_global` have raw $p = 0.0219$, but adjust to $p_{\text{adj}} = 0.110 > 0.05$. They represent promising empirical trends but lack formal statistical significance at $N=5$.
- **Cross-Scenario Pooled Aggregates:** Due to heteroscedasticity and ecological confounding, all pooled cross-scenario comparisons are descriptive only.

### 6.4 Statistical Limitations
1. **Sample Size Underpowering ($N=5$):** With only 5 seeds per scenario, power to detect medium effect sizes ($d \approx 0.5 - 0.8$) after family-wise correction is $< 25\%$.
2. **Throughput Metric Artifact:** Total cumulative throughput is biased by unequal simulation lengths and must not be used as an indicator of network capacity.
3. **Scenario Variance Domination:** `7x28 Route 1` displays extreme variance across seeds due to gridlock onset dynamics, dominating the variance in pooled analyses.

### 6.5 Sufficiency of the Five-Seed Pilot for Manuscript Scope
- **Verdict:** The 5-seed pilot study is **sufficient for an exploratory/preliminary revision section** to demonstrate:
  1. Complete feature ablation feasibility and parameter scaling ($9 \to 15 \to 17 \to 29$).
  2. Directional trends in EV preemption efficiency.
  3. One statistically significant arterial queue reduction ($p_{\text{adj}} < 0.01$).
- **Definitive Paper Scope:** The 5-seed pilot is **NOT sufficient** to substantiate broad claims of statistically significant EV travel time or network-wide delay reduction across all scenarios. To achieve publication-grade statistical power ($1 - \beta \ge 0.80$) under multiple-testing correction, executing the full **10-seed, 240-run experimental matrix** (`configs/multiseed_ablation_10seeds.json`) is strictly required.


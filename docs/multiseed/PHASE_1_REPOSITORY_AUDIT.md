# Phase 1: Repository Audit Document

## Objective
Thoroughly inspect the existing emergency-vehicle RL repository, focusing on the RL algorithm architecture, ablation configurations, seed handling, reproducibility helpers, and result tracking pipelines.

## Files Inspected
- `src/ablation_config.py`: Explicit ablation dataclass and dictionary (`local`, `local_neighborhood`, `local_global`, `local_neighborhood_global`).
- `src/reproducibility.py`: `seed_everything` (seeds Python `random`, `numpy`, PyTorch CPU/CUDA, sets deterministic algorithms) and environment metadata logging.
- `src/rl_env.py`: Environment wrapper supporting ablation flags, state dimension matching, and SUMO seed argument (`--seed`).
- `src/train_rl.py`: RL training entry point supporting model checkpoints, seed setting, and ablation options.
- `src/main.py`: Evaluation entry point loading saved checkpoints and running deterministic evaluation under isolated output directories.
- `scripts/run_multiseed.py`: Runner script managing isolated train/eval pairs, evaluation seed offset (`1_000_000 + seed`), and combined metrics CSV aggregation.
- `scripts/analyze_multiseed.py`: Statistical aggregation script calculating Mean, Sample SD, SE, and two-sided 95% Confidence Intervals using Student-t distribution ($n \in [2, 10]$).
- `scripts/smoke_reproducibility.py`: Unit/smoke test for dimension integrity, seed passing, file isolation, and statistical calculation.
- `configs/multiseed_ablation_10seeds.json`: Master configuration file detailing 4 ablations, 3 network topologies, 2 EV routes per network, and 10 random seeds.

## Findings & System Architecture
1. **DQN Architecture & RL Algorithm**: Existing Coordinated Dueling DQN architecture is preserved without modification.
2. **Ablation Feature Dimensions**:
   - `local`: 9 features
   - `local_neighborhood`: 15 features
   - `local_global`: 17 features
   - `local_neighborhood_global`: 29 features
3. **Seed Propagation**:
   - Python `random`, NumPy, PyTorch CPU/GPU properly seeded before model construction.
   - SUMO process initialized with explicit `--seed` parameter matching the experiment seed for training, and `1_000_000 + seed` for evaluation.
4. **Checkpoint & Output Isolation**:
   - Directory hierarchy: `outputs/multiseed_ablation/<network>/route_<route>/<ablation>/seed_<seed>/[train|eval]`.
5. **Experiment Matrix Scope**:
   - Ablations: 4
   - Networks: 3 (`4x4`, `6x6_bidirectional`, `7x28`)
   - Routes: 2 (`ev_0`, `ev_1`)
   - Seeds: 10 (`101`, `202`, `303`, `404`, `505`, `606`, `707`, `808`, `909`, `1010`)
   - Total run pairs: $4 \times 3 \times 2 \times 10 = 240$ training + evaluation pairs (80 episodes, 3600 max steps per episode).

## Current Status
COMPLETED. All core components audited and confirmed intact.

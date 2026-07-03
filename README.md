# EV_BASELINE2

SUMO + TraCI project for emergency vehicle priority on a reusable `4x4` urban traffic grid.

## What This Project Does

- Runs rule-based emergency vehicle signal priority on a 4x4 traffic-light grid.
- Supports RL-based signal control with a DQN model.
- Uses route-aware EV handling, so the controller adapts automatically when EV routes change.
- Generates plots, CSV metrics, logs, and trained RL checkpoints under `outputs/`.

## Setup
1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Grid Scenario

The project now works primarily on the generated `4x4` grid scenario:

```bash
python scenario/grid/generate_grid.py
```

## Main Run Commands

Open SUMO GUI:

```bash
sumo-gui -c scenario/grid/grid.sumocfg
```

Run rule-based controller:

```bash
python src/main.py --sumocfg scenario/grid/grid.sumocfg
```

Run RL model:

```bash
python src/main.py --sumocfg scenario/grid/grid.sumocfg --mode rl_model --model-path outputs/models/dqn.pt
```

Train RL:

```bash
python src/train_rl.py --sumocfg scenario/grid/grid.sumocfg --episodes 200
```

Train congestion-aware coordinated PPO:

```bash
python src/train_rl.py --sumocfg scenario/grid/grid.sumocfg --controller-type congestion_aware_coordinated_ppo --episodes 200 --max-steps 3600
```

Run congestion-aware coordinated PPO:

```bash
python src/main.py --sumocfg scenario/grid/grid.sumocfg --mode congestion_aware_coordinated_ppo --congestion-aware-coordinated-ppo-model-path outputs/models/congestion_aware_coordinated_ppo.pt
```

Train multi-level coordinated PPO:

```bash
python src/train_rl.py --sumocfg scenario/grid/grid.sumocfg --controller-type multi_level_coordinated_ppo --episodes 200 --max-steps 3600
```

Run multi-level coordinated PPO:

```bash
python src/main.py --sumocfg scenario/grid/grid.sumocfg --mode multi_level_coordinated_ppo --multi-level-coordinated-ppo-model-path outputs/models/multi_level_coordinated_ppo.pt --ev-id ev_2
```

## Testing Different EV Routes

Run against the predefined emergency vehicles:

```bash
python src/main.py --sumocfg scenario/grid/grid.sumocfg --ev-id ev_0
python src/main.py --sumocfg scenario/grid/grid.sumocfg --ev-id ev_1
python src/main.py --sumocfg scenario/grid/grid.sumocfg --ev-id ev_2
```

You can also edit the EV route in `scenario/grid/grid_routes.rou.xml` and rerun without changing controller code.

## Important Notes

- Rule-based and RL runs both support `--max-steps` and `--post-ev-seconds`.
- RL mode now keeps the simulation alive briefly after the EV exits, just like the rule-based mode.
- The current controller is route-aware and works for horizontal, vertical, and turning EV paths on the grid.
- If you change the reward function or state representation, retrain the RL model before evaluating it.
- The congestion-aware coordinated PPO uses a larger state vector than the original coordinated PPO, so its checkpoint is separate from `outputs/models/coordinated_ppo.pt`.
- The multi-level coordinated PPO uses an expanded 29-dimensional observation that combines local, neighbor, and global traffic features, so its checkpoint is separate from all earlier PPO checkpoints.
- Its default checkpoint is `outputs/models/multi_level_coordinated_ppo.pt` and its training log is `outputs/logs/multi_level_coordinated_ppo_training.csv`.

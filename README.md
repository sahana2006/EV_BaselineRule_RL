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

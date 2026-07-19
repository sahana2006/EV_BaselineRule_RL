# Grid 6x6 Bidirectional Outputs

Use one output root per route to avoid overwriting evaluation or training artifacts:

- `outputs/grid6x6_bidirectional/rule_based_0/eval`
- `outputs/grid6x6_bidirectional/global_dqn_0/train`
- `outputs/grid6x6_bidirectional/global_dqn_0/eval`
- `outputs/grid6x6_bidirectional/coordinated_marl_0/train`
- `outputs/grid6x6_bidirectional/coordinated_marl_0/eval`
- `outputs/grid6x6_bidirectional/multi_level_coordinated_dqn_0/train`
- `outputs/grid6x6_bidirectional/multi_level_coordinated_dqn_0/eval`

Repeat the same structure for `_1` and `_2` with `--ev-id ev_1` and `--ev-id ev_2`.

Each `train` folder is expected to contain:

- `checkpoints/latest.pt`
- `checkpoints/episode_XXX.pt`
- `logs/`
- `plots/`

Each `eval` folder is expected to contain:

- `csv/`
- `plots/`
- `logs/`

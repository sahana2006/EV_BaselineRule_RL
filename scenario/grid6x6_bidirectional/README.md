# Grid 6x6 Bidirectional

This scenario mirrors the `grid_7x28` workflow, but with a 6x6 bidirectional grid and three emergency routes:

- `ev_0` on `ev_route_0` with 10-12 intersections
- `ev_1` on `ev_route_1` with 16-18 intersections
- `ev_2` on `ev_route_2` with 22-25 intersections

Generate the SUMO assets with:

```bash
python scenario/grid6x6_bidirectional/generate_grid6x6_bidirectional.py
```

The generator writes:

- `grid6x6_bidirectional.net.xml`
- `grid6x6_bidirectional_routes.rou.xml`
- `grid6x6_bidirectional.sumocfg`
- `grid6x6_bidirectional_view.settings.xml`

Use route-scoped output roots to keep results separate across EV routes, for example:

- `outputs/grid6x6_bidirectional/global_dqn_0/train`
- `outputs/grid6x6_bidirectional/global_dqn_0/eval`
- `outputs/grid6x6_bidirectional/global_dqn_1/train`
- `outputs/grid6x6_bidirectional/global_dqn_1/eval`
- `outputs/grid6x6_bidirectional/global_dqn_2/train`
- `outputs/grid6x6_bidirectional/global_dqn_2/eval`

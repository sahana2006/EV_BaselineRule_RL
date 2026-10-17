# Phase 2: Ablation Validation Document

## Objective
Verify that all four explicit feature ablation configurations build correct feature state vectors matching exact specification without runtime errors.

## Implementation & Verification Details
- **Configurations Tested**:
  1. `local`: `state_dim = 9` (only intersection local state)
  2. `local_neighborhood`: `state_dim = 15` (local + 1-hop neighbor features)
  3. `local_global`: `state_dim = 17` (local + global network features)
  4. `local_neighborhood_global`: `state_dim = 29` (local + neighbor + global features)

- **Verification Executed**:
  - `scripts/smoke_reproducibility.py` instantiated `TrafficEnv` for each ablation mode and verified that state vector lengths strictly match 9, 15, 17, and 29 respectively.
  - Checked `src/rl_env.py` helper `_state_from_context` to confirm feature ordering and dimensionality.

## Commands Executed
```bash
python scripts/smoke_reproducibility.py
```

## Validation Output
`[SMOKE_PASS] dimensions={'local': 9, 'local_neighborhood': 15, 'local_global': 17, 'local_neighborhood_global': 29}`

## Current Status
COMPLETED & PASSED.

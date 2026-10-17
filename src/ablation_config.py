"""Explicit observation configurations for the DQN ablation study."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str
    include_neighborhood: bool
    include_global: bool
    state_dim: int


ABLATIONS = {
    "local": AblationConfig("local", False, False, 9),
    "local_neighborhood": AblationConfig("local_neighborhood", True, False, 15),
    "local_global": AblationConfig("local_global", False, True, 17),
    "local_neighborhood_global": AblationConfig("local_neighborhood_global", True, True, 29),
}


def get_ablation(name: str | None) -> AblationConfig | None:
    if name is None:
        return None
    try:
        return ABLATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation {name!r}; choose from {', '.join(ABLATIONS)}") from exc

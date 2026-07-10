from __future__ import annotations

import os
from typing import List

import traci


DEBUG_SIGNAL_LOGS = os.environ.get("SUMO_SIGNAL_DEBUG", "0") == "1"


def infer_ev_green_phase(tl_id: str, route_lanes: List[str]) -> int:
    logic = traci.trafficlight.getAllProgramLogics(tl_id)
    if not logic:
        return traci.trafficlight.getPhase(tl_id)
    phases = logic[0].phases
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    if not phases or not controlled_lanes:
        return traci.trafficlight.getPhase(tl_id)

    lane_index = [i for i, lane in enumerate(controlled_lanes) if lane in set(route_lanes)]
    if not lane_index:
        return traci.trafficlight.getPhase(tl_id)

    for idx, phase in enumerate(phases):
        if any(i < len(phase.state) and phase.state[i] in ("G", "g") for i in lane_index):
            return idx
    return traci.trafficlight.getPhase(tl_id)


def stage1_saturation_reduction(tl_id: str, edge_id: str, distance: float) -> None:
    vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id) if edge_id else 0
    erl = 1
    if vehicle_count > 15:
        clrs = 4
    elif vehicle_count > 10:
        clrs = 3
    elif vehicle_count > 5:
        clrs = 2
    else:
        clrs = 1

    if distance <= 50:
        tul = 3
    elif distance <= 150:
        tul = 2
    else:
        tul = 1

    drrs = (erl**0.1031) * (clrs**0.6053) * (tul**0.2915)
    if drrs > 3:
        extension = 10
    elif drrs > 2:
        extension = 7
    elif drrs > 1:
        extension = 5
    else:
        extension = 3

    current_duration = traci.trafficlight.getPhaseDuration(tl_id)
    traci.trafficlight.setPhaseDuration(tl_id, current_duration + extension)
    if DEBUG_SIGNAL_LOGS:
        print(f"[{tl_id}] Stage=SATURATION DRRS={drrs:.3f} CLRS={clrs} TUL={tul} Extension={extension}s")


def stage2_non_intrusive(tl_id: str, ev_phase: int, distance: float) -> None:
    current_phase = traci.trafficlight.getPhase(tl_id)
    current_duration = traci.trafficlight.getPhaseDuration(tl_id)
    if current_phase != ev_phase:
        traci.trafficlight.setPhaseDuration(tl_id, max(2, current_duration - 2))
        if DEBUG_SIGNAL_LOGS:
            print(f"[{tl_id}] Stage=NON_INTRUSIVE prepare_ev_phase distance={distance:.1f}")
        return
    traci.trafficlight.setPhaseDuration(tl_id, current_duration + 3)
    if DEBUG_SIGNAL_LOGS:
        print(f"[{tl_id}] Stage=NON_INTRUSIVE extend_green distance={distance:.1f}")


def stage2_intrusive(tl_id: str, ev_phase: int, distance: float) -> None:
    traci.trafficlight.setPhase(tl_id, ev_phase)
    traci.trafficlight.setPhaseDuration(tl_id, 5)
    if DEBUG_SIGNAL_LOGS:
        print(f"[{tl_id}] Stage=INTRUSIVE FORCE GREEN distance={distance:.1f}")


def stage3_recovery(tl_id: str, steps_left: int, default_program: str) -> int:
    if steps_left <= 0:
        steps_left = 6
    steps_left -= 1
    if steps_left <= 0:
        traci.trafficlight.setProgram(tl_id, default_program)
        if DEBUG_SIGNAL_LOGS:
            print(f"[{tl_id}] Stage=RECOVERY restored default cycle")
        return 0

    phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
    traci.trafficlight.setPhaseDuration(tl_id, max(3, phase_duration - 1))
    if DEBUG_SIGNAL_LOGS:
        print(f"[{tl_id}] Stage=RECOVERY gradual step={steps_left}")
    return steps_left


def apply_green_wave(tl_id: str, ev_phase: int, time_to_arrival: float, arrival: float) -> None:
    traci.trafficlight.setPhase(tl_id, ev_phase)
    traci.trafficlight.setPhaseDuration(tl_id, max(3, int(time_to_arrival)))
    if DEBUG_SIGNAL_LOGS:
        print(f"[{tl_id}] GreenWave arrival={arrival:.1f} prepare_green")


# Discrete RL action space (global controller applies to nearest upcoming TL).
ACTION_KEEP_PHASE = 0
ACTION_SWITCH_EV_GREEN = 1
ACTION_EXTEND_GREEN = 2


def apply_discrete_rl_action(tl_id: str, action: int, ev_phase: int, distance: float) -> None:
    """Map discrete RL actions to existing signal logic."""
    if action == ACTION_KEEP_PHASE:
        return
    if action == ACTION_SWITCH_EV_GREEN:
        stage2_intrusive(tl_id, ev_phase, distance)
        return
    if action == ACTION_EXTEND_GREEN:
        stage2_non_intrusive(tl_id, ev_phase, distance)
        return

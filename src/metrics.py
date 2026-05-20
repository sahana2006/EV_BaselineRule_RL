from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import traci

from .config import STOP_SPEED_THRESHOLD


@dataclass
class TimeSeriesMetrics:
    time_s: List[float] = field(default_factory=list)
    ev_speed: List[float] = field(default_factory=list)
    ev_distance_to_goal: List[float] = field(default_factory=list)
    ev_position_xy: List[Tuple[float, float]] = field(default_factory=list)
    ev_waiting_time: List[float] = field(default_factory=list)
    avg_waiting_time: List[float] = field(default_factory=list)
    queue_length: List[int] = field(default_factory=list)
    total_waiting_time: List[float] = field(default_factory=list)
    throughput: List[int] = field(default_factory=list)
    ev_stops: int = 0
    _ev_prev_stopped: bool = False

    def capture(self, ev_id: str) -> None:
        t = traci.simulation.getTime()
        speed = traci.vehicle.getSpeed(ev_id)
        pos = traci.vehicle.getPosition(ev_id)
        dist_goal = traci.vehicle.getDrivingDistance(ev_id, traci.vehicle.getRoute(ev_id)[-1], 0.0)
        if dist_goal < 0:
            dist_goal = 0.0
        ev_wait = traci.vehicle.getWaitingTime(ev_id)

        veh_ids = traci.vehicle.getIDList()
        waits = [traci.vehicle.getWaitingTime(v_id) for v_id in veh_ids]
        total_wait = float(sum(waits))
        avg_wait = total_wait / len(waits) if waits else 0.0
        queue = sum(1 for v_id in veh_ids if traci.vehicle.getSpeed(v_id) < STOP_SPEED_THRESHOLD)
        arrived = traci.simulation.getArrivedNumber()

        ev_stopped = speed < STOP_SPEED_THRESHOLD
        if ev_stopped and not self._ev_prev_stopped:
            self.ev_stops += 1
        self._ev_prev_stopped = ev_stopped

        self.time_s.append(t)
        self.ev_speed.append(speed)
        self.ev_distance_to_goal.append(dist_goal)
        self.ev_position_xy.append(pos)
        self.ev_waiting_time.append(ev_wait)
        self.avg_waiting_time.append(avg_wait)
        self.queue_length.append(queue)
        self.total_waiting_time.append(total_wait)
        self.throughput.append(arrived)

    def save_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "ev_speed_mps",
                    "ev_pos_x_m",
                    "ev_pos_y_m",
                    "ev_distance_to_goal_m",
                    "ev_waiting_time_s",
                    "avg_waiting_time_s",
                    "total_waiting_time_s",
                    "queue_length_veh",
                    "throughput_veh_step",
                ]
            )
            for i in range(len(self.time_s)):
                x, y = self.ev_position_xy[i]
                writer.writerow(
                    [
                        float(self.time_s[i]),
                        float(self.ev_speed[i]),
                        float(x),
                        float(y),
                        float(self.ev_distance_to_goal[i]),
                        float(self.ev_waiting_time[i]),
                        float(self.avg_waiting_time[i]),
                        float(self.total_waiting_time[i]),
                        int(self.queue_length[i]),
                        int(self.throughput[i]),
                    ]
                )

    def final_summary(self) -> Dict[str, float]:
        travel_time = self.time_s[-1] - self.time_s[0] if len(self.time_s) >= 2 else 0.0
        avg_wait = self.avg_waiting_time[-1] if self.avg_waiting_time else 0.0
        return {"ev_travel_time": travel_time, "avg_waiting_time": avg_wait, "ev_stops": float(self.ev_stops)}

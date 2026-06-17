from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationConfig:
    sumo_config: str = "scenario/simulation.sumocfg"
    use_gui: bool = True
    ev_id: str = "ev_0"
    max_steps: int = 3600
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("outputs/logs")
    plot_dir: Path = Path("outputs/plots")
    csv_dir: Path = Path("outputs/csv")
    csv_name: str = "full_model_metrics.csv"
    fixed_csv_name: str = "fixed_time_metrics.csv"
    intrusive_csv_name: str = "intrusive_only_metrics.csv"
    rl_model_csv_name: str = "rl_model_metrics.csv"
    global_ppo_model_csv_name: str = "global_ppo_model_metrics.csv"
    coordinated_ppo_model_csv_name: str = "coordinated_ppo_model_metrics.csv"
    coord_dueling_dqn_model_csv_name: str = "coord_dueling_dqn_model_metrics.csv"
    step_length: float = 1.0
    post_ev_buffer_seconds: int = 90
    ev_color_rgba: tuple[int, int, int, int] = (255, 0, 0, 255)
    normal_vehicle_color_rgba: tuple[int, int, int, int] = (0, 90, 255, 255)


SATURATION_DISTANCE_THRESHOLD = 150.0
INTRUSIVE_DISTANCE_THRESHOLD = 50.0
STOP_SPEED_THRESHOLD = 0.1
LARGE_ARRIVAL_TIME = 1e9

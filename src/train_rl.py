from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Run as: python src/train_rl.py (from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.dqn_agent import DQNAgent
from src.rl_env import (
    ACTION_DIM,
    CONTROLLER_COORDINATED_MARL,
    CONTROLLER_INDEPENDENT_MARL,
    CONTROLLER_SINGLE_AGENT,
    TrafficEnv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN traffic signal controller (SUMO + TraCI).")
    p.add_argument("--sumocfg", default="scenario/simulation.sumocfg")
    p.add_argument("--ev-id", default="ev_0")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=3600, help="Max steps per episode")
    p.add_argument("--learn-every", type=int, default=1, help="Call agent.learn() every N steps")
    p.add_argument("--model-out", default=str(PROJECT_ROOT / "outputs" / "models" / "dqn.pt"))
    p.add_argument(
        "--reward-log",
        default=str(PROJECT_ROOT / "outputs" / "logs" / "rl_training_rewards.csv"),
        help="Append episode total reward",
    )
    p.add_argument(
        "--controller-type",
        choices=[CONTROLLER_SINGLE_AGENT, CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL, "multi_agent"],
        default=CONTROLLER_COORDINATED_MARL,
        help="Train single-agent RL, independent MARL, or coordinated MARL",
    )
    p.add_argument(
        "--traffic-scale",
        type=float,
        default=1.0,
        help="SUMO demand scaling for moderate/heavy traffic experiments",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.controller_type == "multi_agent":
        args.controller_type = CONTROLLER_INDEPENDENT_MARL
    cfg = SimulationConfig(
        sumo_config=args.sumocfg,
        use_gui=False,
        ev_id=args.ev_id,
        max_steps=args.max_steps,
        output_dir=PROJECT_ROOT / "outputs",
        log_dir=PROJECT_ROOT / "outputs" / "logs",
        plot_dir=PROJECT_ROOT / "outputs" / "plots",
        csv_dir=PROJECT_ROOT / "outputs" / "csv",
    )
    env = TrafficEnv(
        cfg,
        headless=True,
        max_episode_steps=args.max_steps,
        controller_type=args.controller_type,
        traffic_scale=args.traffic_scale,
    )
    agent = DQNAgent(state_dim=env.state_dim, action_dim=ACTION_DIM)

    log_path = Path(args.reward_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["episode", "controller_type", "traffic_scale", "total_reward", "steps", "epsilon", "loss_last"])

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        last_loss: float | None = None
        while not done:
            if args.controller_type in {CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL}:
                if not isinstance(state, dict):
                    raise TypeError("Multi-agent training expected a dictionary of states.")
                actions = agent.choose_actions(state, greedy=False)
                next_state, rewards, done, _ = env.step(actions)
                if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                    raise TypeError("Multi-agent environment returned unexpected types.")
                agent.store_multi_agent_transition(state, actions, rewards, next_state, done)
                if steps % args.learn_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        last_loss = loss
                state = next_state
                total_r += float(sum(rewards.values()))
            else:
                if isinstance(state, dict):
                    raise TypeError("Single-agent training expected a single state vector.")
                action = agent.choose_action(state, greedy=False)
                next_state, reward, done, _ = env.step(action)
                if isinstance(next_state, dict) or isinstance(reward, dict):
                    raise TypeError("Single-agent environment returned unexpected types.")
                agent.store_transition(state, action, reward, next_state, done)
                if steps % args.learn_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        last_loss = loss
                state = next_state
                total_r += reward
            steps += 1

        agent.decay_epsilon()
        agent.save(args.model_out)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    ep + 1,
                    args.controller_type,
                    f"{args.traffic_scale:.2f}",
                    f"{total_r:.4f}",
                    steps,
                    f"{agent.epsilon:.4f}",
                    "" if last_loss is None else f"{last_loss:.6f}",
                ]
            )
        print(
            f"episode {ep + 1}/{args.episodes} controller={args.controller_type} traffic_scale={args.traffic_scale:.2f} "
            f"reward={total_r:.2f} steps={steps} eps={agent.epsilon:.3f} saved={args.model_out}"
        )

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set.")
    main()

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import traci

# Run as: python src/train_rl.py (from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.coordinated_ppo_agent import CoordinatedPPOAgent
from src.coord_dueling_dqn_agent import CoordinatedDuelingDQNAgent
from src.dqn_agent import DQNAgent
from src.global_ppo_agent import GlobalPPOAgent
from src.rl_env import (
    ACTION_DIM,
    CONTROLLER_COORDINATED_PPO,
    CONTROLLER_GLOBAL_PPO,
    CONTROLLER_COORDINATED_MARL,
    CONTROLLER_INDEPENDENT_MARL,
    CONTROLLER_SINGLE_AGENT,
    TrafficEnv,
)

CONTROLLER_COORDINATED_DUELING_DQN = "coordinated_dueling_dqn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN traffic signal controller (SUMO + TraCI).")
    p.add_argument("--sumocfg", default="scenario/simulation.sumocfg")
    p.add_argument("--ev-id", default="ev_0")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=3600, help="Max steps per episode")
    p.add_argument("--learn-every", type=int, default=1, help="Call agent.learn() every N steps")
    p.add_argument("--model-out", default=None)
    p.add_argument(
        "--reward-log",
        default=None,
        help="Append episode total reward",
    )
    p.add_argument(
        "--controller-type",
        choices=[
            CONTROLLER_SINGLE_AGENT,
            CONTROLLER_COORDINATED_PPO,
            CONTROLLER_GLOBAL_PPO,
            CONTROLLER_INDEPENDENT_MARL,
            CONTROLLER_COORDINATED_MARL,
            CONTROLLER_COORDINATED_DUELING_DQN,
            "multi_agent",
        ],
        default=CONTROLLER_COORDINATED_MARL,
        help="Train single-agent RL, independent MARL, coordinated MARL, coordinated PPO, or global PPO",
    )
    p.add_argument(
        "--traffic-scale",
        type=float,
        default=1.0,
        help="SUMO demand scaling for moderate/heavy traffic experiments",
    )
    return p.parse_args()


def _default_paths(args: argparse.Namespace) -> None:
    if args.controller_type == "multi_agent":
        args.controller_type = CONTROLLER_INDEPENDENT_MARL

    if args.controller_type == CONTROLLER_COORDINATED_PPO:
        if args.model_out is None:
            args.model_out = str(PROJECT_ROOT / "outputs" / "models" / "coordinated_ppo.pt")
        if args.reward_log is None:
            args.reward_log = str(PROJECT_ROOT / "outputs" / "logs" / "coordinated_ppo_training.csv")
    if args.controller_type == CONTROLLER_GLOBAL_PPO:
        if args.model_out is None:
            args.model_out = str(PROJECT_ROOT / "outputs" / "models" / "global_ppo.pt")
        if args.reward_log is None:
            args.reward_log = str(PROJECT_ROOT / "outputs" / "logs" / "global_ppo_training.csv")
    elif args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
        if args.model_out is None:
            args.model_out = str(PROJECT_ROOT / "outputs" / "models" / "coord_dueling_dqn.pt")
        if args.reward_log is None:
            args.reward_log = str(PROJECT_ROOT / "outputs" / "logs" / "coord_dueling_dqn_training.csv")
    else:
        if args.model_out is None:
            args.model_out = str(PROJECT_ROOT / "outputs" / "models" / "dqn.pt")
        if args.reward_log is None:
            args.reward_log = str(PROJECT_ROOT / "outputs" / "logs" / "rl_training_rewards.csv")


def main() -> None:
    args = parse_args()
    _default_paths(args)

    env_controller_type = args.controller_type
    if args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
        env_controller_type = CONTROLLER_COORDINATED_MARL
    elif args.controller_type == CONTROLLER_COORDINATED_PPO:
        env_controller_type = CONTROLLER_COORDINATED_PPO

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
        controller_type=env_controller_type,
        traffic_scale=args.traffic_scale,
    )

    try:
        if args.controller_type == CONTROLLER_COORDINATED_PPO:
            agent: DQNAgent | CoordinatedDuelingDQNAgent | GlobalPPOAgent | CoordinatedPPOAgent = CoordinatedPPOAgent(
                state_dim=env.state_dim,
                action_dim=ACTION_DIM,
                learning_rate=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                entropy_coef=0.001,
                value_coef=0.5,
                ppo_epochs=4,
                batch_size=256,
            )
        elif args.controller_type == CONTROLLER_GLOBAL_PPO:
            agent: DQNAgent | CoordinatedDuelingDQNAgent | GlobalPPOAgent = GlobalPPOAgent(
                state_dim=env.state_dim,
                action_dim=ACTION_DIM,
                learning_rate=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                clip_eps=0.2,
                entropy_coef=0.001,
                value_coef=0.5,
                ppo_epochs=4,
                batch_size=256,
            )
            print(
                "[GLOBAL_PPO_INIT]\n"
                f"state_dim={env.state_dim}\n"
                f"action_dim={env.action_dim}\n"
                "lr=1e-3\n"
                "entropy_coef=0.001\n"
                "reward_normalization=True"
            )
        elif args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
            agent = CoordinatedDuelingDQNAgent(
                state_dim=env.state_dim,
                action_dim=ACTION_DIM,
            )
        else:
            agent = DQNAgent(state_dim=env.state_dim, action_dim=ACTION_DIM)

        log_path = Path(args.reward_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                if args.controller_type == CONTROLLER_COORDINATED_PPO:
                    writer.writerow(["episode", "reward", "policy_loss", "value_loss", "entropy", "runtime_seconds"])
                elif args.controller_type == CONTROLLER_GLOBAL_PPO:
                    writer.writerow(["episode", "reward", "policy_loss", "value_loss", "entropy", "runtime_seconds"])
                elif args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
                    writer.writerow(["episode", "reward", "loss", "epsilon", "avg_td_error", "runtime_seconds"])
                else:
                    writer.writerow(["episode", "controller_type", "traffic_scale", "total_reward", "steps", "epsilon", "loss_last"])

        for ep in range(args.episodes):
            episode_start = time.perf_counter()
            state = env.reset()
            agent_ids = env.get_agent_ids() or (sorted(state.keys()) if isinstance(state, dict) else [])
            if args.controller_type == CONTROLLER_COORDINATED_PPO and not isinstance(state, dict):
                raise TypeError("Coordinated PPO training expected a dictionary of states.")
            if args.controller_type == CONTROLLER_GLOBAL_PPO and isinstance(state, dict):
                raise TypeError("Global PPO training expected a single state vector.")
            if args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN and not isinstance(state, dict):
                raise TypeError("Coordinated dueling DQN training expected a multi-agent state dictionary.")
            if ep == 0 and args.controller_type == CONTROLLER_COORDINATED_PPO:
                print(
                    "[COORDINATED_PPO_INIT]\n"
                    f"agents={len(agent_ids)}\n"
                    "shared_policy=True\n"
                    f"state_dim={env.state_dim}\n"
                    f"action_dim={env.action_dim}\n"
                    "coordination=True"
                )
            if ep == 0 and args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
                print(
                    "[COORD_DUELING_DQN_INIT]\n"
                    f"agents={len(env.get_agent_ids())}\n"
                    "shared_policy=True\n"
                    "double_dqn=True\n"
                    "dueling_network=True\n"
                    "prioritized_replay=True\n"
                    f"state_dim={env.state_dim}\n"
                    f"action_dim={env.action_dim}"
                )

            done = False
            total_r = 0.0
            steps = 0
            last_loss: float | None = None
            last_td_error: float | None = None
            last_value_loss: float = 0.0
            last_entropy: float = 0.0
            per_agent_states: dict[str, list[np.ndarray]] = {agent_id: [] for agent_id in agent_ids}
            per_agent_actions: dict[str, list[int]] = {agent_id: [] for agent_id in agent_ids}
            per_agent_log_probs: dict[str, list[float]] = {agent_id: [] for agent_id in agent_ids}
            per_agent_values: dict[str, list[float]] = {agent_id: [] for agent_id in agent_ids}
            per_agent_rewards: dict[str, list[float]] = {agent_id: [] for agent_id in agent_ids}
            per_agent_dones: dict[str, list[bool]] = {agent_id: [] for agent_id in agent_ids}
            episode_states: list[np.ndarray] = []
            episode_actions: list[int] = []
            episode_log_probs: list[float] = []
            episode_values: list[float] = []
            episode_rewards: list[float] = []
            episode_dones: list[bool] = []
            episode_reward_components: dict[str, float] = {
                "ev_reward": 0.0,
                "queue_penalty": 0.0,
                "congestion_penalty": 0.0,
                "throughput": 0.0,
            }

            while not done:
                if args.controller_type == CONTROLLER_COORDINATED_PPO:
                    if not isinstance(state, dict):
                        raise TypeError("Coordinated PPO training expected a dictionary of states.")
                    actions_map, log_prob_map, value_map, _entropy_map = agent.act(state, deterministic=False)
                    next_state, rewards, done, info = env.step(actions_map)
                    if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                        raise TypeError("Coordinated PPO environment returned unexpected single-agent outputs.")
                    total_r += float(sum(rewards.values()))
                    reward_breakdowns = info.get("reward_breakdowns", {}) if isinstance(info, dict) else {}
                    for agent_id in agent_ids:
                        if (
                            agent_id not in state
                            or agent_id not in actions_map
                            or agent_id not in log_prob_map
                            or agent_id not in value_map
                            or agent_id not in rewards
                        ):
                            continue
                        per_agent_states[agent_id].append(np.asarray(state[agent_id], dtype=np.float32))
                        per_agent_actions[agent_id].append(int(actions_map[agent_id]))
                        per_agent_log_probs[agent_id].append(float(log_prob_map[agent_id]))
                        per_agent_values[agent_id].append(float(value_map[agent_id]))
                        per_agent_rewards[agent_id].append(float(rewards[agent_id]) / 100.0)
                        per_agent_dones[agent_id].append(bool(done))
                        breakdown = reward_breakdowns.get(agent_id, {}) if isinstance(reward_breakdowns, dict) else {}
                        episode_reward_components["ev_reward"] += float(
                            breakdown.get("ev_delay_penalty", 0.0)
                            + breakdown.get("ev_stop_penalty", 0.0)
                            + breakdown.get("low_speed_penalty", 0.0)
                            + breakdown.get("intersection_clear_reward", 0.0)
                        )
                        episode_reward_components["queue_penalty"] += float(
                            breakdown.get("queue_penalty", 0.0) + breakdown.get("queue_growth_penalty", 0.0)
                        )
                        episode_reward_components["congestion_penalty"] += float(
                            breakdown.get("neighbor_congestion_penalty", 0.0)
                            + breakdown.get("network_congestion_penalty", 0.0)
                            + breakdown.get("downstream_blockage_penalty", 0.0)
                            + breakdown.get("anti_gridlock_penalty", 0.0)
                        )
                        episode_reward_components["throughput"] += float(breakdown.get("throughput_reward", 0.0))
                    state = next_state
                elif args.controller_type == CONTROLLER_GLOBAL_PPO:
                    if isinstance(state, dict):
                        raise TypeError("Global PPO training expected a single state vector.")
                    action, log_prob, value, _entropy = agent.act(state, deterministic=False)
                    next_state, reward, done, _ = env.step(action)
                    if isinstance(next_state, dict) or isinstance(reward, dict):
                        raise TypeError("Global PPO environment returned unexpected multi-agent outputs.")
                    episode_states.append(np.asarray(state, dtype=np.float32))
                    episode_actions.append(action)
                    episode_log_probs.append(log_prob)
                    episode_values.append(value)
                    episode_rewards.append(float(reward) / 100.0)
                    episode_dones.append(bool(done))
                    state = next_state
                    total_r += float(reward)
                elif args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
                    if not isinstance(state, dict):
                        raise TypeError("Coordinated dueling DQN training expected a dictionary of states.")
                    actions = agent.choose_actions(state, greedy=False)
                    next_state, rewards, done, _ = env.step(actions)
                    if not isinstance(next_state, dict) or not isinstance(rewards, dict):
                        raise TypeError("Coordinated dueling DQN environment returned unexpected types.")
                    agent.store_multi_agent_transition(state, actions, rewards, next_state, done)
                    if steps % args.learn_every == 0:
                        learn_result = agent.learn()
                        if learn_result is not None:
                            last_loss, last_td_error = learn_result
                    state = next_state
                    total_r += float(sum(rewards.values()))
                elif args.controller_type in {CONTROLLER_INDEPENDENT_MARL, CONTROLLER_COORDINATED_MARL}:
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
                    total_r += float(reward)
                steps += 1

            if args.controller_type == CONTROLLER_COORDINATED_PPO:
                flat_states: list[np.ndarray] = []
                flat_actions: list[int] = []
                flat_log_probs: list[float] = []
                flat_values: list[float] = []
                flat_advantages: list[float] = []
                flat_returns: list[float] = []
                for agent_id in agent_ids:
                    advantages, returns = CoordinatedPPOAgent.compute_gae(
                        per_agent_rewards[agent_id],
                        per_agent_dones[agent_id],
                        per_agent_values[agent_id],
                        next_value=0.0,
                        gamma=agent.hyperparams.gamma,
                        gae_lambda=agent.hyperparams.gae_lambda,
                    )
                    flat_states.extend(per_agent_states[agent_id])
                    flat_actions.extend(per_agent_actions[agent_id])
                    flat_log_probs.extend(per_agent_log_probs[agent_id])
                    flat_values.extend(per_agent_values[agent_id])
                    flat_advantages.extend(advantages.tolist())
                    flat_returns.extend(returns.tolist())

                policy_loss, value_loss, entropy, _diagnostics = agent.update_from_batch(
                    flat_states,
                    flat_actions,
                    flat_log_probs,
                    flat_values,
                    flat_advantages,
                    flat_returns,
                )
                last_loss = policy_loss
                last_value_loss = value_loss
                last_entropy = entropy
                print(
                    f"[EPISODE_REWARD] episode={ep + 1} "
                    f"ev_reward={episode_reward_components['ev_reward']:.3f} "
                    f"queue_penalty={episode_reward_components['queue_penalty']:.3f} "
                    f"congestion_penalty={episode_reward_components['congestion_penalty']:.3f} "
                    f"throughput={episode_reward_components['throughput']:.3f}"
                )
                agent.save(args.model_out)
            elif args.controller_type == CONTROLLER_GLOBAL_PPO:
                ev_present = args.ev_id in traci.vehicle.getIDList()
                next_value = 0.0
                if ev_present and not isinstance(state, dict):
                    next_value = agent.value(state)
                advantages, returns = GlobalPPOAgent.compute_gae(
                    episode_rewards,
                    episode_dones,
                    episode_values,
                    next_value=next_value,
                    gamma=agent.hyperparams.gamma,
                    gae_lambda=agent.hyperparams.gae_lambda,
                )
                policy_loss, value_loss, entropy, _diagnostics = agent.update_from_batch(
                    episode_states,
                    episode_actions,
                    episode_log_probs,
                    episode_values,
                    advantages,
                    returns,
                )
                last_loss = policy_loss
                last_value_loss = value_loss
                last_entropy = entropy
                agent.save(args.model_out)
            else:
                agent.decay_epsilon()
                agent.save(args.model_out)

            runtime_seconds = time.perf_counter() - episode_start
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if args.controller_type in {CONTROLLER_COORDINATED_PPO, CONTROLLER_GLOBAL_PPO}:
                    writer.writerow(
                        [
                            ep + 1,
                            f"{total_r:.4f}",
                            f"{last_loss:.6f}",
                            f"{last_value_loss:.6f}",
                            f"{last_entropy:.6f}",
                            f"{runtime_seconds:.4f}",
                        ]
                    )
                    print(
                        f"episode {ep + 1}/{args.episodes} controller={args.controller_type} traffic_scale={args.traffic_scale:.2f} "
                        f"reward={total_r:.2f} policy_loss={last_loss:.6f} value_loss={last_value_loss:.6f} "
                        f"entropy={last_entropy:.6f} runtime_seconds={runtime_seconds:.2f} saved={args.model_out}"
                    )
                elif args.controller_type == CONTROLLER_COORDINATED_DUELING_DQN:
                    loss_display = 0.0 if last_loss is None else last_loss
                    td_error_display = 0.0 if last_td_error is None else last_td_error
                    writer.writerow(
                        [
                            ep + 1,
                            f"{total_r:.4f}",
                            "" if last_loss is None else f"{last_loss:.6f}",
                            f"{agent.epsilon:.4f}",
                            "" if last_td_error is None else f"{last_td_error:.6f}",
                            f"{runtime_seconds:.4f}",
                        ]
                    )
                    print(
                        f"episode {ep + 1}/{args.episodes} reward={total_r:.2f} loss={loss_display:.6f} "
                        f"eps={agent.epsilon:.3f} avg_td_error={td_error_display:.6f} "
                        f"runtime_seconds={runtime_seconds:.2f} saved={args.model_out}"
                    )
                else:
                    writer.writerow(
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
    finally:
        env.close()

    print("Training finished.")


if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set.")
    main()

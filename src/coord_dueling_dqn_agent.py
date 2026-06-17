from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


@dataclass
class PrioritizedReplayConfig:
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_frames: int = 100_000
    priority_epsilon: float = 1e-5


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        priority_epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = max(1, beta_frames)
        self.priority_epsilon = priority_epsilon
        self.storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.frame = 1

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        priority: float | None = None,
    ) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition

        if priority is None:
            if len(self.storage) == 1:
                priority = 1.0
            else:
                priority = float(self.priorities[: len(self.storage)].max())
                if priority <= 0.0:
                    priority = 1.0
        self.priorities[self.position] = float(priority)
        self.position = (self.position + 1) % self.capacity

    def _beta(self) -> float:
        fraction = min(1.0, self.frame / float(self.beta_frames))
        return float(self.beta_start + fraction * (self.beta_end - self.beta_start))

    def sample(self, batch_size: int) -> tuple[Tuple[np.ndarray, int, float, np.ndarray, bool], np.ndarray, np.ndarray, float]:
        if len(self.storage) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priorities = self.priorities[: len(self.storage)]
        scaled = priorities ** self.alpha
        scaled_sum = float(scaled.sum())
        if scaled_sum <= 0.0:
            scaled = np.ones_like(scaled, dtype=np.float32)
            scaled_sum = float(scaled.sum())
        probs = scaled / scaled_sum

        replace = len(self.storage) < batch_size
        indices = np.random.choice(len(self.storage), size=batch_size, replace=replace, p=probs)
        samples = [self.storage[idx] for idx in indices]
        sampled_probs = probs[indices]

        beta = self._beta()
        self.frame += 1
        weights = (len(self.storage) * sampled_probs) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        return tuple(samples), indices, weights.astype(np.float32), beta

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        updated = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.priority_epsilon
        updated = np.power(updated, self.alpha)
        for idx, priority in zip(indices, updated):
            self.priorities[int(idx)] = float(priority)


@dataclass
class CoordDuelingDQNHyperParams:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    memory_size: int = 50_000
    batch_size: int = 64
    target_update_every: int = 200
    hidden: int = 128
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_frames: int = 100_000
    priority_epsilon: float = 1e-5


class CoordinatedDuelingDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        memory_size: int = 50_000,
        batch_size: int = 64,
        target_update_every: int = 200,
        hidden: int = 128,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        priority_epsilon: float = 1e-5,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparams = CoordDuelingDQNHyperParams(
            gamma=gamma,
            learning_rate=lr,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update_every=target_update_every,
            hidden=hidden,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            priority_epsilon=priority_epsilon,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(
            memory_size,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            priority_epsilon=priority_epsilon,
        )
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.learn_steps = 0

    def _q_values(self, states: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        net = self.target_net if use_target else self.policy_net
        return net(states)

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self._q_values(state_t, use_target=False)
            return int(q_values.argmax(dim=1).item())

    def choose_actions(self, states: Dict[str, np.ndarray], greedy: bool = False) -> Dict[str, int]:
        if not states:
            return {}
        if not greedy and random.random() < self.epsilon:
            return {agent_id: random.randrange(self.action_dim) for agent_id in states}

        agent_ids = list(states.keys())
        batch = np.stack([states[agent_id] for agent_id in agent_ids])
        with torch.no_grad():
            state_t = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            q_values = self._q_values(state_t, use_target=False)
            best_actions = q_values.argmax(dim=1).tolist()
        return {agent_id: int(best_actions[idx]) for idx, agent_id in enumerate(agent_ids)}

    def predict(self, state: np.ndarray) -> int:
        return self.choose_action(state, greedy=True)

    def predict_actions(self, states: Dict[str, np.ndarray]) -> Dict[str, int]:
        return self.choose_actions(states, greedy=True)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def store_multi_agent_transition(
        self,
        states: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_states: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        for agent_id, state in states.items():
            if agent_id not in actions or agent_id not in rewards or agent_id not in next_states:
                continue
            self.replay_buffer.add(state, int(actions[agent_id]), float(rewards[agent_id]), next_states[agent_id], done)

    def learn(self) -> tuple[float, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, indices, weights, _beta = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + (1.0 - dones_t) * self.hyperparams.gamma * next_q_values

        td_errors = targets - q_values
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        avg_td_error = float(td_errors.abs().mean().item())
        return float(loss.item()), avg_td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hyperparams": self.hyperparams.__dict__,
                "shared_policy": True,
                "double_dqn": True,
                "dueling_network": True,
                "prioritized_replay": True,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt.get("target", ckpt["policy"]))
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon_min))
        self.learn_steps = int(ckpt.get("learn_steps", 0))

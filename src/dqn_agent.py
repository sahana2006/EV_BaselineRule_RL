from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """A single shared DQN that can be reused by one or many traffic-light agents."""

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
        tau: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.tau = tau  # 1.0 = hard update; <1 enables soft update (unused for now)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=memory_size)
        self.learn_steps = 0

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(t)
            return int(q.argmax(dim=1).item())

    def predict(self, state: np.ndarray) -> int:
        """Greedy policy for evaluation."""
        return self.choose_action(state, greedy=True)

    def choose_actions(self, states: Dict[str, np.ndarray], greedy: bool = False) -> Dict[str, int]:
        """
        Shared-policy multi-agent action selection.

        Each intersection contributes its own local state, but all of them reuse
        the same policy network weights.
        """
        if not states:
            return {}
        if not greedy and random.random() < self.epsilon:
            return {agent_id: random.randrange(self.action_dim) for agent_id in states}

        agent_ids = list(states.keys())
        batch = np.stack([states[agent_id] for agent_id in agent_ids])
        with torch.no_grad():
            t = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            q = self.policy_net(t)
            best_actions = q.argmax(dim=1).tolist()
        return {agent_id: int(best_actions[idx]) for idx, agent_id in enumerate(agent_ids)}

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def store_transitions(
        self,
        transitions: Iterable[Tuple[np.ndarray, int, float, np.ndarray, bool]],
    ) -> None:
        for transition in transitions:
            self.memory.append(transition)

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
            self.memory.append((state, int(actions[agent_id]), float(rewards[agent_id]), next_states[agent_id], done))

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_actions = self.policy_net(s2).argmax(dim=1, keepdim=True)
            q_next = self.target_net(s2).gather(1, next_actions)
            target = r + (1.0 - d) * self.gamma * q_next

        loss = nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

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
                "shared_policy": True,
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

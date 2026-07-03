from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class CoordinatedPPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CoordinatedPPOCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class CoordinatedPPOHyperParams:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 256


class CoordinatedPPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.001,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 256,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparams = CoordinatedPPOHyperParams(
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = CoordinatedPPOActor(state_dim, action_dim).to(self.device)
        self.critic = CoordinatedPPOCritic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def _distribution(self, states: torch.Tensor) -> Categorical:
        logits = self.actor(states)
        return Categorical(logits=logits)

    def act(
        self,
        states: Mapping[str, np.ndarray],
        *,
        deterministic: bool = False,
        include_probs: bool = False,
    ) -> tuple[Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, float]] | tuple[
        Dict[str, int],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ]:
        if not states:
            empty: Dict[str, float] = {}
            if include_probs:
                return {}, {}, {}, {}, empty, empty
            return {}, {}, {}, {}

        agent_ids = sorted(states.keys())
        batch = np.stack([states[agent_id] for agent_id in agent_ids])
        state_t = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self._distribution(state_t)
            values = self.critic(state_t)
            if deterministic:
                actions = torch.argmax(dist.probs, dim=-1)
            else:
                actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            probs = dist.probs
            selected_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            max_probs, _ = probs.max(dim=-1)

        action_map = {agent_id: int(actions[idx].item()) for idx, agent_id in enumerate(agent_ids)}
        log_prob_map = {agent_id: float(log_probs[idx].item()) for idx, agent_id in enumerate(agent_ids)}
        value_map = {agent_id: float(values[idx].item()) for idx, agent_id in enumerate(agent_ids)}
        entropy_map = {agent_id: float(entropies[idx].item()) for idx, agent_id in enumerate(agent_ids)}
        if include_probs:
            selected_prob_map = {agent_id: float(selected_probs[idx].item()) for idx, agent_id in enumerate(agent_ids)}
            max_prob_map = {agent_id: float(max_probs[idx].item()) for idx, agent_id in enumerate(agent_ids)}
            return action_map, log_prob_map, value_map, entropy_map, selected_prob_map, max_prob_map
        return action_map, log_prob_map, value_map, entropy_map

    def predict_actions(self, states: Mapping[str, np.ndarray]) -> Dict[str, int]:
        actions, _, _, _ = self.act(states, deterministic=True)
        return actions

    @staticmethod
    def compute_gae(
        rewards: Iterable[float],
        dones: Iterable[bool],
        values: Iterable[float],
        *,
        next_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards_np = np.asarray(list(rewards), dtype=np.float32)
        dones_np = np.asarray(list(dones), dtype=np.float32)
        values_np = np.asarray(list(values), dtype=np.float32)
        if rewards_np.size == 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        returns = np.zeros_like(rewards_np, dtype=np.float32)
        gae = 0.0
        next_val = float(next_value)
        for step in reversed(range(len(rewards_np))):
            mask = 1.0 - dones_np[step]
            delta = rewards_np[step] + gamma * next_val * mask - values_np[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = gae + values_np[step]
            next_val = values_np[step]
        return advantages, returns

    def update_from_batch(
        self,
        local_states: Sequence[np.ndarray],
        actions: Sequence[int],
        log_probs: Sequence[float],
        values: Sequence[float],
        advantages: Sequence[float],
        returns: Sequence[float],
    ) -> tuple[float, float, float, Dict[str, float]]:
        states_np = np.asarray(list(local_states), dtype=np.float32)
        actions_np = np.asarray(list(actions), dtype=np.int64)
        old_log_probs_np = np.asarray(list(log_probs), dtype=np.float32)
        values_np = np.asarray(list(values), dtype=np.float32)
        advantages_np = np.asarray(list(advantages), dtype=np.float32)
        returns_np = np.asarray(list(returns), dtype=np.float32)

        if states_np.size == 0:
            return 0.0, 0.0, 0.0, {"samples": 0.0, "updates": 0.0}

        adv_mean = float(advantages_np.mean())
        adv_std = float(advantages_np.std())
        if adv_std > 1e-8:
            advantages_np = (advantages_np - adv_mean) / (adv_std + 1e-8)
        else:
            advantages_np = advantages_np - adv_mean

        states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions_np, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs_np, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(values_np, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)

        batch_size = max(1, int(self.hyperparams.batch_size))
        indices = np.arange(len(states_np))
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        selected_prob_total = 0.0
        max_prob_total = 0.0
        approx_kl_total = 0.0
        clip_fraction_total = 0.0
        ratio_total = 0.0
        ratio_sq_total = 0.0
        sample_total = 0
        update_total = 0

        for _ in range(self.hyperparams.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_old_values = old_values_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                dist = self._distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values_pred = self.critic(batch_states)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_mask = torch.abs(ratio - 1.0) > self.hyperparams.clip_eps
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.hyperparams.clip_eps,
                    1.0 + self.hyperparams.clip_eps,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                values_pred_clipped = batch_old_values + torch.clamp(
                    values_pred - batch_old_values,
                    -self.hyperparams.clip_eps,
                    self.hyperparams.clip_eps,
                )
                value_loss_unclipped = (values_pred - batch_returns).pow(2)
                value_loss_clipped = (values_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = policy_loss + self.hyperparams.value_coef * value_loss - self.hyperparams.entropy_coef * entropy
                selected_probs = dist.probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                max_probs, _ = dist.probs.max(dim=-1)
                approx_kl = (batch_old_log_probs - new_log_probs).mean()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                batch_size_actual = len(batch_idx)
                policy_loss_total += float(policy_loss.item()) * batch_size_actual
                value_loss_total += float(value_loss.item()) * batch_size_actual
                entropy_total += float(entropy.item()) * batch_size_actual
                selected_prob_total += float(selected_probs.mean().item()) * batch_size_actual
                max_prob_total += float(max_probs.mean().item()) * batch_size_actual
                approx_kl_total += float(approx_kl.item()) * batch_size_actual
                clip_fraction_total += float(clipped_mask.float().mean().item()) * batch_size_actual
                ratio_mean = float(ratio.mean().item())
                ratio_total += ratio_mean * batch_size_actual
                ratio_sq_total += float(ratio.pow(2).mean().item()) * batch_size_actual
                sample_total += batch_size_actual
                update_total += 1

        sample_total = max(sample_total, 1)
        diagnostics = {
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "samples": float(sample_total),
            "updates": float(update_total),
            "selected_prob_mean": selected_prob_total / sample_total,
            "max_prob_mean": max_prob_total / sample_total,
            "approx_kl": approx_kl_total / sample_total,
            "clip_fraction": clip_fraction_total / sample_total,
            "ratio_mean": ratio_total / sample_total,
            "ratio_std": float(max(ratio_sq_total / sample_total - (ratio_total / sample_total) ** 2, 0.0) ** 0.5),
        }
        return (
            policy_loss_total / sample_total,
            value_loss_total / sample_total,
            entropy_total / sample_total,
            diagnostics,
        )

    def save(self, path: str | Path, *, controller_type: str = "coordinated_ppo") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hyperparams": self.hyperparams.__dict__,
                "controller_type": controller_type,
                "shared_policy": True,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

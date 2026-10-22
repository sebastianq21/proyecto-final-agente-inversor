"""Double DQN (van Hasselt et al., 2016) on top of Stable-Baselines3's DQN.

Contents:
  - DoubleDQN : SB3 DQN whose TD target picks the next action with the online
                network and scores it with the target network

SB3's DQN takes max_a Q_target(s', a): the same network picks and scores the
action, so noise in the estimates turns into an upward bias. Double DQN
decouples the two: a* = argmax_a Q_online(s', a), target = Q_target(s', a*).
Everything else (replay, epsilon schedule, Huber loss, target updates) is
SB3's DQN unchanged.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional as F


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # The only change from DQN.train: online net picks, target net scores
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations), dim=1, index=next_actions
                )
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = th.gather(
                self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

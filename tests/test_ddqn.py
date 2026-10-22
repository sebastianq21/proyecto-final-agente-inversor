"""DoubleDQN: the online network picks the next action, the target network scores it."""
import torch as th

from rl_games_sb import registry
from rl_games_sb.ddqn import DoubleDQN


def test_ddqn_is_registered_and_trains():
    model = registry.create(
        "ddqn", "googl-log-v0", learning_starts=50, buffer_size=1_000, seed=0, device="cpu"
    )
    assert isinstance(model, DoubleDQN)
    model.learn(total_timesteps=200)
    assert model._n_updates > 0


def test_ddqn_target_uses_online_argmax(monkeypatch):
    model = registry.create("ddqn", "googl-log-v0", learning_starts=50, buffer_size=1_000, seed=0, device="cpu")
    model.learn(total_timesteps=100)  # fill the buffer

    # Online net prefers action 2, target net rates action 2 at 1.0 and action 0 at 9.0.
    # Double DQN must bootstrap from 1.0 (target value of the online choice), not 9.0.
    online = th.tensor([[0.0, 0.0, 5.0]])
    target = th.tensor([[9.0, 0.0, 1.0]])
    seen = {}

    def fake_q_net(obs):
        n = obs.shape[0]
        out = online.repeat(n, 1).requires_grad_()
        return out

    monkeypatch.setattr(model, "q_net", fake_q_net)
    monkeypatch.setattr(model, "q_net_target", lambda obs: target.repeat(obs.shape[0], 1))
    monkeypatch.setattr(model.policy.optimizer, "step", lambda: None)

    def fake_loss(current, target_q):
        seen["target"] = target_q
        return (current - target_q).abs().mean()

    monkeypatch.setattr("rl_games_sb.ddqn.F.smooth_l1_loss", fake_loss)
    data = model.replay_buffer.sample(8)
    monkeypatch.setattr(model.replay_buffer, "sample", lambda *a, **k: data)
    model.train(gradient_steps=1, batch_size=8)

    expected = data.rewards + (1 - data.dones) * model.gamma * 1.0
    assert th.allclose(seen["target"], expected)

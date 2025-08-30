import os
import math
import logging
from typing import Tuple

import numpy as np

try:  # pragma: no cover - heavy deps not needed in CI
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore

logger = logging.getLogger(__name__)


class MockClashEnv(gym.Env if gym else object):  # type: ignore[misc]
    """A mock environment to enable PPO training/eval scaffolding.

    State: [elixir (0..10), my_towers(3), enemy_towers(3), hand(4 one-hot placeholder)]
    Action: MultiDiscrete([4, 9, 5]) -> (card_idx, grid_x, grid_y)
    Reward: +1 when enemy tower HP decreases; -1 when we 'lose' (enemy takes all towers)
    """

    metadata = {"render_modes": []}

    def __init__(self):  # pragma: no cover - simple stochastic mock
        if gym is None:
            raise RuntimeError("gymnasium not available")
        self.action_space = spaces.MultiDiscrete([4, 9, 5])
        # Observation space: 1 + 3 + 3 + 4 = 11 dims; keep bounded
        low = np.zeros(11, dtype=np.float32)
        high = np.ones(11, dtype=np.float32) * 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.reset()

    def _obs(self):
        elixir_norm = self.elixir / 10.0
        my = np.array(self.my_towers, dtype=np.float32) / 100.0
        en = np.array(self.enemy_towers, dtype=np.float32) / 100.0
        hand = np.ones(4, dtype=np.float32) / 4.0
        return np.concatenate([[elixir_norm], my, en, hand]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.elixir = 5
        self.my_towers = [100, 100, 100]
        self.enemy_towers = [100, 100, 100]
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        card_idx, gx, gy = int(action[0]), int(action[1]), int(action[2])
        # Simple stochastic outcome: higher gy slightly more offensive
        attack_factor = (gy + 1) / 5.0
        damage = np.random.binomial(1, min(0.6, 0.2 + 0.6 * attack_factor)) * np.random.randint(1, 5)
        # Apply to a random enemy tower
        ti = np.random.randint(0, 3)
        pre = self.enemy_towers[ti]
        self.enemy_towers[ti] = max(0, self.enemy_towers[ti] - damage)
        reward = 1.0 if self.enemy_towers[ti] < pre else 0.0
        # Chance of taking damage back
        retaliation = np.random.binomial(1, 0.3)
        if retaliation:
            mi = np.random.randint(0, 3)
            self.my_towers[mi] = max(0, self.my_towers[mi] - np.random.randint(0, 3))
        # Lose condition
        done = all(v == 0 for v in self.my_towers) or self.steps >= 180
        if all(v == 0 for v in self.my_towers):
            reward -= 1.0
        return self._obs(), float(reward), done, False, {}


def build_and_train_ppo(total_timesteps: int = 2_000_000, lr: float = 3e-4, ent_coef: float = 0.01, log_dir: str = "runs/ppo"):
    if PPO is None or gym is None:
        raise RuntimeError("Stable-Baselines3 or gymnasium not available")
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: MockClashEnv()])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr, ent_coef=ent_coef, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps)
    return model


class OfflineLogEnv(gym.Env if gym else object):  # type: ignore[misc]
    """Replay environment from logged JSONL transitions.

    Each line in logs/episodes/*.jsonl is expected to contain keys:
    - elixir, gold, daily_wins, action {card, nx, ny}, reward
    """

    def __init__(self, logs_dir: str):  # pragma: no cover - IO heavy
        if gym is None:
            raise RuntimeError("gymnasium not available")
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.files = [os.path.join(self.logs_dir, f) for f in os.listdir(self.logs_dir) if f.endswith(".jsonl")]
        self.file_idx = 0
        self.fp = None
        self.action_space = spaces.MultiDiscrete([4, 9, 5])
        low = np.zeros(11, dtype=np.float32)
        high = np.ones(11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.curr_obs = None

    def _obs_from_record(self, rec: dict) -> np.ndarray:
        elixir_norm = float(rec.get("elixir", 0)) / 10.0
        my = np.ones(3, dtype=np.float32)  # placeholder
        en = np.ones(3, dtype=np.float32)
        hand = np.ones(4, dtype=np.float32) / 4.0
        return np.concatenate([[elixir_norm], my, en, hand]).astype(np.float32)

    def _open_next(self):
        import json
        if self.fp:
            self.fp.close()
            self.fp = None
        while self.file_idx < len(self.files):
            try:
                self.fp = open(self.files[self.file_idx], "r", encoding="utf-8")
                return
            except Exception:
                self.file_idx += 1

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if not self.files:
            self.curr_obs = np.zeros(11, dtype=np.float32)
            return self.curr_obs, {}
        self._open_next()
        self.curr_obs = np.zeros(11, dtype=np.float32)
        return self.curr_obs, {}

    def step(self, action):  # pragma: no cover - IO heavy
        import json
        if not self.fp:
            return self.curr_obs, 0.0, True, False, {}
        line = self.fp.readline()
        if not line:
            self.file_idx += 1
            return self.curr_obs, 0.0, True, False, {}
        try:
            rec = json.loads(line)
        except Exception:
            return self.curr_obs, 0.0, False, False, {}
        reward = float(rec.get("reward", 0.0))
        self.curr_obs = self._obs_from_record(rec)
        done = False
        return self.curr_obs, reward, done, False, {}


def build_and_train_offline(logs_dir: str = "logs/episodes", total_timesteps: int = 1_000_000, lr: float = 3e-4, ent_coef: float = 0.01, log_dir: str = "runs/offline"):
    if PPO is None or gym is None:
        raise RuntimeError("Stable-Baselines3 or gymnasium not available")
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: OfflineLogEnv(logs_dir)])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr, ent_coef=ent_coef, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps)
    return model

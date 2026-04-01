"""
Gymnasium-compatible wrapper for the 2048 game engine.

Provides a standard Gymnasium interface with a 16-channel binary
observation space (one channel per power-of-2 tile value), suitable
for CNN-based agents (DQN, PPO) following the Saligram paper approach.

Observation Space:
    Box(0, 1, shape=(16, 4, 4), dtype=float32)
    Channel k represents positions of tiles with value 2^k.
    Channel 0 represents empty cells (value 0).

Action Space:
    Discrete(4): {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}

Reward:
    Merge score delta per step (configurable via reward_mode).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.game_2048 import Game2048


class Gym2048Env(gym.Env):
    """
    Gymnasium wrapper for 2048.

    Supports multiple observation encodings and reward shaping modes.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 4,
        reward_mode: str = "score_delta",
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            size: Board dimension (default 4).
            reward_mode: One of 'score_delta', 'log_score', 'shaped'.
                - 'score_delta': raw merge score per step
                - 'log_score': log2(score_delta + 1) for scale normalization
                - 'shaped': score_delta + bonus for reaching milestones
            max_steps: Maximum steps per episode (None for unlimited).
            render_mode: 'human' for pretty print, 'ansi' for string.
            seed: Random seed.
        """
        super().__init__()

        self.size = size
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._seed = seed

        # 16 channels: channel 0 = empty, channel k = tile 2^k (k=1..15)
        # Covers tiles from 2 to 2^15 = 32768
        self.n_channels = 16
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_channels, self.size, self.size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self.game: Optional[Game2048] = None
        self._step_count = 0
        self._milestone_tiles_reached: set[int] = set()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self.game = Game2048(size=self.size, seed=effective_seed)
        self._step_count = 0
        self._milestone_tiles_reached = set()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self.game is not None, "Call reset() before step()"

        _, score_delta, done, step_info = self.game.step(action)
        self._step_count += 1

        # Compute reward based on mode
        reward = self._compute_reward(score_delta, step_info)

        # Truncation check
        truncated = False
        if self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True

        terminated = done
        obs = self._get_observation()
        info = self._get_info()
        info.update(step_info)

        if self.render_mode == "human":
            print(self.game.render())

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, score_delta: float, info: dict) -> float:
        """Compute reward based on the configured reward mode."""
        if not info["valid"]:
            # Penalty for invalid moves (no board change)
            return -1.0

        if self.reward_mode == "score_delta":
            return score_delta

        elif self.reward_mode == "log_score":
            return float(np.log2(score_delta + 1)) if score_delta > 0 else 0.0

        elif self.reward_mode == "shaped":
            reward = score_delta
            # Milestone bonuses (only awarded once per tile value)
            max_tile = info["max_tile"]
            milestones = {256: 50, 512: 100, 1024: 200, 2048: 500, 4096: 1000}
            for tile, bonus in milestones.items():
                if max_tile >= tile and tile not in self._milestone_tiles_reached:
                    reward += bonus
                    self._milestone_tiles_reached.add(tile)
            return reward

        else:
            return score_delta

    def _get_observation(self) -> np.ndarray:
        """
        Convert the board to a 16-channel binary representation.

        Channel 0: empty cells (board == 0)
        Channel k (1-15): cells with value 2^k
        """
        obs = np.zeros(
            (self.n_channels, self.size, self.size), dtype=np.float32
        )
        board = self.game.board

        # Channel 0: empty cells
        obs[0] = (board == 0).astype(np.float32)

        # Channels 1-15: tile values 2^1 through 2^15
        for k in range(1, self.n_channels):
            obs[k] = (board == (1 << k)).astype(np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Return current environment info."""
        return {
            "score": self.game.score,
            "max_tile": self.game.max_tile,
            "step_count": self._step_count,
            "valid_actions": self.game.get_valid_actions(),
        }

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (required by sb3_contrib MaskablePPO/A2C)."""
        mask = np.zeros(4, dtype=bool)
        if self.game is not None:
            for a in self.game.get_valid_actions():
                mask[a] = True
        return mask

    def render(self) -> Optional[str]:
        """Render the current board."""
        if self.game is None:
            return None
        rendered = self.game.render()
        if self.render_mode == "human":
            print(rendered)
        return rendered

    def close(self):
        """Clean up."""
        self.game = None


# Register the environment with Gymnasium
gym.register(
    id="Game2048-v0",
    entry_point="src.env.gym_wrapper:Gym2048Env",
    kwargs={"size": 4, "reward_mode": "score_delta"},
)

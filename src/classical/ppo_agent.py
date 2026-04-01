"""
PPO Agent for 2048 using Stable-Baselines3.

Wraps the SB3 PPO implementation with a custom CNN feature extractor
to handle the 16-channel binary observation space from the Gym2048Env.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sb3_contrib import MaskablePPO as PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


# ─── Custom Feature Extractor ────────────────────────────────────────

class Game2048CNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 16-channel 2048 observations.

    Architecture matches the DQN agent for fair comparison:
        Conv2D(16, 128, 2×2) → Conv2D(128, 128, 2×2) → Conv2D(128, 128, 2×2)
        → Flatten → FC(128, 256)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute output size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ─── Metrics Callback ────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Callback to log episode metrics during PPO training."""

    def __init__(
        self,
        logger_obj: TrainingLogger,
        eval_freq: int = 10_000,
        checkpoint_freq: int = 50_000,
        log_dir: str = "logs/ppo",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.logger_obj = logger_obj
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_dir = log_dir
        self.episode_num = 0
        self.episode_start = time.time()

    def _on_step(self) -> bool:
        # Check for completed episodes in the info buffer
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_num += 1
                metrics = EpisodeMetrics(
                    episode=self.episode_num,
                    total_score=info.get("score", info["episode"]["r"]),
                    max_tile=info.get("max_tile", 0),
                    num_moves=info["episode"]["l"],
                    valid_moves=info["episode"]["l"],  # SB3 doesn't track invalid
                    invalid_moves=0,
                    wall_clock_seconds=info["episode"]["t"],
                    training_steps=self.num_timesteps,
                )
                self.logger_obj.log_episode(metrics)

        # Periodic reporting
        if self.num_timesteps % self.eval_freq == 0 and self.verbose:
            summary = self.logger_obj.get_summary(last_n=50)
            if summary:
                tqdm.write(
                    f"  Step {self.num_timesteps:>7,} | Ep {self.episode_num:>5} | "
                    f"Avg Score: {summary['avg_score']:>7.0f} | "
                    f"Max Tile: {summary['max_tile_ever']:>5} | "
                    f"512+: {summary['reach_512']:.1%}"
                )

        # Checkpointing
        if self.num_timesteps % self.checkpoint_freq == 0:
            path = os.path.join(
                self.log_dir, f"ppo_step_{self.num_timesteps}"
            )
            self.model.save(path)
            self.logger_obj.plot_training_curves()

        return True


# ─── Training Function ───────────────────────────────────────────────

def train_ppo(
    total_steps: int = 1_000_000,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    log_dir: str = "logs/ppo",
    reward_mode: str = "shaped",          # milestone bonuses at 256/512/1024/2048
    seed: int = 42,
    lr: float = 2e-4,                     # slightly lower for stability
    n_steps: int = 512,                   # shorter rollouts → faster policy updates
    batch_size: int = 128,
    n_epochs: int = 4,
    gamma: float = 0.995,
    clip_range: float = 0.2,
    ent_coef: float = 0.05,               # stronger entropy prevents premature collapse
    n_envs: int = 8,
    device: str = "auto",
) -> PPO:
    """
    Train a PPO agent on 2048 using Stable-Baselines3.

    Returns:
        Trained PPO model.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environments
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    import src.env.gym_wrapper  # ensure "Game2048-v0" is registered in each subprocess

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        "Game2048-v0",
        n_envs=n_envs,
        env_kwargs={"reward_mode": reward_mode},
        vec_env_cls=vec_cls,
        seed=seed,
    )

    # Bigger feature extractor for harder task
    policy_kwargs = {
        "features_extractor_class": Game2048CNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
    }

    # Create PPO model
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,                # keep entropy alive
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device=device,
    )

    # Setup logging
    logger = TrainingLogger(log_dir=log_dir, experiment_name="ppo")
    callback = MetricsCallback(
        logger_obj=logger,
        eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq,
        log_dir=log_dir,
        verbose=1,
    )

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  PPO Training — {total_steps:,} steps              ║")
    print(f"║  Device: {model.device}                          ║")
    print(f"║  Reward: {reward_mode}                    ║")
    print(f"║  Parallel envs: {n_envs}                           ║")
    print(f"╚══════════════════════════════════════════╝")

    # Train with progress bar
    pbar = tqdm(total=total_steps, desc="PPO Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    class ProgressCallback(BaseCallback):
        """Updates tqdm progress bar."""
        def _on_step(self):
            pbar.update(self.model.n_envs)
            return True

    model.learn(
        total_timesteps=total_steps,
        callback=[callback, ProgressCallback()],
    )
    pbar.close()

    # Final save
    model.save(os.path.join(log_dir, "ppo_final"))
    logger.plot_training_curves()
    env.close()

    print(f"\n{'='*50}")
    print("Training complete!")
    summary = logger.get_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return model

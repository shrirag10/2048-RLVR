"""
QR-DQN Agent for 2048 using SB3-Contrib.

Quantile Regression DQN (QR-DQN) is a distributional RL algorithm that
learns the full distribution of returns rather than just the expected value.
This leads to better risk-aware decision making and improved sample efficiency.

References:
    Dabney et al., "Distributional Reinforcement Learning with Quantile Regression" (AAAI 2018)
    Horizon-DQN framework showed QR-DQN reaches tile 4096 in 2048 (arXiv 2024)
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

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


# ─── Custom Feature Extractor ────────────────────────────────────────

class Game2048CNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 16-channel 2048 observations.

    Same architecture as DQN/PPO/A2C for fair comparison:
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
    """Callback to log episode metrics during QR-DQN training."""

    def __init__(
        self,
        logger_obj: TrainingLogger,
        eval_freq: int = 10_000,
        checkpoint_freq: int = 50_000,
        log_dir: str = "logs/qrdqn",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.logger_obj = logger_obj
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_dir = log_dir
        self.episode_num = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_num += 1
                metrics = EpisodeMetrics(
                    episode=self.episode_num,
                    total_score=info.get("score", info["episode"]["r"]),
                    max_tile=info.get("max_tile", 0),
                    num_moves=info["episode"]["l"],
                    valid_moves=info["episode"]["l"],
                    invalid_moves=0,
                    wall_clock_seconds=info["episode"]["t"],
                    training_steps=self.num_timesteps,
                )
                self.logger_obj.log_episode(metrics)

        if self.num_timesteps % self.eval_freq == 0 and self.verbose:
            summary = self.logger_obj.get_summary(last_n=50)
            if summary:
                tqdm.write(
                    f"  Step {self.num_timesteps:>7,} | Ep {self.episode_num:>5} | "
                    f"Avg Score: {summary['avg_score']:>7.0f} | "
                    f"Max Tile: {summary['max_tile_ever']:>5} | "
                    f"512+: {summary['reach_512']:.1%}"
                )

        if self.num_timesteps % self.checkpoint_freq == 0:
            path = os.path.join(self.log_dir, f"qrdqn_step_{self.num_timesteps}")
            self.model.save(path)
            self.logger_obj.plot_training_curves()

        return True


# ─── Training Function ───────────────────────────────────────────────

def train_qrdqn(
    total_steps: int = 500_000,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    log_dir: str = "logs/qrdqn",
    reward_mode: str = "score_delta",              # match DQN — distributional RL works best with simple rewards
    seed: int = 42,
    lr: float = 6.3e-5,                            # slightly lower than DQN — distributional needs stability
    buffer_size: int = 500_000,                     # bigger buffer for off-policy data diversity
    batch_size: int = 128,                          # larger batch for better quantile gradient estimates
    gamma: float = 0.99,                            # match DQN — 0.995 was too high
    exploration_fraction: float = 0.4,              # explore longer — 40% of total steps
    exploration_final_eps: float = 0.02,            # slightly higher floor than DQN
    n_quantiles: int = 200,                         # more quantiles = finer return distribution
    target_update_interval: int = 1000,             # less frequent target sync for stability
    train_freq: int = 4,                            # train every 4 steps — less aggressive
    learning_starts: int = 20_000,                  # collect more diverse initial data
    gradient_steps: int = 1,
    n_envs: int = 8,
    device: str = "auto",
) -> QRDQN:
    """
    Train a QR-DQN agent on 2048 using SB3-Contrib.

    QR-DQN models the full return distribution using quantile regression,
    enabling better risk-sensitive learning compared to standard DQN.

    Returns:
        Trained QR-DQN model.
    """
    os.makedirs(log_dir, exist_ok=True)

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    import src.env.gym_wrapper  # ensure registration in subprocesses

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        "Game2048-v0",
        n_envs=n_envs,
        env_kwargs={"reward_mode": reward_mode},
        vec_env_cls=vec_cls,
        seed=seed,
    )

    policy_kwargs = {
        "features_extractor_class": Game2048CNN,
        "features_extractor_kwargs": {"features_dim": 512},  # bigger extractor
        "n_quantiles": n_quantiles,
    }

    model = QRDQN(
        "CnnPolicy",
        env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device=device,
    )

    logger = TrainingLogger(log_dir=log_dir, experiment_name="qrdqn")
    callback = MetricsCallback(
        logger_obj=logger,
        eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq,
        log_dir=log_dir,
        verbose=1,
    )

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  QR-DQN Training — {total_steps:,} steps           ║")
    print(f"║  Quantiles: {n_quantiles}                           ║")
    print(f"║  Device: {model.device}                          ║")
    print(f"║  Reward: {reward_mode}                    ║")
    print(f"║  Parallel envs: {n_envs}                           ║")
    print(f"╚══════════════════════════════════════════╝")

    pbar = tqdm(total=total_steps, desc="QR-DQN Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            pbar.update(self.model.n_envs)
            return True

    model.learn(
        total_timesteps=total_steps,
        callback=[callback, ProgressCallback()],
    )
    pbar.close()

    model.save(os.path.join(log_dir, "qrdqn_final"))
    logger.plot_training_curves()
    env.close()

    print(f"\n{'='*50}")
    print("Training complete!")
    summary = logger.get_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return model

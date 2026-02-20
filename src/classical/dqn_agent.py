"""
DQN Agent for 2048.

Implements Deep Q-Network following the Saligram paper methodology:
- CNN architecture handling 16-channel board observations
- Experience replay buffer with configurable size
- Target network with periodic syncing
- Epsilon-greedy exploration with linear decay
- Gradient clipping for stability

Reference: Saligram et al., "2048: Reinforcement Learning in a Delayed
Reward Environment" (arXiv:2507.05465)
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger


# ─── DQN Network Architecture ────────────────────────────────────────

class DQN_CNN(nn.Module):
    """
    CNN-based Q-Network for 2048.

    Architecture:
        Input: (batch, 16, 4, 4) — 16 binary channels
        → Conv2D(16, 128, 2×2) + ReLU
        → Conv2D(128, 128, 2×2) + ReLU
        → Conv2D(128, 128, 2×2) + ReLU (output: 128×1×1 for 4×4 board)
        → Flatten
        → FC(128, 256) + ReLU
        → FC(256, 4)  — Q-values for 4 actions
    """

    def __init__(self, n_channels: int = 16, n_actions: int = 4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        # After three 2×2 convs on 4×4 input: 4→3→2→1, so output is 128×1×1
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.fc_layers(x)


# ─── Experience Replay Buffer ────────────────────────────────────────

@dataclass
class Transition:
    """Single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent for 2048.

    Features:
        - Double networks (policy + target) for stability
        - Epsilon-greedy with linear decay
        - Experience replay
        - Gradient clipping
        - Periodic checkpointing
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100_000,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_sync_freq: int = 1000,
        grad_clip: float = 1.0,
        device: str = "auto",
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = DQN_CNN().to(self.device)
        self.target_net = DQN_CNN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.target_sync_freq = target_sync_freq

        # Counters
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(
        self, state: np.ndarray, valid_actions: list[int] | None = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        If valid_actions is provided, only consider those actions.
        """
        if random.random() < self.epsilon:
            # Random action from valid ones
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, 3)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)

            if valid_actions:
                # Mask invalid actions with -inf
                mask = torch.full((4,), float("-inf"), device=self.device)
                for a in valid_actions:
                    mask[a] = 0.0
                q_values = q_values + mask

            return int(q_values.argmax().item())

    def update(self) -> Optional[float]:
        """
        Perform one gradient step on a batch from the replay buffer.

        Returns:
            Loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)

        # Batch tensors
        states = torch.FloatTensor(
            np.array([t.state for t in transitions])
        ).to(self.device)
        actions = torch.LongTensor(
            [t.action for t in transitions]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [t.reward for t in transitions]
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in transitions])
        ).to(self.device)
        dones = torch.FloatTensor(
            [float(t.done) for t in transitions]
        ).to(self.device)

        # Current Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Huber loss (smooth L1), more robust than MSE
        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip
        )
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        """Linearly decay epsilon."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.steps_done
            / self.epsilon_decay_steps,
        )

    def sync_target(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.episodes_done = checkpoint["episodes_done"]
        self.epsilon = checkpoint["epsilon"]


# ─── Training Loop ────────────────────────────────────────────────────

def train_dqn(
    total_steps: int = 500_000,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    log_dir: str = "logs/dqn",
    reward_mode: str = "score_delta",
    seed: int = 42,
    **agent_kwargs,
) -> DQNAgent:
    """
    Train a DQN agent on 2048.

    Args:
        total_steps: Total environment steps.
        eval_freq: Steps between evaluation reports.
        checkpoint_freq: Steps between checkpoint saves.
        log_dir: Directory for logs and checkpoints.
        reward_mode: Reward mode for the environment.
        seed: Random seed.
        **agent_kwargs: Additional DQNAgent parameters.

    Returns:
        Trained DQNAgent.
    """
    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup
    env = Gym2048Env(reward_mode=reward_mode, seed=seed)
    agent = DQNAgent(**agent_kwargs)
    logger = TrainingLogger(log_dir=log_dir, experiment_name="dqn")

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  DQN Training — {total_steps:,} steps               ║")
    print(f"║  Device: {agent.device}                          ║")
    print(f"║  Reward: {reward_mode}                    ║")
    print(f"╚══════════════════════════════════════════╝")

    obs, info = env.reset()
    episode_score = 0.0
    episode_moves = 0
    episode_valid = 0
    episode_invalid = 0
    episode_start = time.time()
    episode_num = 0

    for step in range(1, total_steps + 1):
        agent.steps_done = step

        # Select and execute action
        valid_actions = info.get("valid_actions", list(range(4)))
        action = agent.select_action(obs, valid_actions)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track metrics
        if info.get("valid", True):
            episode_valid += 1
        else:
            episode_invalid += 1
        episode_moves += 1
        episode_score = info.get("score", episode_score)

        # Store transition
        agent.replay_buffer.push(
            Transition(obs, action, reward, next_obs, done)
        )

        # Train
        loss = agent.update()
        agent.decay_epsilon()

        # Sync target network
        if step % agent.target_sync_freq == 0:
            agent.sync_target()

        obs = next_obs

        if done:
            episode_num += 1
            agent.episodes_done = episode_num
            elapsed = time.time() - episode_start

            metrics = EpisodeMetrics(
                episode=episode_num,
                total_score=episode_score,
                max_tile=info.get("max_tile", 0),
                num_moves=episode_moves,
                valid_moves=episode_valid,
                invalid_moves=episode_invalid,
                wall_clock_seconds=elapsed,
                training_steps=step,
            )
            logger.log_episode(metrics)

            # Reset
            obs, info = env.reset()
            episode_score = 0.0
            episode_moves = 0
            episode_valid = 0
            episode_invalid = 0
            episode_start = time.time()

        # Periodic reporting
        if step % eval_freq == 0:
            summary = logger.get_summary(last_n=50)
            if summary:
                print(
                    f"  Step {step:>7,} | Ep {episode_num:>5} | "
                    f"ε={agent.epsilon:.3f} | "
                    f"Avg Score: {summary['avg_score']:>7.0f} | "
                    f"Max Tile: {summary['max_tile_ever']:>5} | "
                    f"512+: {summary['reach_512']:.1%}"
                )

        # Checkpointing
        if step % checkpoint_freq == 0:
            ckpt_path = os.path.join(log_dir, f"dqn_step_{step}.pt")
            agent.save(ckpt_path)
            logger.plot_training_curves()

    # Final save
    agent.save(os.path.join(log_dir, "dqn_final.pt"))
    logger.plot_training_curves()
    env.close()

    print(f"\n{'='*50}")
    print("Training complete!")
    summary = logger.get_summary()
    if summary:
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return agent

"""
Shared metrics and logging utilities.

Provides CSV-based metric logging and matplotlib plot generation
for training curves across both classical RL and LLM-RLVR tracks.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    total_score: float
    max_tile: int
    num_moves: int
    valid_moves: int
    invalid_moves: int
    wall_clock_seconds: float
    training_steps: int = 0


@dataclass
class TrainingLogger:
    """
    CSV-based training logger with matplotlib plot generation.

    Usage:
        logger = TrainingLogger("logs/dqn")
        logger.log_episode(metrics)
        logger.plot_training_curves()
    """
    log_dir: str
    experiment_name: str = "experiment"
    _episodes: list[EpisodeMetrics] = field(default_factory=list)
    _csv_path: Optional[str] = None
    _start_time: float = field(default_factory=time.time)

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(
            self.log_dir, f"{self.experiment_name}_metrics.csv"
        )
        # Write CSV header
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "total_score", "max_tile", "num_moves",
                "valid_moves", "invalid_moves", "wall_clock_seconds",
                "training_steps"
            ])

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """Log an episode's metrics to CSV and internal buffer."""
        self._episodes.append(metrics)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.episode, metrics.total_score, metrics.max_tile,
                metrics.num_moves, metrics.valid_moves, metrics.invalid_moves,
                metrics.wall_clock_seconds, metrics.training_steps,
            ])

    def get_summary(self, last_n: int = 100) -> dict:
        """Get summary statistics for the last N episodes."""
        if not self._episodes:
            return {}
        episodes = self._episodes[-last_n:]
        scores = [e.total_score for e in episodes]
        tiles = [e.max_tile for e in episodes]
        moves = [e.num_moves for e in episodes]
        return {
            "n_episodes": len(self._episodes),
            "avg_score": np.mean(scores),
            "max_score": np.max(scores),
            "std_score": np.std(scores),
            "avg_max_tile": np.mean(tiles),
            "max_tile_ever": max(e.max_tile for e in self._episodes),
            "avg_moves": np.mean(moves),
            "win_rate_2048": sum(1 for t in tiles if t >= 2048) / len(tiles),
            "reach_1024": sum(1 for t in tiles if t >= 1024) / len(tiles),
            "reach_512": sum(1 for t in tiles if t >= 512) / len(tiles),
        }

    def plot_training_curves(self, save_dir: Optional[str] = None) -> None:
        """Generate and save matplotlib training curve plots."""
        if not self._episodes:
            return

        save_dir = save_dir or self.log_dir
        os.makedirs(save_dir, exist_ok=True)

        episodes = self._episodes
        ep_nums = [e.episode for e in episodes]
        scores = [e.total_score for e in episodes]
        tiles = [e.max_tile for e in episodes]
        moves = [e.num_moves for e in episodes]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"{self.experiment_name} — Training Curves",
            fontsize=14, fontweight="bold"
        )

        # 1. Score over episodes (with rolling average)
        ax = axes[0, 0]
        ax.plot(ep_nums, scores, alpha=0.3, color="steelblue", linewidth=0.5)
        if len(scores) >= 50:
            window = min(100, len(scores) // 5)
            rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
            ax.plot(
                ep_nums[window - 1:], rolling,
                color="darkblue", linewidth=2, label=f"Rolling avg ({window})"
            )
            ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Episode Score")
        ax.grid(True, alpha=0.3)

        # 2. Max tile distribution
        ax = axes[0, 1]
        unique_tiles, counts = np.unique(tiles, return_counts=True)
        tile_labels = [str(int(t)) for t in unique_tiles]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(unique_tiles)))
        ax.bar(tile_labels, counts, color=colors)
        ax.set_xlabel("Max Tile")
        ax.set_ylabel("Count")
        ax.set_title("Max Tile Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Max tile over time
        ax = axes[1, 0]
        ax.scatter(ep_nums, tiles, alpha=0.3, s=5, color="coral")
        if len(tiles) >= 50:
            window = min(100, len(tiles) // 5)
            rolling = np.convolve(
                [float(t) for t in tiles],
                np.ones(window) / window, mode="valid"
            )
            ax.plot(
                ep_nums[window - 1:], rolling,
                color="darkred", linewidth=2, label=f"Rolling avg ({window})"
            )
            ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Max Tile")
        ax.set_title("Max Tile Over Time")
        ax.set_yscale("log", base=2)
        ax.grid(True, alpha=0.3)

        # 4. Moves per episode
        ax = axes[1, 1]
        ax.plot(ep_nums, moves, alpha=0.3, color="seagreen", linewidth=0.5)
        if len(moves) >= 50:
            window = min(100, len(moves) // 5)
            rolling = np.convolve(moves, np.ones(window) / window, mode="valid")
            ax.plot(
                ep_nums[window - 1:], rolling,
                color="darkgreen", linewidth=2, label=f"Rolling avg ({window})"
            )
            ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Moves")
        ax.set_title("Moves Per Episode")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{self.experiment_name}_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to: {path}")

    def plot_comparison(
        self,
        other_loggers: list["TrainingLogger"],
        save_path: str = "comparison.png",
    ) -> None:
        """Plot comparison of multiple experiments."""
        all_loggers = [self] + other_loggers

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Cross-Track Comparison", fontsize=14, fontweight="bold")

        colors = plt.cm.Set2(np.linspace(0, 1, len(all_loggers)))

        for idx, logger in enumerate(all_loggers):
            episodes = logger._episodes
            if not episodes:
                continue

            ep_nums = [e.episode for e in episodes]
            scores = [e.total_score for e in episodes]
            window = min(100, max(1, len(scores) // 5))
            if len(scores) >= window:
                rolling = np.convolve(
                    scores, np.ones(window) / window, mode="valid"
                )
                axes[0].plot(
                    ep_nums[window - 1:], rolling,
                    color=colors[idx], linewidth=2,
                    label=logger.experiment_name,
                )

        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Score (Rolling Avg)")
        axes[0].set_title("Score Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Max tile box plot
        tile_data = []
        labels = []
        for logger in all_loggers:
            if logger._episodes:
                tile_data.append([e.max_tile for e in logger._episodes])
                labels.append(logger.experiment_name)

        if tile_data:
            axes[1].boxplot(tile_data, labels=labels)
            axes[1].set_ylabel("Max Tile")
            axes[1].set_title("Max Tile Distribution")
            axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot saved to: {save_path}")

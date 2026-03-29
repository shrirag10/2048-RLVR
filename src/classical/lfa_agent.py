"""
Linear Function Approximation Agent for 2048.

Implements Semi-gradient SARSA with linear function approximation
exactly as described in Sutton & Barto "Reinforcement Learning: An
Introduction" (2nd ed., 2018):

  Algorithm:  Semi-gradient SARSA (Chapter 10, Algorithm 10.1)
  Value fn:   q̂(s, a, w) = w_a · φ(s)             [Eq. 9.3, linear case]
  Update:     w_a ← w_a + α [R + γ q̂(s', a', w) - q̂(s, a, w)] φ(s)
                                                    [Eq. 10.1, semi-gradient TD]

Feature vector φ(s) ∈ ℝ^d encodes the 2048 board state into a fixed-size
real-valued vector. Each action has its own weight vector w_a ∈ ℝ^d, making
the combined weight tensor w ∈ ℝ^{4 × d}.

Feature Groups (Sutton & Barto §9.5 — Feature Construction):
  1. Log-tile encoding     — φ_i = log2(tile_i + 1) / 15,  i ∈ {0..15}
     Coarse representation of tile magnitudes, normalized to [0,1].
  2. Empty-cell ratio      — n_empty / 16
     Captures board density; strongly correlated with game stage.
  3. Merge potential       — adjacent same-tile pairs (normalized)
     Proxy for immediate reward opportunity.
  4. Monotonicity          — left-right and up-down gradient measures
     Key heuristic: monotone boards are easier to keep organized.
  5. Max tile (log-norm)   — log2(max_tile + 1) / 15
     Tracks game progress.
  6. Corner bonus          — max_tile in corner indicator
     Cornerstone strategy: keep the highest tile in a corner.
  7. Tile distribution     — fraction of empty/low/mid/high tiles
     Coarse summary of board composition.

Total feature dimension d = 16 + 1 + 1 + 4 + 1 + 1 + 3 = 27

References:
  Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An
  Introduction (2nd ed.). MIT Press.
    - §9.3  Stochastic-gradient and Semi-gradient Methods
    - §9.5  Feature Construction (linear methods, tile coding)
    - §10.1 Episodic Semi-gradient Control (SARSA)
    - §6.5  Q-learning (off-policy update variant used in --off-policy mode)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.env.gym_wrapper import Gym2048Env
from src.utils.metrics import EpisodeMetrics, TrainingLogger

# ─── Feature Engineering ─────────────────────────────────────────────────────

# Feature dimension breakdown (must match _phi implementation)
_D_TILE    = 16   # log-encoded tile values at each cell
_D_EMPTY   =  1   # empty-cell ratio
_D_MERGE   =  1   # normalized merge-pair count
_D_MONO    =  4   # monotonicity in 4 directions
_D_MAXTILE =  1   # log-normalized max tile
_D_CORNER  =  1   # max-tile-in-corner indicator
_D_DIST    =  3   # tile distribution (low / mid / high)
N_FEATURES = _D_TILE + _D_EMPTY + _D_MERGE + _D_MONO + _D_MAXTILE + _D_CORNER + _D_DIST  # 27


def _board_from_obs(obs: np.ndarray) -> np.ndarray:
    """
    Reconstruct the flat integer board from a 16-channel binary observation.

    obs shape: (16, 4, 4).  Channel 0 = empty, channel k = tile 2^k.
    Returns: flat int32 array of length 16.
    """
    board = np.zeros(16, dtype=np.int32)
    for k in range(1, 16):
        positions = np.where(obs[k].flatten() == 1.0)[0]
        board[positions] = 1 << k
    return board


def phi(obs: np.ndarray) -> np.ndarray:
    """
    Compute the feature vector φ(s) ∈ ℝ^{N_FEATURES} for a board state.

    Sutton & Barto §9.5: "features are functions of the state and action
    that we can reasonably expect to be relevant to predicting value."

    Args:
        obs: Gym2048Env observation, shape (16, 4, 4), binary channels.

    Returns:
        Feature vector of shape (N_FEATURES,), dtype float32.
    """
    board = _board_from_obs(obs)        # flat int32, length 16
    b2d   = board.reshape(4, 4)

    # 1. Log-tile encoding — §9.5 linear fn approx feature representation
    #    Normalize by log2(2^15) = 15 so each feature ∈ [0, 1]
    tile_feats = np.log2(board.astype(np.float32) + 1.0) / 15.0   # (16,)

    # 2. Empty-cell ratio — board density
    n_empty = np.sum(board == 0)
    empty_feat = np.array([n_empty / 16.0], dtype=np.float32)

    # 3. Merge potential — count adjacent pairs with equal nonzero values
    #    (horizontal and vertical), normalized by max possible pairs (24)
    merge_count = 0
    for r in range(4):
        for c in range(3):
            if b2d[r, c] != 0 and b2d[r, c] == b2d[r, c + 1]:
                merge_count += 1
    for c in range(4):
        for r in range(3):
            if b2d[r, c] != 0 and b2d[r, c] == b2d[r + 1, c]:
                merge_count += 1
    merge_feat = np.array([merge_count / 24.0], dtype=np.float32)

    # 4. Monotonicity — measure how monotone each row/column direction is.
    #    A perfectly monotone row scores 1.0; a reversed row scores 0.0.
    #    We compute four metrics: LR, RL, UD, DU (§9.5, spatial features).
    def _mono_score(seq: np.ndarray) -> float:
        """Fraction of adjacent pairs that are non-decreasing (in log space)."""
        log_seq = np.log2(seq.astype(np.float32) + 1.0)
        diffs = np.diff(log_seq)
        return float(np.sum(diffs >= 0)) / max(1, len(diffs))

    lr = np.mean([_mono_score(b2d[r, :])  for r in range(4)])
    rl = np.mean([_mono_score(b2d[r, ::-1]) for r in range(4)])
    ud = np.mean([_mono_score(b2d[:, c])  for c in range(4)])
    du = np.mean([_mono_score(b2d[::-1, c]) for c in range(4)])
    mono_feat = np.array([lr, rl, ud, du], dtype=np.float32)

    # 5. Max tile (log-normalized)
    max_tile = int(board.max())
    maxtile_feat = np.array([np.log2(max_tile + 1) / 15.0], dtype=np.float32)

    # 6. Corner bonus — highest tile is in one of the four corners
    #    (canonical 2048 strategy; strong heuristic predictor of success)
    corners = {b2d[0,0], b2d[0,3], b2d[3,0], b2d[3,3]}
    corner_feat = np.array([1.0 if max_tile in corners else 0.0], dtype=np.float32)

    # 7. Tile distribution — fraction of cells in each value range
    n = 16
    low  = np.sum((board > 0)   & (board <= 8))   / n
    mid  = np.sum((board >= 16) & (board <= 128))  / n
    high = np.sum(board >= 256) / n
    dist_feat = np.array([low, mid, high], dtype=np.float32)

    return np.concatenate([
        tile_feats,    # 16
        empty_feat,    #  1
        merge_feat,    #  1
        mono_feat,     #  4
        maxtile_feat,  #  1
        corner_feat,   #  1
        dist_feat,     #  3
    ])  # total: 27


# ─── Agent ────────────────────────────────────────────────────────────────────

@dataclass
class LFAConfig:
    """Hyperparameters for the Linear FA SARSA agent."""
    alpha:       float = 5e-4   # Step size (Sutton & Barto α)
    gamma:       float = 0.99   # Discount factor (γ)
    epsilon:     float = 1.0    # Initial ε for ε-greedy policy
    epsilon_min: float = 0.05   # Final ε
    epsilon_decay: float = 0.9999  # Multiplicative decay applied each step


class LinearFAAgent:
    """
    Semi-gradient SARSA agent with linear function approximation.

    Sutton & Barto Algorithm 10.1 (Episodic Semi-gradient SARSA):

        Input: differentiable q̂: S × A × ℝ^d → ℝ, step size α, small ε > 0
        Initialize w ∈ ℝ^d arbitrarily (here: zeros)

        Loop for each episode:
            S ← initial state
            A ← ε-greedy(S, w)
            Loop for each step:
                R, S' ← take action A
                if S' is terminal:
                    w ← w + α [R - q̂(S,A,w)] ∇q̂(S,A,w)
                    break
                A' ← ε-greedy(S', w)
                w ← w + α [R + γ q̂(S',A',w) - q̂(S,A,w)] ∇q̂(S,A,w)
                S ← S';  A ← A'

    Linear case:  q̂(s, a, w) = w_a · φ(s)
    Gradient:     ∇_w q̂(s, a, w) = φ(s)   (w.r.t. w_a; 0 for other actions)
    Update:       w_a ← w_a + α · δ · φ(s)   where δ = TD error
    """

    def __init__(self, cfg: LFAConfig = LFAConfig()):
        self.cfg = cfg
        # Weight matrix w ∈ ℝ^{4 × d}: row a = w_a
        # Initialized to zero (Sutton & Barto: "arbitrarily, often 0")
        self.w = np.zeros((4, N_FEATURES), dtype=np.float64)
        self.epsilon = cfg.epsilon

    # ── Value function ──────────────────────────────────────────────────────

    def q_hat(self, phi_s: np.ndarray, action: int) -> float:
        """
        Linear approximation of action-value function.

        q̂(s, a, w) = w_a · φ(s)    [Sutton & Barto Eq. 9.3, linear case]
        """
        return float(self.w[action] @ phi_s)

    def q_hat_all(self, phi_s: np.ndarray) -> np.ndarray:
        """Return q̂(s, a, w) for all 4 actions: shape (4,)."""
        return self.w @ phi_s   # (4, d) @ (d,) → (4,)

    # ── Policy ─────────────────────────────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: list[int],
        greedy: bool = False,
    ) -> tuple[int, np.ndarray]:
        """
        ε-greedy action selection restricted to valid actions.

        Sutton & Barto §10.1: "ε-greedy with respect to q̂".
        Invalid actions (those that don't change the board) are masked
        to avoid the agent wasting steps — a domain-specific constraint.

        Returns:
            (action, phi_s) — action chosen and cached feature vector.
        """
        phi_s = phi(obs)
        if (not greedy) and (np.random.random() < self.epsilon):
            return int(np.random.choice(valid_actions)), phi_s

        # Greedy: argmax q̂(s, a, w) over valid actions
        q_vals = self.q_hat_all(phi_s)
        # Mask invalid actions with -inf
        masked = np.full(4, -np.inf)
        masked[valid_actions] = q_vals[valid_actions]
        return int(np.argmax(masked)), phi_s

    # ── Semi-gradient SARSA update ─────────────────────────────────────────

    def update(
        self,
        phi_s:  np.ndarray,
        action: int,
        reward: float,
        phi_s_next: Optional[np.ndarray],
        next_action: Optional[int],
        terminal: bool,
    ) -> float:
        """
        One semi-gradient SARSA weight update.

        TD error (Sutton & Barto Eq. 10.1):
            δ = R + γ q̂(S', A', w) - q̂(S, A, w)   [non-terminal]
            δ = R - q̂(S, A, w)                       [terminal]

        Weight update (semi-gradient, linear case):
            w_a ← w_a + α · δ · φ(s)

        Note: "semi-gradient" because we do not differentiate through
        the bootstrap target q̂(S', A', w) (Sutton & Barto §9.3).

        Returns:
            TD error δ (for logging).
        """
        q_sa = self.q_hat(phi_s, action)

        if terminal:
            # No bootstrap: target is just R  (Sutton & Barto §10.1)
            delta = reward - q_sa
        else:
            # Bootstrap from next state-action pair (SARSA: on-policy)
            q_next = self.q_hat(phi_s_next, next_action)
            delta = reward + self.cfg.gamma * q_next - q_sa

        # Semi-gradient update: only w_a is updated, gradient = φ(s)
        self.w[action] += self.cfg.alpha * delta * phi_s

        return delta

    def decay_epsilon(self):
        """Exponential epsilon decay, clamped at epsilon_min."""
        self.epsilon = max(self.cfg.epsilon_min,
                           self.epsilon * self.cfg.epsilon_decay)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, w=self.w, epsilon=np.array([self.epsilon]))

    def load(self, path: str):
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        self.w       = data["w"]
        self.epsilon = float(data["epsilon"][0])


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_lfa(
    total_steps:     int   = 2_000_000,
    alpha:           float = 5e-4,
    gamma:           float = 0.99,
    epsilon:         float = 1.0,
    epsilon_min:     float = 0.05,
    epsilon_decay:   float = 0.9999,
    reward_mode:     str   = "score_delta",
    seed:            int   = 42,
    log_dir:         str   = "logs/lfa",
    checkpoint_freq: int   = 100_000,
) -> LinearFAAgent:
    """
    Train a LinearFAAgent via Semi-gradient SARSA.

    Sutton & Barto Algorithm 10.1, run for `total_steps` environment steps.

    Args:
        total_steps:     Total environment interaction steps.
        alpha:           SARSA step size (α).
        gamma:           Discount factor (γ).
        epsilon:         Initial exploration rate.
        epsilon_min:     Minimum exploration rate.
        epsilon_decay:   Per-step multiplicative ε decay.
        reward_mode:     Gym2048Env reward shaping mode.
        seed:            RNG seed for reproducibility.
        log_dir:         Output directory for checkpoints + CSV metrics.
        checkpoint_freq: Save weights every N steps.

    Returns:
        Trained LinearFAAgent.
    """
    np.random.seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    cfg = LFAConfig(
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
    )
    agent  = LinearFAAgent(cfg)
    env    = Gym2048Env(reward_mode=reward_mode, seed=seed)
    logger = TrainingLogger(log_dir=log_dir, experiment_name="lfa")

    step      = 0
    episode   = 0
    best_tile = 0

    bar = tqdm(total=total_steps, desc="LFA SARSA", unit="step",
               dynamic_ncols=True, colour="blue")

    while step < total_steps:
        episode += 1
        obs, info = env.reset(seed=seed + episode)
        valid     = info.get("valid_actions", list(range(4)))
        ep_start  = time.time()

        # S₀, A₀ — initial state-action pair (Algorithm 10.1)
        action, phi_s = agent.select_action(obs, valid)

        ep_reward     = 0.0
        ep_moves      = 0
        ep_td_abs_sum = 0.0

        while True:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done    = terminated or truncated
            ep_reward += reward
            ep_moves  += 1
            step      += 1
            agent.decay_epsilon()

            if done:
                # Terminal update: no bootstrap (Algorithm 10.1 terminal case)
                delta = agent.update(phi_s, action, reward,
                                     phi_s_next=None, next_action=None,
                                     terminal=True)
                ep_td_abs_sum += abs(delta)
                bar.update(1)
                break

            # A' ← ε-greedy(S', w)
            next_valid  = next_info.get("valid_actions", list(range(4)))
            next_action, phi_s_next = agent.select_action(next_obs, next_valid)

            # Semi-gradient SARSA update (Algorithm 10.1)
            delta = agent.update(phi_s, action, reward,
                                 phi_s_next=phi_s_next, next_action=next_action,
                                 terminal=False)
            ep_td_abs_sum += abs(delta)

            # S ← S';  A ← A'
            obs, info, phi_s = next_obs, next_info, phi_s_next
            action = next_action

            bar.update(1)
            if step >= total_steps:
                break

        # ── Episode metrics ────────────────────────────────────────────────
        max_tile = next_info.get("max_tile", 0) if done else info.get("max_tile", 0)
        score    = next_info.get("score", 0)    if done else info.get("score", 0)
        best_tile = max(best_tile, max_tile)

        metrics = EpisodeMetrics(
            episode=episode,
            total_score=score,
            max_tile=max_tile,
            num_moves=ep_moves,
            valid_moves=ep_moves,
            invalid_moves=0,
            wall_clock_seconds=time.time() - ep_start,
            training_steps=step,
        )
        logger.log_episode(metrics)

        bar.set_postfix(
            ep=episode, tile=max_tile, best=best_tile,
            score=f"{score:,}", eps=f"{agent.epsilon:.3f}",
            refresh=False,
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        if step % checkpoint_freq < ep_moves:   # crossed a checkpoint boundary
            ckpt = os.path.join(log_dir, f"lfa_step_{step // checkpoint_freq * checkpoint_freq}")
            agent.save(ckpt)

    bar.close()
    env.close()

    # Final save
    agent.save(os.path.join(log_dir, "lfa_final"))
    logger.plot_training_curves()

    summary = logger.get_summary()
    print(f"\n{'='*55}")
    print(f"  LFA SARSA Training Complete")
    print(f"  Steps:     {step:,}")
    print(f"  Episodes:  {episode:,}")
    print(f"  Avg score: {summary.get('avg_score', 0):.0f}")
    print(f"  Best tile: {best_tile}")
    print(f"{'='*55}")

    return agent

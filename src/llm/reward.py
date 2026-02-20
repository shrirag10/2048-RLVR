"""
Verifier and Reward Functions for LLM-RLVR 2048 Agent.

Implements a multi-component reward function for GRPO training,
combining format correctness, move validity, score delta, and
milestone bonuses. Designed to be passed as reward_funcs to
TRL's GRPOTrainer.

Reward Components:
┌─────────────────────┬────────┬─────────────────────────────────────────┐
│ Component           │ Value  │ Description                             │
├─────────────────────┼────────┼─────────────────────────────────────────┤
│ Format correctness  │ +0.5   │ Valid XML <think> and <answer> tags     │
│ Valid direction      │ +0.5   │ Answer ∈ {UP, DOWN, LEFT, RIGHT}       │
│ Move validity       │ +1/-2  │ Move changes board / no-op penalty     │
│ Score delta         │ Δs/max │ Normalized merge score                  │
│ Game over penalty   │ -1.0   │ Game ends after this move               │
│ Milestone bonus     │ +2..+5 │ Reaching 256/512/1024+ tiles           │
└─────────────────────┴────────┴─────────────────────────────────────────┘
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from src.env.game_2048 import Game2048
from src.env.text_wrapper import parse_llm_response


# ─── Completion Text Extraction ───────────────────────────────────────
# TRL 0.24+ GRPOTrainer can pass completions in different formats:
#   - list[str]: plain text completions
#   - list[list[dict]]: chat-format completions (list of message dicts)
# This helper normalizes both to plain strings.


def _extract_text(completion) -> str:
    """
    Extract the text content from a TRL completion.

    Handles:
        - str: returned as-is
        - list[dict]: extracts 'content' from the last message
        - dict: extracts 'content' field
    """
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        # Chat format: list of {"role": ..., "content": ...}
        if completion and isinstance(completion[0], dict):
            # Get the assistant's (last) message content
            return completion[-1].get("content", "")
        # List of strings — join them
        return " ".join(str(c) for c in completion)
    elif isinstance(completion, dict):
        return completion.get("content", str(completion))
    else:
        return str(completion)


# ─── Individual Reward Functions ──────────────────────────────────────
# Each function follows the TRL GRPOTrainer signature:
#   reward_fn(completions, **kwargs) -> list[float]
# where completions are the model's generated outputs (may be strings
# or chat-format lists) and kwargs contain additional context.


def format_reward_fn(completions, **kwargs) -> list[float]:
    """
    Reward for correct XML formatting.

    Awards +0.5 if the response contains both <think>...</think>
    and <answer>...</answer> tags in the correct structure.
    Partial credit (+0.25) if only one tag is present.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        parsed = parse_llm_response(text)
        if parsed.format_valid:
            rewards.append(0.5)
        elif parsed.thinking is not None or parsed.action is not None:
            rewards.append(0.25)  # Partial format
        else:
            rewards.append(0.0)
    return rewards


def direction_reward_fn(completions, **kwargs) -> list[float]:
    """
    Reward for valid direction in <answer> tag.

    Awards +0.5 if the extracted direction is one of
    {UP, DOWN, LEFT, RIGHT}.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        parsed = parse_llm_response(text)
        rewards.append(0.5 if parsed.direction_valid else 0.0)
    return rewards


def game_reward_fn(
    completions: list[str],
    board_states: Optional[list[list[list[int]]]] = None,
    scores: Optional[list[int]] = None,
    **kwargs,
) -> list[float]:
    """
    Core game-play reward using the verifier.

    Executes the proposed move on the game engine and rewards based on:
    - Move validity (does the move change the board?)
    - Score delta (normalized by the theoretical max single-step score)
    - Game state consequences

    Args:
        completions: List of model-generated response strings.
        board_states: List of board states (4×4 nested lists).
        scores: List of current scores for each board state.

    Returns:
        List of reward floats.
    """
    rewards = []

    if board_states is None:
        # Can't verify without board states — return neutral
        return [0.0] * len(completions)

    for i, completion in enumerate(completions):
        text = _extract_text(completion)
        parsed = parse_llm_response(text)
        reward = 0.0

        if parsed.action_id is None:
            # Couldn't parse a valid action
            rewards.append(-1.0)
            continue

        # Create a game from the board state
        board = np.array(board_states[i], dtype=np.int32)
        game = Game2048.__new__(Game2048)
        game.size = 4
        game.board = board.copy()
        game.score = scores[i] if scores else 0
        game.done = False
        game.rng = np.random.default_rng(42)  # Deterministic for scoring

        # Check if move is valid BEFORE executing
        is_valid = game.is_valid_action(parsed.action_id)

        if not is_valid:
            # Invalid move: board wouldn't change
            rewards.append(-2.0)
            continue

        # Execute the move
        _, score_delta, done, info = game.step(parsed.action_id)

        # Score delta reward (normalized)
        # Max single-step score in 2048 is merging four 16384s = 65536
        # But practically, normalize by a more common range
        max_practical_score = 1024.0  # Reasonable normalization constant
        reward += score_delta / max_practical_score

        # Valid move bonus
        reward += 1.0

        # Game over penalty
        if done:
            reward -= 1.0

        # Milestone bonuses (based on max tile AFTER the move)
        max_tile = info.get("max_tile", 0)
        milestone_bonuses = {
            256: 2.0,
            512: 3.0,
            1024: 5.0,
            2048: 10.0,
            4096: 15.0,
        }
        for tile, bonus in milestone_bonuses.items():
            if max_tile >= tile:
                # Check if this tile was newly achieved by this move
                old_max = int(np.max(board))
                if old_max < tile:
                    reward += bonus

        rewards.append(reward)

    return rewards


def thinking_quality_reward_fn(completions, **kwargs) -> list[float]:
    """
    Reward for reasoning quality in the <think> block.

    Heuristically measures the quality of the thinking process:
    - Mentions of specific tile values (shows board awareness)
    - Mentions of strategic concepts (corner, merge, monotonic, etc.)
    - Appropriate length (not too short, not too long)

    This is a soft signal that encourages reasoning without being
    overly prescriptive.
    """
    STRATEGIC_TERMS = [
        "corner", "merge", "monoton", "gradient", "empty", "space",
        "highest", "largest", "adjacent", "slide", "combine", "row",
        "column", "edge", "trap", "stuck", "block", "strategy",
    ]

    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        parsed = parse_llm_response(text)
        reward = 0.0

        if parsed.thinking is None:
            rewards.append(0.0)
            continue

        thinking = parsed.thinking.lower()
        words = thinking.split()
        n_words = len(words)

        # Length bonus: 20-150 words is ideal
        if 20 <= n_words <= 150:
            reward += 0.2
        elif 10 <= n_words < 20 or 150 < n_words <= 250:
            reward += 0.1
        # Too short or too long gets 0

        # Board awareness: mentions specific numbers
        tile_mentions = sum(
            1 for w in words
            if w.isdigit() and int(w) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        )
        reward += min(0.15, tile_mentions * 0.03)

        # Strategic vocabulary
        strategy_mentions = sum(
            1 for term in STRATEGIC_TERMS if term in thinking
        )
        reward += min(0.15, strategy_mentions * 0.03)

        rewards.append(reward)

    return rewards


# ─── Combined Reward Function ────────────────────────────────────────

def combined_reward_fn(
    completions: list[str],
    board_states: Optional[list[list[list[int]]]] = None,
    scores: Optional[list[int]] = None,
    **kwargs,
) -> list[float]:
    """
    Combined reward function aggregating all components.

    This is a convenience function that sums all reward components.
    For GRPO training, it's better to pass individual functions
    as separate reward_funcs to GRPOTrainer for better gradient signals.
    """
    format_r = format_reward_fn(completions, **kwargs)
    direction_r = direction_reward_fn(completions, **kwargs)
    game_r = game_reward_fn(
        completions, board_states=board_states, scores=scores, **kwargs
    )
    thinking_r = thinking_quality_reward_fn(completions, **kwargs)

    return [
        f + d + g + t
        for f, d, g, t in zip(format_r, direction_r, game_r, thinking_r)
    ]

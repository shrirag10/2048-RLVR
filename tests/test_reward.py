"""
Test suite for LLM reward functions.

Validates that each reward component produces the expected values
for known inputs, ensuring the verifier correctly assesses move
validity, format compliance, and score calculation.
"""

import numpy as np
import pytest

from src.llm.reward import (
    format_reward_fn,
    direction_reward_fn,
    game_reward_fn,
    thinking_quality_reward_fn,
    combined_reward_fn,
)


# ─── Format Reward Tests ─────────────────────────────────────────────

class TestFormatReward:
    """Test XML format validation reward."""

    def test_perfect_format(self):
        """Both <think> and <answer> tags present → +0.5."""
        completions = ["<think>reasoning</think><answer>UP</answer>"]
        rewards = format_reward_fn(completions)
        assert rewards[0] == 0.5

    def test_partial_format_think_only(self):
        """Only <think> present → +0.25."""
        completions = ["<think>reasoning</think>UP"]
        rewards = format_reward_fn(completions)
        assert rewards[0] == 0.25

    def test_partial_format_answer_only(self):
        """Only <answer> present → +0.25."""
        completions = ["reasoning <answer>UP</answer>"]
        rewards = format_reward_fn(completions)
        assert rewards[0] == 0.25

    def test_no_format(self):
        """No tags → 0.0."""
        completions = ["I think UP is best"]
        rewards = format_reward_fn(completions)
        assert rewards[0] == 0.0

    def test_batch(self):
        """Batch of completions processed correctly."""
        completions = [
            "<think>a</think><answer>UP</answer>",
            "no format",
            "<think>b</think>missing answer",
        ]
        rewards = format_reward_fn(completions)
        assert len(rewards) == 3
        assert rewards[0] == 0.5
        assert rewards[1] == 0.0
        assert rewards[2] == 0.25


# ─── Direction Reward Tests ───────────────────────────────────────────

class TestDirectionReward:
    """Test direction validity reward."""

    def test_valid_directions(self):
        """All four valid directions get +0.5."""
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            completions = [f"<think>r</think><answer>{direction}</answer>"]
            rewards = direction_reward_fn(completions)
            assert rewards[0] == 0.5, f"Failed for {direction}"

    def test_invalid_direction(self):
        """Invalid direction → 0.0."""
        completions = ["<think>r</think><answer>DIAGONAL</answer>"]
        rewards = direction_reward_fn(completions)
        assert rewards[0] == 0.0

    def test_no_answer_tag(self):
        """Missing answer tag → 0.0."""
        completions = ["<think>r</think>UP"]
        rewards = direction_reward_fn(completions)
        assert rewards[0] == 0.0


# ─── Game Reward Tests ────────────────────────────────────────────────

class TestGameReward:
    """Test game-play verifier reward."""

    def test_valid_move_positive_reward(self):
        """A valid move that changes the board gets positive reward."""
        board = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        completions = ["<think>merge</think><answer>LEFT</answer>"]
        rewards = game_reward_fn(
            completions, board_states=[board], scores=[0]
        )
        assert rewards[0] > 0  # Valid move + score delta

    def test_invalid_move_penalty(self):
        """A move that doesn't change the board gets -2.0."""
        board = [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        completions = ["<think>left</think><answer>LEFT</answer>"]
        rewards = game_reward_fn(
            completions, board_states=[board], scores=[0]
        )
        assert rewards[0] == -2.0

    def test_unparseable_response(self):
        """Unparseable response gets -1.0."""
        board = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        completions = ["garbled output"]
        rewards = game_reward_fn(
            completions, board_states=[board], scores=[0]
        )
        assert rewards[0] == -1.0

    def test_no_board_states(self):
        """Without board states, returns neutral 0.0."""
        completions = ["<think>r</think><answer>UP</answer>"]
        rewards = game_reward_fn(completions)
        assert rewards[0] == 0.0


# ─── Thinking Quality Tests ──────────────────────────────────────────

class TestThinkingQuality:
    """Test thinking quality heuristic reward."""

    def test_good_reasoning(self):
        """Reasoning with strategic terms and tile references gets bonus."""
        thinking = (
            "The highest tile is 512 in the corner. "
            "I see two 64 tiles adjacent that can merge. "
            "Moving UP maintains the monotonic gradient along the edge. "
            "There are 3 empty spaces for new tiles."
        )
        completions = [f"<think>{thinking}</think><answer>UP</answer>"]
        rewards = thinking_quality_reward_fn(completions)
        assert rewards[0] > 0.2  # Should get length + tile + strategy bonuses

    def test_empty_thinking(self):
        """No thinking block → 0.0."""
        completions = ["<answer>UP</answer>"]
        rewards = thinking_quality_reward_fn(completions)
        assert rewards[0] == 0.0

    def test_too_short_thinking(self):
        """Very short thinking gets less reward."""
        completions = ["<think>go up</think><answer>UP</answer>"]
        rewards = thinking_quality_reward_fn(completions)
        assert rewards[0] < 0.2  # Short = less reward


# ─── Combined Reward Tests ───────────────────────────────────────────

class TestCombinedReward:
    """Test combined reward function."""

    def test_combined_sums_components(self):
        """Combined reward is sum of all components."""
        board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        completions = [
            "<think>The two 2-tiles can merge LEFT for a 4 tile.</think>"
            "<answer>LEFT</answer>"
        ]
        combined = combined_reward_fn(
            completions, board_states=[board], scores=[0]
        )
        assert len(combined) == 1
        assert isinstance(combined[0], float)
        assert combined[0] > 0  # Should be positive for a valid move


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

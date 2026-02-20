"""
Test suite for the 2048 game environment.

Tests cover:
- Game engine mechanics (init, moves, scoring, game over)
- Gymnasium wrapper API compliance
- Text wrapper prompt/response formatting
"""

import numpy as np
import pytest

from src.env.game_2048 import Game2048
from src.env.gym_wrapper import Gym2048Env
from src.env.text_wrapper import TextGame2048, parse_llm_response


# ─── Game Engine Tests ────────────────────────────────────────────────

class TestGame2048:
    """Test core 2048 game mechanics."""

    def test_init_board(self):
        """Board starts with exactly 2 tiles."""
        game = Game2048(seed=42)
        non_zero = np.count_nonzero(game.board)
        assert non_zero == 2

    def test_init_tile_values(self):
        """Initial tiles are 2 or 4."""
        game = Game2048(seed=42)
        values = game.board[game.board > 0]
        assert all(v in [2, 4] for v in values)

    def test_reset(self):
        """Reset returns board to initial state."""
        game = Game2048(seed=42)
        game.step(Game2048.UP)
        game.reset(seed=42)
        assert game.score == 0
        assert not game.done
        assert np.count_nonzero(game.board) == 2

    def test_deterministic_seed(self):
        """Same seed produces same initial board."""
        g1 = Game2048(seed=123)
        g2 = Game2048(seed=123)
        np.testing.assert_array_equal(g1.board, g2.board)

    def test_different_seeds(self):
        """Different seeds produce different boards (with high probability)."""
        g1 = Game2048(seed=1)
        g2 = Game2048(seed=999)
        # Not strictly guaranteed but extremely likely
        assert not np.array_equal(g1.board, g2.board)

    def test_slide_left_simple(self):
        """Test basic left slide and merge."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        _, delta, _, info = game.step(Game2048.LEFT)
        # 2+2 = 4, score delta should be 4
        assert delta == 4.0
        assert info["valid"]
        assert info["merges"] == 1
        # First cell in first row should be 4
        assert game.board[0, 0] == 4

    def test_slide_right(self):
        """Test right slide."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        _, delta, _, info = game.step(Game2048.RIGHT)
        assert delta == 4.0
        assert game.board[0, 3] == 4

    def test_slide_up(self):
        """Test up slide."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        _, delta, _, info = game.step(Game2048.UP)
        assert delta == 4.0
        assert game.board[0, 0] == 4

    def test_slide_down(self):
        """Test down slide."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        _, delta, _, info = game.step(Game2048.DOWN)
        assert delta == 4.0
        assert game.board[3, 0] == 4

    def test_no_merge_slide(self):
        """Tiles with different values slide but don't merge."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        _, delta, _, info = game.step(Game2048.LEFT)
        assert delta == 0.0
        assert info["merges"] == 0

    def test_invalid_move_no_change(self):
        """Invalid move returns valid=False."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)

        # LEFT won't change anything (tiles already packed left)
        _, delta, _, info = game.step(Game2048.LEFT)
        assert not info["valid"]
        assert delta == 0.0

    def test_is_valid_action(self):
        """is_valid_action correctly identifies valid moves."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)

        assert not game.is_valid_action(Game2048.LEFT)  # Already packed left
        assert not game.is_valid_action(Game2048.RIGHT)  # All distinct, no merge possible
        assert not game.is_valid_action(Game2048.UP)  # Already at top
        assert game.is_valid_action(Game2048.DOWN)  # Can slide down to empty rows

        # Board where RIGHT is valid (has gaps)
        game.board = np.array([
            [2, 0, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        assert game.is_valid_action(Game2048.RIGHT)  # Tiles can slide right

    def test_game_over_detection(self):
        """Game correctly detects when no moves are available."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ], dtype=np.int32)
        assert not game._has_valid_moves()

    def test_game_not_over_with_merge(self):
        """Game is not over when merges are possible."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 4],  # Two 4s can merge
        ], dtype=np.int32)
        assert game._has_valid_moves()

    def test_max_tile(self):
        """max_tile returns the highest value on the board."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        assert game.max_tile == 1024

    def test_clone(self):
        """Clone creates an independent copy."""
        game = Game2048(seed=42)
        clone = game.clone()
        clone.step(Game2048.UP)
        assert game.score != clone.score or not np.array_equal(
            game.board, clone.board
        )

    def test_score_accumulates(self):
        """Score accumulates across multiple moves."""
        game = Game2048(seed=42)
        game.board = np.array([
            [2, 2, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        game.score = 0

        game.step(Game2048.LEFT)
        # Should merge 2+2=4 and 4+4=8, score = 4+8 = 12
        assert game.score == 12

    def test_render_no_crash(self):
        """Render returns a string without crashing."""
        game = Game2048(seed=42)
        rendered = game.render()
        assert isinstance(rendered, str)
        assert "Score" in rendered

    def test_to_list(self):
        """to_list returns proper nested list."""
        game = Game2048(seed=42)
        lst = game.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 4
        assert all(len(row) == 4 for row in lst)


# ─── Gymnasium Wrapper Tests ─────────────────────────────────────────

class TestGym2048Env:
    """Test Gymnasium API compliance."""

    def test_reset_returns_correct_shape(self):
        """Reset returns observation with correct shape."""
        env = Gym2048Env(seed=42)
        obs, info = env.reset()
        assert obs.shape == (16, 4, 4)
        assert obs.dtype == np.float32

    def test_observation_is_binary(self):
        """Each channel contains only 0s and 1s."""
        env = Gym2048Env(seed=42)
        obs, _ = env.reset()
        assert np.all((obs == 0) | (obs == 1))

    def test_observation_channels_exclusive(self):
        """Each cell is represented in exactly one channel."""
        env = Gym2048Env(seed=42)
        obs, _ = env.reset()
        # Sum across channels — each position should have exactly 1
        channel_sum = obs.sum(axis=0)
        np.testing.assert_array_equal(channel_sum, np.ones((4, 4)))

    def test_step_returns_correct_types(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = Gym2048Env(seed=42)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (16, 4, 4)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        """Action space is Discrete(4)."""
        env = Gym2048Env(seed=42)
        assert env.action_space.n == 4

    def test_reward_modes(self):
        """Different reward modes produce different rewards."""
        # score_delta mode
        env1 = Gym2048Env(reward_mode="score_delta", seed=42)
        env1.reset()
        _, r1, _, _, _ = env1.step(0)

        # log_score mode
        env2 = Gym2048Env(reward_mode="log_score", seed=42)
        env2.reset()
        _, r2, _, _, _ = env2.step(0)

        # Both should be valid floats
        assert isinstance(r1, float)
        assert isinstance(r2, float)

    def test_max_steps_truncation(self):
        """Environment truncates after max_steps."""
        env = Gym2048Env(max_steps=5, seed=42)
        obs, _ = env.reset()
        for i in range(5):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated:
                break
        # Should be truncated by step 5 if not terminated
        if not terminated:
            assert truncated

    def test_invalid_move_penalty(self):
        """Invalid moves get negative reward."""
        env = Gym2048Env(seed=42)
        obs, info = env.reset()
        # Try many moves — at least one should be invalid
        for _ in range(50):
            _, reward, terminated, _, info = env.step(0)
            if not info.get("valid", True):
                assert reward < 0
                break
            if terminated:
                break


# ─── Text Wrapper Tests ──────────────────────────────────────────────

class TestTextWrapper:
    """Test LLM text format and parsing."""

    def test_parse_valid_response(self):
        """Parse a correctly formatted LLM response."""
        response = (
            "<think>The highest tile is 64 in the top-left corner. "
            "I should move UP to maintain the gradient.</think>"
            "<answer>UP</answer>"
        )
        parsed = parse_llm_response(response)
        assert parsed.format_valid
        assert parsed.direction_valid
        assert parsed.action == "UP"
        assert parsed.action_id == 0
        assert "64" in parsed.thinking

    def test_parse_all_directions(self):
        """All four directions are parsed correctly."""
        for direction, expected_id in [
            ("UP", 0), ("RIGHT", 1), ("DOWN", 2), ("LEFT", 3)
        ]:
            response = f"<think>reason</think><answer>{direction}</answer>"
            parsed = parse_llm_response(response)
            assert parsed.action == direction
            assert parsed.action_id == expected_id

    def test_parse_case_insensitive(self):
        """Direction parsing handles case variations."""
        response = "<think>reason</think><answer>up</answer>"
        parsed = parse_llm_response(response)
        assert parsed.action == "UP"
        assert parsed.direction_valid

    def test_parse_invalid_direction(self):
        """Invalid direction is detected."""
        response = "<think>reason</think><answer>DIAGONAL</answer>"
        parsed = parse_llm_response(response)
        assert parsed.format_valid  # Tags are present
        assert not parsed.direction_valid
        assert parsed.action is None

    def test_parse_missing_tags(self):
        """Missing tags are detected."""
        response = "I think UP is best"
        parsed = parse_llm_response(response)
        assert not parsed.format_valid
        assert not parsed.direction_valid

    def test_parse_partial_tags(self):
        """Partial tags (only think or answer)."""
        response = "<answer>LEFT</answer>"
        parsed = parse_llm_response(response)
        assert not parsed.format_valid  # Missing think
        assert parsed.direction_valid  # But direction is valid

    def test_text_game_prompt_format(self):
        """TextGame2048 generates valid prompts."""
        game = TextGame2048(seed=42)
        prompt = game.get_prompt()
        assert "Current board state:" in prompt
        assert "Current score:" in prompt
        assert "Max tile:" in prompt
        assert "Valid moves:" in prompt

    def test_text_game_system_prompt(self):
        """System prompt contains game rules."""
        game = TextGame2048(seed=42)
        system = game.get_system_prompt()
        assert "4×4" in system or "4x4" in system
        assert "UP" in system
        assert "<think>" in system
        assert "<answer>" in system

    def test_text_game_step(self):
        """TextGame2048 correctly processes a move."""
        game = TextGame2048(seed=42)
        game.reset(seed=42)
        response = "<think>Moving up seems good.</think><answer>UP</answer>"
        result = game.step_from_response(response)
        assert "parsed" in result
        assert "score_delta" in result
        assert "done" in result
        assert "info" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

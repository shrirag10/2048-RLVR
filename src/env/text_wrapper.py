"""
Text-based 2048 wrapper for the LLM-RLVR agent.

Converts the 2048 board into structured text prompts suitable for
language model input, and parses XML-style model outputs
(<think>...</think><answer>DIRECTION</answer>).

This module serves as the bridge between the Game2048 engine and the
GRPO training pipeline where the LLM acts as a policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.env.game_2048 import Game2048


# ─── Prompt Templates ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert player of the puzzle game 2048.

GAME RULES:
1. The board is a 4×4 grid of numbered tiles (powers of 2) and empty cells (0).
2. Each turn, you choose one of four moves: UP, DOWN, LEFT, or RIGHT.
3. All tiles slide as far as possible in that direction.
4. If two adjacent tiles with the same value collide during a slide, they merge into one tile with double the value. Each merged pair can only merge once per move.
5. After each valid move, a new tile (2 with 90% probability, 4 with 10%) appears on a random empty cell.
6. The game ends when no valid moves remain (the board is full and no adjacent tiles can merge).
7. OBJECTIVE: Maximize your total score. The score increases by the value of each newly merged tile.

STRATEGY GUIDELINES:
- Keep your highest tile in a corner.
- Build a monotonic gradient along edges.
- Avoid moves that scatter high-value tiles.
- Prioritize merges that consolidate the board.

RESPONSE FORMAT — you MUST respond in EXACTLY this format:
<think>
[Analyze the current board state. Identify the highest tiles, possible merges,
empty spaces, and potential dangers. Reason about which move best advances
your position.]
</think>
<answer>[exactly one of: UP, DOWN, LEFT, RIGHT]</answer>"""


USER_PROMPT_TEMPLATE = """Current board state:
{board_text}

Current score: {score}
Max tile: {max_tile}
Valid moves: {valid_moves}

Choose your next move."""


# ─── Response Parsing ─────────────────────────────────────────────────

@dataclass
class ParsedResponse:
    """Parsed LLM response with thinking and action."""
    raw: str
    thinking: Optional[str]
    action: Optional[str]  # "UP", "DOWN", "LEFT", "RIGHT" or None if parse failed
    action_id: Optional[int]  # 0-3 or None
    format_valid: bool
    direction_valid: bool


def parse_llm_response(response: str) -> ParsedResponse:
    """
    Parse the XML-style LLM response.

    Extracts the <think>...</think> reasoning and <answer>...</answer> direction.
    Robust to whitespace variations and case differences.

    Args:
        response: Raw LLM output string.

    Returns:
        ParsedResponse with parsed components and validity flags.
    """
    raw = response.strip()

    # Extract thinking block
    think_match = re.search(
        r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE
    )
    thinking = think_match.group(1).strip() if think_match else None

    # Extract answer block
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL | re.IGNORECASE
    )

    action = None
    action_id = None
    direction_valid = False

    if answer_match:
        raw_action = answer_match.group(1).strip().upper()
        if raw_action in Game2048.NAME_TO_ACTION:
            action = raw_action
            action_id = Game2048.NAME_TO_ACTION[action]
            direction_valid = True

    # Format validity: both tags present
    format_valid = think_match is not None and answer_match is not None

    return ParsedResponse(
        raw=raw,
        thinking=thinking,
        action=action,
        action_id=action_id,
        format_valid=format_valid,
        direction_valid=direction_valid,
    )


# ─── Text Game Wrapper ────────────────────────────────────────────────

class TextGame2048:
    """
    Text-based 2048 game wrapper for LLM interaction.

    Provides methods to generate prompts from board states and
    execute parsed LLM responses, returning structured feedback.
    """

    def __init__(self, seed: Optional[int] = None):
        self.game = Game2048(seed=seed)

    def reset(self, seed: Optional[int] = None) -> str:
        """Reset game and return the initial prompt."""
        self.game.reset(seed=seed)
        return self.get_prompt()

    def get_system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return SYSTEM_PROMPT

    def get_prompt(self) -> str:
        """Generate a user prompt from the current board state."""
        board_text = self._board_to_text()
        valid_moves = ", ".join(
            Game2048.ACTION_NAMES[a] for a in self.game.get_valid_actions()
        )
        return USER_PROMPT_TEMPLATE.format(
            board_text=board_text,
            score=self.game.score,
            max_tile=self.game.max_tile,
            valid_moves=valid_moves,
        )

    def step_from_response(self, response: str) -> dict:
        """
        Parse an LLM response and execute the action.

        Args:
            response: Raw LLM output string.

        Returns:
            Dict with:
                - parsed: ParsedResponse object
                - score_delta: float
                - done: bool
                - info: dict from game.step()
                - next_prompt: str (prompt for next turn, if not done)
        """
        parsed = parse_llm_response(response)

        if parsed.action_id is None:
            # Invalid or unparseable response — treat as no-op
            return {
                "parsed": parsed,
                "score_delta": 0.0,
                "done": self.game.done,
                "info": {
                    "valid": False,
                    "max_tile": self.game.max_tile,
                    "total_score": self.game.score,
                    "merges": 0,
                },
                "next_prompt": self.get_prompt() if not self.game.done else None,
            }

        _, score_delta, done, info = self.game.step(parsed.action_id)

        return {
            "parsed": parsed,
            "score_delta": score_delta,
            "done": done,
            "info": info,
            "next_prompt": self.get_prompt() if not done else None,
        }

    def _board_to_text(self) -> str:
        """
        Convert board to a clean text representation.

        Format: [[2, 4, 8, 0], [0, 2, 4, 16], [2, 0, 0, 4], [0, 0, 2, 2]]
        """
        rows = self.game.to_list()
        row_strs = [str(row) for row in rows]
        return "[" + ", ".join(row_strs) + "]"

    @property
    def done(self) -> bool:
        return self.game.done

    @property
    def score(self) -> int:
        return self.game.score

    @property
    def max_tile(self) -> int:
        return self.game.max_tile


def generate_board_states(
    n: int = 1000,
    seed: int = 42,
    stage_distribution: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Generate diverse board states for GRPO training dataset.

    Creates board states at different game stages by playing random moves.

    Args:
        n: Number of board states to generate.
        seed: Base random seed.
        stage_distribution: Dict mapping stage to fraction, e.g.,
            {"early": 0.3, "mid": 0.4, "late": 0.3}.
            Stages are defined by max tile: early (<64), mid (64-512), late (>512).

    Returns:
        List of dicts with 'board', 'score', 'max_tile', 'system_prompt', 'user_prompt'.
    """
    if stage_distribution is None:
        stage_distribution = {"early": 0.3, "mid": 0.4, "late": 0.3}

    rng = np.random.default_rng(seed)
    states = []
    text_game = TextGame2048()
    max_attempts = n * 10  # Prevent infinite loops
    attempts = 0

    stage_counts = {
        stage: int(n * frac) for stage, frac in stage_distribution.items()
    }
    # Adjust for rounding
    remaining = n - sum(stage_counts.values())
    if remaining > 0:
        stage_counts["mid"] += remaining

    collected = {stage: 0 for stage in stage_counts}
    needed = dict(stage_counts)

    pbar = tqdm(total=n, desc="Generating board states", unit="state")

    while sum(collected.values()) < n and attempts < max_attempts:
        attempts += 1
        game_seed = int(rng.integers(0, 2**31))
        text_game.reset(seed=game_seed)

        # Play random moves to reach various game stages
        num_moves = int(rng.integers(1, 200))
        for _ in range(num_moves):
            valid = text_game.game.get_valid_actions()
            if not valid:
                break
            action = valid[int(rng.integers(0, len(valid)))]
            text_game.game.step(action)

        if text_game.done:
            continue

        # Classify stage
        mt = text_game.max_tile
        if mt < 64:
            stage = "early"
        elif mt <= 512:
            stage = "mid"
        else:
            stage = "late"

        if collected[stage] < needed.get(stage, 0):
            states.append({
                "board": text_game.game.to_list(),
                "score": text_game.score,
                "max_tile": mt,
                "system_prompt": text_game.get_system_prompt(),
                "user_prompt": text_game.get_prompt(),
            })
            collected[stage] += 1
            pbar.update(1)

    pbar.close()
    return states

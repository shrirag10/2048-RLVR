"""
GRPO Dataset Generation for 2048 LLM Training.

Generates diverse board states as a Hugging Face Dataset for use
with TRL's GRPOTrainer. Each row contains the formatted prompt
(system + user) and the board state metadata needed by the reward functions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from datasets import Dataset

from src.env.text_wrapper import TextGame2048, SYSTEM_PROMPT, generate_board_states


def create_grpo_dataset(
    n_states: int = 10_000,
    seed: int = 42,
    stage_distribution: Optional[dict[str, float]] = None,
) -> Dataset:
    """
    Create a Hugging Face Dataset of 2048 board states for GRPO training.

    Each row contains:
        - prompt: list of message dicts (system + user) for chat template
        - board_state: the 4×4 board as nested list (for reward verification)
        - score: current game score
        - max_tile: current max tile value

    Args:
        n_states: Number of board states to generate.
        seed: Random seed for reproducibility.
        stage_distribution: Distribution across game stages.

    Returns:
        Hugging Face Dataset ready for GRPOTrainer.
    """
    raw_states = generate_board_states(
        n=n_states, seed=seed, stage_distribution=stage_distribution
    )

    records = []
    for state in raw_states:
        # Format as chat messages for the model
        # Prefill assistant with <think> to kickstart CoT reasoning
        # (following DeepSeek-R1, TinyZero, and Open-R1 best practices)
        messages = [
            {"role": "system", "content": state["system_prompt"]},
            {"role": "user", "content": state["user_prompt"]},
            {"role": "assistant", "content": "<think>\n"},
        ]

        records.append({
            "prompt": messages,
            "board_state": state["board"],
            "score": state["score"],
            "max_tile": state["max_tile"],
        })

    dataset = Dataset.from_list(records)

    print(f"Created GRPO dataset: {len(dataset)} board states")
    print(f"  Stage distribution:")

    # Report stage stats
    early = sum(1 for r in records if r["max_tile"] < 64)
    mid = sum(1 for r in records if 64 <= r["max_tile"] <= 512)
    late = sum(1 for r in records if r["max_tile"] > 512)
    total = len(records)
    print(f"    Early (<64):   {early:>5} ({early/total:.1%})")
    print(f"    Mid (64-512):  {mid:>5} ({mid/total:.1%})")
    print(f"    Late (>512):   {late:>5} ({late/total:.1%})")

    return dataset

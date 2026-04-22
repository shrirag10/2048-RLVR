"""
Replay generator for the GRPO LLM agent.

Plays one game with the trained Qwen2.5 adapter, records every step in the
same replays.json format used by classical agents, and updates manifest.json.

Usage:
    python3 -m src.llm.replay_gen
    python3 -m src.llm.replay_gen --model-path logs/grpo/adapter --max-turns 500
"""

from __future__ import annotations

import argparse
import json
import os
import time

from src.env.text_wrapper import TextGame2048, parse_llm_response
from src.llm.predict import load_model, generate_move, fallback_parse_direction

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


def _pseudo_probs(chosen: int, valid: list[int]) -> list[float]:
    """
    Pseudo action probabilities for the LLM — the model outputs text, not logits.
    Assign 0.7 to the chosen action, split remainder across other valid actions.
    """
    probs = [0.0, 0.0, 0.0, 0.0]
    if not valid:
        return probs
    other_valid = [a for a in valid if a != chosen]
    probs[chosen] = 0.7
    if other_valid:
        rest = 0.3 / len(other_valid)
        for a in other_valid:
            probs[a] = rest
    return probs


def generate_llm_replay(
    model,
    tokenizer,
    seed: int = 0,
    max_turns: int = 500,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> tuple[list[dict], int, int]:
    """
    Play one game and record every step.

    Returns:
        (steps, final_score, max_tile)
    """
    game = TextGame2048(seed=seed)
    game.reset(seed=seed)
    system_prompt = game.get_system_prompt()

    steps = []
    consecutive_invalid = 0

    while not game.done and len(steps) < max_turns:
        board_snapshot = [row.tolist() for row in game.game.board]
        valid = game.game.get_valid_actions()

        user_prompt = game.get_prompt()
        response = generate_move(
            model, tokenizer, system_prompt, user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Parse action
        parsed = parse_llm_response(response)
        action_id = parsed.action_id

        # Fallback if XML parse failed
        if action_id is None:
            _, action_id = fallback_parse_direction(response)

        # If still no action or invalid, pick the first valid action
        if action_id is None or action_id not in valid:
            if valid:
                action_id = valid[0]
                consecutive_invalid += 1
            else:
                break  # game over
        else:
            consecutive_invalid = 0

        _, score_delta, done, info = game.game.step(action_id)

        steps.append(dict(
            board=board_snapshot,
            action=action_id,
            action_name=ACTION_NAMES[action_id],
            probs=_pseudo_probs(action_id, valid),
            score_delta=int(score_delta),
            total_score=int(game.score),
            max_tile=int(game.game.board.max()),
            valid_actions=valid,
        ))

        if consecutive_invalid >= 10:
            break

    final_score = game.score
    max_tile = int(game.game.board.max()) if steps else 0
    return steps, final_score, max_tile


def generate_replay(
    model_path: str = "logs/grpo/adapter",
    log_dir: str = "logs/grpo",
    manifest_path: str = "logs/manifest.json",
    max_turns: int = 500,
    seed: int = 42,
):
    print(f"\n{'='*55}")
    print(f"  GRPO LLM  →  generating replay")
    print(f"{'='*55}")

    model, tokenizer = load_model(model_path, max_seq_length=896)

    print(f"  Playing game (seed={seed}, max_turns={max_turns})...")
    from tqdm import tqdm
    steps, final_score, max_tile = generate_llm_replay(
        model, tokenizer, seed=seed, max_turns=max_turns,
    )
    print(f"  Done — score={final_score:,}  max_tile={max_tile}  moves={len(steps)}")

    # Build replays.json in same format as classical agents
    replay = {
        "agent": "grpo",
        "model_path": os.path.abspath(model_path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "avg_score": float(final_score),
            "max_score": int(final_score),
            "max_tile_ever": int(max_tile),
            "avg_moves": float(len(steps)),
        },
        "episodes": [
            {
                "episode": 1,
                "final_score": int(final_score),
                "max_tile": int(max_tile),
                "num_moves": len(steps),
                "steps": steps,
            }
        ],
    }

    os.makedirs(log_dir, exist_ok=True)
    replay_path = os.path.join(log_dir, "replays.json")
    with open(replay_path, "w") as f:
        json.dump(replay, f, separators=(",", ":"))
    print(f"  Saved: {replay_path}")

    # Update manifest.json
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"agents": [], "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    existing = {a["id"]: a for a in manifest.get("agents", [])}
    if "grpo" in existing:
        existing["grpo"]["replay"] = replay_path
        existing["grpo"]["replay_summary"] = replay["summary"]
    else:
        existing["grpo"] = {
            "id": "grpo",
            "name": "GRPO (Qwen2.5)",
            "checkpoint": os.path.abspath(model_path),
            "replay": replay_path,
            "replay_summary": replay["summary"],
        }

    manifest["agents"] = list(existing.values())
    manifest["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json updated  ({len(manifest['agents'])} agents)")
    print("  Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="logs/grpo/adapter")
    p.add_argument("--log-dir", default="logs/grpo")
    p.add_argument("--manifest", default="logs/manifest.json")
    p.add_argument("--max-turns", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate_replay(
        model_path=args.model_path,
        log_dir=args.log_dir,
        manifest_path=args.manifest,
        max_turns=args.max_turns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

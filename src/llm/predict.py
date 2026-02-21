"""
LLM Prediction / Evaluation for 2048 GRPO Agent.

Loads a trained Qwen2.5-0.5B GRPO model and plays complete 2048 games,
collecting metrics (score, max tile, move validity, reasoning quality).

Supports:
  - Adapter-based loading (LoRA weights on top of base model)
  - Merged model loading (full 16-bit weights)
  - Random baseline for comparison
  - CSV export and matplotlib plots via TrainingLogger

Usage:
  python -m src.llm.predict --model-path logs/grpo/adapter --num-games 20
  python -m src.llm.predict --baseline random --num-games 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import re

import numpy as np
import torch

from src.env.game_2048 import Game2048
from src.env.text_wrapper import TextGame2048, parse_llm_response, SYSTEM_PROMPT
from src.utils.metrics import TrainingLogger, EpisodeMetrics


# Direction name to action ID mapping
_NAME_TO_ACTION = {"UP": 0, "DOWN": 2, "LEFT": 3, "RIGHT": 1}


def fallback_parse_direction(text: str) -> tuple[str | None, int | None]:
    """
    Fallback parser: extract a direction from raw text when XML parsing fails.

    Checks:
    1. First word of the response (model often starts with the direction)
    2. Any standalone direction keyword in the text

    Returns:
        Tuple of (direction_name, action_id) or (None, None).
    """
    text = text.strip()
    if not text:
        return None, None

    # Check first word
    first_word = text.split()[0].strip(".,;:!?[](){}\"'").upper()
    if first_word in _NAME_TO_ACTION:
        return first_word, _NAME_TO_ACTION[first_word]

    # Search for standalone direction words (word boundary match)
    for direction in ["RIGHT", "LEFT", "DOWN", "UP"]:
        if re.search(rf'\b{direction}\b', text, re.IGNORECASE):
            return direction, _NAME_TO_ACTION[direction]

    return None, None


# ─── Model Loading ────────────────────────────────────────────────────

def load_model(
    model_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length: int = 512,
):
    """
    Load a trained GRPO model for inference.

    Uses Unsloth's native loading for both adapter and merged models.
    For adapters, loads directly via model_name=adapter_path which
    Unsloth handles natively (better than manual PeftModel + merge).

    Args:
        model_path: Path to adapter/ or merged/ directory.
        base_model: HuggingFace base model name (for adapter loading).
        max_seq_length: Maximum sequence length.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Disable torch.compile to avoid graph break issues with Unsloth
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    from unsloth import FastLanguageModel

    # Check if this is an adapter directory
    adapter_config = os.path.join(model_path, "adapter_config.json")
    is_adapter = os.path.exists(adapter_config)

    if is_adapter:
        print(f"Loading adapter weights from: {model_path}")
        print(f"Base model: {base_model}")

        # Load base model first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

        # Apply LoRA adapters (same config as training)
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Load trained adapter weights
        from peft import set_peft_model_state_dict
        import safetensors.torch

        adapter_weights = safetensors.torch.load_file(
            os.path.join(model_path, "adapter_model.safetensors")
        )
        set_peft_model_state_dict(model, adapter_weights)
        print("Adapter weights loaded successfully.")
    else:
        print(f"Loading merged model from: {model_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        print("Merged model loaded successfully.")

    # Switch to inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


# ─── LLM Inference ────────────────────────────────────────────────────

def generate_move(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate a move from the LLM given a board state prompt.

    Uses the chat template with add_generation_prompt=True so the
    model starts generating from the assistant turn. We do NOT
    include the <think> prefill at inference time — the model should
    produce it on its own (trained via GRPO to output this format).

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        system_prompt: System prompt with game rules.
        user_prompt: User prompt with current board state.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Generated text response from the model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # Prevent immediate EOS
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


# ─── Random Baseline ──────────────────────────────────────────────────

def play_random_game(seed: Optional[int] = None, max_turns: int = 500) -> dict:
    """
    Play a 2048 game with random moves as a baseline.

    Returns:
        Dict with game metrics.
    """
    game = TextGame2048(seed=seed)
    game.reset(seed=seed)
    num_moves = 0
    valid_moves = 0

    while not game.done and num_moves < max_turns:
        valid_actions = game.game.get_valid_actions()
        if not valid_actions:
            break
        action = valid_actions[np.random.randint(len(valid_actions))]
        game.game.step(action)
        num_moves += 1
        valid_moves += 1

    return {
        "score": game.score,
        "max_tile": game.max_tile,
        "num_moves": num_moves,
        "valid_moves": valid_moves,
        "invalid_moves": 0,
    }


# ─── LLM Game Loop ───────────────────────────────────────────────────

def play_llm_game(
    model,
    tokenizer,
    seed: Optional[int] = None,
    max_turns: int = 500,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    verbose: bool = False,
) -> dict:
    """
    Play a complete 2048 game using the LLM as the policy.

    Args:
        model: Loaded language model.
        tokenizer: Tokenizer.
        seed: Game seed for reproducibility.
        max_turns: Maximum turns per game.
        max_new_tokens: Max generation tokens per turn.
        temperature: Sampling temperature.
        verbose: Print turn-by-turn output.

    Returns:
        Dict with game metrics and sample responses.
    """
    game = TextGame2048(seed=seed)
    game.reset(seed=seed)
    system_prompt = game.get_system_prompt()

    num_moves = 0
    valid_moves = 0
    invalid_moves = 0
    consecutive_invalid = 0
    sample_responses = []

    while not game.done and num_moves < max_turns:
        user_prompt = game.get_prompt()

        # Generate move from LLM
        response = generate_move(
            model, tokenizer,
            system_prompt, user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Parse and execute — try XML parser first, fallback to raw direction
        parsed = parse_llm_response(response)

        # If XML parsing didn't find a valid direction, try fallback
        if parsed.action_id is None:
            fb_dir, fb_id = fallback_parse_direction(response)
            if fb_id is not None:
                # Execute directly via the game engine
                if game.game.is_valid_action(fb_id):
                    _, score_delta, done, info = game.game.step(fb_id)
                    num_moves += 1
                    valid_moves += 1
                    consecutive_invalid = 0

                    if len(sample_responses) < 3:
                        sample_responses.append({
                            "turn": num_moves,
                            "response": response[:500],
                            "action": fb_dir,
                            "format_valid": False,
                            "direction_valid": True,
                            "score_delta": score_delta,
                            "fallback": True,
                        })

                    if verbose:
                        print(
                            f"  Turn {num_moves:3d} | {fb_dir:>6s}* | "
                            f"Score: {game.score:>6d} | Max: {game.max_tile:>4d} | "
                            f"✓ (fallback)"
                        )
                        preview = response.replace('\n', ' ')[:120]
                        print(f"    └─ {preview}")
                    continue
                else:
                    # Direction found but move is invalid on this board
                    num_moves += 1
                    invalid_moves += 1
                    consecutive_invalid += 1

                    if len(sample_responses) < 3:
                        sample_responses.append({
                            "turn": num_moves,
                            "response": response[:500],
                            "action": fb_dir,
                            "format_valid": False,
                            "direction_valid": True,
                            "score_delta": 0.0,
                            "fallback": True,
                            "invalid_reason": "move_doesnt_change_board",
                        })

                    if verbose:
                        print(
                            f"  Turn {num_moves:3d} | {fb_dir:>6s}* | "
                            f"Score: {game.score:>6d} | Max: {game.max_tile:>4d} | "
                            f"✗ (invalid move)"
                        )
                        preview = response.replace('\n', ' ')[:120]
                        print(f"    └─ {preview}")

                    if consecutive_invalid >= 10:
                        if verbose:
                            print(f"  ⚠ Stuck: {consecutive_invalid} consecutive invalid moves. Ending game.")
                        break
                    continue

        # Standard XML-parsed path
        result = game.step_from_response(response)
        parsed = result["parsed"]
        num_moves += 1

        if result["info"].get("valid", False):
            valid_moves += 1
            consecutive_invalid = 0
        else:
            invalid_moves += 1
            consecutive_invalid += 1

        # Save first few responses for inspection
        if len(sample_responses) < 3:
            sample_responses.append({
                "turn": num_moves,
                "response": response[:500],  # Truncate for readability
                "action": parsed.action,
                "format_valid": parsed.format_valid,
                "direction_valid": parsed.direction_valid,
                "score_delta": result["score_delta"],
            })

        if verbose:
            action_str = parsed.action or "INVALID"
            print(
                f"  Turn {num_moves:3d} | {action_str:>6s} | "
                f"Score: {game.score:>6d} | Max: {game.max_tile:>4d} | "
                f"{'✓' if result['info'].get('valid') else '✗'}"
            )
            preview = response.replace('\n', ' ')[:120]
            print(f"    └─ {preview}")

        # Break if stuck in invalid move loop
        if consecutive_invalid >= 10:
            if verbose:
                print(f"  ⚠ Stuck: {consecutive_invalid} consecutive invalid moves. Ending game.")
            break

    return {
        "score": game.score,
        "max_tile": game.max_tile,
        "num_moves": num_moves,
        "valid_moves": valid_moves,
        "invalid_moves": invalid_moves,
        "sample_responses": sample_responses,
    }


# ─── Evaluation Runner ────────────────────────────────────────────────

def evaluate(
    model=None,
    tokenizer=None,
    num_games: int = 20,
    max_turns: int = 500,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    baseline: Optional[str] = None,
    output_dir: str = "logs/grpo",
    verbose: bool = False,
    seed: int = 42,
):
    """
    Run evaluation: play N games and collect metrics.

    Args:
        model: Loaded model (None if using baseline).
        tokenizer: Tokenizer (None if using baseline).
        num_games: Number of games to play.
        max_turns: Max turns per game.
        max_new_tokens: Max generation tokens per turn.
        temperature: Sampling temperature.
        baseline: Baseline mode ('random') or None for LLM.
        output_dir: Directory to save results.
        verbose: Print detailed output.
        seed: Base seed for reproducibility.
    """
    mode = baseline if baseline else "llm"
    experiment_name = f"eval_{mode}"
    logger = TrainingLogger(
        log_dir=output_dir,
        experiment_name=experiment_name,
    )

    rng = np.random.default_rng(seed)
    all_results = []

    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {mode.upper()} Agent — {num_games} games")
    print(f"  Max turns: {max_turns} | Output: {output_dir}")
    print(f"{'=' * 60}\n")

    total_start = time.time()

    for i in range(num_games):
        game_seed = int(rng.integers(0, 2**31))
        game_start = time.time()

        if baseline == "random":
            result = play_random_game(seed=game_seed, max_turns=max_turns)
        else:
            result = play_llm_game(
                model, tokenizer,
                seed=game_seed,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                verbose=verbose,
            )

        game_time = time.time() - game_start

        metrics = EpisodeMetrics(
            episode=i + 1,
            total_score=result["score"],
            max_tile=result["max_tile"],
            num_moves=result["num_moves"],
            valid_moves=result["valid_moves"],
            invalid_moves=result["invalid_moves"],
            wall_clock_seconds=game_time,
        )
        logger.log_episode(metrics)
        all_results.append(result)

        print(
            f"  Game {i+1:3d}/{num_games} | "
            f"Score: {result['score']:>6d} | "
            f"Max Tile: {result['max_tile']:>4d} | "
            f"Moves: {result['num_moves']:>3d} "
            f"(valid: {result['valid_moves']}, invalid: {result['invalid_moves']}) | "
            f"Time: {game_time:.1f}s"
        )

    total_time = time.time() - total_start

    # ─── Summary ──────────────────────────────────────────────────
    summary = logger.get_summary()

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY — {mode.upper()} Agent")
    print(f"{'=' * 60}")
    print(f"  Games played:     {num_games}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/num_games:.1f}s/game)")
    print(f"  Avg Score:        {summary.get('avg_score', 0):.1f} ± {summary.get('std_score', 0):.1f}")
    print(f"  Max Score:        {summary.get('max_score', 0):.0f}")
    print(f"  Avg Max Tile:     {summary.get('avg_max_tile', 0):.0f}")
    print(f"  Highest Tile:     {summary.get('max_tile_ever', 0)}")
    print(f"  Avg Moves/Game:   {summary.get('avg_moves', 0):.1f}")
    print(f"  Reach 512:        {summary.get('reach_512', 0):.1%}")
    print(f"  Reach 1024:       {summary.get('reach_1024', 0):.1%}")
    print(f"  Win (2048):       {summary.get('win_rate_2048', 0):.1%}")
    print(f"{'=' * 60}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, f"{experiment_name}_summary.json")
    summary["mode"] = mode
    summary["num_games"] = num_games
    summary["total_time_seconds"] = total_time
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved: {summary_path}")

    # Generate plots
    logger.plot_training_curves()

    # Save sample responses (LLM mode only)
    if mode == "llm" and all_results:
        samples_path = os.path.join(output_dir, "sample_responses.json")
        samples = []
        for r in all_results[:5]:  # First 5 games
            if "sample_responses" in r:
                samples.extend(r["sample_responses"])
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Sample responses saved: {samples_path}")

    return summary, all_results


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GRPO model on 2048 games"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to adapter/ or merged/ directory"
    )
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name for adapter loading"
    )
    parser.add_argument(
        "--baseline", type=str, default=None, choices=["random"],
        help="Run baseline mode instead of LLM"
    )
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="logs/grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    model = None
    tokenizer = None

    if args.baseline is None:
        if args.model_path is None:
            parser.error("--model-path is required when not using --baseline")
        model, tokenizer = load_model(
            args.model_path,
            base_model=args.base_model,
            max_seq_length=512,
        )

    evaluate(
        model=model,
        tokenizer=tokenizer,
        num_games=args.num_games,
        max_turns=args.max_turns,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        baseline=args.baseline,
        output_dir=args.output_dir,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

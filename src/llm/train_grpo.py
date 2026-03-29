"""
GRPO Training Pipeline for 2048 LLM Agent.

Trains Qwen2.5-0.5B-Instruct using Unsloth (QLoRA 4-bit) + TRL's GRPOTrainer
to play 2048. The model learns to reason about board states and select moves
through self-play with verifiable rewards.

Hardware target: NVIDIA RTX 4060 (6GB VRAM)
Expected VRAM usage: ~3-4GB (0.5B model in 4-bit + GRPO overhead)
"""

from __future__ import annotations

import argparse
import os
import json
import time
from typing import Optional

import numpy as np
import torch
import yaml

from src.llm.dataset import create_grpo_dataset
from src.llm.reward import (
    format_reward_fn,
    direction_reward_fn,
    game_reward_fn,
    thinking_quality_reward_fn,
    length_reward_fn,
)


def make_game_reward_adapter(dataset):
    """
    Create a reward function adapter that passes board states
    from the dataset to the game reward function.

    GRPOTrainer calls reward_fn(completions=..., prompts=..., ...)
    but our game reward needs board_states and scores from the dataset.
    This adapter bridges that gap.
    """
    # Pre-index the dataset by prompt content for fast lookup
    prompt_to_state = {}
    for i in range(len(dataset)):
        row = dataset[i]
        # Use the user message as key (unique per board state)
        key = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else str(row["prompt"])
        prompt_to_state[key] = {
            "board_state": row["board_state"],
            "score": row["score"],
        }

    def adapted_game_reward(completions, prompts=None, **kwargs):
        """
        Adapted game reward that looks up board states from prompts.
        """
        board_states = []
        scores = []

        if prompts is not None:
            for prompt in prompts:
                # Extract user message from prompt
                if isinstance(prompt, list):
                    user_msg = prompt[-1]["content"] if prompt else ""
                elif isinstance(prompt, str):
                    user_msg = prompt
                else:
                    user_msg = str(prompt)

                state = prompt_to_state.get(user_msg)
                if state:
                    board_states.append(state["board_state"])
                    scores.append(state["score"])
                else:
                    # Fallback: empty board
                    board_states.append([[0]*4]*4)
                    scores.append(0)
        else:
            board_states = [[[0]*4]*4] * len(completions)
            scores = [0] * len(completions)

        return game_reward_fn(
            completions,
            board_states=board_states,
            scores=scores,
        )

    return adapted_game_reward


def train_grpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_size: int = 10_000,
    num_generations: int = 8,
    max_completion_length: int = 256,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_seq_length: int = 1024,
    output_dir: str = "logs/grpo",
    seed: int = 42,
    config_path: Optional[str] = None,
    steps: Optional[int] = None,
):
    """
    Train the LLM agent using GRPO.

    This function:
    1. Loads Qwen2.5-0.5B-Instruct with Unsloth QLoRA 4-bit
    2. Generates diverse 2048 board states as training data
    3. Trains with GRPOTrainer using multi-component reward functions
    4. Saves the trained adapter weights

    Args:
        model_name: HuggingFace model identifier.
        dataset_size: Number of board states to generate.
        num_generations: GRPO group size G (8 fits in 6GB VRAM).
        max_completion_length: Max tokens for model response.
        learning_rate: Learning rate for GRPO.
        num_train_epochs: Number of training epochs.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        max_seq_length: Maximum sequence length for the model.
        output_dir: Directory for outputs and checkpoints.
        seed: Random seed.
        config_path: Optional YAML config file to override defaults.
        steps: Optional max training steps (overrides epochs).
    """
    # Collect all params into a dict so YAML overrides actually apply
    cfg = dict(
        model_name=model_name, dataset_size=dataset_size,
        num_generations=num_generations, max_completion_length=max_completion_length,
        learning_rate=learning_rate, num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r, lora_alpha=lora_alpha, max_seq_length=max_seq_length,
        output_dir=output_dir, seed=seed, steps=steps,
    )

    # Load config overrides — mutating cfg directly works correctly
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        unknown = set(overrides) - set(cfg)
        if unknown:
            print(f"  [warn] Unknown config keys ignored: {unknown}")
        cfg.update({k: v for k, v in overrides.items() if k in cfg})
        print(f"  [config] Loaded overrides from {config_path}: "
              f"{[k for k in overrides if k in cfg]}")

    # Unpack cfg back to named variables for readability below
    model_name             = cfg["model_name"]
    dataset_size           = cfg["dataset_size"]
    num_generations        = cfg["num_generations"]
    max_completion_length  = cfg["max_completion_length"]
    learning_rate          = cfg["learning_rate"]
    num_train_epochs       = cfg["num_train_epochs"]
    per_device_batch_size  = cfg["per_device_batch_size"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    lora_r                 = cfg["lora_r"]
    lora_alpha             = cfg["lora_alpha"]
    max_seq_length         = cfg["max_seq_length"]
    output_dir             = cfg["output_dir"]
    seed                   = cfg["seed"]
    steps                  = cfg["steps"]

    os.makedirs(output_dir, exist_ok=True)

    # Disable torch.compile to avoid graph break issues with Unsloth
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    print("=" * 60)
    print("  GRPO Training Pipeline — 2048 LLM Agent")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  LoRA r/α:        {lora_r}/{lora_alpha}")
    print(f"  Group size G:    {num_generations}")
    print(f"  Max completion:  {max_completion_length} tokens")
    print(f"  Dataset size:    {dataset_size}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Epochs:          {num_train_epochs}")
    print(f"  Output:          {output_dir}")
    print("=" * 60)

    # ─── Step 1: Generate dataset FIRST (before model load to save RAM) ─
    print("\n[1/4] Generating 2048 board state dataset...")

    dataset = create_grpo_dataset(n_states=dataset_size, seed=seed)

    # Free memory from dataset generation before loading model
    import gc
    gc.collect()

    # ─── Step 2: Load model with Unsloth ───────────────────────────
    print("\n[2/4] Loading model with Unsloth QLoRA 4-bit...")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # Auto-detect
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    print(f"  Model loaded. Trainable params: {model.print_trainable_parameters()}")

    # ─── Step 3: Configure GRPO Trainer ────────────────────────────
    print("\n[3/4] Configuring GRPOTrainer...")

    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=512,  # Prompts run ~400 tokens; give headroom
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=steps if steps else -1,
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        seed=seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",  # Use our custom logging
        optim="adamw_8bit",
        # Research-backed improvements:
        temperature=0.9,          # High temp for diverse completions (TinyZero uses 1.0)
        top_p=0.95,               # Nucleus sampling
        num_iterations=2,         # Reuse samples for efficiency (Open-R1: 2-4)
        # Reward weights: format > direction > length > game > thinking
        # Prioritize format learning first since everything depends on it
        reward_weights=[0.3, 0.3, 0.1, 0.2, 0.1],
    )

    # Create adapted game reward function
    game_reward_adapted = make_game_reward_adapter(dataset)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[
            format_reward_fn,
            direction_reward_fn,
            game_reward_adapted,
            thinking_quality_reward_fn,
            length_reward_fn,
        ],
    )

    # ─── Step 4: Train ─────────────────────────────────────────────
    print("\n[4/4] Training...")
    start_time = time.time()

    # Progress callback — tqdm bar + per-step reward/loss logging
    from transformers import TrainerCallback
    from tqdm import tqdm

    total_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else (
        len(dataset) // (per_device_batch_size * gradient_accumulation_steps * num_generations)
        * num_train_epochs
    )

    class GRPOProgressCallback(TrainerCallback):
        def __init__(self):
            self.bar = tqdm(total=total_steps, desc="  GRPO", unit="step",
                            dynamic_ncols=True, colour="green")
            self._log_path = os.path.join(output_dir, "train_log.jsonl")
            # clear previous log
            open(self._log_path, "w").close()

        def on_log(self, _args, state, _control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            loss    = logs.get("loss", logs.get("train_loss"))
            reward  = logs.get("reward", logs.get("rewards/mean"))
            fmt_r   = logs.get("rewards/format_reward_fn")
            dir_r   = logs.get("rewards/direction_reward_fn")
            elapsed = time.time() - start_time

            postfix = {}
            if loss   is not None: postfix["loss"]   = f"{loss:.4f}"
            if reward is not None: postfix["reward"] = f"{reward:.3f}"
            if fmt_r  is not None: postfix["fmt"]    = f"{fmt_r:.2f}"
            if dir_r  is not None: postfix["dir"]    = f"{dir_r:.2f}"
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(max(0, step - self.bar.n))

            # Append to jsonl log
            record = {"step": step, "elapsed": round(elapsed, 1), **{
                k: round(v, 5) if isinstance(v, float) else v
                for k, v in logs.items()
            }}
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        def on_train_end(self, _args, _state, _control, **kwargs):
            self.bar.close()

    trainer.add_callback(GRPOProgressCallback())

    trainer.train()

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    # ─── Save ──────────────────────────────────────────────────────
    print("\nSaving model and adapter weights...")
    model.save_pretrained(os.path.join(output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))

    # Save merged model for inference
    print("Saving merged model...")
    model.save_pretrained_merged(
        os.path.join(output_dir, "merged"),
        tokenizer,
        save_method="merged_16bit",
    )

    # Save training config
    config_out = {
        "model_name": model_name,
        "dataset_size": dataset_size,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "training_time_seconds": elapsed,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}/")
    print("  adapter/     — LoRA adapter weights")
    print("  merged/      — Full merged model (16-bit)")
    print("  training_config.json — Hyperparameters")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for 2048 LLM Agent")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset-size", type=int, default=10_000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--output-dir", default="logs/grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="YAML config file")

    args = parser.parse_args()

    train_grpo(
        model_name=args.model,
        dataset_size=args.dataset_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output_dir,
        seed=args.seed,
        config_path=args.config,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()

"""
Live Terminal Visualization for 2048 Agents.

Shows the game board updating in real-time as an AI agent makes moves.
Supports: DQN, PPO, LLM (GRPO), and random agents.

Usage:
  python -m src.visualize --agent random
  python -m src.visualize --agent dqn --model-path logs/dqn/dqn_step_499976.zip
  python -m src.visualize --agent ppo --model-path logs/ppo/ppo_final.zip
  python -m src.visualize --agent llm --model-path logs/grpo_long/adapter
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

from src.env.game_2048 import Game2048


# ─── Tile Colors ──────────────────────────────────────────────────────

TILE_COLORS = {
    0:     ("grey23",       "grey50"),
    2:     ("grey93",       "grey11"),
    4:     ("wheat1",       "grey11"),
    8:     ("dark_orange",  "white"),
    16:    ("orange_red1",  "white"),
    32:    ("red1",         "white"),
    64:    ("red3",         "white"),
    128:   ("yellow1",      "grey11"),
    256:   ("yellow2",      "grey11"),
    512:   ("gold1",        "grey11"),
    1024:  ("bright_green", "grey11"),
    2048:  ("green1",       "grey11"),
    4096:  ("cyan1",        "grey11"),
    8192:  ("deep_sky_blue1","white"),
    16384: ("magenta1",     "white"),
    32768: ("purple",       "white"),
    65536: ("bright_white", "grey11"),
}

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_ARROWS = {0: "⬆", 1: "➡", 2: "⬇", 3: "⬅"}


# ─── Board Rendering ─────────────────────────────────────────────────

def render_board(board: np.ndarray) -> Table:
    """Render the 4×4 board as a rich Table with colored tiles."""
    table = Table(
        show_header=False, show_edge=True, pad_edge=True,
        border_style="bright_white", padding=(0, 1),
        expand=False,
    )
    for _ in range(4):
        table.add_column(width=7, justify="center")

    for row in board:
        cells = []
        for val in row:
            val = int(val)
            bg, fg = TILE_COLORS.get(val, ("white", "grey11"))
            if val == 0:
                cells.append(Text("  ·  ", style=f"{fg} on {bg}"))
            else:
                label = f"{val:^5}"
                cells.append(Text(label, style=f"bold {fg} on {bg}"))
        table.add_row(*cells)

    return table


def build_display(
    game: Game2048,
    agent_name: str,
    turn: int,
    last_action: Optional[int],
    last_reasoning: str,
    valid_moves: int,
    invalid_moves: int,
    game_num: int,
    total_games: int,
    elapsed: float,
    game_over: bool = False,
) -> Panel:
    """Build the full display panel."""

    # Header stats
    validity = valid_moves / max(turn, 1) * 100
    header = Text()
    header.append(f"  Score: ", style="dim")
    header.append(f"{game.score:,}", style="bold bright_yellow")
    header.append(f"   Max Tile: ", style="dim")
    header.append(f"{game.max_tile}", style="bold bright_cyan")
    header.append(f"   Turn: ", style="dim")
    header.append(f"{turn}", style="bold")
    header.append(f"   Valid: ", style="dim")
    header.append(f"{validity:.0f}%", style="bold bright_green" if validity > 70 else "bold red")
    header.append(f"   Time: ", style="dim")
    header.append(f"{elapsed:.1f}s", style="dim")

    # Board
    board_table = render_board(game.board)

    # Last move
    move_text = Text()
    if last_action is not None:
        arrow = ACTION_ARROWS.get(last_action, "?")
        name = ACTION_NAMES.get(last_action, "?")
        move_text.append(f"  {arrow} ", style="bold bright_yellow")
        move_text.append(f"{name}", style="bold bright_white")
    elif game_over:
        move_text.append("  💀 GAME OVER", style="bold red")
    else:
        move_text.append("  ⏳ Thinking...", style="dim italic")

    # Reasoning (truncated)
    reason_text = Text()
    if last_reasoning:
        preview = last_reasoning.replace("\n", " ")[:100]
        reason_text.append(f"  💭 {preview}", style="dim italic")

    # Combine
    output = Text()
    output.append("\n")
    output.append_text(header)
    output.append("\n\n")

    title = f"🎮 2048 — {agent_name}   [{game_num}/{total_games}]"
    subtitle = None
    if game_over:
        subtitle = f"Final Score: {game.score:,}  |  Max Tile: {game.max_tile}  |  Moves: {turn}"

    # Build layout string
    lines = [str(header), ""]

    panel = Panel(
        Align.center(
            Text.from_ansi(
                str(header) + "\n\n" +
                "  (board renders above)\n\n" +
                str(move_text) + "\n" +
                str(reason_text)
            )
        ),
        title=title,
        border_style="bright_blue" if not game_over else "red",
    )

    return board_table, header, move_text, reason_text, title, subtitle, game_over


def make_panel(
    board_table: Table,
    header: Text,
    move_text: Text,
    reason_text: Text,
    title: str,
    subtitle: Optional[str],
    game_over: bool,
) -> Panel:
    """Assemble all parts into a single Panel for Live display."""
    from rich.console import Group

    parts = [header, Text(""), board_table, Text(""), move_text]
    if reason_text.plain.strip():
        parts.append(reason_text)

    group = Group(*parts)

    return Panel(
        group,
        title=title,
        subtitle=subtitle,
        border_style="bright_blue" if not game_over else "red",
        padding=(1, 2),
    )


# ─── Agent Loaders ───────────────────────────────────────────────────

def load_agent(agent_type: str, model_path: str = "", device: str = "auto"):
    """Load an agent and return a callable (state, game) -> (action, reasoning)."""

    def encode_board(board: np.ndarray) -> np.ndarray:
        """Convert 4x4 board to 16-channel binary observation."""
        obs = np.zeros((16, 4, 4), dtype=np.float32)
        obs[0] = (board == 0).astype(np.float32)
        for k in range(1, 16):
            obs[k] = (board == (1 << k)).astype(np.float32)
        return obs

    if agent_type == "random":
        def random_agent(state, game):
            valid = game.get_valid_actions()
            if not valid:
                return None, ""
            action = np.random.choice(valid)
            return action, f"Randomly chose {ACTION_NAMES[action]}"
        return random_agent, "Random Agent"

    elif agent_type == "dqn":
        from src.classical.dqn_agent import DQNAgent

        agent = DQNAgent(device=device)
        agent.load(model_path)
        agent.epsilon = 0.0

        def dqn_agent(state, game):
            obs = encode_board(game.board)
            valid = game.get_valid_actions()
            action = agent.select_action(obs, valid_actions=valid)
            return action, f"Q-network chose {ACTION_NAMES[action]} (ε=0)"
        return dqn_agent, "DQN Agent"

    elif agent_type == "ppo":
        from stable_baselines3 import PPO as SB3_PPO

        model = SB3_PPO.load(model_path, device=device)

        def ppo_agent(state, game):
            obs = encode_board(game.board)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            return action, f"PPO policy chose {ACTION_NAMES[action]}"
        return ppo_agent, "PPO Agent"

    elif agent_type == "a2c":
        from stable_baselines3 import A2C as SB3_A2C

        model = SB3_A2C.load(model_path, device=device)

        def a2c_agent(state, game):
            obs = encode_board(game.board)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            return action, f"A2C policy chose {ACTION_NAMES[action]}"
        return a2c_agent, "A2C Agent"

    elif agent_type == "qrdqn":
        from sb3_contrib import QRDQN

        model = QRDQN.load(model_path, device=device)

        def qrdqn_agent(state, game):
            obs = encode_board(game.board)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            return action, f"QR-DQN chose {ACTION_NAMES[action]} (50 quantiles)"
        return qrdqn_agent, "QR-DQN Agent"

    elif agent_type == "sac":
        from src.classical.sac_agent import DiscreteSACAgent

        agent = DiscreteSACAgent(device=device)
        agent.load(model_path)

        def sac_agent(state, game):
            obs = encode_board(game.board)
            valid = game.get_valid_actions()
            action = agent.select_action(obs, valid_actions=valid, deterministic=True)
            return action, f"SAC policy chose {ACTION_NAMES[action]} (α={agent.alpha:.3f})"
        return sac_agent, "SAC-Discrete Agent"

    elif agent_type == "llm":
        import re
        from src.env.text_wrapper import TextGame2048, SYSTEM_PROMPT
        from src.llm.predict import load_model, generate_move, fallback_parse_direction

        model, tokenizer = load_model(model_path)

        def llm_agent(state, game):
            # Build prompt from the game state
            text_game = TextGame2048.__new__(TextGame2048)
            text_game.game = game
            prompt = text_game.get_prompt()

            response = generate_move(
                model, tokenizer,
                SYSTEM_PROMPT, prompt,
                max_new_tokens=256,
                temperature=0.7,
            )

            # Parse direction
            from src.env.text_wrapper import parse_llm_response
            parsed = parse_llm_response(response)
            if parsed.action_id is not None:
                return parsed.action_id, response.replace("\n", " ")[:150]

            # Fallback
            fb_dir, fb_id = fallback_parse_direction(response)
            if fb_id is not None:
                return fb_id, response.replace("\n", " ")[:150]

            return None, response.replace("\n", " ")[:150]

        return llm_agent, f"LLM Agent (GRPO)"

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ─── Main Game Loop ──────────────────────────────────────────────────

def play_live(
    agent_fn,
    agent_name: str,
    num_games: int = 5,
    max_turns: int = 500,
    delay: float = 0.0,
    seed: int = 42,
):
    """Play games with live visualization."""
    console = Console()

    all_scores = []
    all_tiles = []

    for game_idx in range(num_games):
        game = Game2048(seed=seed + game_idx)
        game.reset()

        turn = 0
        valid_moves = 0
        invalid_moves = 0
        last_action = None
        last_reasoning = ""
        start_time = time.time()
        consecutive_invalid = 0

        with Live(console=console, refresh_per_second=10, screen=False) as live:
            # Initial board
            parts = build_display(
                game, agent_name, turn, None, "", 0, 0,
                game_idx + 1, num_games, 0.0,
            )
            live.update(make_panel(*parts))
            time.sleep(max(delay, 0.3))

            while not game.done and turn < max_turns:
                elapsed = time.time() - start_time

                # Get agent's move
                action, reasoning = agent_fn(None, game)

                if action is None:
                    # No valid action found
                    turn += 1
                    invalid_moves += 1
                    consecutive_invalid += 1
                    last_action = None
                    last_reasoning = reasoning
                else:
                    if game.is_valid_action(action):
                        game.step(action)
                        turn += 1
                        valid_moves += 1
                        consecutive_invalid = 0
                        last_action = action
                        last_reasoning = reasoning
                    else:
                        turn += 1
                        invalid_moves += 1
                        consecutive_invalid += 1
                        last_action = action
                        last_reasoning = f"[INVALID] {reasoning}"

                # Update display
                elapsed = time.time() - start_time
                parts = build_display(
                    game, agent_name, turn, last_action, last_reasoning,
                    valid_moves, invalid_moves,
                    game_idx + 1, num_games, elapsed,
                    game_over=game.done,
                )
                live.update(make_panel(*parts))

                if consecutive_invalid >= 10:
                    break

                if delay > 0:
                    time.sleep(delay)

            # Game over display
            elapsed = time.time() - start_time
            parts = build_display(
                game, agent_name, turn, last_action, last_reasoning,
                valid_moves, invalid_moves,
                game_idx + 1, num_games, elapsed,
                game_over=True,
            )
            live.update(make_panel(*parts))
            time.sleep(1.5)

        all_scores.append(game.score)
        all_tiles.append(game.max_tile)

        console.print(
            f"  Game {game_idx+1}/{num_games} | "
            f"Score: {game.score:,} | Max Tile: {game.max_tile} | "
            f"Moves: {turn} (valid: {valid_moves}, invalid: {invalid_moves}) | "
            f"Time: {elapsed:.1f}s",
            style="dim",
        )

    # Final summary
    console.print()
    console.rule("Summary", style="bright_blue")
    console.print(f"  Agent: [bold]{agent_name}[/bold]")
    console.print(f"  Games: {num_games}")
    console.print(f"  Avg Score: [bold bright_yellow]{np.mean(all_scores):,.0f}[/bold bright_yellow] ± {np.std(all_scores):,.0f}")
    console.print(f"  Max Score: [bold]{max(all_scores):,}[/bold]")
    console.print(f"  Avg Max Tile: [bold bright_cyan]{np.mean(all_tiles):,.0f}[/bold bright_cyan]")
    console.print(f"  Highest Tile: [bold]{max(all_tiles)}[/bold]")
    console.print()


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live terminal visualization for 2048 agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.visualize --agent random --num-games 3 --delay 0.1
  python -m src.visualize --agent dqn --model-path logs/dqn/dqn_step_499976
  python -m src.visualize --agent ppo --model-path logs/ppo/ppo_final
  python -m src.visualize --agent llm --model-path logs/grpo_long/adapter
        """,
    )
    parser.add_argument("--agent", type=str, required=True,
                        choices=["random", "dqn", "ppo", "a2c", "qrdqn", "sac", "llm"],
                        help="Agent type to visualize")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to model weights/checkpoint")
    parser.add_argument("--num-games", type=int, default=3,
                        help="Number of games to play (default: 3)")
    parser.add_argument("--max-turns", type=int, default=500,
                        help="Max turns per game (default: 500)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between moves in seconds (default: 0 for LLM, 0.15 for classical)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for model inference (default: auto)")

    args = parser.parse_args()

    # Auto-set delay for fast agents
    if args.delay == 0.0 and args.agent in ("random", "dqn", "ppo"):
        args.delay = 0.15  # Make it watchable

    agent_fn, agent_name = load_agent(
        args.agent, args.model_path, args.device
    )

    play_live(
        agent_fn, agent_name,
        num_games=args.num_games,
        max_turns=args.max_turns,
        delay=args.delay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


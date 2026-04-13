"""
Replay Generator for Track A agents.

Loads a trained checkpoint, runs N evaluation episodes while recording
every board state and action probabilities, then saves a replay JSON
that the frontend can load for live playback.

Usage (via train.py):
    python3 -m src.classical.train replay \\
        --agent dqn \\
        --model logs/dqn/dqn_final.pt \\
        --episodes 10

Output: logs/<agent>/replays.json
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from src.env.gym_wrapper import Gym2048Env

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-9)


# ── Per-agent episode runners ──────────────────────────────────────────────────

def _dqn_episode(agent, env: Gym2048Env) -> tuple[list[dict], dict]:
    """Run one DQN greedy episode and record every transition."""
    import torch

    obs, info = env.reset()
    steps: list[dict] = []
    done = False

    while not done:
        # Batched GPU forward pass for Q-values
        with torch.no_grad():
            st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q = agent.policy_net(st).squeeze().cpu().numpy()  # (4,)

        valid = info.get("valid_actions", list(range(4)))
        mq = q.copy()
        for i in range(4):
            if i not in valid:
                mq[i] = -1e9
        probs = _softmax(mq)
        action = int(np.argmax(mq))
        board_2d = [row.tolist() for row in env.game.board]  # 4×4 int list
        score_before = int(env.game.score)

        next_obs, _reward, term, trunc, next_info = env.step(action)
        done = term or trunc

        steps.append({
            "board": board_2d,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "probs": probs.tolist(),
            "score_delta": int(env.game.score) - score_before,
            "total_score": int(env.game.score),
            "max_tile": int(env.game.board.max()),
            "valid_actions": valid,
        })
        obs, info = next_obs, next_info

    return steps, next_info


def _sac_episode(agent, env: Gym2048Env) -> tuple[list[dict], dict]:
    """Run one Discrete-SAC greedy episode."""
    import torch

    obs, info = env.reset()
    steps: list[dict] = []
    done = False

    while not done:
        with torch.no_grad():
            st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action_probs, _, _ = agent.online(st)   # (1, 4)
        probs_np = action_probs.squeeze().cpu().numpy()

        valid = info.get("valid_actions", list(range(4)))
        mask = np.zeros(4)
        for i in valid:
            mask[i] = 1.0
        probs_np = probs_np * mask
        s = probs_np.sum()
        probs_np = probs_np / s if s > 0 else mask / mask.sum()

        action = int(np.argmax(probs_np))
        board_2d = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)

        next_obs, _reward, term, trunc, next_info = env.step(action)
        done = term or trunc

        steps.append({
            "board": board_2d,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "probs": probs_np.tolist(),
            "score_delta": int(env.game.score) - score_before,
            "total_score": int(env.game.score),
            "max_tile": int(env.game.board.max()),
            "valid_actions": valid,
        })
        obs, info = next_obs, next_info

    return steps, next_info


def _sb3_episode(model, env: Gym2048Env, algo: str) -> tuple[list[dict], dict]:
    """Run one SB3 (PPO / A2C / QR-DQN) greedy episode."""
    import torch

    obs, info = env.reset()
    steps: list[dict] = []
    done = False

    while not done:
        valid = info.get("valid_actions", list(range(4)))

        # Extract raw logits/q-values and mask invalid actions
        try:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                if algo == "qrdqn":
                    # SB3-contrib QR-DQN uses 'quantile_net' not 'q_net'
                    qnet = getattr(model.policy, 'quantile_net', None) or model.policy.q_net
                    q = qnet(obs_t).mean(dim=1).squeeze().cpu().numpy()
                    mask = np.full(4, -1e9); mask[valid] = q[valid]
                    action = int(np.argmax(mask))
                    probs = _softmax(q).tolist()
                else:
                    dist = model.policy.get_distribution(obs_t)
                    logits = dist.distribution.logits.squeeze().cpu().numpy()
                    mask = np.full(4, -1e9); mask[valid] = logits[valid]
                    action = int(np.argmax(mask))
                    probs = _softmax(logits).tolist()
        except Exception:
            action = int(model.predict(obs, deterministic=True)[0])
            probs = [0.25, 0.25, 0.25, 0.25]

        board_2d = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)

        next_obs, _reward, term, trunc, next_info = env.step(action)
        done = term or trunc

        steps.append({
            "board": board_2d,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "probs": probs,
            "score_delta": int(env.game.score) - score_before,
            "total_score": int(env.game.score),
            "max_tile": int(env.game.board.max()),
            "valid_actions": valid,
        })
        obs, info = next_obs, next_info

    return steps, next_info


def generate_replays(
    agent_type: str,
    model_path: str,
    target_tile: int = 0,
    max_attempts: int = 200,
    log_dir: str | None = None,
    seed: int = 0,
    output_path: str | None = None,
) -> str:
    """
    Generate a single best-episode replay for the given trained agent.

    Runs up to max_attempts episodes and saves the one that reaches
    target_tile (or the best tile achieved if target is never hit).

    Args:
        agent_type:   One of 'dqn', 'ppo', 'a2c', 'qrdqn', 'sac'.
        model_path:   Path to the saved checkpoint (.pt or .zip).
        target_tile:  Stop as soon as an episode hits this tile.
                      0 = run all max_attempts and keep the best.
        max_attempts: Maximum episodes to try before giving up.
        log_dir:      Directory to save replays.json (defaults to checkpoint dir).
        seed:         RNG seed for the first episode (incremented each attempt).
        output_path:  Full output path (overrides log_dir).

    Returns:
        Absolute path to the saved replays.json.
    """
    if log_dir is None:
        log_dir = os.path.dirname(os.path.abspath(model_path)) or f"logs/{agent_type}"
    if output_path is None:
        output_path = os.path.join(log_dir, "replays.json")

    agent_type = agent_type.lower()

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Replay Generator                        ║")
    print(f"║  Agent   : {agent_type:<30}║")
    print(f"║  Target  : {target_tile:<30}║")
    print(f"║  Max tries: {max_attempts:<29}║")
    print(f"╚══════════════════════════════════════════╝")

    # ── Load agent ────────────────────────────────────────────────────────────
    runner = None   # callable(env) -> (steps, info)

    if agent_type == "dqn":
        from src.classical.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(model_path)
        agent.epsilon = 0.05  # small noise breaks deterministic cycles at high tiles
        runner = lambda env: _dqn_episode(agent, env)

    elif agent_type == "sac":
        from src.classical.sac_agent import DiscreteSACAgent
        agent = DiscreteSACAgent()
        agent.load(model_path)
        runner = lambda env: _sac_episode(agent, env)

    elif agent_type == "ppo":
        from src.classical.ppo_agent import Game2048CNN
        _env0 = Gym2048Env(seed=seed)
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, env=_env0,
                                     custom_objects={"features_extractor_class": Game2048CNN})
        except Exception:
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=_env0,
                             custom_objects={"features_extractor_class": Game2048CNN})
        _env0.close()
        runner = lambda env: _sb3_episode(model, env, "ppo")

    elif agent_type == "a2c":
        from stable_baselines3 import A2C
        from src.classical.a2c_agent import Game2048CNN as A2CCNN
        _env0 = Gym2048Env(seed=seed)
        model = A2C.load(model_path, env=_env0,
                         custom_objects={"features_extractor_class": A2CCNN})
        _env0.close()
        runner = lambda env: _sb3_episode(model, env, "a2c")

    elif agent_type == "qrdqn":
        from sb3_contrib import QRDQN
        from src.classical.qrdqn_agent import Game2048CNN as QRDQNCNN
        _env0 = Gym2048Env(seed=seed)
        model = QRDQN.load(model_path, env=_env0,
                           custom_objects={"features_extractor_class": QRDQNCNN})
        _env0.close()
        runner = lambda env: _sb3_episode(model, env, "qrdqn")

    elif agent_type == "lfa":
        from src.classical.lfa_agent import LinearFAAgent, phi as lfa_phi

        agent = LinearFAAgent()
        agent.load(model_path)
        agent.epsilon = 0.05  # small noise breaks deterministic cycles at high tiles   # greedy evaluation

        def _lfa_episode(env: Gym2048Env) -> tuple[list[dict], dict]:
            obs, info = env.reset()
            steps: list[dict] = []
            done = False
            while not done:
                valid = info.get("valid_actions", list(range(4)))
                action, phi_s = agent.select_action(obs, valid, greedy=True)
                q_vals = agent.q_hat_all(phi_s)
                # Softmax over valid actions for display probabilities
                masked = np.full(4, -np.inf)
                masked[valid] = q_vals[valid]
                e = np.exp(masked - np.nanmax(masked[np.isfinite(masked)]))
                e[~np.isfinite(masked)] = 0.0
                probs = (e / (e.sum() + 1e-9)).tolist()

                board_2d = [row.tolist() for row in env.game.board]
                score_before = int(env.game.score)
                next_obs, _reward, term, trunc, next_info = env.step(action)
                done = term or trunc
                steps.append({
                    "board": board_2d,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "probs": probs,
                    "score_delta": int(env.game.score) - score_before,
                    "total_score": int(env.game.score),
                    "max_tile": int(env.game.board.max()),
                    "valid_actions": valid,
                })
                obs, info = next_obs, next_info
            return steps, next_info

        runner = _lfa_episode

    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'. "
                         "Choose from: dqn, ppo, a2c, qrdqn, sac, lfa")

    # ── Hunt for the best episode ─────────────────────────────────────────────
    best_steps: list | None = None
    best_tile = 0
    best_score = 0

    bar = tqdm(range(max_attempts), desc=f"{agent_type.upper():6}",
               unit="ep", dynamic_ncols=True, colour="green")
    for attempt in bar:
        env = Gym2048Env(seed=seed + attempt, max_steps=2000)
        try:
            steps, info = runner(env)
        finally:
            env.close()

        ep_tile  = steps[-1]["max_tile"]   if steps else 0
        ep_score = steps[-1]["total_score"] if steps else 0

        if ep_tile > best_tile or (ep_tile == best_tile and ep_score > best_score):
            best_steps = steps
            best_tile  = ep_tile
            best_score = ep_score

        bar.set_postfix(best_tile=best_tile, best_score=f"{best_score:,}",
                        cur_tile=ep_tile, refresh=False)

        if target_tile > 0 and best_tile >= target_tile:
            bar.write(f"  ✓ [{agent_type.upper()}] Target tile {target_tile} reached at attempt {attempt+1}")
            bar.close()
            break

    if not best_steps:
        raise RuntimeError(f"No valid episode produced for {agent_type}")

    # ── Build single-episode output ───────────────────────────────────────────
    ep_data = {
        "episode": 1,
        "final_score": best_score,
        "max_tile":    best_tile,
        "num_moves":   len(best_steps),
        "steps":       best_steps,
    }
    result: dict[str, Any] = {
        "agent":      agent_type,
        "model_path": os.path.abspath(model_path),
        "created":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "avg_score":     float(best_score),
            "max_score":     int(best_score),
            "max_tile_ever": best_tile,
            "avg_moves":     float(len(best_steps)),
        },
        "episodes": [ep_data],
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ Replay saved → {output_path}  ({size_kb:.1f} KB)")
    print(f"  Score : {best_score:,}")
    print(f"  Tile  : {best_tile}")
    print(f"  Moves : {len(best_steps)}")
    return output_path


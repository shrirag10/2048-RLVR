"""
hunt_2048.py — Run each trained agent until it reaches tile 2048.

Runs sequentially on GPU. Logs every move to logs/<agent>/hunt_2048.jsonl
and prints a tqdm progress bar per attempt.

Usage:
    python3 -m src.classical.hunt_2048
    python3 -m src.classical.hunt_2048 --agents dqn ppo   # subset
    python3 -m src.classical.hunt_2048 --max-attempts 500
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np
from tqdm import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")

from src.env.gym_wrapper import Gym2048Env

TARGET = 2048
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

AGENTS = [
    dict(id="dqn",   ckpt="logs/dqn/dqn_final.pt",      display="DQN"),
    dict(id="ppo",   ckpt="logs/ppo/ppo_final.zip",      display="PPO"),
    dict(id="a2c",   ckpt="logs/a2c/a2c_final.zip",      display="A2C"),
    dict(id="qrdqn", ckpt="logs/qrdqn/qrdqn_final.zip",  display="QR-DQN"),
    dict(id="sac",   ckpt="logs/sac/sac_final.pt",       display="SAC"),
    dict(id="lfa",   ckpt="logs/lfa/lfa_final.npz",      display="LFA"),
]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


# ── Agent loaders ─────────────────────────────────────────────────────────────

def _load_dqn(ckpt: str):
    from src.classical.dqn_agent import DQNAgent
    agent = DQNAgent()
    agent.load(ckpt)
    agent.epsilon = 0.05  # small noise to break deterministic cycles at post-1024 states
    return agent

def _load_sac(ckpt: str):
    from src.classical.sac_agent import DiscreteSACAgent
    agent = DiscreteSACAgent()
    agent.load(ckpt)
    return agent

def _load_ppo(ckpt: str):
    from stable_baselines3 import PPO
    from src.classical.ppo_agent import Game2048CNN
    env0 = Gym2048Env(seed=0)
    model = PPO.load(ckpt, env=env0, custom_objects={"features_extractor_class": Game2048CNN})
    env0.close()
    return model

def _load_a2c(ckpt: str):
    from stable_baselines3 import A2C
    from src.classical.a2c_agent import Game2048CNN
    env0 = Gym2048Env(seed=0)
    model = A2C.load(ckpt, env=env0, custom_objects={"features_extractor_class": Game2048CNN})
    env0.close()
    return model

def _load_qrdqn(ckpt: str):
    from sb3_contrib import QRDQN
    from src.classical.qrdqn_agent import Game2048CNN
    env0 = Gym2048Env(seed=0)
    model = QRDQN.load(ckpt, env=env0, custom_objects={"features_extractor_class": Game2048CNN})
    env0.close()
    return model

def _load_lfa(ckpt: str):
    from src.classical.lfa_agent import LinearFAAgent
    agent = LinearFAAgent()
    agent.load(ckpt)
    agent.epsilon = 0.0   # greedy at eval time
    return agent


# ── Episode runners ───────────────────────────────────────────────────────────

def _run_dqn(agent, env: Gym2048Env) -> tuple[list[dict], int]:
    import torch
    obs, info = env.reset()
    steps, done = [], False
    while not done:
        valid = info.get("valid_actions", list(range(4)))
        with torch.no_grad():
            st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            q = agent.policy_net(st).squeeze().cpu().numpy()
        mask = np.full(4, -1e9); mask[valid] = q[valid]
        action = int(np.argmax(mask))
        probs = _softmax(q)
        board = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps.append(dict(board=board, action=action, action_name=ACTION_NAMES[action],
                          probs=probs, score_delta=int(env.game.score)-score_before,
                          total_score=int(env.game.score), max_tile=int(env.game.board.max()),
                          valid_actions=valid))
    return steps, steps[-1]["max_tile"] if steps else 0


def _run_lfa(agent, env: Gym2048Env) -> tuple[list[dict], int]:
    obs, info = env.reset()
    steps, done = [], False
    while not done:
        valid = info.get("valid_actions", list(range(4)))
        greedy = np.random.random() > 0.05
        action, phi_s = agent.select_action(obs, valid, greedy=greedy)
        q_vals = agent.q_hat_all(phi_s)
        masked = np.full(4, -np.inf); masked[valid] = q_vals[valid]
        e = np.exp(masked - np.nanmax(masked[np.isfinite(masked)]))
        e[~np.isfinite(masked)] = 0.0
        probs = (e / (e.sum() + 1e-9)).tolist()
        board = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps.append(dict(board=board, action=action, action_name=ACTION_NAMES[action],
                          probs=probs, score_delta=int(env.game.score)-score_before,
                          total_score=int(env.game.score), max_tile=int(env.game.board.max()),
                          valid_actions=valid))
    return steps, steps[-1]["max_tile"] if steps else 0


def _run_sac(agent, env: Gym2048Env) -> tuple[list[dict], int]:
    import torch
    obs, info = env.reset()
    steps, done = [], False
    while not done:
        valid = info.get("valid_actions", list(range(4)))
        with torch.no_grad():
            st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            probs_t, _, _ = agent.online(st)
        p = probs_t.squeeze().cpu().numpy()
        mask = np.zeros(4); mask[valid] = p[valid]
        s = mask.sum(); mask = mask / s if s > 0 else np.ones(4) / 4
        # Sample from distribution — SAC is stochastic by design
        action = int(np.random.choice(4, p=mask))
        board = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps.append(dict(board=board, action=action, action_name=ACTION_NAMES[action],
                          probs=mask.tolist(), score_delta=int(env.game.score)-score_before,
                          total_score=int(env.game.score), max_tile=int(env.game.board.max()),
                          valid_actions=valid))
    return steps, steps[-1]["max_tile"] if steps else 0


def _run_sb3(model, env: Gym2048Env, algo: str) -> tuple[list[dict], int]:
    import torch
    obs, info = env.reset()
    steps, done = [], False
    while not done:
        valid = info.get("valid_actions", list(range(4)))
        try:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                if algo == "qrdqn":
                    q = model.policy.q_net(obs_t).mean(dim=1).squeeze().cpu().numpy()
                    mask = np.full(4, -1e9); mask[valid] = q[valid]
                    # epsilon-greedy to break cycles
                    if np.random.random() < 0.05:
                        action = int(np.random.choice(valid))
                    else:
                        action = int(np.argmax(mask))
                    probs = _softmax(q)
                else:
                    dist = model.policy.get_distribution(obs_t)
                    logits = dist.distribution.logits.squeeze().cpu().numpy()
                    # Sample from policy distribution (PPO/A2C are stochastic policies)
                    valid_logits = logits[valid]
                    valid_probs = _softmax(valid_logits)
                    action = int(np.random.choice(valid, p=valid_probs))
                    probs = _softmax(logits)
        except Exception:
            action = int(model.predict(obs, deterministic=True)[0])
            probs = [0.25, 0.25, 0.25, 0.25]

        board = [row.tolist() for row in env.game.board]
        score_before = int(env.game.score)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps.append(dict(board=board, action=action, action_name=ACTION_NAMES[action],
                          probs=probs, score_delta=int(env.game.score)-score_before,
                          total_score=int(env.game.score), max_tile=int(env.game.board.max()),
                          valid_actions=valid))
    return steps, steps[-1]["max_tile"] if steps else 0


# ── Main hunt loop ────────────────────────────────────────────────────────────

def hunt(agent_id: str, ckpt: str, display: str, log_dir: str):
    print(f"\n{'═'*55}")
    print(f"  {display}  →  hunting tile {TARGET}  (unlimited attempts)")
    print(f"{'═'*55}")

    # Load model
    loaders = dict(dqn=_load_dqn, sac=_load_sac, ppo=_load_ppo,
                   a2c=_load_a2c, qrdqn=_load_qrdqn, lfa=_load_lfa)
    runners = dict(dqn=_run_dqn, sac=_run_sac, lfa=_run_lfa,
                   ppo=lambda m, e: _run_sb3(m, e, "ppo"),
                   a2c=lambda m, e: _run_sb3(m, e, "a2c"),
                   qrdqn=lambda m, e: _run_sb3(m, e, "qrdqn"))

    print(f"  Loading checkpoint: {ckpt}")
    model = loaders[agent_id](ckpt)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hunt_2048.jsonl")
    # Clear previous log for this agent
    open(log_path, "w").close()

    best_tile = 0
    best_score = 0
    best_steps: list = []
    t0 = time.time()

    bar = tqdm(desc=f"  {display:6}", unit="ep", dynamic_ncols=True, colour="cyan")

    for attempt in itertools.count():
        env = Gym2048Env(seed=attempt, max_steps=5000)
        try:
            steps, tile = runners[agent_id](model, env)
        finally:
            env.close()

        score = steps[-1]["total_score"] if steps else 0
        if tile > best_tile or (tile == best_tile and score > best_score):
            best_tile = tile
            best_score = score
            best_steps = steps

        bar.update(1)
        bar.set_postfix(best=best_tile, score=f"{best_score:,}", cur=tile, attempt=attempt+1, refresh=False)

        # Log every move of this attempt to jsonl
        record = dict(attempt=attempt + 1, max_tile=tile, final_score=score,
                      num_moves=len(steps), steps=steps)
        with open(log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        if best_tile >= TARGET:
            bar.write(f"\n  ✓ {display} reached {TARGET}! "
                      f"score={best_score:,}  moves={len(steps)}  attempt={attempt+1}")
            bar.close()
            break

    elapsed = time.time() - t0

    # ── Write hunt_replays.json (separate from normal replays.json) ───────────
    hunt_replay_path = os.path.join(log_dir, "hunt_replays.json")
    replay = {
        "agent":      agent_id,
        "model_path": os.path.abspath(ckpt),
        "mode":       "hunt",
        "target":     TARGET,
        "created":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "avg_score":     float(best_score),
            "max_score":     int(best_score),
            "max_tile_ever": best_tile,
            "avg_moves":     float(len(best_steps)),
        },
        "episodes": [{
            "episode":     1,
            "final_score": best_score,
            "max_tile":    best_tile,
            "num_moves":   len(best_steps),
            "steps":       best_steps,
        }],
    }
    with open(hunt_replay_path, "w") as f:
        json.dump(replay, f, separators=(",", ":"))

    print(f"  Logged       → {log_path}  ({os.path.getsize(log_path)//1024} KB)")
    print(f"  Hunt replay  → {hunt_replay_path}  ({os.path.getsize(hunt_replay_path)//1024} KB)  [{elapsed:.0f}s]\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=[a["id"] for a in AGENTS],
                   choices=[a["id"] for a in AGENTS],
                   help="Agents to run (default: all). lfa only available after training.")
    p.add_argument("--log-root", default="logs", help="Root log directory")
    args = p.parse_args()

    selected = [a for a in AGENTS if a["id"] in args.agents]
    print(f"\n  Hunting tile {TARGET} with {len(selected)} agent(s): "
          f"{', '.join(a['display'] for a in selected)}")

    for agent in selected:
        hunt(
            agent_id=agent["id"],
            ckpt=agent["ckpt"],
            display=agent["display"],
            log_dir=os.path.join(args.log_root, agent["id"]),
        )

    # ── Update manifest.json — merge hunt_replay into existing entries ────────
    manifest_path = os.path.join(args.log_root, "manifest.json")
    # Load existing manifest if present (preserves normal replay entries)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"agents": [], "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    display_names = {a["id"]: a["display"] for a in AGENTS}
    existing = {a["id"]: a for a in manifest.get("agents", [])}

    for agent in AGENTS:
        hunt_replay_path = os.path.join(args.log_root, agent["id"], "hunt_replays.json")
        if not os.path.exists(hunt_replay_path):
            continue
        with open(hunt_replay_path) as f:
            hunt_summary = json.load(f).get("summary", {})

        if agent["id"] in existing:
            # Patch hunt_replay into the existing entry — don't overwrite replay
            existing[agent["id"]]["hunt_replay"]         = hunt_replay_path
            existing[agent["id"]]["hunt_replay_summary"] = hunt_summary
        else:
            # Agent has no normal replay yet — add a minimal entry
            existing[agent["id"]] = {
                "id":                 agent["id"],
                "name":               display_names[agent["id"]],
                "checkpoint":         os.path.abspath(agent["ckpt"]),
                "hunt_replay":         hunt_replay_path,
                "hunt_replay_summary": hunt_summary,
            }

    manifest["agents"] = list(existing.values())
    manifest["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json updated → {len(manifest['agents'])} agent(s)")
    print("  All done.")

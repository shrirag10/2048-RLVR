"""
scaling_eval.py — Evaluate checkpoints at milestone steps to produce scaling curves.

Loads each agent's checkpoint at 200k, 500k, 1M, 2M, 5M steps (where available),
runs N evaluation episodes, and saves results to logs/scaling_curves.json.

Usage:
    python3 -m src.classical.scaling_eval
    python3 -m src.classical.scaling_eval --episodes 50
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from src.env.gym_wrapper import Gym2048Env

MILESTONES = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]

AGENTS = {
    "dqn":   {"ext": ".pt",   "prefix": "dqn_step_"},
    "ppo":   {"ext": ".zip",  "prefix": "ppo_step_"},
    "a2c":   {"ext": ".zip",  "prefix": "a2c_step_"},
    "qrdqn": {"ext": ".zip",  "prefix": "qrdqn_step_"},
    "sac":   {"ext": ".pt",   "prefix": "sac_step_"},
    "lfa":   {"ext": ".npz",  "prefix": "lfa_step_"},
}


def _find_closest_checkpoint(agent_dir: str, agent_id: str, target_step: int) -> tuple[str, int] | None:
    """Find the checkpoint closest to target_step (within 20% tolerance)."""
    info = AGENTS[agent_id]
    prefix, ext = info["prefix"], info["ext"]

    best_path, best_step, best_diff = None, None, float("inf")
    for f in os.listdir(agent_dir):
        if f.startswith(prefix) and f.endswith(ext):
            step = int(f.replace(prefix, "").replace(ext, ""))
            diff = abs(step - target_step)
            if diff < best_diff:
                best_diff = diff
                best_step = step
                best_path = os.path.join(agent_dir, f)

    if best_path and best_diff <= target_step * 0.25:
        return best_path, best_step
    return None


def _load_agent(agent_id: str, ckpt_path: str):
    """Load agent from checkpoint."""
    if agent_id == "dqn":
        from src.classical.dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.load(ckpt_path)
        agent.epsilon = 0.0
        return agent
    elif agent_id == "sac":
        from src.classical.sac_agent import DiscreteSACAgent
        agent = DiscreteSACAgent()
        agent.load(ckpt_path)
        return agent
    elif agent_id == "lfa":
        from src.classical.lfa_agent import LinearFAAgent
        agent = LinearFAAgent()
        agent.load(ckpt_path)
        return agent
    elif agent_id in ("ppo", "a2c", "qrdqn"):
        import torch
        if agent_id == "ppo":
            from sb3_contrib import MaskablePPO
            from src.classical.ppo_agent import Game2048CNN
            env0 = Gym2048Env(seed=0)
            model = MaskablePPO.load(ckpt_path, env=env0,
                                     custom_objects={"features_extractor_class": Game2048CNN})
            env0.close()
            return model
        elif agent_id == "a2c":
            from stable_baselines3 import A2C
            from src.classical.a2c_agent import Game2048CNN
            env0 = Gym2048Env(seed=0)
            model = A2C.load(ckpt_path, env=env0,
                             custom_objects={"features_extractor_class": Game2048CNN})
            env0.close()
            return model
        elif agent_id == "qrdqn":
            from sb3_contrib import QRDQN
            from src.classical.qrdqn_agent import Game2048CNN
            env0 = Gym2048Env(seed=0)
            model = QRDQN.load(ckpt_path, env=env0,
                               custom_objects={"features_extractor_class": Game2048CNN})
            env0.close()
            return model
    return None


def _run_episode(agent, agent_id: str, seed: int) -> dict:
    """Run one evaluation episode and return metrics."""
    import torch
    env = Gym2048Env(seed=seed, max_steps=3000)
    obs, info = env.reset()
    done = False
    moves = 0

    while not done:
        valid = info.get("valid_actions", list(range(4)))

        if agent_id == "dqn":
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                q = agent.policy_net(st).squeeze().cpu().numpy()
            mask = np.full(4, -1e9)
            mask[valid] = q[valid]
            action = int(np.argmax(mask))
        elif agent_id == "sac":
            action = agent.select_action(obs, valid_actions=valid, deterministic=True)
        elif agent_id == "lfa":
            action, _ = agent.select_action(obs, valid, greedy=True)
        elif agent_id in ("ppo", "a2c", "qrdqn"):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                if agent_id == "qrdqn":
                    q = agent.policy.q_net(obs_t).mean(dim=1).squeeze().cpu().numpy()
                    mask = np.full(4, -1e9)
                    mask[valid] = q[valid]
                    action = int(np.argmax(mask))
                else:
                    action = int(agent.predict(obs, deterministic=True)[0])
        else:
            action = np.random.choice(valid)

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        moves += 1

    score = int(env.game.score)
    max_tile = int(env.game.board.max())
    env.close()
    return {"score": score, "max_tile": max_tile, "moves": moves}


def evaluate_scaling(
    log_root: str = "logs",
    n_episodes: int = 30,
    agents: list[str] | None = None,
):
    """Evaluate all agents at milestone checkpoints."""
    if agents is None:
        agents = list(AGENTS.keys())

    results = {}

    for agent_id in agents:
        agent_dir = os.path.join(log_root, agent_id)
        if not os.path.isdir(agent_dir):
            print(f"  [{agent_id}] no directory — skipping")
            continue

        results[agent_id] = []
        print(f"\n{'='*50}")
        print(f"  {agent_id.upper()} — scaling evaluation")
        print(f"{'='*50}")

        for milestone in MILESTONES:
            match = _find_closest_checkpoint(agent_dir, agent_id, milestone)
            if match is None:
                continue

            ckpt_path, actual_step = match
            print(f"  Step {actual_step:>10,} ({ckpt_path.split('/')[-1]})")

            try:
                agent = _load_agent(agent_id, ckpt_path)
            except Exception as e:
                print(f"    ✗ load failed: {e}")
                continue

            scores, tiles, moves_list = [], [], []
            bar = tqdm(range(n_episodes), desc=f"    eval", unit="ep",
                       dynamic_ncols=True, leave=False)
            for ep in bar:
                r = _run_episode(agent, agent_id, seed=ep * 7 + 42)
                scores.append(r["score"])
                tiles.append(r["max_tile"])
                moves_list.append(r["moves"])
                bar.set_postfix(avg=f"{np.mean(scores):.0f}", top=max(tiles))

            entry = {
                "step": actual_step,
                "avg_score": float(np.mean(scores)),
                "max_score": int(max(scores)),
                "avg_max_tile": float(np.mean(tiles)),
                "max_tile": int(max(tiles)),
                "reach_256": float(np.mean([t >= 256 for t in tiles])),
                "reach_512": float(np.mean([t >= 512 for t in tiles])),
                "reach_1024": float(np.mean([t >= 1024 for t in tiles])),
                "avg_moves": float(np.mean(moves_list)),
            }
            results[agent_id].append(entry)
            print(f"    avg={entry['avg_score']:.0f}  top={entry['max_tile']}  "
                  f"512+={entry['reach_512']:.0%}  1024+={entry['reach_1024']:.0%}")

            # Free GPU memory
            del agent
            import gc; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

    # Save results
    output_path = os.path.join(log_root, "scaling_curves.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {output_path}")
    return output_path


def main():
    p = argparse.ArgumentParser(description="Evaluate checkpoints at scaling milestones")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--agents", nargs="+", default=None,
                   choices=list(AGENTS.keys()))
    args = p.parse_args()

    evaluate_scaling(
        log_root=args.log_dir,
        n_episodes=args.episodes,
        agents=args.agents,
    )


if __name__ == "__main__":
    main()

"""
export_all.py — parallel GPU replay export + manifest writer.

Launches all agents simultaneously in separate processes so GPU inference
runs in parallel. Each process has its own timeout; failures never block others.

Usage:
    python3 -m src.classical.export_all
    python3 -m src.classical.export_all --episodes 5 --timeout 180
    python3 -m src.classical.export_all --force   # regenerate all replays
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time

# (checkpoint filename, display name) relative to logs/<agent>/
AGENT_CKPTS = {
    "dqn":   ("dqn_final.pt",    "DQN"),
    "ppo":   ("ppo_final.zip",   "PPO"),
    "a2c":   ("a2c_final.zip",   "A2C"),
    "qrdqn": ("qrdqn_final.zip", "QR-DQN"),
    "sac":   ("sac_final.pt",    "SAC"),
    "lfa":   ("lfa_final.npz",   "LFA"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_summary(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    scores = [float(r["total_score"]) for r in rows]
    tiles  = [int(r["max_tile"])      for r in rows]
    moves  = [int(r["num_moves"])     for r in rows]
    return {
        "n_episodes":    len(rows),
        "avg_score":     round(sum(scores) / len(scores), 1),
        "max_score":     max(scores),
        "max_tile_ever": max(tiles),
        "avg_moves":     round(sum(moves)  / len(moves),  1),
        "reach_512":     round(sum(t >= 512  for t in tiles) / len(tiles), 4),
        "reach_1024":    round(sum(t >= 1024 for t in tiles) / len(tiles), 4),
    }


def _run_one_agent(agent_id: str, ckpt_path: str, replay_path: str,
                   target_tile: int, timeout: int) -> tuple[str, bool, str]:
    """
    Spawn a child process that generates the replay for one agent.
    Returns (agent_id, success, message).
    """
    script = (
        "from src.classical.replay_gen import generate_replays; "
        f"generate_replays('{agent_id}', r'{ckpt_path}', "
        f"target_tile={target_tile}, output_path=r'{replay_path}')"
    )
    try:
        # stdout/stderr=None → child writes tqdm bars directly to terminal
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=None, stderr=None,
        )
        proc.wait(timeout=timeout)
        if proc.returncode == 0 and os.path.exists(replay_path):
            kb = os.path.getsize(replay_path) // 1024
            return agent_id, True, f"✓ done ({kb} KB)"
        return agent_id, False, f"✗ process exited {proc.returncode}"
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        return agent_id, False, f"✗ timed out after {timeout}s"
    except Exception as e:
        return agent_id, False, f"✗ {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def scan_and_export(log_root: str = "logs", timeout: int = 180, mode: str = None) -> str:
    os.makedirs(log_root, exist_ok=True)
    suffix = f"_{mode}" if mode else ""

    # ── Determine which agents need replay generation ────────────────────────
    to_generate: list[tuple] = []   # (agent_id, ckpt_path, replay_path, target_tile)
    already_done: list[str]  = []

    for agent_id, (ckpt_file, _) in AGENT_CKPTS.items():
        agent_dir   = os.path.join(log_root, f"{agent_id}{suffix}")
        ckpt_path   = os.path.join(agent_dir, ckpt_file)
        replay_path = os.path.join(agent_dir, "replays.json")

        if not os.path.exists(ckpt_path):
            print(f"  [{agent_id:6}] ✗ no checkpoint ({ckpt_file}) — skipping")
            continue
        if os.path.exists(replay_path):
            kb = os.path.getsize(replay_path) // 1024
            print(f"  [{agent_id:6}] ✓ replay exists ({kb} KB)")
            already_done.append(agent_id)
        else:
            # Use training CSV max_tile as target; default 1024 for DQN
            summary = _csv_summary(os.path.join(agent_dir, f"{agent_id}_metrics.csv"))
            if agent_id == "dqn":
                target_tile = 1024
            else:
                target_tile = int(summary.get("max_tile_ever", 0))
            print(f"  [{agent_id:6}] target_tile={target_tile}")
            to_generate.append((agent_id, ckpt_path, replay_path, target_tile))

    # ── Run agents sequentially on GPU ──────────────────────────────────────
    if to_generate:
        print(f"\n  Running {len(to_generate)} agent(s) sequentially on GPU (timeout={timeout}s each)…\n")
        for aid, cp, rp, tt in to_generate:
            _, ok, msg = _run_one_agent(aid, cp, rp, tt, timeout)
            print(f"  [{aid:6}] {msg}")
        print()

    # ── Build manifest from all agents that now have a replay ────────────────
    manifest: dict = {"agents": [], "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    mode_label = f" ({mode.upper()})" if mode else ""
    for agent_id, (ckpt_file, display_name) in AGENT_CKPTS.items():
        agent_dir   = os.path.join(log_root, f"{agent_id}{suffix}")
        ckpt_path   = os.path.join(agent_dir, ckpt_file)
        replay_path = os.path.join(agent_dir, "replays.json")

        if not os.path.exists(replay_path):
            continue

        replay_summary: dict = {}
        try:
            with open(replay_path) as f:
                replay_summary = json.load(f).get("summary", {})
        except Exception:
            pass

        training = _csv_summary(os.path.join(agent_dir, f"{agent_id}_metrics.csv"))
        manifest_id = f"{agent_id}_{mode}" if mode else agent_id

        manifest["agents"].append({
            "id":             manifest_id,
            "name":           f"{display_name}{mode_label}",
            "checkpoint":     ckpt_path,
            "replay":         replay_path,
            "replay_summary": replay_summary,
            "training":       training,
        })
        print(f"  [{agent_id:6}] → manifest  "
              f"avg={training.get('avg_score','?')}  "
              f"top={training.get('max_tile_ever','?')}")

    manifest_name = f"manifest_{mode}.json" if mode else "manifest.json"
    manifest_path = os.path.join(log_root, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ manifest.json written — {len(manifest['agents'])} agent(s)")
    return manifest_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export trained agents for frontend showcase")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--timeout", type=int, default=180, help="Seconds per agent (wall clock)")
    p.add_argument("--force",   action="store_true",   help="Delete & regenerate all replays")
    p.add_argument("--mode", default=None, choices=["1m", "5m"],
                   help="Training mode suffix (e.g. --mode 5m → logs/dqn_5m/)")
    args = p.parse_args()

    suffix = f"_{args.mode}" if args.mode else ""
    if args.force:
        for aid in AGENT_CKPTS:
            rp = os.path.join(args.log_dir, f"{aid}{suffix}", "replays.json")
            if os.path.exists(rp):
                os.remove(rp)
                print(f"  [{aid:6}] removed old replay")

    scan_and_export(args.log_dir, args.timeout, mode=args.mode)

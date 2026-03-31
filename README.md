# Solving 2048: Classical RL vs. LLM-Guided RLVR

> **CS 5180 — Reinforcement Learning** &nbsp;|&nbsp; Northeastern University

A comprehensive study comparing **six classical Reinforcement Learning agents** (DQN, PPO, A2C, QR-DQN, SAC, LFA) against an **LLM-based agent** trained with Reinforcement Learning from Verifiable Rewards (GRPO + Qwen2.5-0.5B) on the puzzle game **2048**.

The project includes a custom Gymnasium environment, a multi-component verifiable reward system, an interactive web dashboard with live agent replay, and a full LaTeX research report.

---

## ✨ Highlights

- **7 agents** trained and benchmarked end-to-end on identical 2048 environments
- **RLVR pipeline** — Qwen2.5-0.5B fine-tuned with GRPO using structured `<think>/<answer>` reasoning
- **Interactive dashboard** — play 2048, view benchmark charts, and watch agent replays step-by-step with move probabilities
- **Hunt-2048 mode** — a greedy-search wrapper that turns any trained agent into a stronger player via multi-step lookahead
- **Full reproducibility** — YAML configs, CSV metrics, JSON replays, and a LaTeX report

---

## 🏗 Project Structure

```
RLVR/
├── src/
│   ├── env/                     # 2048 Game Engine & Wrappers
│   │   ├── game_2048.py         # Core engine (pure NumPy, 4×4 board)
│   │   ├── gym_wrapper.py       # Gymnasium env (16-channel CNN input)
│   │   └── text_wrapper.py      # Text wrapper (LLM prompt formatting)
│   │
│   ├── classical/               # Track A — Classical RL Agents
│   │   ├── dqn_agent.py         # Custom DQN (CNN, replay buffer, target net)
│   │   ├── ppo_agent.py         # PPO via Stable-Baselines3
│   │   ├── a2c_agent.py         # A2C via Stable-Baselines3
│   │   ├── qrdqn_agent.py       # QR-DQN via sb3-contrib
│   │   ├── sac_agent.py         # Discrete SAC (custom, entropy-tuned)
│   │   ├── lfa_agent.py         # Linear Function Approximation (n-tuple)
│   │   ├── hunt_2048.py         # Hunt-2048 greedy lookahead wrapper
│   │   ├── replay_gen.py        # Replay JSON generator for the dashboard
│   │   ├── export_all.py        # Batch export replays + manifest
│   │   └── train.py             # Unified CLI: train / eval / replay
│   │
│   ├── llm/                     # Track B — LLM-RLVR Agent
│   │   ├── train_grpo.py        # GRPO training pipeline (Unsloth + TRL)
│   │   ├── reward.py            # Multi-component verifiable reward functions
│   │   ├── dataset.py           # Board-state dataset generator
│   │   ├── predict.py           # Inference / evaluation for trained LLM
│   │   ├── prompt.py            # System prompt template
│   │   └── replay_gen.py        # LLM replay generator
│   │
│   ├── utils/
│   │   └── metrics.py           # CSV logger + matplotlib training curves
│   └── visualize.py             # Standalone visualization utilities
│
├── configs/                     # YAML hyperparameter configs
│   ├── dqn_config.yaml
│   ├── ppo_config.yaml
│   └── grpo_config.yaml
│
├── tests/                       # pytest suite
│   ├── test_env.py              # Environment correctness tests
│   └── test_reward.py           # Reward function tests
│
├── report/                      # LaTeX research report
│   ├── main.tex
│   ├── references.bib
│   └── fig_*.png                # Training curve figures
│
├── logs/                        # Training outputs (git-ignored)
│   ├── manifest.json            # Agent registry for the dashboard
│   ├── <agent>/
│   │   ├── *_metrics.csv        # Per-episode training metrics
│   │   ├── replays.json         # Replay data for dashboard playback
│   │   └── hunt_replays.json    # Hunt-2048 enhanced replays
│   └── grpo/
│       ├── adapter/             # LoRA adapter weights
│       ├── merged/              # Full merged model (16-bit)
│       └── train_log.jsonl      # Per-step GRPO training log
│
├── requirements/
│   ├── base.txt                 # numpy, gymnasium, matplotlib, etc.
│   ├── classical.txt            # + torch, stable-baselines3
│   └── llm.txt                  # + unsloth, trl, transformers, peft
│
├── index.html                   # Interactive dashboard (main)
├── Front_end.html               # Alternate dashboard layout
├── serve.py                     # Dev HTTP server for the dashboard
├── Design.md                    # UI design system document
└── RL_Project.pdf               # Compiled research report
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements/base.txt
```

### 2. Play 2048 Interactively (Terminal)

```bash
python -m src.env.game_2048
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Launch the Web Dashboard

```bash
python serve.py          # opens http://localhost:8080
python serve.py 9000     # or use a custom port
```

The dashboard has three views:
| View | Description |
|------|-------------|
| **Play 2048** | Full interactive game with undo, score tracking, and tile spectrum |
| **Benchmark** | Score convergence charts, tile milestone distributions, and comparative metrics across all agents |
| **Agent Playback** | Step-by-step replay of trained agents with move probability bars, reasoning logs, and score progression charts |

---

## 🎓 Training Agents

### Track A — Classical RL

```bash
pip install -r requirements/classical.txt

# DQN (custom CNN + replay buffer + target network)
python -m src.classical.train train --agent dqn --steps 500000

# PPO (CnnPolicy + custom feature extractor)
python -m src.classical.train train --agent ppo --steps 1000000

# A2C
python -m src.classical.train train --agent a2c --steps 1000000

# QR-DQN (quantile regression)
python -m src.classical.train train --agent qrdqn --steps 500000

# SAC (discrete, entropy-tuned)
python -m src.classical.train train --agent sac --steps 500000

# LFA (linear function approximation with n-tuple features)
python -m src.classical.train train --agent lfa --steps 500000
```

**Common options:**
```bash
--reward {score_delta,log_score,shaped}   # reward mode
--lr 0.0003                                # learning rate
--num-envs 8                               # parallel environments
--seed 42                                  # random seed
--log-dir logs                             # output directory
```

### Track B — LLM-RLVR (requires GPU)

```bash
pip install -r requirements/llm.txt

# GRPO training — Qwen2.5-0.5B + QLoRA 4-bit
python -m src.llm.train_grpo \
  --dataset-size 10000 \
  --epochs 3 \
  --num-generations 4 \
  --lr 5e-6 \
  --config configs/grpo_config.yaml

# Run inference on a trained model
python -m src.llm.predict --model logs/grpo/merged --episodes 50
```

### Evaluation & Replay Generation

```bash
# Evaluate a checkpoint
python -m src.classical.train eval --agent dqn --model logs/dqn/dqn_final.pt --episodes 100

# Generate replay JSONs for the dashboard
python -m src.classical.train replay --agent dqn --model logs/dqn/dqn_final.pt --episodes 10
```

---

## 📊 Agent Comparison

### Trained Agents

| Track | Agent | Method | Framework | Key Features |
|-------|-------|--------|-----------|--------------|
| A | **DQN** | Custom CNN + Replay + Target Net | PyTorch | 16-channel board encoding, ε-greedy, prioritized replay |
| A | **PPO** | CnnPolicy + Custom Extractor | Stable-Baselines3 | Clipped surrogate, GAE, multi-env rollouts |
| A | **A2C** | CnnPolicy + Custom Extractor | Stable-Baselines3 | Synchronous advantage actor-critic |
| A | **QR-DQN** | Quantile Regression DQN | sb3-contrib | Distributional value function |
| A | **SAC** | Discrete Soft Actor-Critic | PyTorch (custom) | Entropy-tuned, twin Q-networks |
| A | **LFA** | N-Tuple Linear FA | NumPy | Handcrafted features, no neural network |
| B | **GRPO** | Qwen2.5-0.5B + GRPO + QLoRA | Unsloth + TRL | Structured reasoning with `<think>/<answer>` |

### Results Summary

| Agent | Avg Score | Max Tile | 512+ Rate | 1024+ Rate |
|-------|-----------|----------|-----------|------------|
| **DQN** | 2,825 | 1024 | 16.9% | 1.2% |
| **LFA** | 2,299 | 1024 | 4.4% | 0.01% |
| **A2C** | 1,244 | 512 | 0.2% | — |
| **PPO** | 1,099 | 512 | 0.01% | — |
| **QR-DQN** | 1,090 | 256 | — | — |
| **GRPO** | 1,816 | 128 | — | — |

> **Hunt-2048 Enhancement:** DQN + Hunt achieves a best score of **21,572** with max tile **2048**, demonstrating that greedy search amplification can push learned policies past their standalone ceiling.

### Reward System (LLM Track)

The GRPO training uses decomposed verifiable rewards:

| Component | Value | Description |
|-----------|-------|-------------|
| Format correctness | +0.5 | Valid `<think>` and `<answer>` XML tags |
| Valid direction | +0.5 | Answer ∈ {UP, DOWN, LEFT, RIGHT} |
| Move validity | +1 / −2 | Move changes board / no-op penalty |
| Score delta | Δs / 1024 | Normalized merge score |
| Game over penalty | −1.0 | Terminal state reached |
| Milestone bonus | +2 … +15 | Reaching 256 / 512 / 1024 / 2048+ tiles |

---

## 🎯 Metrics Tracked

- Max tile reached (distribution across episodes)
- Average & max episode score
- Sample efficiency (score / training steps)
- Tile milestone reach rates (256+, 512+, 1024+)
- Moves per episode
- Wall-clock training time
- Per-step reward decomposition (LLM track)

---

## 🔧 Hardware & Requirements

| Resource | Specification |
|----------|---------------|
| **GPU** | NVIDIA RTX 4060 (6 GB VRAM) |
| **GRPO VRAM** | ~3–4 GB (0.5B model, QLoRA 4-bit, G=4) |
| **Python** | 3.10+ |
| **OS** | Linux (tested on Ubuntu) |

---

## 📚 References

1. Guo et al. (2025). _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ Nature.
2. Wen et al. (2025). _RLVR Implicitly Incentivizes Reasoning in LLMs._
3. Saligram et al. (2025). _2048: RL in a Delayed Reward Environment._
4. Hu et al. (2025). _lmgame-Bench: How Good are LLMs at Playing Games?_

---

## 📄 License

This project was developed as part of the CS 5180 Reinforcement Learning course at Northeastern University.

# Solving 2048: Classical RL vs. LLM-Guided RLVR

> **CS 5180 — Reinforcement Learning** &nbsp;|&nbsp; Northeastern University

A comprehensive study comparing **six classical Reinforcement Learning agents** (DQN, PPO, A2C, QR-DQN, SAC, LFA) against an **LLM-based agent** trained with Reinforcement Learning from Verifiable Rewards (GRPO + Qwen2.5-0.5B) on the puzzle game **2048**.

The project includes a custom Gymnasium environment, a multi-component verifiable reward system, an interactive web dashboard with live agent replay, a scaling-curve evaluator, and a full LaTeX research report.

---

## ✨ Highlights

- **7 agents** trained and benchmarked end-to-end on identical 2048 environments
- **RLVR pipeline** — Qwen2.5-0.5B fine-tuned with GRPO using structured `<think>/<answer>` reasoning
- **Interactive dashboard** — play 2048, view benchmark charts, and watch agent replays step-by-step with move probabilities
- **Hunt-2048 mode** — runs each agent until it reaches tile 2048, logging the best episode as a dashboard replay
- **Scaling curves** — `scaling_eval.py` evaluates every checkpoint milestone (50k → 5M steps) to plot sample-efficiency curves
- **MaskablePPO** — PPO and A2C upgraded to `sb3_contrib.MaskablePPO` / `MaskableA2C` so illegal moves are masked at the policy level
- **Full reproducibility** — YAML configs, CSV metrics, JSON replays, and a LaTeX report

---

## 🏗 Project Structure

```
RLVR/
├── src/
│   ├── env/                     # 2048 Game Engine & Wrappers
│   │   ├── game_2048.py         # Core engine (pure NumPy, 4×4 board)
│   │   ├── gym_wrapper.py       # Gymnasium env (16-channel CNN input, action masking)
│   │   └── text_wrapper.py      # Text wrapper (LLM prompt formatting)
│   │
│   ├── classical/               # Track A — Classical RL Agents
│   │   ├── dqn_agent.py         # Custom DQN (CNN, replay buffer, target net)
│   │   ├── ppo_agent.py         # MaskablePPO via sb3_contrib (illegal-move masking)
│   │   ├── a2c_agent.py         # MaskableA2C via sb3_contrib
│   │   ├── qrdqn_agent.py       # QR-DQN via sb3_contrib
│   │   ├── sac_agent.py         # Discrete SAC (custom, entropy-tuned, twin Q-nets)
│   │   ├── lfa_agent.py         # Linear Function Approximation (n-tuple features)
│   │   ├── hunt_2048.py         # Hunt-2048: run agents until tile 2048 is reached
│   │   ├── replay_gen.py        # Best-episode replay JSON generator for the dashboard
│   │   ├── export_all.py        # Batch export replays + manifest (parallel GPU workers)
│   │   ├── scaling_eval.py      # Evaluate milestone checkpoints → scaling_curves.json
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
│   └── fig_*.png                # Training-curve figures
│
├── logs/                        # Training outputs (git-ignored)
│   ├── manifest.json            # Agent registry for the dashboard
│   ├── manifest_5m.json         # Registry for 5M-step runs
│   ├── scaling_curves.json      # Scaling evaluation results (generated)
│   ├── <agent>/
│   │   ├── *_metrics.csv        # Per-episode training metrics
│   │   ├── replays.json         # Best-episode replay for dashboard
│   │   ├── hunt_replays.json    # Hunt-2048 best episode replay
│   │   └── hunt_2048.jsonl      # Every hunt attempt, one JSON per line
│   └── grpo/
│       ├── adapter/             # LoRA adapter weights
│       ├── merged/              # Full merged model (16-bit)
│       └── train_log.jsonl      # Per-step GRPO training log
│
├── requirements/
│   ├── base.txt                 # numpy, gymnasium, matplotlib, pandas, tqdm
│   ├── classical.txt            # + torch, stable-baselines3, sb3_contrib
│   └── llm.txt                  # + unsloth, trl, transformers, peft
│
├── index.html                   # Interactive dashboard (primary)
├── Front_end.html               # Alternate dashboard layout
├── serve.py                     # Dev HTTP server for the dashboard
├── Design.md                    # UI design-system document
└── RL_Project.pdf               # Compiled research report PDF
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
# Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Launch the Web Dashboard

```bash
python serve.py          # opens http://localhost:8080
python serve.py 9000     # or specify a custom port
```

Dashboard views:

| View | Description |
|------|-------------|
| **Play 2048** | Full interactive game with undo, score tracking, and tile spectrum |
| **Benchmark** | Score convergence charts, tile milestone distributions, and comparative metrics across all agents |
| **Agent Playback** | Step-by-step replay of trained agents with move-probability bars, reasoning logs (LLM), and score-progression charts |

---

## 🎓 Training Agents

### Track A — Classical RL

```bash
pip install -r requirements/classical.txt

# DQN (custom CNN + replay buffer + target network)
python -m src.classical.train train --agent dqn --steps 500000

# PPO (MaskablePPO + custom CNN feature extractor)
python -m src.classical.train train --agent ppo --steps 1000000

# A2C (MaskableA2C + custom CNN feature extractor)
python -m src.classical.train train --agent a2c --steps 1000000

# QR-DQN (quantile regression, distributional RL)
python -m src.classical.train train --agent qrdqn --steps 500000

# SAC (discrete, entropy-tuned, twin Q-networks)
python -m src.classical.train train --agent sac --steps 500000

# LFA (linear function approximation with n-tuple features)
python -m src.classical.train train --agent lfa --steps 500000
```

**Common options:**
```bash
--reward {score_delta,log_score,shaped}   # reward shaping mode
--lr 0.0003                               # learning rate override
--num-envs 8                              # parallel environments (on-policy)
--seed 42                                 # random seed
--log-dir logs                            # output directory
--mode {1m,5m}                            # training mode (appends suffix, e.g. logs/dqn_5m/)
```

### Track B — LLM-RLVR (requires GPU)

```bash
pip install -r requirements/llm.txt

# GRPO training — Qwen2.5-0.5B + QLoRA 4-bit
python -m src.llm.train_grpo \
  --dataset-size 10000 \
  --epochs 3 \
  --num-generations 8 \
  --lr 5e-6 \
  --config configs/grpo_config.yaml

# Run inference on a trained model
python -m src.llm.predict --model logs/grpo/merged --episodes 50
```

### Evaluation & Replay Generation

```bash
# Evaluate a checkpoint
python -m src.classical.train eval --agent dqn --model logs/dqn/dqn_final.pt --episodes 100

# Generate the best-episode replay JSON for the dashboard
python -m src.classical.train replay --agent dqn --model logs/dqn/dqn_final.pt --episodes 10
```

### Batch Export (All Agents → Dashboard Ready)

```bash
# Generate replays + manifest for all trained agents
python -m src.classical.export_all

# Force regenerate even if replays already exist
python -m src.classical.export_all --force

# Export 5M-step run variant
python -m src.classical.export_all --mode 5m
```

### Hunt-2048 (Run Until Tile 2048)

```bash
# Run all agents until each reaches tile 2048 (unlimited attempts)
python -m src.classical.hunt_2048

# Run a subset
python -m src.classical.hunt_2048 --agents dqn lfa

# Use 5M-step checkpoints
python -m src.classical.hunt_2048 --mode 5m
```

### Scaling Curve Evaluation

```bash
# Evaluate all milestone checkpoints (50k, 100k, 200k, 500k, 1M, 2M, 5M steps)
python -m src.classical.scaling_eval

# Evaluate specific agents with 50 episodes per checkpoint
python -m src.classical.scaling_eval --agents dqn lfa --episodes 50
```

Outputs `logs/scaling_curves.json` with per-step metrics for each agent.

---

## 📊 Agent Comparison

### Trained Agents

| Track | Agent | Method | Framework | Key Features |
|-------|-------|--------|-----------|--------------|
| A | **DQN** | Custom CNN + Replay + Target Net | PyTorch | 16-channel board encoding, ε-greedy, prioritized replay |
| A | **PPO** | MaskablePPO + Custom CNN Extractor | sb3_contrib | Clipped surrogate, GAE, invalid-move masking, multi-env |
| A | **A2C** | MaskableA2C + Custom CNN Extractor | sb3_contrib | Synchronous advantage actor-critic, invalid-move masking |
| A | **QR-DQN** | Quantile Regression DQN | sb3_contrib | Distributional value function, risk-sensitive learning |
| A | **SAC** | Discrete Soft Actor-Critic | PyTorch (custom) | Entropy-tuned, twin Q-networks, temperature auto-tuning |
| A | **LFA** | N-Tuple Linear FA | NumPy | Handcrafted tile-pattern features, no neural network |
| B | **GRPO** | Qwen2.5-0.5B + GRPO + QLoRA | Unsloth + TRL | Structured `<think>/<answer>` chain-of-thought reasoning |

### Results Summary

| Agent | Avg Score | Max Tile | 512+ Rate | 1024+ Rate | Hunt Best |
|-------|-----------|----------|-----------|------------|-----------|
| **DQN** | 2,825 | 1024 | 16.9% | 1.2% | 21,572 (2048✓) |
| **LFA** | 2,299 | 1024 | 4.4% | 0.01% | 12,300 (1024) |
| **GRPO** | 1,816 | 128 | — | — | — |
| **A2C** | 1,244 | 512 | 0.2% | — | 7,844 (512) |
| **PPO** | 1,099 | 512 | 0.01% | — | 5,860 (512) |
| **QR-DQN** | 1,090 | 256 | — | — | 228 |
| **SAC** | ~700* | 256 | — | — | 3,648 (256) |

> *SAC training metrics CSV not available; replay score is 2,644.

> **Hunt-2048 Enhancement:** DQN + Hunt achieves a best score of **21,572** with max tile **2048**, demonstrating that greedy search amplification can push learned policies past their standalone ceiling.

### Reward System (LLM Track)

The GRPO training uses five decomposed verifiable reward functions:

| Component | Value | Description |
|-----------|-------|-------------|
| Length reward | 0 → +0.3 | Encourages substantive completions; breaks zero-variance GRPO trap |
| Format correctness | +0.5 | Valid `<think>` and `<answer>` XML tags (partial: +0.25) |
| Valid direction | +0.5 | Answer ∈ {UP, DOWN, LEFT, RIGHT} |
| Move validity | +1 / −2 | Move changes board / no-op penalty |
| Score delta | Δs / 1024 | Normalized merge score |
| Game over penalty | −1.0 | Terminal state reached |
| Milestone bonus | +2 … +10 | First time reaching 256 / 512 / 1024 / 2048+ |
| Thinking quality | 0 → +0.5 | Tile-value awareness + strategic vocabulary in `<think>` |

---

## 🎯 Metrics Tracked

- Max tile reached (distribution across episodes)
- Average & max episode score
- Sample efficiency (score vs. training steps — see scaling curves)
- Tile milestone reach rates (256+, 512+, 1024+)
- Moves per episode (including invalid-move rate)
- Wall-clock training time
- Per-step reward decomposition (LLM track)

---

## 🔧 Hardware & Requirements

| Resource | Specification |
|----------|---------------|
| **GPU** | NVIDIA RTX 4060 (6 GB VRAM) |
| **GRPO VRAM** | ~3–4 GB (0.5B model, QLoRA 4-bit, G=8) |
| **Python** | 3.10+ |
| **OS** | Linux (tested on Ubuntu) |

### Dependency Matrix

| Package | Version | Used By |
|---------|---------|---------|
| numpy | ≥ 1.24 | All |
| gymnasium | ≥ 0.29 | All |
| torch | ≥ 2.0 | DQN, SAC, PPO, A2C, QR-DQN |
| stable-baselines3 | ≥ 2.3 | PPO, A2C |
| sb3_contrib | ≥ 2.3 | QR-DQN, MaskablePPO, MaskableA2C |
| unsloth | latest | GRPO training |
| trl | ≥ 0.24 | GRPO training |
| transformers | ≥ 4.40 | GRPO inference |
| peft | latest | LoRA adapter |

---

## 📚 References

1. Guo et al. (2025). _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ Nature.
2. Wen et al. (2025). _RLVR Implicitly Incentivizes Reasoning in LLMs._
3. Saligram et al. (2025). _2048: RL in a Delayed Reward Environment._
4. Hu et al. (2025). _lmgame-Bench: How Good are LLMs at Playing Games?_
5. Schulman et al. (2017). _Proximal Policy Optimization Algorithms._
6. Dabney et al. (2018). _Distributional Reinforcement Learning with Quantile Regression._
7. Haarnoja et al. (2018). _Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning._

---

## 📄 License

This project was developed as part of the CS 5180 Reinforcement Learning course at Northeastern University.

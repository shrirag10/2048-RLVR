# Solving 2048: Classical RL vs. LLM-Guided RLVR

> **CS 5180 — Reinforcement Learning** | Shriman + Partner

A comparative study of classical Reinforcement Learning (DQN, PPO) versus
LLM-based Reinforcement Learning with Verifiable Rewards (GRPO with
Qwen2.5-0.5B) on the puzzle game 2048.

## 🏗 Project Structure

```
RLVR/
├── src/
│   ├── env/              # 2048 game engine + wrappers
│   │   ├── game_2048.py  # Core engine (pure Python)
│   │   ├── gym_wrapper.py # Gymnasium env (16-ch CNN input)
│   │   └── text_wrapper.py # Text wrapper (LLM prompts)
│   ├── classical/        # Track A: DQN + PPO
│   │   ├── dqn_agent.py  # Custom DQN (CNN, replay, target net)
│   │   ├── ppo_agent.py  # SB3 PPO wrapper
│   │   └── train.py      # Unified training CLI
│   ├── llm/              # Track B: LLM-RLVR
│   │   ├── reward.py     # Multi-component verifier/reward
│   │   ├── dataset.py    # Board state dataset generator
│   │   └── train_grpo.py # GRPO training pipeline
│   └── utils/
│       └── metrics.py    # Logging + matplotlib plots
├── configs/              # YAML hyperparameter configs
├── tests/                # pytest test suite
├── report/               # LaTeX final report
└── requirements/         # Dependency files
```

## 🚀 Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements/base.txt

# Play interactively
python -m src.env.game_2048

# Run tests
python -m pytest tests/ -v

# Train DQN
pip install -r requirements/classical.txt
python -m src.classical.train train --agent dqn --steps 500000

# Train PPO
python -m src.classical.train train --agent ppo --steps 1000000

# Train LLM (GRPO) — requires GPU
pip install -r requirements/llm.txt
python -m src.llm.train_grpo --dataset-size 10000 --epochs 3
```

## 📊 Tracks

| Track | Agent | Method | Framework |
|-------|-------|--------|-----------|
| A | DQN | Custom CNN + Replay + Target Net | PyTorch |
| A | PPO | CnnPolicy + Custom Extractor | Stable-Baselines3 |
| B | LLM | Qwen2.5-0.5B + GRPO + QLoRA | Unsloth + TRL |

## 🎯 Metrics

- Max tile reached (distribution)
- Average episode score
- Sample efficiency (score/steps)
- Win rate (% reaching 2048)
- Wall-clock training time
- Reasoning quality (LLM track)

## 📚 References

1. Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning via RL. *Nature*.
2. Wen et al. (2025). RLVR Implicitly Incentivizes Correct Reasoning.
3. Saligram et al. (2025). 2048: RL in a Delayed Reward Environment.
4. Hu et al. (2025). lmgame-Bench: How Good are LLMs at Playing Games?

## 🔧 Hardware

- **GPU**: NVIDIA RTX 4060 (6GB VRAM)
- **GRPO VRAM**: ~3-4GB (0.5B model, QLoRA 4-bit, G=8)

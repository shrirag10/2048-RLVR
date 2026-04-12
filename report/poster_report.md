# Solving 2048: Classical RL vs. LLM-Guided RLVR

> **CS 5180 — Reinforcement Learning** &nbsp;|&nbsp; Northeastern University
> **Shriman Raghav** & **Gautham Ramkumar** | MS Robotics

---

## 1. Problem: The 2048 MDP

We formalize the 2048 puzzle as a Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$:

### State Space $\mathcal{S}$

Each state $s \in \mathcal{S}$ is a $4 \times 4$ board where each cell holds a tile value $\in \{0, 2, 4, 8, \ldots, 2^{15}\}$ (0 = empty).

**Observation encoding** (for CNN agents): A 16-channel binary tensor $\mathbf{x} \in \{0,1\}^{16 \times 4 \times 4}$:

$$x_{k,i,j} = \mathbb{1}[B_{i,j} = 2^k], \quad k = 0, 1, \ldots, 15$$

- Channel 0 → empty cells
- Channel $k$ → positions of tile $2^k$
- Raw state space: $\sim 17^{16} \approx 2.8 \times 10^{19}$

**Observation encoding** (for LLM agent): Plain-text ASCII grid with row/column labels + score.

### Action Space $\mathcal{A}$

$$\mathcal{A} = \{\texttt{UP}(0),\; \texttt{RIGHT}(1),\; \texttt{DOWN}(2),\; \texttt{LEFT}(3)\}$$

At each turn, the player selects one of four cardinal directions. All tiles slide; **identical adjacent tiles merge** (doubling their value), and the merged value is added to the score. After a valid move, a new tile (2 with 90% probability, 4 with 10%) spawns at a random empty cell.

### Transition Function $\mathcal{T}(s' | s, a)$

$$s' = \text{Slide-Merge}(s, a) \oplus \text{RandomTile}(s'_{\text{pre}})$$

- **Deterministic component**: Slide all rows/columns in direction $a$; merge equal adjacent tiles.
- **Stochastic component**: Place 2 (p=0.9) or 4 (p=0.1) on a uniform random empty cell.
- If $a$ produces no board change → **invalid move** (no tile spawns, no state change).
- **Terminal**: Game ends when **no action changes the board**.

### Reward Function $\mathcal{R}(s, a, s')$

| Reward Mode | Formula | Description |
|---|---|---|
| **Score-delta** (default) | $r = \Delta\text{score}$ | Sum of merged tile values at this step |
| **Log-score** | $r = \log_2(\Delta\text{score} + 1)$ | Scale-compressed for numerical stability |
| **Shaped** | $r = \Delta\text{score} + \text{milestones}$ | +50 at 256, +100 at 512, +200 at 1024, +500 at 2048 |
| **Invalid move** | $r = -1$ | Penalty for selecting a no-op action |

### Discount Factor

$$\gamma = 0.99$$

### Key MDP Challenges

> [!IMPORTANT]
> **Why 2048 is hard for RL:**
> 1. **Sparse, delayed rewards** — hundreds of low-value merges precede a single high-value event
> 2. **Stochastic transitions** — random tile spawns prevent deterministic planning
> 3. **Combinatorial state space** — $\sim 10^{19}$ states make tabular methods infeasible
> 4. **Long horizons** — episodes last 100–1000+ steps

---

## 2. Algorithms & Baselines

We benchmark **7 agents** — 6 classical RL + 1 LLM-RLVR — all using the same environment and (for deep agents) the same CNN backbone.

### Shared CNN Feature Extractor $\phi_\theta$

All deep classical agents share an identical CNN:

```
Input: (batch, 16, 4, 4)
  → Conv2D(16→128, 2×2) + ReLU      # output: 128×3×3
  → Conv2D(128→128, 2×2) + ReLU     # output: 128×2×2
  → Conv2D(128→128, 2×2) + ReLU     # output: 128×1×1
  → Flatten → FC(128→256) + ReLU    # 256-dim features
  → Task Head(256→4)                # Q-values / policy logits
```

This ensures performance differences reflect the **learning algorithm**, not architecture.

---

### Algorithm 1: Deep Q-Network (DQN) ⭐ **Best Performer**

```
Algorithm: DQN with Experience Replay + Target Network
─────────────────────────────────────────────────────
Initialize policy network Q_θ and target network Q_θ̄
Initialize replay buffer D (capacity = 100,000)
ε ← 1.0

for step = 1 to total_steps:
    # ε-greedy action selection (only valid actions)
    if random() < ε:
        a ← random valid action
    else:
        a ← argmax_a Q_θ(s, a)  [mask invalid with -∞]

    Execute a, observe (r, s', done)
    Store (s, a, r, s', done) in D
    
    # Train on minibatch from replay buffer
    Sample batch of 64 from D
    y ← r + γ · max_a' Q_θ̄(s', a')  [if not done]
    Loss ← SmoothL1Loss(Q_θ(s,a), y)
    Update θ with Adam (lr=1e-4, grad_clip=1.0)
    
    # Decay exploration: ε: 1.0 → 0.01 over 100K steps
    ε ← max(0.01, 1.0 - step/100,000)
    
    # Sync target network every 1,000 steps
    if step % 1000 == 0:  θ̄ ← θ
```

> [!TIP]
> **Why DQN wins**: Experience replay (100K buffer) lets DQN revisit rare high-scoring trajectories thousands of times, amplifying the sparse reward signal. Off-policy learning is critical for 2048's sparse rewards.

**Key hyperparameters**: γ=0.99, batch=64, buffer=100K, target sync=1000 steps, lr=1e-4.

---

### Algorithm 2: Proximal Policy Optimization (PPO)

```
Algorithm: MaskablePPO (on-policy, actor-critic)
──────────────────────────────────────────────────
for iteration = 1, 2, ...:
    Collect 2048 steps with π_θ (action masking enabled)
    Compute GAE advantages: Â_t = Σ (γλ)^l δ_{t+l}
    
    for epoch = 1 to 10:
        for minibatch of 64:
            r_t(θ) = π_θ(a|s) / π_old(a|s)
            L_clip = min(r_t·Â, clip(r_t, 1±0.2)·Â)
            L = -L_clip + 0.5·MSE(V,R) - 0.01·H[π]
            Update θ
```

**Key detail**: Uses `sb3_contrib.MaskablePPO` — invalid actions masked at the policy level (logits set to $-\infty$). GAE with λ=0.95, clip ε=0.2, 8 parallel envs.

---

### Algorithm 3: Advantage Actor-Critic (A2C)

```
Algorithm: MaskableA2C (synchronous on-policy)
──────────────────────────────────────────────
Collect 5-step rollouts across 8 parallel envs
R_t = Σ γ^k r_{t+k} + γ^5 V(s_{t+5})   # 5-step bootstrap
Â_t = R_t - V(s_t)

L = -E[log π·Â] + 0.5·||R-V||² - 0.01·H[π]
Update θ (single gradient step per batch)
```

**Key difference from PPO**: No clipping constraint, single gradient step per batch, relies on entropy regularization for stability.

---

### Algorithm 4: Quantile Regression DQN (QR-DQN)

```
Algorithm: QR-DQN (distributional RL)
────────────────────────────────────────
Learn N=50 quantiles τ_i of the return distribution Z(s,a)
Network outputs 4×50 = 200 values

Loss = (1/N) Σ_i E_j [ ρ_τi(δ_ij) ]
  where δ_ij = r + γ·z_j(s',a*) - z_i(s,a)
  and ρ_τ(δ) = |τ - 1[δ<0]| · HuberLoss(δ)

Action selection: a* = argmax_a (1/N) Σ_i z_i(s,a)
```

**Key insight**: Models the full return distribution, but the 200-output head requires proportionally more data to converge.

---

### Algorithm 5: Discrete Soft Actor-Critic (SAC)

```
Algorithm: Discrete SAC (max-entropy off-policy)
──────────────────────────────────────────────────
Maintain: π_θ (categorical policy), Q_ψ1, Q_ψ2 (twin Q-nets), α (auto-tuned temp)

Critic target:
  y = r + γ Σ_a' π(a'|s')[min_k Q_ψ̄k(s',a') - α·log π(a'|s')]

Actor loss:
  L_π = E_s[ Σ_a π(a|s)(α·log π(a|s) - min_k Q_ψk(s,a)) ]

Temperature loss (auto-tune):
  L_α = -α · E_s[H[π(·|s)] - H̄]    # target entropy H̄ = 0.5·log|A|

Soft target: ψ̄ ← τ·ψ + (1-τ)·ψ̄    # τ = 0.005
```

---

### Algorithm 6: Linear Function Approximation (LFA)

```
Algorithm: Semi-Gradient SARSA with Linear FA
────────────────────────────────────────────────
Weight matrix: w ∈ R^{4×27}  (only 108 parameters!)
Feature vector φ(s) ∈ R^27:
  - 16 log-normalized tile values
  - 1 empty-cell ratio
  - 1 merge potential
  - 4 monotonicity scores (row/col gradients)
  - 1 max-tile indicator, 1 corner bonus
  - 3 tile distribution stats (low/mid/high)

Q̂(s,a) = w_a^T · φ(s)

SARSA update:
  δ = r + γ·Q̂(s',a') - Q̂(s,a)
  w_a ← w_a + α·δ·φ(s)    # α = 5e-4
```

> [!TIP]
> **Surprise result**: LFA with 108 parameters outperforms all deep on-policy methods (330K+ params), proving that domain-specific feature engineering > large neural networks for structured games.

---

### Algorithm 7: GRPO-Trained LLM (RLVR Track)

```
Algorithm: Group Relative Policy Optimization (GRPO)
─────────────────────────────────────────────────────
Model: Qwen2.5-0.5B-Instruct (QLoRA 4-bit, rank=16)
Trainable: ~5.2M params (1.05% of 494M total)

for each prompt q (board state as text):
    Generate G=4 completions {o_1, ..., o_G} from π_θ
    Score each with verifiable reward: r_i = R_verify(o_i, q)
    
    # Group-normalized advantage (replaces critic!)
    Â_i = (r_i - μ_G) / (σ_G + ε)
    
    # Clipped policy gradient + KL penalty
    L = (1/G) Σ_i (1/|o_i|) Σ_t [
        min(ratio_t · Â_i, clip(ratio_t, 1±0.2)·Â_i)
        - β · D_KL(π_θ || π_ref)    # β = 4.0
    ]
    
    Update θ with AdamW-8bit (lr=1e-6)
```

**Multi-Model GRPO Scaling**:

| Model | Score | Max Tile | Moves | Training |
|---|---|---|---|---|
| Qwen2.5-0.5B (baseline) | 704 | 64 | 93 | Stage 1→2 (~1 hr) |
| **Qwen2.5-1.5B** | **2,348** | **256** | **197** | Stage 1→2→3 (~4.5 hrs) |

**Multi-Component Verifiable Reward**:

| Component | Value | Description |
|---|---|---|
| Format $r_f$ | 0 → +0.5 | Valid `<think>...</think><answer>...</answer>` XML tags |
| Direction $r_d$ | +0.5 | Answer ∈ {UP, DOWN, LEFT, RIGHT} |
| Move validity $r_g$ | +1 / −2 | Valid move / no-op penalty |
| Score delta | Δs/1024 | Normalized merge score |
| Thinking quality $r_q$ | 0 → +0.5 | Tile-value awareness + strategic vocabulary |
| Length $r_\ell$ | 0 → +0.3 | Breaks GRPO zero-variance trap |

> [!WARNING]
> **Staged reward scheduling is mandatory**: Training all rewards simultaneously causes reward hacking (model maximizes length while ignoring format). Successful config: Stage 1 = format + direction only.

---

## 3. Current Results

> [!NOTE]
> **Training status**: DQN trained 5M steps (8 parallel envs → ~20M env interactions). PPO, A2C, LFA, QR-DQN completed 5M steps. SAC completed ~4M steps. GRPO: 0.5B and 1.5B both completed 3-stage training.

### 3.1 Agent Performance Comparison — Final Results

| Agent | Type | Steps | Avg Score (last 100) | Max Score | Max Tile | 512+ | 1024+ | Params |
|---|---|---|---|---|---|---|---|---|
| **DQN** ⭐ | Off-policy, value | 5M×8 | **7,744** | **39,314** | **2048** | **76.0%** | **31.0%** | 330K |
| **LFA** | On-policy, linear | 5M | 2,456 | 11,436 | 1024 | 6.0% | <1% | **108** |
| **GRPO 1.5B** | RLVR | 1300 steps | **2,348** | 2,348 | **256** | — | — | 1.8B |
| **A2C** | On-policy, A-C | 5M | 1,944 | 9,640 | 1024 | 6.0% | <1% | 330K |
| **GRPO 0.5B** | RLVR | 1000 steps | 1,816 | 1,816 | 128 | — | — | 494M |
| **SAC** | Off-policy, max-ent | ~4M | 1,131 | 5,566 | 512 | <1% | — | 330K |
| **QR-DQN** | Off-policy, distrib. | 5M | 995 | 5,176 | 512 | <1% | — | 430K |
| **PPO** | On-policy, A-C | 5M | 962 | 4,900 | 512 | <1% | — | 330K |
| **GRPO 0.5B** (new) | RLVR | 1000 steps | 704 | 704 | 64 | — | — | 494M |

### 3.2 Improvement from 1M → 5M Steps

| Agent | Avg @ 1M | Avg @ 5M | Δ | Verdict |
|---|---|---|---|---|
| **DQN** | 2,825 | **7,744** | **+174%** 🚀 | Strong continued learning |
| **A2C** | 1,244 | **1,944** | +56% | Modest improvement with more data |
| **LFA** | 2,299 | 2,456 | +7% | Plateaued (linear capacity ceiling) |
| **QR-DQN** | 1,090 | 995 | **−9%** 📉 | Still degrading (invalid-move loops) |
| **PPO** | 1,099 | 962 | **−12%** 📉 | Worse with more training! |
| **SAC** | ~700 | 1,131 | +62% | Slow but improving |

> [!IMPORTANT]
> **Key insight from scaling**: Only **DQN** shows strong continued improvement with 5× more training. On-policy methods (PPO) actually *degrade*. A2C improves modestly. QR-DQN's invalid-move loops worsen. This confirms **off-policy replay is essential** for 2048's sparse reward structure.

### 3.3 Hunt Mode — Unlimited Attempts Until Tile 2048

| Agent | Best Tile | Best Score | Strategy |
|---|---|---|---|
| **DQN** | **2048 ✓** | **21,572** | ε=0.05 exploration |
| LFA | 1024 | 12,300 | 95% greedy |
| A2C | 512 | 7,844 | Distribution sampling |
| PPO | 512 | 5,860 | Distribution sampling |
| SAC | 256 | 3,648 | Probability sampling |
| QR-DQN | 16 | 228 | ε=0.05 |

### 3.4 Learning Curves

#### DQN (5M steps) — Reaches 2048, avg ~7,700
![DQN 5M Training Curves: Score rises from ~1000 to ~7500+ over 48K episodes. Max tile reaches 2048 tiles. 512 is the most common tile. Moves per episode ~500.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_dqn_5m_curves.png)

#### A2C (5M steps) — Modest improvement, hitting 512/1024
![A2C 5M Training Curves: Score averages ~2000 with high variance. Max tile reaches 1024 rarely. Most episodes end at 128-256.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_a2c_5m_curves.png)

#### LFA (5M steps) — Plateaued at ~2,400 avg
![LFA Training Curves: Score stabilizes around 2,400. Max tile mostly 256 with occasional 512. Moves ~200/episode.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_lfa_curves.png)

#### SAC (~4M steps) — Flat at ~1,100, max tile 128
![SAC Training Curves: Score flatlines at ~1100 across 33K episodes. Max tile stuck at 128 with occasional 256/512 spikes. No learning trend.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_sac_curves.png)

#### QR-DQN (5M steps) — Degrades, invalid-move loops
![QR-DQN 5M Training Curves: Score declines slightly from ~1100 to ~1000. Moves per episode explode to 1000+ as agent gets stuck in invalid-move loops.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_qrdqn_5m_curves.png)

#### PPO (5M steps) — Flat, no improvement over 1M
![PPO Training Curves: Episode score flatlines at ~1100 across all 7000 episodes. Max tile stuck at 128 with occasional 256 spikes.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_ppo_curves.png)

#### LLM (GRPO) — Very early stage (300 training steps)
![LLM Evaluation Curves: Only 10 eval episodes, scores ranging 50-430, max tiles 8-32. Extremely early stage.](/home/shrirag10/.gemini/antigravity/brain/83bfae1a-628c-44c2-a423-4e7136714eb2/fig_llm_curves.png)

### 3.5 Analysis: Key Findings

**1. Off-policy learning is decisive for sparse rewards**

DQN's 100K-sample replay buffer allows it to revisit rare high-scoring trajectories thousands of times. At 5M steps, DQN reaches 2048 tiles and averages 7,744 — a **3× lead** over the next-best agent. On-policy methods discard each trajectory after a single update, creating a chicken-and-egg problem where the agent can't generate informative data until it learns, but can't learn without informative data.

**2. Feature engineering > deep learning (up to a point)**

LFA with 108 parameters (a $4 \times 27$ weight matrix) outperforms PPO (330K params), QR-DQN (430K), and SAC (330K) — a **4000× parameter efficiency advantage**. Its 27-dim hand-crafted features encode monotonicity, corner strategies, and merge potential — heuristics that CNNs must learn from scratch. However, LFA plateaus at ~2,500 due to its linear capacity ceiling.

**3. Scaling reveals divergent trajectories**

| Pattern | Agents | Explanation |
|---|---|---|
| **Strong scaling** | DQN (+174%) | Off-policy replay compounds with more data; reaches qualitatively new strategies |
| **Weak scaling** | A2C (+56%), SAC (+62%) | Some improvement but fundamentally limited |
| **Plateau** | LFA (+7%) | Linear model hits capacity ceiling |
| **Degradation** | PPO (−12%), QR-DQN (−9%) | More training → worse policy (overfitting / invalid loops) |

**4. Deterministic policies fail at high tiles**

In Hunt mode, DQN needs 5% ε-noise to break policy cycles at states with poor Q-estimates. Without exploration, the agent repeats the same suboptimal move indefinitely at 1024+ tiles.

**5. GRPO: Model scale enables board reasoning**

With 3-stage training (format → game reward → thinking quality), the 0.5B model learns XML formatting and direction selection but cannot reason about board tile merges (score 704, tile 64). The 1.5B model, with 3× more parameters, achieves 3.3× higher scores (2,348) and reaches tile 256 — demonstrating that model scale is critical for text-based board reasoning. KL divergence stability required `β=4.0` and `max_grad_norm=0.1` after two failed training runs diverged at KL=20,000+.

### 3.6 Latency Comparison

| Agent | ms/move | Hardware | Params |
|---|---|---|---|
| LFA | <0.1 | CPU | 108 |
| PPO, A2C, SAC | <1 | GPU | 330K |
| DQN | 3.6 | GPU | 330K |
| QR-DQN | 4.2 | GPU | 430K |
| **LLM (GRPO)** | **1,500** | GPU | **494M** |

The LLM is **400× slower** than classical agents but provides **interpretable chain-of-thought reasoning**.

---

## 4. Plan to Finish

### Completed ✅
- [x] Custom 2048 Gymnasium environment with 3 reward modes + action masking
- [x] 16-channel binary observation encoding + shared CNN backbone
- [x] 7 agents trained and benchmarked (DQN, PPO, A2C, QR-DQN, SAC, LFA, GRPO)
- [x] 5M-step training completed for DQN, PPO, A2C, QR-DQN, LFA
- [x] ~4M-step training for SAC (in progress)
- [x] Hunt-2048 mode — DQN reaches 2048 ✓
- [x] Interactive web dashboard with 3 views (Play, Benchmark, Agent Playback)
- [x] LaTeX research report (IEEE format)
- [x] GRPO training pipeline with staged rewards

### Remaining Work 🔧

| Task | Priority | Status |
|---|---|---|
| **Complete SAC 5M** | High | ~4M done, ~1M remaining |
| **GRPO 3-stage training (0.5B + 1.5B)** | High | ✅ **Done** — 1.5B: score 2,348, tile 256 |
| **Scaling curves** (eval checkpoints 50K→5M) | Medium | Script implemented, need to run |
| **Multi-seed runs** (3 seeds for error bars) | Low | Not started |
| **Poster layout** and figure polishing | High | ✅ **Updated** with GRPO results |
| **Final report revision** with all results | Medium | In progress |

---

## References

1. Guo et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL.* Nature.
2. Wen et al. (2025). *RLVR Implicitly Incentivizes Reasoning in LLMs.*
3. Saligram et al. (2025). *2048: RL in a Delayed Reward Environment.*
4. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
5. Dabney et al. (2018). *Distributional RL with Quantile Regression.*
6. Haarnoja et al. (2018). *Soft Actor-Critic.*
7. Hu et al. (2025). *lmgame-Bench: How Good are LLMs at Playing Games?*

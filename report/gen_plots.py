"""Generate 3 new comparison plots for the poster."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT = os.path.dirname(__file__)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DQN_COLOR   = '#D35400'
QRDQN_COLOR = '#1F618D'
PPO_COLOR   = '#C0392B'
A2C_COLOR   = '#1A7A4C'
SAC_COLOR   = '#6C3483'
SARSA_COLOR = '#7D6608'
GRPO_COLOR  = '#6C3483'

# ── helpers ────────────────────────────────────────────────────────────────

def smooth(series, w=40):
    return pd.Series(series).rolling(w, min_periods=1).mean().values

def load(path):
    return pd.read_csv(path)

# ══════════════════════════════════════════════════════════════════════════
# Plot 1 — GRPO Scaling Law
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.2, 3.4))

params   = [494e6, 1.8e9]
scores   = [1816, 2348]
labels   = ['Qwen2.5\n0.5B', 'Qwen2.5\n1.5B']
colors   = ['#9B59B6', '#6C3483']
bars = ax.bar(labels, scores, color=colors, width=0.45, zorder=3, edgecolor='white', linewidth=0.5)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            f'{score:,}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')

# trend annotation
ax.annotate('', xy=(1, 2348), xytext=(0, 1816),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.0))
ax.text(0.55, 2100, '3x params\n3.3x score', fontsize=9, color='#C0392B', fontweight='bold')

ax.set_ylabel('Avg Score (300 prompts)', fontsize=10)
ax.set_title('GRPO: LLM Scaling Law', fontsize=12, fontweight='bold', color='#2C3E50')
ax.set_ylim(0, 2900)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_grpo_scaling.png'), dpi=180, bbox_inches='tight')
plt.close()
print('fig_grpo_scaling.png done')

# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — DQN vs QR-DQN learning curves (same axes)
# ══════════════════════════════════════════════════════════════════════════
dqn   = load('/home/shrirag10/Projects/RLVR/logs/dqn_5m/dqn_metrics.csv')
qrdqn = load('/home/shrirag10/Projects/RLVR/logs/qrdqn_5m/qrdqn_metrics.csv')

fig, ax = plt.subplots(figsize=(5.6, 3.6))

for df, color, label in [
    (dqn,   DQN_COLOR,   'DQN  (avg 7,744)'),
    (qrdqn, QRDQN_COLOR, 'QR-DQN (avg 995)'),
]:
    steps = df['training_steps'].values
    raw   = df['total_score'].values
    sm    = smooth(raw, w=60)
    ax.plot(steps/1e6, sm, color=color, lw=2.0, label=label, zorder=3)
    ax.fill_between(steps/1e6, sm*0.75, sm*1.25, color=color, alpha=0.10, zorder=2)

# 1M marker
ax.axvline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6)
ax.text(1.02, ax.get_ylim()[1]*0.92, '1M', fontsize=8, color='gray')

ax.set_xlabel('Training Steps (M)', fontsize=10)
ax.set_ylabel('Smoothed Score', fontsize=10)
ax.set_title('DQN vs. QR-DQN: Distributional Modeling Fails', fontsize=11, fontweight='bold', color='#2C3E50')
ax.legend(fontsize=9, framealpha=0.85)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.25, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_dqn_vs_qrdqn.png'), dpi=180, bbox_inches='tight')
plt.close()
print('fig_dqn_vs_qrdqn.png done')

# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — Sample efficiency: score @ 1M vs @ 5M (grouped bars)
# ══════════════════════════════════════════════════════════════════════════
agents = ['DQN', 'SARSA', 'A2C', 'SAC', 'QR-DQN', 'PPO']
colors_agents = [DQN_COLOR, SARSA_COLOR, A2C_COLOR, SAC_COLOR, QRDQN_COLOR, PPO_COLOR]

scores_1m = [2825, 2299, 1244, 700, 1090, 1099]
scores_5m = [7744, 2456, 1944, 1255, 995,  962]

x = np.arange(len(agents))
w = 0.38

fig, ax = plt.subplots(figsize=(6.5, 3.8))

bars1 = ax.bar(x - w/2, scores_1m, w, label='@ 1M steps',
               color=[c + '99' for c in ['#D35400','#7D6608','#1A7A4C','#6C3483','#1F618D','#C0392B']],
               edgecolor='white', linewidth=0.5, zorder=3)
bars5 = ax.bar(x + w/2, scores_5m, w, label='@ 5M steps',
               color=colors_agents,
               edgecolor='white', linewidth=0.5, zorder=3)

# delta labels
for i, (s1, s5, c) in enumerate(zip(scores_1m, scores_5m, colors_agents)):
    pct = (s5 - s1) / s1 * 100
    sign = '+' if pct >= 0 else ''
    color = '#1A7A4C' if pct >= 0 else '#C0392B'
    ax.text(x[i] + w/2, s5 + 80, f'{sign}{pct:.0f}%', ha='center', va='bottom',
            fontsize=7.5, fontweight='bold', color=color)

ax.set_xticks(x)
ax.set_xticklabels(agents, fontsize=9)
ax.set_ylabel('Avg Score', fontsize=10)
ax.set_title('Sample Efficiency: 1M vs. 5M Steps', fontsize=11, fontweight='bold', color='#2C3E50')
ax.legend(fontsize=9, framealpha=0.85)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_sample_efficiency.png'), dpi=180, bbox_inches='tight')
plt.close()
print('fig_sample_efficiency.png done')

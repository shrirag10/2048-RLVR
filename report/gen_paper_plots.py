"""
Generate all plots for the AAAI-26 paper.

NOTE: DQN and SAC use custom vectorized training with n_envs=4. When configured
for 5M training steps, each step advances 4 envs in parallel, producing ~20M
total env interactions. SB3-based agents (PPO, A2C, QR-DQN) count total env
steps directly. All agents were configured for 5M training steps.

  1. fig_all_agents_learning.png      — All 6 classical agents learning curves
  2. fig_dqn_training.png             — DQN: dual-axis score + max tile
  3. fig_onpolicy_degradation.png     — PPO & QR-DQN over 5M steps
  4. fig_sample_efficiency.png        — Grouped bar: 1M vs 5M with delta labels
  5. fig_dqn_vs_qrdqn.png             — DQN vs QR-DQN head-to-head
  6. fig_grpo_training_dynamics.png   — GRPO: reward + format + direction + KL
  7. fig_grpo_scaling.png             — GRPO scaling bar chart
  8. fig_classical_vs_rlvr.png        — All 8 agents comparison
  9. fig_hunt_mode.png                — Hunt mode attempts vs best tile
 10. fig_param_efficiency.png         — Parameter count vs avg score (log-x)
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT = os.path.dirname(os.path.abspath(__file__))
LOGS = '/home/shrirag10/Projects/RLVR/logs'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
})

# Colors
DQN_COLOR   = '#D35400'
QRDQN_COLOR = '#1F618D'
PPO_COLOR   = '#C0392B'
A2C_COLOR   = '#1A7A4C'
SAC_COLOR   = '#6C3483'
SARSA_COLOR = '#7D6608'
GRPO05_COLOR = '#9B59B6'
GRPO15_COLOR = '#6C3483'

def smooth(series, w=50):
    return pd.Series(series).rolling(w, min_periods=1).mean().values

# DQN and SAC use n_envs=4 in custom vectorized training. Their training_steps
# column counts total env interactions (step * n_envs), so we divide by 4 to get
# the actual training iteration count, making them comparable to SB3 agents.
NENVS_CUSTOM = {'dqn': 4, 'sac': 4}

def load(path):
    df = pd.read_csv(path)
    # Normalize training_steps for custom agents (DQN, SAC)
    basename = os.path.basename(path).replace('_metrics.csv', '')
    if basename in NENVS_CUSTOM:
        df['training_steps'] = df['training_steps'] / NENVS_CUSTOM[basename]
    return df

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1 — All Classical Agents: Learning Curves
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.5))

agents_data = [
    ('DQN',    f'{LOGS}/dqn_5m/dqn_metrics.csv',    DQN_COLOR,   2.0),
    ('SARSA',  f'{LOGS}/lfa_5m/lfa_metrics.csv',     SARSA_COLOR, 1.5),
    ('A2C',    f'{LOGS}/a2c_5m/a2c_metrics.csv',     A2C_COLOR,   1.2),
    ('SAC',    f'{LOGS}/sac_5m/sac_metrics.csv',     SAC_COLOR,   1.2),
    ('PPO',    f'{LOGS}/ppo_5m/ppo_metrics.csv',     PPO_COLOR,   1.2),
    ('QR-DQN', f'{LOGS}/qrdqn_5m/qrdqn_metrics.csv', QRDQN_COLOR, 1.2),
]

for name, path, color, lw in agents_data:
    df = load(path)
    steps = df['training_steps'].values
    scores = smooth(df['total_score'].values, w=80)
    ax.plot(steps / 1e6, scores, color=color, lw=lw, label=name, alpha=0.9, zorder=3 if name == 'DQN' else 2)

ax.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.text(1.05, ax.get_ylim()[1] * 0.02, '1M', fontsize=7, color='gray')
ax.set_xlabel('Training Steps (millions)', fontsize=9)
ax.set_ylabel('Score (smoothed, w=80)', fontsize=9)
ax.set_title('Learning Curves: All Classical Agents (5M Training Steps)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, ncol=2, framealpha=0.8, loc='upper left')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_all_agents_learning.png'))
plt.close()
print('[1/10] fig_all_agents_learning.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 2 — DQN Training: Score + Max Tile (dual axis)
# ══════════════════════════════════════════════════════════════════════════
dqn = load(f'{LOGS}/dqn_5m/dqn_metrics.csv')
fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

steps = dqn['training_steps'].values / 1e6
score_sm = smooth(dqn['total_score'].values, w=100)
tile_sm  = smooth(np.log2(dqn['max_tile'].values.clip(min=2)), w=100)

ax1.plot(steps, score_sm, color=DQN_COLOR, lw=1.8, label='Score')
ax1.set_xlabel('Training Steps (millions)', fontsize=9)
ax1.set_ylabel('Score (smoothed)', fontsize=9, color=DQN_COLOR)
ax1.tick_params(axis='y', labelcolor=DQN_COLOR)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

ax2 = ax1.twinx()
ax2.plot(steps, tile_sm, color='#2980B9', lw=1.5, ls='--', label='Max Tile (log\u2082)', alpha=0.8)
ax2.set_ylabel('Max Tile (log\u2082)', fontsize=9, color='#2980B9')
ax2.tick_params(axis='y', labelcolor='#2980B9')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', framealpha=0.8)

ax1.set_title('DQN Training Over 5M Steps', fontsize=10, fontweight='bold')
ax1.grid(alpha=0.2, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_dqn_training.png'))
plt.close()
print('[2/10] fig_dqn_training.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 3 — On-Policy Degradation: PPO & QR-DQN
# ══════════════════════════════════════════════════════════════════════════
ppo   = load(f'{LOGS}/ppo_5m/ppo_metrics.csv')
qrdqn = load(f'{LOGS}/qrdqn_5m/qrdqn_metrics.csv')

fig, ax = plt.subplots(figsize=(5.5, 3.2))

for df, color, label in [
    (ppo,   PPO_COLOR,   'PPO'),
    (qrdqn, QRDQN_COLOR, 'QR-DQN'),
]:
    steps = df['training_steps'].values / 1e6
    scores = smooth(df['total_score'].values, w=80)
    ax.plot(steps, scores, color=color, lw=1.8, label=label, zorder=3)
    ax.fill_between(steps, scores * 0.8, scores * 1.2, color=color, alpha=0.08, zorder=1)

ax.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.text(1.05, ax.get_ylim()[1] * 0.9, '1M checkpoint', fontsize=7, color='gray')

ax.set_xlabel('Training Steps (millions)', fontsize=9)
ax.set_ylabel('Score (smoothed)', fontsize=9)
ax.set_title('PPO and QR-DQN Over 5M Steps', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_onpolicy_degradation.png'))
plt.close()
print('[3/10] fig_onpolicy_degradation.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 4 — Sample Efficiency: 1M vs 5M
# ══════════════════════════════════════════════════════════════════════════
agents = ['DQN', 'SARSA', 'A2C', 'SAC', 'QR-DQN', 'PPO']
colors_agents = [DQN_COLOR, SARSA_COLOR, A2C_COLOR, SAC_COLOR, QRDQN_COLOR, PPO_COLOR]

scores_1m = [6384, 2455, 1205, 1132, 782, 1038]
scores_5m = [7744, 2456, 1964, 1255, 1003, 957]

x = np.arange(len(agents))
w = 0.36

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars1 = ax.bar(x - w/2, scores_1m, w, label='1M steps',
               color=[c + '80' for c in colors_agents],
               edgecolor='white', linewidth=0.5, zorder=3)
bars5 = ax.bar(x + w/2, scores_5m, w, label='5M steps',
               color=colors_agents,
               edgecolor='white', linewidth=0.5, zorder=3)

for i, (s1, s5) in enumerate(zip(scores_1m, scores_5m)):
    pct = (s5 - s1) / s1 * 100
    sign = '+' if pct >= 0 else ''
    color = '#1A7A4C' if pct >= 0 else '#C0392B'
    ax.text(x[i] + w/2, s5 + 100, f'{sign}{pct:.0f}%', ha='center', va='bottom',
            fontsize=7, fontweight='bold', color=color)

ax.set_xticks(x)
ax.set_xticklabels(agents, fontsize=8)
ax.set_ylabel('Average Score', fontsize=9)
ax.set_title('Sample Efficiency: 1M vs. 5M Training Steps', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_sample_efficiency.png'))
plt.close()
print('[4/10] fig_sample_efficiency.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 5 — DQN vs QR-DQN
# ══════════════════════════════════════════════════════════════════════════
dqn   = load(f'{LOGS}/dqn_5m/dqn_metrics.csv')
qrdqn = load(f'{LOGS}/qrdqn_5m/qrdqn_metrics.csv')

fig, ax = plt.subplots(figsize=(5.5, 3.4))

for df, color, label in [
    (dqn,   DQN_COLOR,   'DQN (avg 7,744)'),
    (qrdqn, QRDQN_COLOR, 'QR-DQN (avg 1,003)'),
]:
    steps = df['training_steps'].values / 1e6
    sm = smooth(df['total_score'].values, w=80)
    ax.plot(steps, sm, color=color, lw=1.8, label=label, zorder=3)
    ax.fill_between(steps, sm * 0.75, sm * 1.25, color=color, alpha=0.08, zorder=1)

ax.axvline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.text(1.05, ax.get_ylim()[1] * 0.9, '1M', fontsize=7, color='gray')

ax.set_xlabel('Training Steps (millions)', fontsize=9)
ax.set_ylabel('Score (smoothed)', fontsize=9)
ax.set_title('DQN vs. QR-DQN (5M Training Steps)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_dqn_vs_qrdqn.png'))
plt.close()
print('[5/10] fig_dqn_vs_qrdqn.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 6 — GRPO Training Dynamics
# ══════════════════════════════════════════════════════════════════════════
grpo_05 = load_jsonl(f'{LOGS}/grpo_0.5b/train_log.jsonl')
grpo_15 = load_jsonl(f'{LOGS}/grpo_1.5b/train_log.jsonl')

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

for df, color, label in [(grpo_05, GRPO05_COLOR, '0.5B'), (grpo_15, GRPO15_COLOR, '1.5B')]:
    sm = smooth(df['reward'].values, w=20)
    axes[0].plot(df['step'], sm, color=color, lw=1.5, label=label)
axes[0].set_title('(a) Total Reward', fontsize=8, fontweight='bold')
axes[0].set_xlabel('Step', fontsize=7)
axes[0].legend(fontsize=7, framealpha=0.8)
axes[0].grid(alpha=0.2)

for df, color, label in [(grpo_05, GRPO05_COLOR, '0.5B'), (grpo_15, GRPO15_COLOR, '1.5B')]:
    fmt = smooth(df['rewards/format_reward_fn/mean'].values, w=20)
    dire = smooth(df['rewards/direction_reward_fn/mean'].values, w=20)
    axes[1].plot(df['step'], fmt, color=color, lw=1.5, ls='-', label=f'{label} format')
    axes[1].plot(df['step'], dire, color=color, lw=1.5, ls='--', label=f'{label} direction', alpha=0.7)
axes[1].set_title('(b) Format & Direction', fontsize=8, fontweight='bold')
axes[1].set_xlabel('Step', fontsize=7)
axes[1].legend(fontsize=5.5, framealpha=0.8, ncol=2)
axes[1].grid(alpha=0.2)

for df, color, label in [(grpo_05, GRPO05_COLOR, '0.5B'), (grpo_15, GRPO15_COLOR, '1.5B')]:
    kl = smooth(df['kl'].values, w=20)
    axes[2].plot(df['step'], kl, color=color, lw=1.5, label=label)
axes[2].set_title('(c) KL Divergence', fontsize=8, fontweight='bold')
axes[2].set_xlabel('Step', fontsize=7)
axes[2].legend(fontsize=7, framealpha=0.8)
axes[2].set_yscale('log')
axes[2].grid(alpha=0.2)

for ax in axes:
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)

fig.suptitle('GRPO Training Dynamics: 0.5B vs. 1.5B', fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_grpo_training_dynamics.png'))
plt.close()
print('[6/10] fig_grpo_training_dynamics.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 7 — GRPO Scaling Bar
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.0, 3.2))

scores = [704, 2348]
labels = ['Qwen2.5\n0.5B', 'Qwen2.5\n1.5B']
colors = [GRPO05_COLOR, GRPO15_COLOR]
bars = ax.bar(labels, scores, color=colors, width=0.45, zorder=3, edgecolor='white', linewidth=0.5)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            f'{score:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('', xy=(1, 2348), xytext=(0, 704),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.0))
ax.text(0.55, 1500, '3x params\n3.3x score', fontsize=9, color='#C0392B', fontweight='bold')

ax.set_ylabel('Average Score', fontsize=9)
ax.set_title('GRPO: Model Scale Effect', fontsize=10, fontweight='bold')
ax.set_ylim(0, 2900)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_grpo_scaling.png'))
plt.close()
print('[7/10] fig_grpo_scaling.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 8 — Classical vs RLVR: All 8 agents
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.8, 4.2))

names  = ['DQN\n(5M)', 'SARSA\n(5M)', 'Qwen\n1.5B', 'A2C\n(5M)', 'SAC\n(5M)', 'PPO\n(5M)', 'QR-DQN\n(5M)', 'Qwen\n0.5B']
scores = [7744,        2456,          2348,          1964,        1255,        957,         1003,            704]
colors_list = [DQN_COLOR,   SARSA_COLOR,   GRPO15_COLOR,  A2C_COLOR,   SAC_COLOR,   PPO_COLOR,   QRDQN_COLOR,    GRPO05_COLOR]
edge   = ['none']*2 + ['#2C3E50'] + ['none']*4 + ['#2C3E50']

bars = ax.bar(range(len(names)), scores, color=colors_list, edgecolor=edge, linewidth=[0]*2+[1.5]+[0]*4+[1.5],
              width=0.65, zorder=3)

for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{s:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_xlabel('Agent (training budget)', fontsize=9)
ax.set_ylabel('Average Score', fontsize=9)
ax.set_title('All Agents: Classical RL vs. RLVR', fontsize=10, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

from matplotlib.patches import Patch
ax.legend([Patch(facecolor='#555555', edgecolor='none'),
           Patch(facecolor='white', edgecolor='#2C3E50', linewidth=1.5)],
          ['Classical RL', 'GRPO-LLM'], fontsize=8, framealpha=0.9, loc='upper right')

ax.grid(axis='y', alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
fig.savefig(os.path.join(OUT, 'fig_classical_vs_rlvr.png'))
plt.close()
print('[8/10] fig_classical_vs_rlvr.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 9 — Hunt Mode Results
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.0, 2.8))

hunt_agents   = ['DQN', 'SARSA', 'A2C', 'SAC', 'PPO']
hunt_attempts = [3902, 3000, 3000, 3000, 193]
hunt_tiles    = [2048, 1024, 512, 256, 64]
hunt_colors   = [DQN_COLOR, SARSA_COLOR, A2C_COLOR, SAC_COLOR, PPO_COLOR]

y = range(len(hunt_agents))
ax.barh(y, hunt_attempts, color=hunt_colors, height=0.55, zorder=3, edgecolor='white', linewidth=0.5)

for i, (att, tile) in enumerate(zip(hunt_attempts, hunt_tiles)):
    ax.text(att + 50, i, f'Best tile: {tile}', va='center', fontsize=7, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(hunt_agents, fontsize=8)
ax.set_xlabel('Attempts Used', fontsize=9)
ax.set_title('Hunt Mode: Attempts to Reach Tile 2048', fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.axvline(3000, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.text(3020, 4.3, 'Budget\n(3,000)', fontsize=6, color='gray')
ax.grid(axis='x', alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_hunt_mode.png'))
plt.close()
print('[9/10] fig_hunt_mode.png')


# ══════════════════════════════════════════════════════════════════════════
# Plot 10 — Parameter Efficiency (log-x scatter)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.0))

params_all = [120,    330_000,  330_000,  330_000,  330_000,  330_000,  494_000_000, 1_800_000_000]
scores_all = [2456,   7744,     957,      1964,     1255,     1003,     704,         2348]
names_all  = ['SARSA','DQN',    'PPO',    'A2C',    'SAC',    'QR-DQN', 'Qwen 0.5B', 'Qwen 1.5B']
colors_all = [SARSA_COLOR, DQN_COLOR, PPO_COLOR, A2C_COLOR, SAC_COLOR, QRDQN_COLOR, GRPO05_COLOR, GRPO15_COLOR]

# Hand-tuned per-point offsets (in points) and horizontal alignments
annotation_offsets = {
    'SARSA':     ((8,   0),   'left'),
    'DQN':       ((10, -3),   'left'),
    'A2C':       ((10,  8),   'left'),
    'SAC':       ((10, -3),   'left'),
    'QR-DQN':    ((-10, -12), 'right'),
    'PPO':       ((10, -12),  'left'),
    'Qwen 0.5B': ((-12,  -3),  'right'),
    'Qwen 1.5B': ((12,   -3),  'left'),
}

try:
    from adjustText import adjust_text
    _has_adjust_text = True
except ImportError:
    _has_adjust_text = False

texts = []
for p, s, n, c in zip(params_all, scores_all, names_all, colors_all):
    ax.scatter(p, s, color=c, s=80, zorder=3, edgecolor='white', linewidth=1.0)
    if _has_adjust_text:
        texts.append(ax.text(p, s, n, fontsize=7, color=c, fontweight='bold'))
    else:
        (dx, dy), ha = annotation_offsets[n]
        ax.annotate(n, (p, s), textcoords='offset points',
                    xytext=(dx, dy), ha=ha,
                    fontsize=7, color=c, fontweight='bold')

if _has_adjust_text:
    adjust_text(texts, ax=ax, expand_points=(1.4, 1.6),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5))

ax.set_xscale('log')

def _fmt_params(x, _):
    if x <= 0:
        return ''
    if x >= 1e9:
        return f'{x/1e9:g}B'
    if x >= 1e6:
        return f'{x/1e6:g}M'
    if x >= 1e3:
        return f'{x/1e3:g}K'
    return f'{x:g}'

ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_params))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.tick_params(axis='x', labelsize=7.5)
ax.set_xlim(80, 3e9)
ax.set_xlabel('Parameters (log scale)', fontsize=9)
ax.set_ylabel('Average Score (5M training steps)', fontsize=9)
ax.set_title('Parameter Efficiency: Score vs. Model Size', fontsize=10, fontweight='bold')
ax.grid(alpha=0.2, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_param_efficiency.png'))
plt.close()
print('[10/10] fig_param_efficiency.png')

print('\nAll plots generated successfully.')

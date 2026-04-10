#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  GRPO Multi-Agent Pipeline
#  Runs Agent 1 (0.5B) and Agent 2 (1.5B) with staged training
#  Total time: ~3 hours on RTX 4060 8GB
# ═══════════════════════════════════════════════════════════
set -e

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

echo "═══════════════════════════════════════════"
echo "  GRPO Pipeline — $(date)"
echo "═══════════════════════════════════════════"

# ─── Agent 1: Qwen2.5-0.5B ───────────────────────────────
AGENT1_DIR="logs/grpo_0.5b"

echo ""
echo "▶ Agent 1: Qwen2.5-0.5B — Stage 1 (format + direction)"
$PYTHON -m src.llm.train_grpo \
    --config configs/grpo_0.5b.yaml \
    --output-dir "$AGENT1_DIR" \
    --stage 1 --steps 500

echo ""
echo "▶ Agent 1: Qwen2.5-0.5B — Stage 2 (+ game reward)"
$PYTHON -m src.llm.train_grpo \
    --config configs/grpo_0.5b.yaml \
    --output-dir "$AGENT1_DIR" \
    --resume "$AGENT1_DIR/adapter" \
    --stage 2 --steps 500

echo ""
echo "▶ Agent 1: Generating replay..."
$PYTHON -m src.llm.replay_gen \
    --model-path "$AGENT1_DIR/adapter" \
    --log-dir "$AGENT1_DIR" \
    --manifest logs/manifest.json \
    --max-turns 500 || echo "  [warn] Replay gen failed for 0.5B — skipping"

echo "✅ Agent 1 complete: $AGENT1_DIR"

# ─── Agent 2: Qwen2.5-1.5B ───────────────────────────────
AGENT2_DIR="logs/grpo_1.5b"

echo ""
echo "▶ Agent 2: Qwen2.5-1.5B — Stage 1 (format + direction)"
$PYTHON -m src.llm.train_grpo \
    --config configs/grpo_1.5b.yaml \
    --output-dir "$AGENT2_DIR" \
    --stage 1 --steps 300

echo ""
echo "▶ Agent 2: Qwen2.5-1.5B — Stage 2 (+ game reward)"
$PYTHON -m src.llm.train_grpo \
    --config configs/grpo_1.5b.yaml \
    --output-dir "$AGENT2_DIR" \
    --resume "$AGENT2_DIR/adapter" \
    --stage 2 --steps 500

echo ""
echo "▶ Agent 2: Qwen2.5-1.5B — Stage 3 (all rewards)"
$PYTHON -m src.llm.train_grpo \
    --config configs/grpo_1.5b.yaml \
    --output-dir "$AGENT2_DIR" \
    --resume "$AGENT2_DIR/adapter" \
    --stage 3 --steps 500

echo ""
echo "▶ Agent 2: Generating replay..."
$PYTHON -m src.llm.replay_gen \
    --model-path "$AGENT2_DIR/adapter" \
    --log-dir "$AGENT2_DIR" \
    --manifest logs/manifest.json \
    --max-turns 500 || echo "  [warn] Replay gen failed for 1.5B — skipping"

echo "✅ Agent 2 complete: $AGENT2_DIR"

# ─── Summary ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Pipeline Complete — $(date)"
echo "═══════════════════════════════════════════"
echo "  Agent 1 (0.5B): $AGENT1_DIR"
echo "  Agent 2 (1.5B): $AGENT2_DIR"
echo ""
echo "  Dashboard manifests auto-updated."

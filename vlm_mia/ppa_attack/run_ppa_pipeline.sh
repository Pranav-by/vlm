#!/bin/bash
# =============================================================================
# Prompt Perturbation Attack (Attack 6) — Standalone Pipeline Runner
#
# A novel membership inference attack against Vision-Language Models that
# exploits cross-prompt response consistency as a memorization signal.
#
# This script is fully self-contained. It only requires:
#   1. The vlm_mia conda environment
#   2. Member/non-member data in vlm_mia/data/ (from 01_generate_synthetic_data.py)
#
# Usage:
#   conda activate vlm_mia
#   bash ppa_attack/run_ppa_pipeline.sh
#
# Options (pass as env vars):
#   GRANULARITY=50    Number of samples per group per iteration (default: 50)
#   METRIC=rouge2_f   Similarity metric: rouge2_f | embedding_mpn (default: rouge2_f)
#   COMPUTE=0         Set COMPUTE=1 to compute real Rouge-2+MPNet from conversations
#                     (slow, ~minutes on CPU). Default: direct score generation (fast).
# =============================================================================

set -e

PPA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$PPA_DIR")"
DATA_DIR="$PROJECT_DIR/data"
SIM_DIR="$DATA_DIR/similarity"

GRANULARITY="${GRANULARITY:-50}"
METRIC="${METRIC:-rouge2_f}"
COMPUTE="${COMPUTE:-0}"

echo "============================================================"
echo "  Prompt Perturbation Attack (Attack 6)"
echo "  Novel Membership Inference for Vision-Language Models"
echo "============================================================"
echo "  Project dir:  $PROJECT_DIR"
echo "  Data dir:     $DATA_DIR"
echo "  Granularity:  $GRANULARITY"
echo "  Metric:       $METRIC"
echo "  Compute mode: $([ "$COMPUTE" = "1" ] && echo "REAL (Rouge-2+MPNet)" || echo "DIRECT (fast synthetic)")"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vlm_mia

# ─── Check prerequisites ──────────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/member_data.json" ]; then
    echo "ERROR: $DATA_DIR/member_data.json not found!"
    echo "Run the base synthetic pipeline first:"
    echo "  bash $PROJECT_DIR/scripts/run_fast_pipeline.sh"
    echo "Or at minimum:"
    echo "  conda activate vlm_mia"
    echo "  python $PROJECT_DIR/scripts/01_generate_synthetic_data.py"
    exit 1
fi

mkdir -p "$SIM_DIR"

# ─── STEP 1: Generate PPA conversation data ───────────────────────────────────
echo "============================================================"
echo "  STEP 1: Generating PPA conversation data"
echo "  (K=12 prompt variants × 1000 images)"
echo "============================================================"
python "$PPA_DIR/01_generate_data.py"

# ─── STEP 2: Compute cross-prompt consistency scores ─────────────────────────
echo ""
echo "============================================================"
echo "  STEP 2: Computing cross-prompt consistency scores"
echo "============================================================"

if [ "$COMPUTE" = "1" ]; then
    python "$PPA_DIR/02_generate_similarity.py" --compute
else
    python "$PPA_DIR/02_generate_similarity.py"
fi

# ─── STEP 3: Run PPA inference attack ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 3: Running Prompt Perturbation Attack"
echo "============================================================"
python "$PPA_DIR/03_run_attack.py" \
    --member_similarity_file  "$SIM_DIR/similarity_member_ppa.json" \
    --non_member_similarity_file "$SIM_DIR/similarity_non_member_ppa.json" \
    --granularity "$GRANULARITY" \
    --similarity_metric "$METRIC"

echo ""
echo "============================================================"
echo "  PROMPT PERTURBATION ATTACK — COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  $DATA_DIR/conversation_member_ppa.json"
echo "  $DATA_DIR/conversation_non_member_ppa.json"
echo "  $SIM_DIR/similarity_member_ppa.json"
echo "  $SIM_DIR/similarity_non_member_ppa.json"
echo ""
echo "To compare with all 5 original attacks:"
echo "  bash $PROJECT_DIR/scripts/run_fast_pipeline.sh"
echo "============================================================"

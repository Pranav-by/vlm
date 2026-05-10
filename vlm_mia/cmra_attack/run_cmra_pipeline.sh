#!/bin/bash
# =============================================================================
# Contrastive Misinformation Resistance Attack (CMRA) — Standalone Pipeline
#
# Attack 7: Novel black-box membership inference for Vision-Language Models.
#
# Signal: How specifically and confidently does the VLM correct a deliberately
#         wrong image description?
#
#   Member   → long, specific, negation-heavy correction (knows the real answer)
#   Non-member → short, vague, hedging response (no memorized anchor)
#
# This script is fully self-contained. Requirements:
#   conda environment: vlm_mia
#   prerequisite data: vlm_mia/data/member_data.json (from 01_generate_synthetic_data.py)
#
# Usage:
#   conda activate vlm_mia
#   bash cmra_attack/run_cmra_pipeline.sh
#
# Environment variables (optional):
#   GRANULARITY=50    Samples per group per iteration (default: 50)
#   METRIC=correction_specificity   Score component to use (default)
#   COMPUTE=0         Set to 1 to compute real scores from conversation files
# =============================================================================

set -e

CMRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$CMRA_DIR")"
DATA_DIR="$PROJECT_DIR/data"
SIM_DIR="$DATA_DIR/similarity"

GRANULARITY="${GRANULARITY:-50}"
METRIC="${METRIC:-correction_specificity}"
COMPUTE="${COMPUTE:-0}"

echo "============================================================"
echo "  Attack 7: Contrastive Misinformation Resistance (CMRA)"
echo "  Novel Membership Inference for Vision-Language Models"
echo "============================================================"
echo "  Project dir:  $PROJECT_DIR"
echo "  Data dir:     $DATA_DIR"
echo "  Granularity:  $GRANULARITY"
echo "  Metric:       $METRIC"
echo "  Mode:         $([ "$COMPUTE" = "1" ] && echo "COMPUTE (real word-count scores)" || echo "DIRECT (fast synthetic)")"
echo ""

# Activate conda
eval "$(conda shell.bash hook)"
conda activate vlm_mia

# ─── Prerequisites check ──────────────────────────────────────────────────────
if [ ! -f "$DATA_DIR/member_data.json" ]; then
    echo "ERROR: $DATA_DIR/member_data.json not found!"
    echo "Run the base synthetic pipeline first:"
    echo "  conda activate vlm_mia"
    echo "  python $PROJECT_DIR/scripts/01_generate_synthetic_data.py"
    exit 1
fi

mkdir -p "$SIM_DIR"

# ─── STEP 1: Generate CMRA conversation data ──────────────────────────────────
echo "============================================================"
echo "  STEP 1: Generating misleading-query conversations"
echo "  (false descriptions + correction responses)"
echo "============================================================"
python "$CMRA_DIR/01_generate_data.py"

# ─── STEP 2: Compute correction specificity scores ────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 2: Computing correction specificity scores"
echo "  (length + negation + content + confidence)"
echo "============================================================"

if [ "$COMPUTE" = "1" ]; then
    python "$CMRA_DIR/02_generate_scores.py" --compute
else
    python "$CMRA_DIR/02_generate_scores.py"
fi

# ─── STEP 3: Run CMRA inference attack ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 3: Running CMRA inference"
echo "  (group-level AUC over correction specificity scores)"
echo "============================================================"
python "$CMRA_DIR/03_run_attack.py" \
    --member_score_file     "$SIM_DIR/similarity_member_cmra.json" \
    --non_member_score_file "$SIM_DIR/similarity_non_member_cmra.json" \
    --granularity           "$GRANULARITY" \
    --similarity_metric     "$METRIC"

echo ""
echo "============================================================"
echo "  CMRA ATTACK COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  $DATA_DIR/conversation_member_cmra.json"
echo "  $DATA_DIR/conversation_non_member_cmra.json"
echo "  $SIM_DIR/similarity_member_cmra.json"
echo "  $SIM_DIR/similarity_non_member_cmra.json"
echo ""
echo "To also run the other attacks:"
echo "  bash $PROJECT_DIR/scripts/run_fast_pipeline.sh   (Attacks 1-5)"
echo "  bash $PROJECT_DIR/ppa_attack/run_ppa_pipeline.sh (Attack 6: PPA)"
echo "============================================================"

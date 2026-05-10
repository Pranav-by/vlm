#!/bin/bash
# =============================================================================
# Run inference attacks on REAL data (after downloading from Lightning AI)
# =============================================================================
# This script runs LOCALLY on your machine (no GPU needed).
# Run AFTER placing real_data/ from Lightning AI at ~/vlm_mia/real_data/
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REAL_DATA_DIR="$PROJECT_DIR/real_data"
SIM_DIR="$REAL_DATA_DIR/similarity"

echo "============================================================"
echo "  VLM-MIA: Inference Attacks on REAL Data"
echo "============================================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vlm_mia

# Check data exists
if [ ! -d "$SIM_DIR" ]; then
    echo "ERROR: $SIM_DIR not found!"
    echo "Please download real_data/ from Lightning AI first."
    echo "See real_pipeline/README_REAL_PIPELINE.md for instructions."
    exit 1
fi

echo "  Data directory: $REAL_DATA_DIR"
echo "  Similarity directory: $SIM_DIR"
echo ""

SHADOW_TEMPS="0.1 0.5 1.0 1.5"

# -------------------------------------------------------
# Attack 1: Shadow Model Inference
# -------------------------------------------------------
echo "--- Attack 1: Shadow Model Inference (Real Data) ---"
echo "    (3-layer binary NN classifier on 4 temperature features)"
if [ -f "$SIM_DIR/similarity_shadow_member_shadow.json" ] && \
   [ -f "$SIM_DIR/similarity_shadow_non_member_shadow.json" ] && \
   [ -f "$SIM_DIR/similarity_target_member_shadow.json" ] && \
   [ -f "$SIM_DIR/similarity_target_non_member_shadow.json" ]; then
    python "$PROJECT_DIR/shadow_model_inference.py" \
        --shadow_member_similarity_file "$SIM_DIR/similarity_shadow_member_shadow.json" \
        --shadow_non_member_similarity_file "$SIM_DIR/similarity_shadow_non_member_shadow.json" \
        --target_member_similarity_file "$SIM_DIR/similarity_target_member_shadow.json" \
        --target_non_member_similarity_file "$SIM_DIR/similarity_target_non_member_shadow.json" \
        --granularity 15 \
        --temperatures $SHADOW_TEMPS \
        --similarity_metric rouge2_f \
        --with_variance \
        --epochs 30
else
    echo "    SKIPPED: Shadow similarity files not found"
fi

# -------------------------------------------------------
# Attack 2: Reference Member Inference
# -------------------------------------------------------
echo ""
echo "--- Attack 2: Reference Member Inference (Real Data) ---"
echo "    (Z-test comparing target vs reference member set)"
if [ -f "$SIM_DIR/similarity_member_reference.json" ] && \
   [ -f "$SIM_DIR/similarity_non_member_reference.json" ]; then
    python "$PROJECT_DIR/reference_member_inference.py" \
        --member_similarity_file "$SIM_DIR/similarity_member_reference.json" \
        --non_member_similarity_file "$SIM_DIR/similarity_non_member_reference.json" \
        --granularity 50 \
        --temperature 0.1 \
        --similarity_metric rouge2_f
else
    echo "    SKIPPED: Reference similarity files not found"
fi

# -------------------------------------------------------
# Attack 3: Reference Non-Member Inference
# -------------------------------------------------------
echo ""
echo "--- Attack 3: Reference Non-Member Inference (Real Data) ---"
echo "    (Z-test comparing target vs reference non-member set)"
if [ -f "$SIM_DIR/similarity_member_reference.json" ] && \
   [ -f "$SIM_DIR/similarity_non_member_reference.json" ]; then
    python "$PROJECT_DIR/reference_non_member_inference.py" \
        --member_similarity_file "$SIM_DIR/similarity_member_reference.json" \
        --non_member_similarity_file "$SIM_DIR/similarity_non_member_reference.json" \
        --granularity 50 \
        --temperature 0.1 \
        --similarity_metric rouge2_f
else
    echo "    SKIPPED: Reference similarity files not found"
fi

# -------------------------------------------------------
# Attack 4: Target-Only Inference
# -------------------------------------------------------
echo ""
echo "--- Attack 4: Target-Only Inference (Real Data) ---"
echo "    (Z-test on temperature sensitivity: 0.1 vs 1.5)"
if [ -f "$SIM_DIR/similarity_member_target_only.json" ] && \
   [ -f "$SIM_DIR/similarity_non_member_target_only.json" ]; then
    python "$PROJECT_DIR/target_only_inference.py" \
        --member_similarity_file "$SIM_DIR/similarity_member_target_only.json" \
        --non_member_similarity_file "$SIM_DIR/similarity_non_member_target_only.json" \
        --granularity 50 \
        --temperature_low 0.1 \
        --temperature_high 1.5 \
        --similarity_metric rouge2_f
else
    echo "    SKIPPED: Target-only similarity files not found"
fi

echo ""
echo "============================================================"
echo "  REAL DATA ATTACKS COMPLETE!"
echo "============================================================"
echo ""
echo "Compare these REAL results with the synthetic results from:"
echo "  bash ~/vlm_mia/scripts/run_fast_pipeline.sh"
echo ""
echo "Synthetic results used crafted overlapping distributions."
echo "Real results use actual LLaVA-7B QLoRA model output."
echo "============================================================"

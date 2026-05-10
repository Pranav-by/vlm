#!/bin/bash
# =============================================================================
# VLM-MIA: Fast Pipeline Runner (CPU-only, no GPU required)
# Runs the 5 original baseline attacks.
# For the novel Prompt Perturbation Attack (Attack 6) run:
#   bash ppa_attack/run_ppa_pipeline.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
SIMILARITY_DIR="$DATA_DIR/similarity"

echo "============================================================"
echo "  VLM-MIA: Membership Inference Attacks Against VLMs"
echo "  Full Pipeline (CPU-only, fast mode)"
echo "============================================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vlm_mia

# -------------------------------------------------
# PHASE 1: Generate Synthetic Data
# -------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 1: Generating Synthetic Data"
echo "============================================================"
python "$SCRIPT_DIR/01_generate_synthetic_data.py"

# -------------------------------------------------
# PHASE 2: Generate Similarity Scores
# -------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 2: Generating Similarity Scores (Direct Mode)"
echo "============================================================"
python "$SCRIPT_DIR/02_generate_similarity_scores.py"

# -------------------------------------------------
# PHASE 3: Run Inference Attacks
# -------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 3: Running Inference Attacks"
echo "============================================================"
SHADOW_TEMPS="0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8"

# Attack 1: Shadow Model Inference (Binary NN classifier)
# granularity=15 because shadow non-member validation has only ~20 samples
echo ""
echo "--- Attack 1: Shadow Model Inference ---"
echo "    (Trains a 3-layer binary neural network classifier)"
python "$PROJECT_DIR/shadow_model_inference.py" \
    --shadow_member_similarity_file "$SIMILARITY_DIR/similarity_shadow_member_shadow.json" \
    --shadow_non_member_similarity_file "$SIMILARITY_DIR/similarity_shadow_non_member_shadow.json" \
    --target_member_similarity_file "$SIMILARITY_DIR/similarity_target_member_shadow.json" \
    --target_non_member_similarity_file "$SIMILARITY_DIR/similarity_target_non_member_shadow.json" \
    --granularity 15 \
    --temperatures $SHADOW_TEMPS \
    --similarity_metric rouge2_f \
    --with_variance \
    --epochs 30

# Attack 2: Reference Member Inference (Z-test)
echo ""
echo "--- Attack 2: Reference Member Inference ---"
echo "    (Z-test comparing target vs reference member set)"
python "$PROJECT_DIR/reference_member_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_reference.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_reference.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

# Attack 3: Reference Non-Member Inference (Z-test)
echo ""
echo "--- Attack 3: Reference Non-Member Inference ---"
echo "    (Z-test comparing target vs reference non-member set)"
python "$PROJECT_DIR/reference_non_member_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_reference.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_reference.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

# Attack 4: Target-Only Inference (Z-test on temperature sensitivity)
echo ""
echo "--- Attack 4: Target-Only Inference ---"
echo "    (Z-test evaluating variance between high/low temperature)"
python "$PROJECT_DIR/target_only_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_target_only.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_target_only.json" \
    --granularity 50 \
    --temperature_low 0.1 \
    --temperature_high 1.5 \
    --similarity_metric rouge2_f

# Attack 5: Image-Only Inference (Pairwise similarity)
echo ""
echo "--- Attack 5: Image-Only Inference ---"
echo "    (Average pairwise similarity of repeated image queries)"
python "$PROJECT_DIR/image_only_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_image_only.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_image_only.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

echo ""
echo "============================================================"
echo "  ALL 5 ATTACKS COMPLETE!"
echo "============================================================"
echo ""
echo "Interpretation of Results:"
echo "  AUC > 0.5 = attack succeeds (can distinguish members/non-members)"
echo "  AUC = 1.0 = perfect attack (complete membership leakage)"
echo "  AUC = 0.5 = random guessing (no leakage detected)"
echo ""
echo "Attack Summary:"
echo "  Attack 1 (Shadow Model):        Most info, strongest signal"
echo "  Attack 2 (Reference Member):    Z-test, member reference set"
echo "  Attack 3 (Reference Non-Member):Z-test, non-member reference set"
echo "  Attack 4 (Target-Only):         Weakest — no reference, 2 temps"
echo "  Attack 5 (Image-Only):          Pairwise repeated query consistency"
echo ""
echo "Higher AUC = more VLM membership leakage = privacy risk"
echo ""
echo "============================================================"

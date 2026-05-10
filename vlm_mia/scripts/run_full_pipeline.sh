#!/bin/bash
# =============================================================================
# VLM-MIA: Full Pipeline Runner (CPU-only, no GPU required)
# =============================================================================
# This script runs the complete Membership Inference Attack pipeline:
#   Phase 1: Generate synthetic data (mimics VLM conversation output)
#   Phase 2: Compute text similarity (Rouge-2 + MPNet)
#   Phase 3: Run all 4 inference attacks and report AUC scores
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
SIMILARITY_DIR="$DATA_DIR/similarity"

echo "============================================================"
echo "  VLM-MIA: Membership Inference Attacks Against VLMs"
echo "  Full Pipeline (CPU-only mode)"
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
# PHASE 2: Compute Similarities
# -------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 2: Computing Text Similarities"
echo "============================================================"
mkdir -p "$SIMILARITY_DIR"

# 2a: Shadow Model Attack - all 16 temperatures
echo ""
echo "--- Shadow Model Attack Similarity ---"
SHADOW_TEMPS="0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8"

for GROUP in shadow_member shadow_non_member target_member target_non_member; do
    echo "  Processing $GROUP..."
    python "$SCRIPT_DIR/02_compute_similarity.py" \
        --mode ground_truth \
        --conversation_json_path "$DATA_DIR/conversation_${GROUP}_shadow.json" \
        --similarity_json_path "$SIMILARITY_DIR/similarity_${GROUP}_shadow.json" \
        --temperatures $SHADOW_TEMPS
done

# 2b: Reference Attack - single temperature 0.1
echo ""
echo "--- Reference Attack Similarity ---"
for GROUP in member non_member; do
    echo "  Processing $GROUP..."
    python "$SCRIPT_DIR/02_compute_similarity.py" \
        --mode ground_truth \
        --conversation_json_path "$DATA_DIR/conversation_${GROUP}_reference.json" \
        --similarity_json_path "$SIMILARITY_DIR/similarity_${GROUP}_reference.json" \
        --temperatures 0.1
done

# 2c: Target-Only Attack - two temperatures (0.1 and 1.5)
echo ""
echo "--- Target-Only Attack Similarity ---"
for GROUP in member non_member; do
    echo "  Processing $GROUP..."
    python "$SCRIPT_DIR/02_compute_similarity.py" \
        --mode ground_truth \
        --conversation_json_path "$DATA_DIR/conversation_${GROUP}_target_only.json" \
        --similarity_json_path "$SIMILARITY_DIR/similarity_${GROUP}_target_only.json" \
        --temperatures 0.1 1.5
done

# 2d: Image-Only Attack - single temperature, 5 repeats
echo ""
echo "--- Image-Only Attack Similarity ---"
for GROUP in member non_member; do
    echo "  Processing $GROUP..."
    python "$SCRIPT_DIR/02_compute_similarity.py" \
        --mode repeating \
        --conversation_json_path "$DATA_DIR/conversation_${GROUP}_image_only.json" \
        --similarity_json_path "$SIMILARITY_DIR/similarity_${GROUP}_image_only.json" \
        --temperatures 0.1 \
        --repeating_num 5
done

# -------------------------------------------------
# PHASE 3: Run Inference Attacks
# -------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 3: Running Inference Attacks"
echo "============================================================"

# 3a: Shadow Model Attack
echo ""
echo "--- Attack 1: Shadow Model Inference ---"
python "$PROJECT_DIR/shadow_model_inference.py" \
    --shadow_member_similarity_file "$SIMILARITY_DIR/similarity_shadow_member_shadow.json" \
    --shadow_non_member_similarity_file "$SIMILARITY_DIR/similarity_shadow_non_member_shadow.json" \
    --target_member_similarity_file "$SIMILARITY_DIR/similarity_target_member_shadow.json" \
    --target_non_member_similarity_file "$SIMILARITY_DIR/similarity_target_non_member_shadow.json" \
    --granularity 50 \
    --temperatures $SHADOW_TEMPS \
    --similarity_metric rouge2_f \
    --with_variance

# 3b: Reference Member Inference
echo ""
echo "--- Attack 2: Reference Member Inference ---"
python "$PROJECT_DIR/reference_member_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_reference.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_reference.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

# 3c: Reference Non-Member Inference
echo ""
echo "--- Attack 3: Reference Non-Member Inference ---"
python "$PROJECT_DIR/reference_non_member_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_reference.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_reference.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

# 3d: Target-Only Inference
echo ""
echo "--- Attack 4: Target-Only Inference ---"
python "$PROJECT_DIR/target_only_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_target_only.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_target_only.json" \
    --granularity 50 \
    --temperature_low 0.1 \
    --temperature_high 1.5 \
    --similarity_metric rouge2_f

# 3e: Image-Only Inference
echo ""
echo "--- Attack 5: Image-Only Inference ---"
python "$PROJECT_DIR/image_only_inference.py" \
    --member_similarity_file "$SIMILARITY_DIR/similarity_member_image_only.json" \
    --non_member_similarity_file "$SIMILARITY_DIR/similarity_non_member_image_only.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity_metric rouge2_f

echo ""
echo "============================================================"
echo "  ALL ATTACKS COMPLETE"
echo "============================================================"
echo ""
echo "Results Summary:"
echo "  - Shadow Model Attack: Uses binary neural network classifier"
echo "  - Reference (Member): Uses Z-test with member reference set"
echo "  - Reference (Non-Member): Uses Z-test with non-member reference set"
echo "  - Target-Only: Uses Z-test on temperature sensitivity"
echo "  - Image-Only: Uses average pairwise similarity of repeated queries"
echo ""
echo "AUC > 0.5 indicates the attack can distinguish members from non-members"
echo "Higher AUC = more successful attack = more membership leakage"
echo "============================================================"

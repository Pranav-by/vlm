"""
Contrastive Misinformation Resistance Attack (CMRA) — Inference

Membership inference based on how specifically and confidently a VLM
corrects a deliberately false image description.

Core hypothesis:
  A VLM that memorized an image-text pair "knows" the correct answer.
  When presented with a wrong description, it pushes back with a
  precise, detailed, confident correction that echoes its training label.

  A non-member VLM has no memorized anchor — it hedges, partially agrees,
  or gives vague generic corrections.

Attack signal:
  High correction specificity  →  MEMBER
  Low correction specificity   →  NON-MEMBER

Score breakdown:
  correction_specificity = mean(length + negation + content + confidence)
    length:     how long the correction is (longer = more specific)
    negation:   how many "not/incorrect/actually" words appear
    content:    density of non-stop content words
    confidence: inverse of hedging words ("maybe/possibly/seems")

Usage:
  python cmra_attack/03_run_attack.py \\
      --member_score_file  data/similarity/similarity_member_cmra.json \\
      --non_member_score_file data/similarity/similarity_non_member_cmra.json \\
      --granularity 50 \\
      --similarity_metric correction_specificity
"""

import json
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_scores(member_file, non_member_file, metric):
    """
    Load CMRA scores. Supports the cmra_score dict format:
      [{"image_id": ..., "cmra_score": {"correction_specificity": 0.67, ...}}, ...]
    """
    with open(member_file) as f:
        member_raw = json.load(f)
    with open(non_member_file) as f:
        non_member_raw = json.load(f)

    def extract(item):
        if "cmra_score" in item:
            return float(item["cmra_score"][metric])
        # Fallback: try root-level key
        if metric in item:
            return float(item[metric])
        raise KeyError(f"Cannot find metric '{metric}' in: {list(item.keys())}")

    member_scores = [extract(x) for x in member_raw]
    non_member_scores = [extract(x) for x in non_member_raw]
    return member_scores, non_member_scores


# ─── Attack inference ─────────────────────────────────────────────────────────

def cmra_inference(member_data, non_member_data, granularity, n_iterations=1000):
    """
    CMRA inference via group-mean AUC.

    Each iteration:
      1. Sample `granularity` member and non-member correction scores
      2. Compute mean correction specificity per group
      3. Append (score, label) pairs: members should have higher mean

    Returns AUC-ROC.
    """
    if len(member_data) < granularity:
        raise ValueError(
            f"Not enough member samples ({len(member_data)}) "
            f"for granularity={granularity}. Reduce --granularity."
        )
    if len(non_member_data) < granularity:
        raise ValueError(
            f"Not enough non-member samples ({len(non_member_data)}) "
            f"for granularity={granularity}. Reduce --granularity."
        )

    score_list = []
    label_list = []

    for _ in range(n_iterations):
        m_sample  = random.sample(member_data,     granularity)
        nm_sample = random.sample(non_member_data, granularity)

        # Higher mean correction specificity → member
        score_list.append(float(np.mean(m_sample)))
        label_list.append(1)  # member

        score_list.append(float(np.mean(nm_sample)))
        label_list.append(0)  # non-member

    return roc_auc_score(label_list, score_list)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Attack 7: Contrastive Misinformation Resistance (CMRA)")
    print("  (Correction specificity when shown a false description)")
    print("=" * 60)
    print(f"  Member file:      {args.member_score_file}")
    print(f"  Non-member file:  {args.non_member_score_file}")
    print(f"  Metric:           {args.similarity_metric}")
    print(f"  Granularity:      {args.granularity}")
    print()

    member_data, non_member_data = load_scores(
        args.member_score_file,
        args.non_member_score_file,
        args.similarity_metric
    )

    print(f"  Loaded {len(member_data)} member scores, "
          f"{len(non_member_data)} non-member scores")
    print(f"  Member mean specificity:     {np.mean(member_data):.4f} "
          f"(std={np.std(member_data):.4f})")
    print(f"  Non-member mean specificity: {np.mean(non_member_data):.4f} "
          f"(std={np.std(non_member_data):.4f})")
    print(f"  Distribution gap:            "
          f"{np.mean(member_data) - np.mean(non_member_data):.4f}")
    print()

    aucs = []
    for run in range(5):
        auc = cmra_inference(member_data, non_member_data, args.granularity)
        aucs.append(auc)
        print(f"  Run {run+1}/5: AUC = {auc:.4f}")

    avg_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    print()
    print(f"Accuracy: {avg_auc:.4f}  (std={std_auc:.4f})")
    print()

    # Individual-level AUC (no grouping) — more conservative diagnostic
    all_scores = member_data + non_member_data
    all_labels = [1] * len(member_data) + [0] * len(non_member_data)
    ind_auc = roc_auc_score(all_labels, all_scores)
    print(f"Individual-level AUC (diagnostic): {ind_auc:.4f}")
    print()

    print("Interpretation:")
    if avg_auc >= 0.90:
        print("  STRONG — correction behavior is a clear membership signal.")
    elif avg_auc >= 0.70:
        print("  MODERATE — detectable signal; distributions overlap.")
    elif avg_auc >= 0.55:
        print("  WEAK — slight signal, close to random.")
    else:
        print("  FAILED — attack is at or below random guessing.")

    print()
    print("Score components (members should score higher on all):")
    print("  length:     length of correction response")
    print("  negation:   density of 'not/incorrect/actually' words")
    print("  content:    density of meaningful content words")
    print("  confidence: inverse of hedging word density")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMRA: membership inference via misinformation correction specificity"
    )
    parser.add_argument(
        "--member_score_file", type=str, required=True,
        help="Path to member CMRA scores JSON"
    )
    parser.add_argument(
        "--non_member_score_file", type=str, required=True,
        help="Path to non-member CMRA scores JSON"
    )
    parser.add_argument(
        "--granularity", type=int, default=50,
        help="Number of samples per group per iteration (default: 50)"
    )
    parser.add_argument(
        "--similarity_metric", type=str, default="correction_specificity",
        choices=["correction_specificity", "length_score",
                 "negation_score", "content_score", "confidence_score"],
        help="Which score component to use (default: correction_specificity)"
    )
    args = parser.parse_args()
    main(args)

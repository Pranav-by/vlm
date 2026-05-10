"""
Prompt Perturbation Attack (Attack 6) — Inference Script

Membership inference based on cross-prompt response consistency.

Core insight (Memorization-Invariance Hypothesis):
  When a VLM has memorized a training sample, it reproduces similar content
  regardless of HOW the question is phrased. For unseen (non-member) images,
  different prompt phrasings activate different generation paths → more varied
  responses.

Signal: average pairwise similarity across K prompt variants per image.
  High consistency score → likely MEMBER
  Low consistency score  → likely NON-MEMBER

Attack design:
  For each iteration (1000 iterations):
    1. Sample 'granularity' members and non-members
    2. Compute mean consistency score per group
    3. Members should have higher mean → AUC over (score, label) pairs

Key advantages over existing attacks:
  - No ground truth needed (unlike Shadow/Reference/Target-Only)
  - Works at temperature=0 (unlike Image-Only which needs sampling variance)
  - More stealthy — varied prompts look like natural usage

Expected AUC: ~0.75-0.85 (stronger than Target-Only at ~0.53,
  competitive with Image-Only at ~0.94)

Usage:
  python prompt_perturbation_inference.py \\
      --member_similarity_file data/similarity/similarity_member_ppa.json \\
      --non_member_similarity_file data/similarity/similarity_non_member_ppa.json \\
      --granularity 50 \\
      --similarity_metric rouge2_f
"""

import json
import numpy as np
import random
import argparse
from sklearn.metrics import roc_auc_score


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(member_file, non_member_file, metric):
    """
    Load PPA consistency scores.

    Supports two JSON formats:
      1. New PPA format:
         [{"image_id": "...", "ppa_consistency": {"rouge2_f": 0.4, "embedding_mpn": 0.7}}, ...]
      2. Legacy similarity format (if someone ran 02_compute_similarity on ppa data):
         [{"image_id": "...", "similarity_0.1": {"rouge2_f": 0.4, "embedding_mpn": 0.7}}, ...]
    """
    with open(member_file) as f:
        member_raw = json.load(f)
    with open(non_member_file) as f:
        non_member_raw = json.load(f)

    def extract_score(item):
        # New PPA format
        if "ppa_consistency" in item:
            return item["ppa_consistency"][metric]
        # Legacy: try to find any similarity_* key
        for key in item:
            if key.startswith("similarity_"):
                return item[key][metric]
        raise KeyError(f"Cannot find PPA consistency score in item: {list(item.keys())}")

    member_scores = [extract_score(item) for item in member_raw]
    non_member_scores = [extract_score(item) for item in non_member_raw]

    return member_scores, non_member_scores


# ─── Attack algorithm ─────────────────────────────────────────────────────────

def prompt_perturbation_inference(member_data, non_member_data, granularity, n_iterations=1000):
    """
    Prompt Perturbation Attack inference.

    For each iteration:
      1. Sample `granularity` members and non-members
      2. Compute mean PPA consistency score per group
      3. Append (score, label) pairs to the AUC computation list

    Members → higher mean consistency → higher score → label=1 should rank higher.

    Returns:
      AUC score (float in [0, 1])
    """
    if len(member_data) < granularity:
        raise ValueError(
            f"Not enough member samples ({len(member_data)}) for granularity={granularity}. "
            f"Reduce --granularity."
        )
    if len(non_member_data) < granularity:
        raise ValueError(
            f"Not enough non-member samples ({len(non_member_data)}) for granularity={granularity}. "
            f"Reduce --granularity."
        )

    score_list = []
    label_list = []

    for _ in range(n_iterations):
        samples_member = random.sample(member_data, granularity)
        samples_non_member = random.sample(non_member_data, granularity)

        # Mean cross-prompt consistency for each group
        score_list.append(float(np.mean(samples_member)))
        label_list.append(1)  # member

        score_list.append(float(np.mean(samples_non_member)))
        label_list.append(0)  # non-member

    auc = roc_auc_score(label_list, score_list)
    return auc


# ─── Z-test variant for group-level analysis ─────────────────────────────────

def prompt_perturbation_ztest(member_data, non_member_data, granularity, n_iterations=1000):
    """
    Optional Z-test variant: directly compare member vs non-member group means.

    This mirrors the Reference attack design — compute Z-statistic between
    member group mean and non-member group mean consistency scores.

    Returns:
      AUC score (float)
    """
    from scipy.stats import norm

    score_list = []
    label_list = []

    for _ in range(n_iterations):
        m_sample = random.sample(member_data, granularity)
        nm_sample = random.sample(non_member_data, granularity)

        mean_m = np.mean(m_sample)
        mean_nm = np.mean(nm_sample)
        var_m = np.var(m_sample, ddof=1)
        var_nm = np.var(nm_sample, ddof=1)
        n = granularity

        # Z: does this group look more like members (high consistency)?
        z = (mean_m - mean_nm) / np.sqrt(var_m / n + var_nm / n)
        p = 1 - norm.cdf(z)  # small p → member group is clearly higher

        score_list.append(p)
        label_list.append(0)  # member group assigned label 0 (lower p = more member-like)

        # Symmetrically: non-member group
        z2 = (mean_nm - mean_m) / np.sqrt(var_nm / n + var_m / n)
        p2 = 1 - norm.cdf(z2)
        score_list.append(p2)
        label_list.append(1)

    auc = roc_auc_score(label_list, score_list)
    return auc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Attack 6: Prompt Perturbation Inference")
    print("  (Cross-prompt consistency of VLM responses)")
    print("=" * 60)
    print(f"  Member file:      {args.member_similarity_file}")
    print(f"  Non-member file:  {args.non_member_similarity_file}")
    print(f"  Granularity:      {args.granularity}")
    print(f"  Metric:           {args.similarity_metric}")
    print(f"  Mode:             {'Z-test' if args.ztest else 'Mean-consistency AUC'}")
    print()

    member_data, non_member_data = load_data(
        args.member_similarity_file,
        args.non_member_similarity_file,
        args.similarity_metric
    )

    print(f"  Loaded {len(member_data)} member scores, "
          f"{len(non_member_data)} non-member scores")
    print(f"  Member mean consistency:     {np.mean(member_data):.4f} "
          f"(std={np.std(member_data):.4f})")
    print(f"  Non-member mean consistency: {np.mean(non_member_data):.4f} "
          f"(std={np.std(non_member_data):.4f})")
    print(f"  Distribution gap:            {np.mean(member_data) - np.mean(non_member_data):.4f}")
    print()

    attack_fn = prompt_perturbation_ztest if args.ztest else prompt_perturbation_inference

    aucs = []
    for run in range(5):
        auc = attack_fn(member_data, non_member_data, args.granularity)
        aucs.append(auc)
        print(f"  Run {run+1}/5: AUC = {auc:.4f}")

    avg_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    print()
    print(f"Accuracy: {avg_auc:.4f}  (std={std_auc:.4f})")
    print()
    print("Interpretation:")
    if avg_auc >= 0.80:
        print("  STRONG attack — cross-prompt memorization signal is clear.")
    elif avg_auc >= 0.65:
        print("  MODERATE attack — detectable signal, but distributions overlap.")
    elif avg_auc >= 0.55:
        print("  WEAK attack — slight signal, close to random.")
    else:
        print("  FAILED — attack is at or below random guessing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt Perturbation Attack: membership inference via cross-prompt consistency"
    )
    parser.add_argument(
        "--member_similarity_file", type=str, required=True,
        help="Path to member PPA consistency scores JSON"
    )
    parser.add_argument(
        "--non_member_similarity_file", type=str, required=True,
        help="Path to non-member PPA consistency scores JSON"
    )
    parser.add_argument(
        "--granularity", type=int, default=50,
        help="Number of samples per group per iteration (default: 50)"
    )
    parser.add_argument(
        "--similarity_metric", type=str, default="rouge2_f",
        choices=["rouge2_f", "embedding_mpn"],
        help="Which similarity metric to use (default: rouge2_f)"
    )
    parser.add_argument(
        "--ztest", action="store_true",
        help="Use Z-test variant instead of mean-consistency AUC (experimental)"
    )

    args = parser.parse_args()
    main(args)

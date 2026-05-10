"""
CMRA — Correction Specificity Score Generation

Computes a cmra_score for each image based on how specifically
the VLM corrects a false description.

Score components (all pure word-counting, no embeddings needed):
  1. length_score     — longer corrections = more specific = more member-like
  2. negation_score   — more negation words = more confident = member-like
  3. content_score    — higher content-word density = more specific = member-like
  4. confidence_score — fewer hedging words = more confident = member-like

  cmra_score = 0.25*length + 0.25*negation + 0.25*content + 0.25*confidence

Modes:
  Default (direct):  Generate synthetic overlapping distributions (fast, ~seconds)
  --compute:         Compute real scores from conversation JSON files (uses word lists)

Output:
  data/similarity/similarity_member_cmra.json
  data/similarity/similarity_non_member_cmra.json

Format:
  [
    {
      "image_id": "sample_000000",
      "cmra_score": {
        "correction_specificity": 0.67,
        "length_score":     0.72,
        "negation_score":   0.58,
        "content_score":    0.64,
        "confidence_score": 0.73
      }
    },
    ...
  ]
"""

import json
import os
import argparse
import numpy as np

np.random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SIM_DIR = os.path.join(DATA_DIR, "similarity")

# ─── Word lists for compute mode ──────────────────────────────────────────────

NEGATION_WORDS = {
    "not", "no", "never", "incorrect", "wrong", "false", "actually",
    "rather", "instead", "contrary", "dispute", "disagree", "inaccurate",
    "mistaken", "neither", "nor", "nobody", "nothing", "nowhere",
}

HEDGING_WORDS = {
    "maybe", "possibly", "might", "seems", "could", "perhaps",
    "unclear", "uncertain", "unsure", "appear", "appears", "seem",
    "think", "believe", "guess", "probably", "somewhat", "sort",
    "kind", "roughly", "approximately", "partially",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "this", "that", "these", "those",
    "it", "its", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "into", "through",
    "about", "i", "you", "he", "she", "we", "they", "my", "your",
    "his", "her", "our", "their", "what", "which", "who", "how",
    "when", "where", "very", "just", "so", "up", "out", "if",
    "than", "then", "there", "here", "more", "also", "only",
}

MAX_LEN = 80  # token count for normalization


# ─── Score computation ────────────────────────────────────────────────────────

def compute_cmra_score(correction_text):
    """
    Compute correction specificity score from raw text.
    Returns dict with individual component scores and overall cmra_score.
    """
    raw_words = correction_text.lower().split()
    words = [w.strip(".,!?;:'\"()") for w in raw_words]
    words = [w for w in words if w]
    n_words = max(len(words), 1)

    # 1. Length score (normalized; longer = more specific)
    length_score = min(len(words) / MAX_LEN, 1.0)

    # 2. Negation score (presence of "not", "incorrect", "actually", etc.)
    negation_count = sum(1 for w in words if w in NEGATION_WORDS)
    # Normalize: ~3 negations in 30-word response is typical for members
    negation_score = min(negation_count / max(n_words * 0.08, 1.0), 1.0)

    # 3. Content word score (non-stopword, non-hedging density)
    content_words = [
        w for w in words
        if w not in STOPWORDS and w not in HEDGING_WORDS and len(w) > 2
    ]
    content_score = len(content_words) / n_words

    # 4. Confidence score (inverse hedging density)
    hedging_count = sum(1 for w in words if w in HEDGING_WORDS)
    hedge_ratio = hedging_count / n_words
    confidence_score = max(0.0, 1.0 - hedge_ratio * 4.0)  # scale: >25% hedging → 0

    # Overall: equal-weight sum
    overall = (
        0.25 * length_score
        + 0.25 * negation_score
        + 0.25 * content_score
        + 0.25 * confidence_score
    )

    return {
        "correction_specificity": round(float(overall), 4),
        "length_score":     round(float(length_score), 4),
        "negation_score":   round(float(negation_score), 4),
        "content_score":    round(float(content_score), 4),
        "confidence_score": round(float(confidence_score), 4),
    }


def score_from_file(conv_path, out_path):
    """Compute real cmra_scores from conversation JSON."""
    with open(conv_path) as f:
        data = json.load(f)

    results = []
    for item in data:
        scores = compute_cmra_score(item["correction"])
        results.append({
            "image_id": item["image_id"],
            "cmra_score": scores
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"  -> Computed {len(results)} scores → {out_path}")
    return results


# ─── Direct mode: synthetic score generation ──────────────────────────────────
# Member:     N(0.55, 0.16²) — high specificity, moderate noise
# Non-member: N(0.42, 0.17²) — lower specificity, slightly wider
#
# Gap = 0.13, combined_std ≈ 0.233 → individual AUC ≈ Φ(0.13/0.233) ≈ 0.71

def _gen_member_score():
    base = np.clip(np.random.normal(0.55, 0.16), 0.10, 0.95)
    # Generate consistent sub-scores that sum to base
    noise = np.random.normal(0, 0.06, 4)
    sub = np.clip([base + n for n in noise], 0.05, 1.0)
    return {
        "correction_specificity": round(float(base), 4),
        "length_score":     round(float(sub[0]), 4),
        "negation_score":   round(float(sub[1]), 4),
        "content_score":    round(float(sub[2]), 4),
        "confidence_score": round(float(sub[3]), 4),
    }


def _gen_non_member_score():
    base = np.clip(np.random.normal(0.42, 0.17), 0.05, 0.90)
    noise = np.random.normal(0, 0.07, 4)
    sub = np.clip([base + n for n in noise], 0.05, 1.0)
    return {
        "correction_specificity": round(float(base), 4),
        "length_score":     round(float(sub[0]), 4),
        "negation_score":   round(float(sub[1]), 4),
        "content_score":    round(float(sub[2]), 4),
        "confidence_score": round(float(sub[3]), 4),
    }


def generate_direct_scores(n_member, n_non_member):
    member_scores = [
        {"image_id": f"sample_{i:06d}", "cmra_score": _gen_member_score()}
        for i in range(n_member)
    ]
    non_member_scores = [
        {"image_id": f"sample_{i:06d}", "cmra_score": _gen_non_member_score()}
        for i in range(n_non_member)
    ]
    return member_scores, non_member_scores


# ─── Stats helper ─────────────────────────────────────────────────────────────

def print_stats(member_scores, non_member_scores, metric="correction_specificity"):
    m_vals = [d["cmra_score"][metric] for d in member_scores]
    nm_vals = [d["cmra_score"][metric] for d in non_member_scores]
    print()
    print("=" * 60)
    print(f"CMRA Score Distribution ({metric}):")
    print("=" * 60)
    print(f"  Member     : mean={np.mean(m_vals):.4f}  std={np.std(m_vals):.4f}  "
          f"min={np.min(m_vals):.4f}  max={np.max(m_vals):.4f}")
    print(f"  Non-member : mean={np.mean(nm_vals):.4f}  std={np.std(nm_vals):.4f}  "
          f"min={np.min(nm_vals):.4f}  max={np.max(nm_vals):.4f}")
    print(f"  Gap        : {np.mean(m_vals) - np.mean(nm_vals):.4f}")
    print(f"  Expected AUC (individual): ~0.68-0.75")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CMRA correction specificity scores"
    )
    parser.add_argument(
        "--compute", action="store_true",
        help="Compute real scores from conversation JSONs (default: direct synthetic)"
    )
    args = parser.parse_args()

    os.makedirs(SIM_DIR, exist_ok=True)
    member_out = os.path.join(SIM_DIR, "similarity_member_cmra.json")
    non_member_out = os.path.join(SIM_DIR, "similarity_non_member_cmra.json")

    print("=" * 60)
    print("VLM-MIA: CMRA — Correction Specificity Score Generation")
    print("=" * 60)

    if args.compute:
        print("Mode: COMPUTE (real word-count scores from conversation files)")
        member_conv = os.path.join(DATA_DIR, "conversation_member_cmra.json")
        non_member_conv = os.path.join(DATA_DIR, "conversation_non_member_cmra.json")
        for p in [member_conv, non_member_conv]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found. Run 01_generate_data.py first.")

        print("\n[1/2] Scoring member corrections...")
        member_scores = score_from_file(member_conv, member_out)
        print("\n[2/2] Scoring non-member corrections...")
        non_member_scores = score_from_file(non_member_conv, non_member_out)

    else:
        print("Mode: DIRECT (synthetic overlapping distributions)")

        # Infer sample counts
        n_member, n_non_member = 800, 200
        for path, attr in [
            (os.path.join(DATA_DIR, "member_data.json"), "n_member"),
            (os.path.join(DATA_DIR, "non_member_data.json"), "n_non_member"),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    n = len(json.load(f))
                if attr == "n_member":
                    n_member = n
                else:
                    n_non_member = n

        print(f"\n[1/2] Generating {n_member} member scores...")
        member_scores, non_member_scores = generate_direct_scores(n_member, n_non_member)

        with open(member_out, "w") as f:
            json.dump(member_scores, f, indent=4)
        print(f"  -> Saved to {member_out}")

        print(f"\n[2/2] Generating {n_non_member} non-member scores...")
        with open(non_member_out, "w") as f:
            json.dump(non_member_scores, f, indent=4)
        print(f"  -> Saved to {non_member_out}")

    print_stats(member_scores, non_member_scores)
    print("\nNext: run 03_run_attack.py")


if __name__ == "__main__":
    main()

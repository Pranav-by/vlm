"""
Generate synthetic conversation data for the Prompt Perturbation Attack (PPA).

Unlike existing attacks that use a SINGLE prompt per image, PPA queries the model
with K=12 semantically equivalent but lexically diverse prompts and measures how
consistent the responses are across prompt phrasings.

Key insight (Memorization-Invariance Hypothesis):
  - Member images: the VLM has a memorized "core" answer — different prompt phrasings
    all trigger the same memorized pathway → HIGH cross-prompt consistency
  - Non-member images: no memorized anchor → different prompts activate different
    generation paths → LOWER cross-prompt consistency

Output:
  data/conversation_member_ppa.json
  data/conversation_non_member_ppa.json

Format per item:
  {
    "image_id": "sample_000000",
    "ppa_responses": [
        {"prompt_id": 0, "prompt": "What is in this image?", "response": "..."},
        {"prompt_id": 1, "prompt": "Describe what you see...", "response": "..."},
        ...  (K=12 prompts)
    ]
  }
"""

import json
import os
import random
import numpy as np

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_DIR = OUTPUT_DIR  # member/non-member data lives here

# ─── K=12 Prompt Variants ───────────────────────────────────────────────────
# Four groups of 3, each stylistically different.
# All ask about the same image; lexically diverse to stress-test memorization invariance.

PROMPT_VARIANTS = [
    # Group A: Direct description requests
    "What is in this image?",
    "Describe what you see in this picture.",
    "Can you tell me what's happening in this photograph?",

    # Group B: Detail-oriented requests
    "Please provide a detailed description of this image.",
    "What are the main objects and activities visible here?",
    "Explain everything you can observe in this picture.",

    # Group C: Casual / conversational requests
    "Tell me about this photo.",
    "What's going on in this image?",
    "Walk me through what you see here.",

    # Group D: Analytical requests
    "Analyze the contents of this image.",
    "What does this picture depict?",
    "Summarize the scene shown in this image.",
]

K = len(PROMPT_VARIANTS)  # 12 prompts → C(12,2)=66 pairwise comparisons

# ─── Helper: controlled-similarity text generation ───────────────────────────

FILLER_WORDS = [
    "the", "a", "an", "is", "are", "with", "near", "on", "in", "at",
    "large", "small", "visible", "present", "appears", "seems",
    "scene", "image", "background", "foreground", "area",
]

SYNONYMS = {
    "cat": ["feline", "kitten", "pet"],
    "dog": ["canine", "puppy", "pet"],
    "person": ["individual", "man", "woman", "figure"],
    "car": ["vehicle", "automobile", "sedan"],
    "sitting": ["resting", "perched", "positioned"],
    "standing": ["positioned", "located", "placed"],
    "playing": ["engaging", "participating", "involved"],
    "running": ["moving", "dashing", "rushing"],
    "windowsill": ["window ledge", "sill", "window area"],
    "birds": ["wildlife", "animals", "creatures"],
    "table": ["surface", "counter", "area"],
    "road": ["street", "path", "lane"],
}


def paraphrase_word(word):
    """Randomly substitute a word with a synonym if available."""
    clean = word.lower().strip(".,!?;:")
    if clean in SYNONYMS and random.random() < 0.6:
        return random.choice(SYNONYMS[clean])
    return word


def generate_member_response(ground_truth, prompt_idx):
    """
    Member responses share a 'memorized core' — key nouns and relationships
    persist regardless of prompt phrasing, with mild paraphrasing variation.

    Consistency level scales with a base similarity that is HIGH and only
    degrades slightly across prompts (simulating prompt-invariant memorization).
    """
    words = ground_truth.split()

    # Members: high retention of core content, some stylistic re-wording
    # Base similarity 0.60–0.80 across all prompts (tight cluster)
    base_sim = 0.68 + np.random.normal(0, 0.06)
    base_sim = max(0.45, min(0.90, base_sim))

    n_keep = max(3, int(len(words) * base_sim))
    kept_indices = sorted(random.sample(range(len(words)), min(n_keep, len(words))))

    result = []
    for i, w in enumerate(words):
        if i in set(kept_indices):
            result.append(paraphrase_word(w))  # keep but may paraphrase
        else:
            result.append(random.choice(FILLER_WORDS))

    # Stylistic prefix varies by prompt group to simulate different phrasings
    prefixes = {
        0: "",                             # Group A: plain
        1: "The image shows ",            # Group B: formal
        2: "Looking at this photo, ",     # Group C: casual
        3: "Upon analysis, ",             # Group D: analytical
    }
    prefix = prefixes[prompt_idx // 3]
    sentence = prefix + " ".join(result).strip()

    # Occasional trailing filler
    if random.random() > 0.6:
        fillers = [
            "The scene appears well-lit.",
            "Overall a clear photograph.",
            "The image quality is good.",
            "This appears to be an outdoor/indoor scene.",
        ]
        sentence += " " + random.choice(fillers)

    return sentence


def generate_non_member_response(ground_truth, prompt_idx):
    """
    Non-member responses have NO memorized core — each prompt activates a
    different interpretation path, producing semantically varied output.

    Consistency level is LOW, with higher variance across prompts.
    """
    words = ground_truth.split()

    # Non-members: lower, more variable retention — each prompt drifts further
    # Base similarity 0.30–0.55 with higher variance (simulating no anchor)
    base_sim = 0.40 + np.random.normal(0, 0.12)
    base_sim = max(0.15, min(0.65, base_sim))

    n_keep = max(1, int(len(words) * base_sim))
    kept_indices = sorted(random.sample(range(len(words)), min(n_keep, len(words))))

    # Larger filler pool to introduce more drift
    extended_fillers = FILLER_WORDS + [
        "object", "item", "element", "feature", "aspect",
        "environment", "setting", "context", "situation",
        "indoor", "outdoor", "daytime", "structure",
        "visible", "apparent", "notable", "present",
    ]

    result = []
    for i, w in enumerate(words):
        if i in set(kept_indices):
            # Non-members more aggressively substitute with broader synonyms
            if random.random() < 0.5:
                result.append(paraphrase_word(w))
            else:
                result.append(random.choice(extended_fillers))
        else:
            result.append(random.choice(extended_fillers))

    # Prompt-dependent framing drifts the response further from the ground truth
    framings = [
        "",
        "There appears to be ",
        "It seems like ",
        "From what I can observe, ",
        "The photograph captures ",
        "This image depicts ",
        "Looking carefully, ",
        "In this scene, ",
        "I can see ",
        "The picture shows what appears to be ",
        "Based on the image, ",
        "The visual content suggests ",
    ]
    framing = framings[prompt_idx % len(framings)]
    sentence = framing + " ".join(result).strip()

    return sentence


# ─── Main generation logic ────────────────────────────────────────────────────

def generate_ppa_conversations(data, is_member):
    """
    For each item, query with all K prompt variants and collect responses.
    Returns list of {image_id, ppa_responses: [{prompt_id, prompt, response}, ...]}.
    """
    results = []
    gen_fn = generate_member_response if is_member else generate_non_member_response

    for item in data:
        ground_truth = item["conversations"][1]["value"]
        image_id = item["id"]

        ppa_item = {
            "image_id": image_id,
            "ground_truth": ground_truth,  # stored for reference; not used in inference
            "ppa_responses": []
        }

        for i, prompt in enumerate(PROMPT_VARIANTS):
            response = gen_fn(ground_truth, i)
            ppa_item["ppa_responses"].append({
                "prompt_id": i,
                "prompt": prompt,
                "response": response
            })

        results.append(ppa_item)

    return results


def main():
    # Load member and non-member data generated by 01_generate_synthetic_data.py
    member_path = os.path.join(DATA_DIR, "member_data.json")
    non_member_path = os.path.join(DATA_DIR, "non_member_data.json")

    if not os.path.exists(member_path):
        raise FileNotFoundError(
            f"{member_path} not found. Run 01_generate_synthetic_data.py first."
        )

    with open(member_path) as f:
        member_data = json.load(f)
    with open(non_member_path) as f:
        non_member_data = json.load(f)

    print("=" * 60)
    print("VLM-MIA: Prompt Perturbation Attack Data Generation")
    print("=" * 60)
    print(f"  Prompt variants (K): {K}")
    print(f"  Pairwise comparisons per image: {K*(K-1)//2}")
    print(f"  Member samples: {len(member_data)}")
    print(f"  Non-member samples: {len(non_member_data)}")

    print("\n[1/2] Generating member PPA conversations...")
    member_convos = generate_ppa_conversations(member_data, is_member=True)
    out_path = os.path.join(DATA_DIR, "conversation_member_ppa.json")
    with open(out_path, "w") as f:
        json.dump(member_convos, f, indent=2)
    print(f"  -> Saved {len(member_convos)} items to {out_path}")

    print("\n[2/2] Generating non-member PPA conversations...")
    non_member_convos = generate_ppa_conversations(non_member_data, is_member=False)
    out_path = os.path.join(DATA_DIR, "conversation_non_member_ppa.json")
    with open(out_path, "w") as f:
        json.dump(non_member_convos, f, indent=2)
    print(f"  -> Saved {len(non_member_convos)} items to {out_path}")

    # Quick sanity check: print first item's response snippet for each group
    print("\nSanity check — first item, prompts 0 & 6:")
    for label, convos in [("Member", member_convos), ("Non-member", non_member_convos)]:
        item = convos[0]
        print(f"\n  [{label}] Ground truth: {item['ground_truth'][:60]}...")
        for idx in [0, 6]:
            r = item["ppa_responses"][idx]
            print(f"    Prompt {idx} ({r['prompt'][:30]}...): {r['response'][:60]}...")

    print("\n" + "=" * 60)
    print("PPA conversation data generation complete!")
    print("Next: run 04_generate_ppa_similarity.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

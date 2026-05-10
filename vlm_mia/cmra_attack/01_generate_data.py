"""
Contrastive Misinformation Resistance Attack (CMRA) — Data Generation

For each image we:
  1. Take the ground-truth description.
  2. Auto-generate a WRONG description by swapping key entities.
  3. Query the VLM (simulated): "I think this image shows: '<wrong>'. Is that right?"
  4. Record the correction response.

Key asymmetry we simulate:
  MEMBER   → model knows the real answer from training → long, confident, specific correction
  NON-MEMBER → model has no memorized anchor → short, vague, hedging correction

Output:
  data/conversation_member_cmra.json
  data/conversation_non_member_cmra.json

Format per item:
  {
    "image_id":          "sample_000000",
    "ground_truth":      "A cat is sitting on a windowsill watching birds.",
    "false_description": "A dog is playing in a park.",
    "correction":        "That is not correct. The image actually shows ..."
  }
"""

import json
import os
import random
import numpy as np

random.seed(42)
np.random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ─── Entity swap tables ───────────────────────────────────────────────────────
# Used to turn a real description into a plausible-but-wrong one.

SUBJECT_SWAPS = {
    "cat":        ["dog", "rabbit", "parrot"],
    "dog":        ["cat", "fox", "squirrel"],
    "person":     ["child", "statue", "mannequin"],
    "child":      ["adult", "dog", "cat"],
    "man":        ["woman", "boy", "robot"],
    "woman":      ["man", "girl", "doll"],
    "car":        ["motorcycle", "bicycle", "truck"],
    "motorcycle": ["car", "scooter", "bicycle"],
    "bicycle":    ["car", "skateboard", "motorcycle"],
    "bird":       ["squirrel", "insect", "lizard"],
    "horse":      ["cow", "donkey", "camel"],
    "elephant":   ["rhinoceros", "hippopotamus", "giraffe"],
    "bear":       ["wolf", "lion", "deer"],
    "cow":        ["horse", "sheep", "goat"],
    "sheep":      ["cow", "dog", "goat"],
    "giraffe":    ["elephant", "camel", "horse"],
    "zebra":      ["horse", "donkey", "cow"],
    "bus":        ["truck", "van", "train"],
    "truck":      ["bus", "car", "van"],
    "train":      ["bus", "tram", "subway"],
    "boat":       ["raft", "kayak", "canoe"],
    "pizza":      ["burger", "salad", "cake"],
    "sandwich":   ["pizza", "burger", "wrap"],
    "cake":       ["pie", "cookie", "muffin"],
    "food":       ["books", "tools", "clothes"],
    "plate":      ["tray", "bowl", "basket"],
    "table":      ["counter", "floor", "shelf"],
    "chair":      ["sofa", "bench", "stool"],
    "sofa":       ["chair", "bed", "hammock"],
    "bed":        ["sofa", "mat", "hammock"],
    "bathroom":   ["bedroom", "kitchen", "living room"],
    "kitchen":    ["bathroom", "garden", "garage"],
    "bedroom":    ["kitchen", "living room", "office"],
    "street":     ["park", "beach", "forest"],
    "park":       ["street", "garden", "mall"],
    "beach":      ["park", "desert", "mountain"],
}

LOCATION_SWAPS = {
    "windowsill":    ["park bench", "kitchen counter", "garden path"],
    "kitchen":       ["bathroom", "garden", "garage"],
    "bathroom":      ["bedroom", "kitchen", "office"],
    "bedroom":       ["kitchen", "living room", "garden"],
    "street":        ["park", "beach", "forest"],
    "park":          ["street", "shopping mall", "airport"],
    "beach":         ["park", "forest", "desert"],
    "forest":        ["beach", "park", "city square"],
    "dining table":  ["kitchen floor", "garden bench", "living room sofa"],
    "counter":       ["floor", "shelf", "table"],
    "yard":          ["indoor room", "parking lot", "alley"],
    "field":         ["urban street", "parking lot", "beach"],
}

ACTION_SWAPS = {
    "sitting":   ["running", "jumping", "swimming"],
    "standing":  ["lying down", "floating", "spinning"],
    "running":   ["sitting", "sleeping", "crawling"],
    "playing":   ["sleeping", "eating", "hiding"],
    "eating":    ["playing", "sleeping", "running"],
    "sleeping":  ["running", "playing", "eating"],
    "watching":  ["ignoring", "chasing", "fleeing"],
    "walking":   ["sitting", "running", "flying"],
    "holding":   ["dropping", "throwing", "hiding"],
    "riding":    ["pushing", "dragging", "avoiding"],
}


def swap_entities(text):
    """
    Replace the first matching entity in the text with an alternative.
    Returns (modified_text, was_swapped).
    """
    words = text.split()
    for i, word in enumerate(words):
        clean = word.lower().strip(".,!?;:'\"")
        for table in [SUBJECT_SWAPS, LOCATION_SWAPS, ACTION_SWAPS]:
            if clean in table:
                replacement = random.choice(table[clean])
                words[i] = replacement
                return " ".join(words), True
    # Fallback: prepend a totally wrong scene
    fallback_scenes = [
        "A dog playing fetch in a park.",
        "A car parked on an empty street.",
        "People swimming at the beach.",
        "A child riding a bicycle in a field.",
        "Two cats fighting on a rooftop.",
    ]
    return random.choice(fallback_scenes), False


# ─── Response generators ──────────────────────────────────────────────────────

NEGATION_STARTERS = [
    "That is not correct.",
    "That description is incorrect.",
    "Actually, that is wrong.",
    "No, that's not right.",
    "That's inaccurate.",
]

MEMBER_CONNECTORS = [
    "The image actually shows",
    "What the image actually depicts is",
    "In reality, the image shows",
    "The photograph actually contains",
    "The correct description is:",
]

MEMBER_DETAILS = [
    "not {false_subject}",
    "not a {false_subject} as you suggested",
    "which is quite different from {false_subject}",
    "rather than the {false_subject} you mentioned",
]

NON_MEMBER_TEMPLATES = [
    "I'm not entirely sure that's right. The image seems to show something slightly different.",
    "That might not be completely accurate. The scene could be different from what you described.",
    "Hmm, it's possible you may have the wrong idea. I can see some differences.",
    "Maybe, though some details seem off to me. It's hard to be certain.",
    "I'm not fully confident in that description. Things might not be as you've described.",
    "That could be partially right, but the image seems to show something else.",
    "It's unclear to me if that's completely accurate. The scene might differ.",
    "Possibly, but I think the description may not match entirely.",
]

MEMBER_SUFFIXES = [
    "The overall scene is quite clear and specific.",
    "The details in the image are unmistakable.",
    "This is a very clear image with distinct features.",
    "The subject and context are clearly visible.",
]


def generate_member_correction(ground_truth, false_description):
    """
    Member: confident, specific, long correction.
    Echoes ground-truth vocabulary; explicit negation of false claim.
    """
    # Extract key phrase from ground truth (first ~10 words)
    gt_words = ground_truth.split()
    core_content = " ".join(gt_words[:min(10, len(gt_words))])

    # Identify what's false in the false description
    false_words = false_description.split()
    false_subject = " ".join(false_words[:3]).strip(".,")

    starter = random.choice(NEGATION_STARTERS)
    connector = random.choice(MEMBER_CONNECTORS)
    detail = random.choice(MEMBER_DETAILS).format(false_subject=false_subject)
    suffix = random.choice(MEMBER_SUFFIXES)

    correction = (
        f"{starter} {connector} {core_content}, {detail}. "
        f"{ground_truth} {suffix}"
    )
    return correction


def generate_non_member_correction(ground_truth, false_description):
    """
    Non-member: vague, hedging, short correction.
    Does not reproduce specific ground-truth vocabulary.
    """
    template = random.choice(NON_MEMBER_TEMPLATES)

    # Occasionally add a slightly more specific but still wrong detail
    if random.random() > 0.6:
        additions = [
            " There appear to be some objects present.",
            " The setting seems different somehow.",
            " Some elements may not match.",
            " The context appears unclear.",
        ]
        template += random.choice(additions)

    return template


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_cmra_conversations(data, is_member):
    results = []
    gen_fn = generate_member_correction if is_member else generate_non_member_correction
    label = "member" if is_member else "non-member"
    swap_fail = 0

    for item in data:
        ground_truth = item["conversations"][1]["value"]
        image_id = item["id"]

        false_description, swapped = swap_entities(ground_truth)
        if not swapped:
            swap_fail += 1

        correction = gen_fn(ground_truth, false_description)

        results.append({
            "image_id": image_id,
            "ground_truth": ground_truth,
            "false_description": false_description,
            "correction": correction,
            "label": label
        })

    print(f"  Entity swap failures (used fallback): {swap_fail}/{len(data)}")
    return results


def main():
    member_path = os.path.join(DATA_DIR, "member_data.json")
    non_member_path = os.path.join(DATA_DIR, "non_member_data.json")

    if not os.path.exists(member_path):
        raise FileNotFoundError(
            f"{member_path} not found. Run scripts/01_generate_synthetic_data.py first."
        )

    with open(member_path) as f:
        member_data = json.load(f)
    with open(non_member_path) as f:
        non_member_data = json.load(f)

    print("=" * 60)
    print("VLM-MIA: CMRA — Conversation Data Generation")
    print("  (Contrastive Misinformation Resistance Attack)")
    print("=" * 60)
    print(f"  Member samples:     {len(member_data)}")
    print(f"  Non-member samples: {len(non_member_data)}")
    print()

    print("[1/2] Generating member CMRA conversations...")
    member_convos = generate_cmra_conversations(member_data, is_member=True)
    out = os.path.join(DATA_DIR, "conversation_member_cmra.json")
    with open(out, "w") as f:
        json.dump(member_convos, f, indent=2)
    print(f"  -> Saved {len(member_convos)} items to {out}")

    print()
    print("[2/2] Generating non-member CMRA conversations...")
    non_member_convos = generate_cmra_conversations(non_member_data, is_member=False)
    out = os.path.join(DATA_DIR, "conversation_non_member_cmra.json")
    with open(out, "w") as f:
        json.dump(non_member_convos, f, indent=2)
    print(f"  -> Saved {len(non_member_convos)} items to {out}")

    # Sanity check
    print()
    print("Sanity check — first item comparison:")
    m0 = member_convos[0]
    nm0 = non_member_convos[0]
    for label, item in [("Member", m0), ("Non-member", nm0)]:
        print(f"\n  [{label}]")
        print(f"    GT:    {item['ground_truth'][:60]}...")
        print(f"    False: {item['false_description'][:60]}")
        print(f"    Corr:  {item['correction'][:80]}...")

    print()
    print("=" * 60)
    print("CMRA conversation data generation complete!")
    print("Next: run 02_generate_scores.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

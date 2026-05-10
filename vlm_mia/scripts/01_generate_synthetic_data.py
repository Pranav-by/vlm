"""
Generate synthetic data that mimics the full VLM-MIA pipeline output.

Since we cannot run actual VLM training/inference without GPU, this script
generates realistic synthetic data matching the expected formats:

1. llava_instruct_158k.json (instruction tuning dataset format)
2. Member/Non-member data splits (80/20)
3. Shadow model splits (4-way: shadow-member, shadow-non-member, target-member, target-non-member)
4. Conversation output files (as if queried from a trained VLM)

The synthetic conversation data simulates the key property from the paper:
- Member data responses will have HIGHER similarity to ground truth
  (model memorized training data)
- Non-member data responses will have LOWER similarity to ground truth
"""

import json
import os
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# --- Configuration ---
NUM_SAMPLES = 1000  # Total synthetic samples (paper uses 158k but we keep small)
MEMBER_RATIO = 0.8
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Temperatures used across attacks
SHADOW_TEMPERATURES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
REFERENCE_TEMPERATURE = [0.1]
TARGET_ONLY_TEMPERATURES = [0.1, 1.5]
IMAGE_ONLY_TEMPERATURE = [0.1]
IMAGE_ONLY_REPEAT = 5

# --- Ground truth text templates ---
GROUND_TRUTH_TEMPLATES = [
    "A person is standing next to a red car in a parking lot.",
    "The kitchen has white cabinets and a stainless steel refrigerator.",
    "A group of people are playing frisbee in a park on a sunny day.",
    "A cat is sitting on a windowsill looking outside at the birds.",
    "There is a traffic light at the intersection with several cars waiting.",
    "A woman is riding a bicycle down a tree-lined street.",
    "The dining table has plates of food and glasses of wine.",
    "A dog is running through a field of tall grass.",
    "The bathroom has a white bathtub and blue tile walls.",
    "Children are playing on a playground with slides and swings.",
    "A man is surfing on a large wave in the ocean.",
    "The living room has a brown leather couch and a flat screen TV.",
    "A bus is stopped at a bus stop with passengers waiting.",
    "Birds are perched on a telephone wire against a cloudy sky.",
    "A skateboarder is performing a trick at a skate park.",
    "The bedroom has a queen-sized bed with blue and white bedding.",
    "A couple is walking hand in hand on the beach at sunset.",
    "Fresh fruits and vegetables are displayed at a farmers market.",
    "A train is crossing a bridge over a river.",
    "Snow-covered mountains are visible in the background of a small town.",
]

HUMAN_QUESTIONS = [
    "What is happening in this image?",
    "Can you describe the scene in the picture?",
    "What do you see in this photograph?",
    "Please describe this image in detail.",
    "What objects can you identify in this image?",
]


def generate_similar_text(ground_truth, similarity_level):
    """
    Generate text with controlled similarity to ground truth.
    Higher similarity_level (0-1) means more words are kept from ground truth.
    """
    words = ground_truth.split()
    n_keep = max(2, int(len(words) * similarity_level))
    
    # Keep a portion of the original words
    kept_indices = sorted(random.sample(range(len(words)), min(n_keep, len(words))))
    
    # Fill remaining with related but different words
    filler_words = [
        "the", "a", "an", "is", "are", "was", "were", "has", "have",
        "with", "near", "beside", "on", "in", "at", "by", "from",
        "large", "small", "bright", "dark", "visible", "present",
        "appears", "seems", "showing", "featuring", "including",
        "scene", "image", "picture", "background", "foreground",
    ]
    
    result_words = []
    kept_set = set(kept_indices)
    for i in range(len(words)):
        if i in kept_set:
            result_words.append(words[i])
        else:
            result_words.append(random.choice(filler_words))
    
    # Sometimes add extra description
    if random.random() > 0.5:
        extras = ["The scene appears to be outdoors.", "The lighting suggests daytime.",
                   "The colors are vivid.", "Overall a clear image."]
        result_words.append(random.choice(extras))
    
    return " ".join(result_words)


def generate_instruct_dataset(num_samples):
    """Generate synthetic llava_instruct_158k.json format data."""
    dataset = []
    for i in range(num_samples):
        ground_truth = random.choice(GROUND_TRUTH_TEMPLATES)
        question = random.choice(HUMAN_QUESTIONS)
        
        item = {
            "id": f"sample_{i:06d}",
            "image": f"COCO_train2017_{i:012d}.jpg",
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": ground_truth}
            ]
        }
        dataset.append(item)
    return dataset


def split_dataset(dataset, member_ratio=0.8):
    """Split dataset into member and non-member (80/20)."""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * member_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def split_for_shadow_attack(member_data, non_member_data):
    """
    Further split for shadow model attack:
    - shadow-member: first half of member data  
    - target-member: second half of member data
    - shadow-non-member: first half of non-member data
    - target-non-member: second half of non-member data
    """
    mid_m = len(member_data) // 2
    mid_nm = len(non_member_data) // 2
    return {
        "shadow_member": member_data[:mid_m],
        "shadow_non_member": non_member_data[:mid_nm],
        "target_member": member_data[mid_m:],
        "target_non_member": non_member_data[mid_nm:],
    }


def generate_conversation_output(data, temperatures, repeat, is_member):
    """
    Generate synthetic conversation output as if a VLM was queried.
    
    Key insight from the paper: member data (seen during training) will produce
    responses that are MORE similar to ground truth, especially at low temperatures.
    Non-member data will have lower similarity.
    """
    results = []
    for item in data:
        ground_truth = item["conversations"][1]["value"]
        question = item["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")
        
        conversation_result = {
            "image_id": item["id"],
        }
        
        for temp in temperatures:
            conversation_result[f"conversations_{temp}"] = []
            conversation_result[f"conversations_{temp}"].append({"from": "human", "value": question})
            
            for r in range(repeat):
                # Member data: higher similarity, especially at low temperature
                # Non-member data: lower similarity
                if is_member:
                    # Members: high similarity at low temp, decreasing with temp
                    base_similarity = 0.75 - (temp * 0.15)
                    noise = random.gauss(0, 0.05)
                else:
                    # Non-members: lower similarity overall
                    base_similarity = 0.45 - (temp * 0.1)
                    noise = random.gauss(0, 0.08)
                
                similarity = max(0.1, min(0.95, base_similarity + noise))
                vlm_response = generate_similar_text(ground_truth, similarity)
                
                conversation_result[f"conversations_{temp}"].append({
                    "from": f"vlm_{r+1}",
                    "value": vlm_response
                })
            
            conversation_result[f"conversations_{temp}"].append({
                "from": "ground truth",
                "value": ground_truth
            })
        
        results.append(conversation_result)
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("VLM-MIA Synthetic Data Generation")
    print("=" * 60)
    
    # Step 1: Generate base instruction dataset
    print("\n[1/5] Generating synthetic instruction dataset...")
    dataset = generate_instruct_dataset(NUM_SAMPLES)
    with open(os.path.join(OUTPUT_DIR, "llava_instruct_synthetic.json"), "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  -> Generated {len(dataset)} samples")
    
    # Step 2: Split into member/non-member (80/20)
    print("\n[2/5] Splitting into member/non-member (80/20)...")
    member_data, non_member_data = split_dataset(dataset, MEMBER_RATIO)
    with open(os.path.join(OUTPUT_DIR, "member_data.json"), "w") as f:
        json.dump(member_data, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "non_member_data.json"), "w") as f:
        json.dump(non_member_data, f, indent=2)
    print(f"  -> Member: {len(member_data)}, Non-member: {len(non_member_data)}")
    
    # Step 3: Further split for shadow model attack
    print("\n[3/5] Splitting for shadow model attack (4-way)...")
    shadow_splits = split_for_shadow_attack(member_data, non_member_data)
    for name, data in shadow_splits.items():
        with open(os.path.join(OUTPUT_DIR, f"{name}_data.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"  -> {name}: {len(data)} samples")
    
    # Step 4: Generate conversation outputs for each attack type
    print("\n[4/5] Generating conversation outputs...")
    
    # 4a: Shadow Model Attack conversations (all 16 temperatures)
    print("  -> Shadow model attack conversations...")
    for group_name in ["shadow_member", "shadow_non_member", "target_member", "target_non_member"]:
        is_member = "member" in group_name and "non" not in group_name
        group_data = shadow_splits[group_name]
        convos = generate_conversation_output(group_data, SHADOW_TEMPERATURES, repeat=1, is_member=is_member)
        with open(os.path.join(OUTPUT_DIR, f"conversation_{group_name}_shadow.json"), "w") as f:
            json.dump(convos, f, indent=2)
    
    # 4b: Reference Attack conversations (single temperature 0.1)
    print("  -> Reference attack conversations...")
    convos_member_ref = generate_conversation_output(member_data, REFERENCE_TEMPERATURE, repeat=1, is_member=True)
    convos_non_member_ref = generate_conversation_output(non_member_data, REFERENCE_TEMPERATURE, repeat=1, is_member=False)
    with open(os.path.join(OUTPUT_DIR, "conversation_member_reference.json"), "w") as f:
        json.dump(convos_member_ref, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "conversation_non_member_reference.json"), "w") as f:
        json.dump(convos_non_member_ref, f, indent=2)
    
    # 4c: Target-Only Attack conversations (two temperatures: 0.1 and 1.5)
    print("  -> Target-only attack conversations...")
    convos_member_target = generate_conversation_output(member_data, TARGET_ONLY_TEMPERATURES, repeat=1, is_member=True)
    convos_non_member_target = generate_conversation_output(non_member_data, TARGET_ONLY_TEMPERATURES, repeat=1, is_member=False)
    with open(os.path.join(OUTPUT_DIR, "conversation_member_target_only.json"), "w") as f:
        json.dump(convos_member_target, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "conversation_non_member_target_only.json"), "w") as f:
        json.dump(convos_non_member_target, f, indent=2)
    
    # 4d: Image-Only Attack conversations (single temp, repeated queries)
    print("  -> Image-only attack conversations...")
    convos_member_img = generate_conversation_output(member_data, IMAGE_ONLY_TEMPERATURE, repeat=IMAGE_ONLY_REPEAT, is_member=True)
    convos_non_member_img = generate_conversation_output(non_member_data, IMAGE_ONLY_TEMPERATURE, repeat=IMAGE_ONLY_REPEAT, is_member=False)
    with open(os.path.join(OUTPUT_DIR, "conversation_member_image_only.json"), "w") as f:
        json.dump(convos_member_img, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "conversation_non_member_image_only.json"), "w") as f:
        json.dump(convos_non_member_img, f, indent=2)
    
    print("\n[5/5] Summary of generated files:")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        f_path = os.path.join(OUTPUT_DIR, f_name)
        size_kb = os.path.getsize(f_path) / 1024
        print(f"  -> {f_name}: {size_kb:.1f} KB")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

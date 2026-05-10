"""
============================================================================
VLM-MIA Real Pipeline — Run on Lightning AI (T4 GPU)
============================================================================

This script runs the complete real VLM-MIA pipeline:
  1. Download llava_instruct_158k.json + selective COCO images (<1GB total)
  2. Split data into member/non-member + shadow 4-way split
  3. QLoRA fine-tune TWO LLaVA-7B models (shadow + target)
  4. Generate conversations at multiple temperatures
  5. Compute Rouge-2 + MPNet similarity scores
  6. Package results for local download

Run on Lightning AI with T4 GPU:
    pip install -r requirements_lightning.txt
    python run_on_lightning.py

Total time: ~2.5-3 hours on T4
Total disk: <1GB downloaded data
============================================================================
"""

import os
import sys
import json
import time
import random
import requests
import argparse
import traceback
import numpy as np
from pathlib import Path
from itertools import combinations

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    "total_samples": 500,          # 500 samples (~50MB COCO images, fits in free tier session)
    "member_ratio": 0.8,           # 80% member, 20% non-member
    "random_seed": 42,
    
    # Model
    "model_name": "liuhaotian/llava-v1.5-7b",
    "quantization_bits": 4,        # 4-bit quantization (QLoRA)
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Training
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    
    # Conversation generation
    "shadow_temperatures": [0.1, 0.5, 1.0, 1.5],   # 4 temps for shadow attack
    "reference_temperatures": [0.1],                  # 1 temp for reference
    "target_only_temperatures": [0.1, 1.5],           # 2 temps for target-only
    "max_new_tokens": 512,
    
    # Paths
    "data_dir": "real_data",
    "coco_dir": "real_data/coco_images",
    "model_dir": "models",
}

# ============================================================================
# STEP 1: DOWNLOAD DATASET
# ============================================================================

def step1_download_data():
    """Download llava_instruct_158k.json and selectively download COCO images."""
    print("\n" + "=" * 70)
    print("  STEP 1: Downloading Data")
    print("=" * 70)
    
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["coco_dir"], exist_ok=True)
    
    # 1a: Download instruction dataset
    instruct_path = os.path.join(CONFIG["data_dir"], "llava_instruct_150k.json")
    if not os.path.exists(instruct_path):
        print("  Downloading llava_instruct_150k.json from HuggingFace...")
        url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(instruct_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  -> Saved to {instruct_path}")
    else:
        print(f"  -> Already exists: {instruct_path}")
    
    # 1b: Load and subsample
    with open(instruct_path, 'r') as f:
        all_data = json.load(f)
    print(f"  -> Loaded {len(all_data)} total instruction samples")
    
    # Filter to samples that have images (some may be text-only)
    image_data = [d for d in all_data if 'image' in d and d['image']]
    print(f"  -> {len(image_data)} samples have images")
    
    # Subsample
    random.seed(CONFIG["random_seed"])
    random.shuffle(image_data)
    subset = image_data[:CONFIG["total_samples"]]
    print(f"  -> Selected {len(subset)} samples for experiment")
    
    # 1c: Download COCO images for our subset
    print(f"\n  Downloading {len(subset)} COCO train2017 images...")
    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0
    
    for i, item in enumerate(subset):
        img_filename = item['image']
        img_path = os.path.join(CONFIG["coco_dir"], img_filename)
        
        if os.path.exists(img_path):
            skipped += 1
            continue
        
        url = f"http://images.cocodataset.org/train2017/{img_filename}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(resp.content)
                downloaded += 1
                total_bytes += len(resp.content)
            else:
                print(f"    Warning: Failed to download {img_filename} (HTTP {resp.status_code})")
                failed += 1
        except Exception as e:
            print(f"    Warning: Error downloading {img_filename}: {e}")
            failed += 1
        
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(subset)} images ({total_bytes/1024/1024:.1f} MB downloaded)")
    
    print(f"  -> Downloaded: {downloaded}, Skipped (exists): {skipped}, Failed: {failed}")
    print(f"  -> Total image data: {total_bytes/1024/1024:.1f} MB")
    
    # Remove samples whose images failed to download
    valid_subset = []
    for item in subset:
        img_path = os.path.join(CONFIG["coco_dir"], item['image'])
        if os.path.exists(img_path):
            valid_subset.append(item)
    
    print(f"  -> Valid samples with images: {len(valid_subset)}")
    
    # Save the subset
    subset_path = os.path.join(CONFIG["data_dir"], "instruct_subset.json")
    with open(subset_path, 'w') as f:
        json.dump(valid_subset, f, indent=2)
    
    return valid_subset


# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================

def step2_split_data(dataset):
    """Split into member/non-member + 4-way shadow split."""
    print("\n" + "=" * 70)
    print("  STEP 2: Splitting Data")
    print("=" * 70)
    
    random.seed(CONFIG["random_seed"])
    random.shuffle(dataset)
    
    # 80/20 member/non-member split
    split_idx = int(len(dataset) * CONFIG["member_ratio"])
    member_data = dataset[:split_idx]
    non_member_data = dataset[split_idx:]
    
    print(f"  Member: {len(member_data)}, Non-member: {len(non_member_data)}")
    
    # 4-way shadow split (split each group in half)
    mid_m = len(member_data) // 2
    mid_nm = len(non_member_data) // 2
    
    splits = {
        "member": member_data,
        "non_member": non_member_data,
        "shadow_member": member_data[:mid_m],
        "shadow_non_member": non_member_data[:mid_nm],
        "target_member": member_data[mid_m:],
        "target_non_member": non_member_data[mid_nm:],
    }
    
    for name, data in splits.items():
        path = os.path.join(CONFIG["data_dir"], f"{name}_data.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  -> {name}: {len(data)} samples -> {path}")
    
    return splits


# ============================================================================
# STEP 3: QLoRA FINE-TUNING
# ============================================================================

def step3_lora_finetune(training_data, adapter_name):
    """QLoRA fine-tune LLaVA-7B on given training data."""
    print("\n" + "=" * 70)
    print(f"  STEP 3: QLoRA Fine-tuning ({adapter_name})")
    print("=" * 70)
    
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        BitsAndBytesConfig, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Check GPU
    if not torch.cuda.is_available():
        print("  ERROR: No GPU available! This must run on Lightning AI with T4.")
        return None
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    
    adapter_path = os.path.join(CONFIG["model_dir"], adapter_name)
    if os.path.exists(adapter_path):
        print(f"  -> Adapter already exists at {adapter_path}, skipping training")
        return adapter_path
    
    # 3a: Load model in 4-bit
    print("  Loading LLaVA-7B in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    
    # 3b: Add LoRA
    print(f"  Adding LoRA (rank={CONFIG['lora_rank']}, alpha={CONFIG['lora_alpha']})...")
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["lora_target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 3c: Prepare training data
    print(f"  Preparing {len(training_data)} training samples...")
    
    train_texts = []
    for item in training_data:
        conversations = item.get("conversations", [])
        text = ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"].replace("<image>\n", "").replace("\n<image>", "")
                text += f"USER: {question}\n"
            elif conv["from"] == "gpt":
                text += f"ASSISTANT: {conv['value']}\n"
        if text:
            train_texts.append(text)
    
    # Tokenize
    encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=CONFIG["max_seq_length"],
        padding="max_length",
        return_tensors="pt",
    )
    
    # Create dataset
    from torch.utils.data import Dataset as TorchDataset
    
    class InstructDataset(TorchDataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.input_ids[idx].clone(),
            }
    
    train_dataset = InstructDataset(encodings)
    
    # 3d: Train
    print(f"  Starting training ({CONFIG['num_epochs']} epoch, batch={CONFIG['batch_size']})...")
    os.makedirs(adapter_path, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=True,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed/60:.1f} minutes")
    
    # Save adapter
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  -> Adapter saved to {adapter_path}")
    
    # Free memory
    del model, trainer
    torch.cuda.empty_cache()
    
    return adapter_path


# ============================================================================
# STEP 4: CONVERSATION GENERATION
# ============================================================================

def step4_generate_conversations(data, adapter_path, output_path, temperatures, repeat=1):
    """Query the LoRA-finetuned model and generate conversations."""
    print(f"\n  Generating conversations: {os.path.basename(output_path)}")
    print(f"    Samples: {len(data)}, Temps: {temperatures}, Repeat: {repeat}")
    
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = json.load(f)
        if len(existing) >= len(data):
            print(f"    -> Already complete ({len(existing)} samples), skipping")
            return
        print(f"    -> Resuming from {len(existing)} samples")
    else:
        existing = []
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    # Load base model + adapter
    print("    Loading model + LoRA adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    completed_ids = {item["image_id"] for item in existing}
    results = existing.copy()
    
    start_time = time.time()
    count = 0
    
    for item in data:
        item_id = item.get("id", item.get("image", "unknown"))
        if item_id in completed_ids:
            continue
        
        try:
            conversations = item.get("conversations", [])
            conversation_result = {"image_id": item_id}
            
            for temp in temperatures:
                conversation_result[f"conversations_{temp}"] = []
                
                for conv in conversations:
                    if conv["from"] == "human":
                        question = conv["value"].replace("<image>\n", "").replace("\n<image>", "")
                        conversation_result[f"conversations_{temp}"].append(
                            {"from": "human", "value": question}
                        )
                        
                        prompt = f"USER: {question}\nASSISTANT:"
                        
                        for r in range(repeat):
                            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=CONFIG["max_new_tokens"],
                                    temperature=max(temp, 0.01),
                                    do_sample=True,
                                    top_p=0.9,
                                )
                            
                            response = tokenizer.decode(
                                outputs[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True
                            ).strip()
                            
                            conversation_result[f"conversations_{temp}"].append(
                                {"from": f"vlm_{r+1}", "value": response}
                            )
                    
                    elif conv["from"] == "gpt":
                        conversation_result[f"conversations_{temp}"].append(
                            {"from": "ground truth", "value": conv["value"]}
                        )
            
            results.append(conversation_result)
            count += 1
            
            if count % 25 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed * 3600
                remaining = (len(data) - len(results)) / (rate / 3600) if rate > 0 else 0
                print(f"    [{count}/{len(data)}] {elapsed/60:.1f}min elapsed, ~{remaining/60:.0f}min remaining")
                
                # Save progress
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"    Error on {item_id}: {e}")
            continue
    
    # Final save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"    -> Done: {len(results)} samples in {total_time/60:.1f} min -> {output_path}")
    
    # Free memory
    del model, base_model
    import torch
    torch.cuda.empty_cache()


def step4_all_conversations(splits):
    """Generate all conversation files for all attacks."""
    print("\n" + "=" * 70)
    print("  STEP 4: Generating Conversations")
    print("=" * 70)
    
    data_dir = CONFIG["data_dir"]
    shadow_adapter = os.path.join(CONFIG["model_dir"], "shadow_adapter")
    target_adapter = os.path.join(CONFIG["model_dir"], "target_adapter")
    
    # Shadow attack: query shadow model on all 4 groups at 4 temperatures
    print("\n--- Shadow Attack Conversations (shadow model) ---")
    for group in ["shadow_member", "shadow_non_member"]:
        step4_generate_conversations(
            splits[group], shadow_adapter,
            os.path.join(data_dir, f"conversation_{group}_shadow.json"),
            CONFIG["shadow_temperatures"]
        )
    
    print("\n--- Shadow Attack Conversations (target model) ---")
    for group in ["target_member", "target_non_member"]:
        step4_generate_conversations(
            splits[group], target_adapter,
            os.path.join(data_dir, f"conversation_{group}_shadow.json"),
            CONFIG["shadow_temperatures"]
        )
    
    # Reference attack: query target model on member + non-member at temp=0.1
    print("\n--- Reference Attack Conversations ---")
    for group in ["member", "non_member"]:
        step4_generate_conversations(
            splits[group], target_adapter,
            os.path.join(data_dir, f"conversation_{group}_reference.json"),
            CONFIG["reference_temperatures"]
        )
    
    # Target-only attack: query target model at two temps
    print("\n--- Target-Only Attack Conversations ---")
    for group in ["member", "non_member"]:
        step4_generate_conversations(
            splits[group], target_adapter,
            os.path.join(data_dir, f"conversation_{group}_target_only.json"),
            CONFIG["target_only_temperatures"]
        )


# ============================================================================
# STEP 5: SIMILARITY COMPUTATION
# ============================================================================

def step5_compute_similarity():
    """Compute Rouge-2 + MPNet similarity for all conversation files."""
    print("\n" + "=" * 70)
    print("  STEP 5: Computing Similarity Scores")
    print("=" * 70)
    
    from rouge import Rouge
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    rouge = Rouge()
    print("  Loading MPNet model...")
    mpnet = SentenceTransformer('all-mpnet-base-v2')
    
    data_dir = CONFIG["data_dir"]
    sim_dir = os.path.join(data_dir, "similarity")
    os.makedirs(sim_dir, exist_ok=True)
    
    # Process each conversation file
    conversation_files = [
        # Shadow attack
        ("conversation_shadow_member_shadow.json", CONFIG["shadow_temperatures"]),
        ("conversation_shadow_non_member_shadow.json", CONFIG["shadow_temperatures"]),
        ("conversation_target_member_shadow.json", CONFIG["shadow_temperatures"]),
        ("conversation_target_non_member_shadow.json", CONFIG["shadow_temperatures"]),
        # Reference attack
        ("conversation_member_reference.json", CONFIG["reference_temperatures"]),
        ("conversation_non_member_reference.json", CONFIG["reference_temperatures"]),
        # Target-only attack
        ("conversation_member_target_only.json", CONFIG["target_only_temperatures"]),
        ("conversation_non_member_target_only.json", CONFIG["target_only_temperatures"]),
    ]
    
    for conv_file, temperatures in conversation_files:
        conv_path = os.path.join(data_dir, conv_file)
        sim_file = conv_file.replace("conversation_", "similarity_")
        sim_path = os.path.join(sim_dir, sim_file)
        
        if not os.path.exists(conv_path):
            print(f"  Skipping {conv_file} (not found)")
            continue
        
        if os.path.exists(sim_path):
            print(f"  Skipping {sim_file} (already computed)")
            continue
        
        print(f"\n  Processing {conv_file}...")
        with open(conv_path, 'r') as f:
            all_data = json.load(f)
        
        results = []
        count = 0
        start_time = time.time()
        
        for item in all_data:
            image_id = item["image_id"]
            similarity_result = {"image_id": image_id}
            
            for temp in temperatures:
                key = f"conversations_{temp}"
                if key not in item:
                    continue
                
                vlm_text = ""
                truth_text = ""
                
                for convo in item[key]:
                    if convo["from"] == "vlm_1":
                        vlm_text += convo["value"].replace('</s>', '') + ' '
                    elif convo["from"] == "ground truth":
                        truth_text += convo["value"] + ' '
                
                vlm_text = vlm_text.strip()
                truth_text = truth_text.strip()
                
                if not vlm_text or not truth_text:
                    similarity_result[f"similarity_{temp}"] = {"rouge2_f": 0.0, "embedding_mpn": 0.0}
                    continue
                
                # Rouge-2
                try:
                    rouge_scores = rouge.get_scores(vlm_text, truth_text)[0]
                    rouge2_f = rouge_scores['rouge-2']['f']
                except Exception:
                    rouge2_f = 0.0
                
                # MPNet cosine similarity
                vlm_emb = mpnet.encode(vlm_text)
                truth_emb = mpnet.encode(truth_text)
                mpn_sim = float(cosine_similarity([vlm_emb], [truth_emb])[0][0])
                
                similarity_result[f"similarity_{temp}"] = {
                    "rouge2_f": rouge2_f,
                    "embedding_mpn": mpn_sim
                }
            
            results.append(similarity_result)
            count += 1
            
            if count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"    [{count}/{len(all_data)}] {elapsed:.1f}s elapsed")
        
        with open(sim_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        total_time = time.time() - start_time
        print(f"    -> Saved {len(results)} similarity scores to {sim_file} ({total_time:.1f}s)")


# ============================================================================
# STEP 6: PACKAGE FOR DOWNLOAD
# ============================================================================

def step6_package_results():
    """Create a summary and package results for local download."""
    print("\n" + "=" * 70)
    print("  STEP 6: Packaging Results")
    print("=" * 70)
    
    data_dir = CONFIG["data_dir"]
    sim_dir = os.path.join(data_dir, "similarity")
    
    # List all output files
    print("\n  Files ready for download:")
    total_size = 0
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith('.json'):
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                total_size += size
                rel_path = os.path.relpath(path, data_dir)
                print(f"    {rel_path}: {size/1024:.1f} KB")
    
    print(f"\n  Total download size: {total_size/1024/1024:.1f} MB")
    print(f"\n  To download to your local machine:")
    print(f"    1. Use Lightning AI's download button on the real_data/ folder")
    print(f"    2. Place it at ~/vlm_mia/real_data/ on your local machine")
    print(f"    3. Run: bash real_pipeline/07_run_attacks_local.sh")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  VLM-MIA REAL PIPELINE — Lightning AI (T4 GPU)")
    print("=" * 70)
    print(f"  Samples: {CONFIG['total_samples']}")
    print(f"  Member ratio: {CONFIG['member_ratio']}")
    print(f"  Model: {CONFIG['model_name']} (4-bit QLoRA)")
    print(f"  LoRA rank: {CONFIG['lora_rank']}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    
    start_time = time.time()
    
    # Step 1: Download
    dataset = step1_download_data()
    
    # Step 2: Split
    splits = step2_split_data(dataset)
    
    # Step 3: Fine-tune TWO models
    print("\n" + "=" * 70)
    print("  STEP 3a: Training SHADOW model")
    print("=" * 70)
    shadow_adapter = step3_lora_finetune(splits["shadow_member"], "shadow_adapter")
    
    print("\n" + "=" * 70)
    print("  STEP 3b: Training TARGET model")
    print("=" * 70)
    target_adapter = step3_lora_finetune(splits["target_member"], "target_adapter")
    
    # Step 4: Generate conversations
    step4_all_conversations(splits)
    
    # Step 5: Compute similarity
    step5_compute_similarity()
    
    # Step 6: Package
    step6_package_results()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {total_time/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()

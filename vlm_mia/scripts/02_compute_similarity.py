"""
Compute text similarity between VLM responses and ground truth.

This is a modified version that works WITHOUT OpenAI API.
Uses only:
  - Rouge-2 F-score (text overlap metric)
  - MPNet embeddings via sentence-transformers (local, no API needed)

Handles both:
  - Ground truth similarity (for Shadow, Reference, Target-Only attacks)
  - Repeating generation similarity (for Image-Only attack)
"""

import json
import os
import time
import argparse
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from itertools import combinations


def similarity_with_ground_truth(conversation_file_name, output_file_name, temperatures):
    """Compute similarity between VLM output and ground truth."""
    with open(conversation_file_name, 'r') as file:
        all_data = json.load(file)

    rouge = Rouge()
    model = SentenceTransformer('all-mpnet-base-v2')
    
    results = []
    count = 0
    start_time = time.time()
    
    for item in all_data:
        image_id = item["image_id"]
        similarity_result = {"image_id": image_id}
        
        for temperature in temperatures:
            vlm_text = ""
            truth_text = ""
            
            for convo in item[f"conversations_{temperature}"]:
                if convo["from"] == "vlm_1":
                    vlm_text += convo["value"].replace('</s>', '') + ' '
                elif convo["from"] == "ground truth":
                    truth_text += convo["value"] + ' '
            
            vlm_text = vlm_text.strip()
            truth_text = truth_text.strip()
            
            if not vlm_text or not truth_text:
                continue
            
            # Rouge-2 F-score
            try:
                rouge_scores = rouge.get_scores(vlm_text, truth_text)[0]
                rouge2_f = rouge_scores['rouge-2']['f']
            except Exception:
                rouge2_f = 0.0
            
            # MPNet embedding cosine similarity
            vlm_embedding = model.encode(vlm_text)
            truth_embedding = model.encode(truth_text)
            embedding_similarity_mpn = float(cosine_similarity([vlm_embedding], [truth_embedding])[0][0])
            
            similarity_result[f"similarity_{temperature}"] = {
                "rouge2_f": rouge2_f,
                "embedding_mpn": embedding_similarity_mpn
            }
        
        results.append(similarity_result)
        count += 1
        if count % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(f"Processed {count} samples. Last 100 took {elapsed_time:.1f}s")
    
    os.makedirs(os.path.dirname(output_file_name) if os.path.dirname(output_file_name) else '.', exist_ok=True)
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file, indent=4)
    print(f"Similarity results saved to {output_file_name} ({len(results)} samples)")


def similarity_with_repeating_generation(conversation_file_name, output_file_name, temperatures, repeating_num):
    """Compute pairwise similarity between repeated VLM responses (Image-Only attack)."""
    with open(conversation_file_name, 'r') as file:
        all_data = json.load(file)

    rouge = Rouge()
    model = SentenceTransformer('all-mpnet-base-v2')
    
    results = []
    count = 0
    start_time = time.time()
    
    for item in all_data:
        image_id = item["image_id"]
        similarity_result = {"image_id": image_id}
        
        for temperature in temperatures:
            vlm_texts = {f'vlm_{i+1}': "" for i in range(repeating_num)}
            
            for convo in item[f"conversations_{temperature}"]:
                if 'vlm_' in convo["from"]:
                    vlm_num = int(convo["from"].split('_')[1])
                    if vlm_num <= repeating_num:
                        index = convo["from"]
                        vlm_texts[index] += convo["value"].replace('</s>', '').strip() + ' '
            
            # Compute embeddings
            embeddings_mpn = {key: model.encode(text).reshape(1, -1) for key, text in vlm_texts.items()}
            
            # Compute pairwise similarities
            similarity_scores_rouge = []
            similarity_scores_mpn = []
            
            for (vlm_i, text_i), (vlm_j, text_j) in combinations(vlm_texts.items(), 2):
                try:
                    similarity_scores_rouge.append(rouge.get_scores(text_i.strip(), text_j.strip())[0]['rouge-2']['f'])
                except Exception:
                    similarity_scores_rouge.append(0.0)
                similarity_scores_mpn.append(float(cosine_similarity(embeddings_mpn[vlm_i], embeddings_mpn[vlm_j])[0][0]))
            
            similarity_result[f"similarity_{temperature}"] = {
                "rouge2_f": sum(similarity_scores_rouge) / max(len(similarity_scores_rouge), 1),
                "embedding_mpn": sum(similarity_scores_mpn) / max(len(similarity_scores_mpn), 1)
            }
        
        results.append(similarity_result)
        count += 1
        if count % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(f"Processed {count} samples. Last 100 took {elapsed_time:.1f}s")
    
    os.makedirs(os.path.dirname(output_file_name) if os.path.dirname(output_file_name) else '.', exist_ok=True)
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file, indent=4)
    print(f"Similarity results saved to {output_file_name} ({len(results)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['ground_truth', 'repeating'], required=True)
    parser.add_argument('--conversation_json_path', type=str, required=True)
    parser.add_argument('--similarity_json_path', type=str, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.1])
    parser.add_argument("--repeating_num", type=int, default=5)
    args = parser.parse_args()
    
    if args.mode == 'ground_truth':
        similarity_with_ground_truth(args.conversation_json_path, args.similarity_json_path, args.temperatures)
    else:
        similarity_with_repeating_generation(args.conversation_json_path, args.similarity_json_path, args.temperatures, args.repeating_num)


if __name__ == '__main__':
    main()

import json
import time
import argparse
import openai
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from itertools import combinations


api_key = '' #Please fill your openai api key here
def similarity_with_repeating_generation(conversation_file_name, output_file_name, temperatures, repeating_num):
    with open(conversation_file_name, 'r') as file:
        all_data = json.load(file)

    client = openai.OpenAI(api_key=api_key)
    rouge = Rouge()
    model = SentenceTransformer('all-mpnet-base-v2')
    
    results = []
    count = 0
    start_time = time.time()
    for item in all_data:
        image_id = item["image_id"]
        similarity_result = {
                "image_id": image_id,
            }
        for temperature in temperatures:
            vlm_texts = {f'vlm_{i+1}': "" for i in range(repeating_num)}
            for convo in item[f"conversations_{temperature}"]:
                if 'vlm_' in convo["from"]:
                    vlm_num = int(convo["from"].split('_')[1])
                    if vlm_num <= repeating_num:
                        index = convo["from"]
                        vlm_texts[index] += convo["value"].replace('</s>', '').strip() + ' '

            embeddings_mpn = {key: model.encode(text).reshape(1, -1) for key, text in vlm_texts.items()}

            embeddings_gpt = {}
            for key, text in vlm_texts.items():
                for attempt in range(3):
                    try:
                        embeddings_gpt[key] = client.embeddings.create(
                            model="text-embedding-3-large",
                            input=text,
                            encoding_format="float"
                        ).data[0].embedding
                        break
                    except openai.OpenAIError as e:
                        print(f"Image: {image_id}_{key}: OpenAI API error on attempt {attempt + 1}: {e}")
                        time.sleep(2**attempt)


            similarity_scores_rouge = []
            similarity_scores_mpn = []
            similarity_scores_gpt = []
            
            for (vlm_i, text_i), (vlm_j, text_j) in combinations(vlm_texts.items(), 2):
                similarity_scores_rouge.append(rouge.get_scores(text_i, text_j)[0]['rouge-2']['f'])
                similarity_scores_mpn.append(float(cosine_similarity(embeddings_mpn[vlm_i], embeddings_mpn[vlm_j])[0][0]))
                similarity_scores_gpt.append(float(cosine_similarity([embeddings_gpt[vlm_i]], [embeddings_gpt[vlm_j]])[0][0]))

            similarity_result[f"similarity_{temperature}"] = {
                    "rouge2_f": sum(similarity_scores_rouge) / len(similarity_scores_rouge),
                    "embedding_gpt": sum(similarity_scores_gpt) / len(similarity_scores_gpt),
                    "embedding_mpn": sum(similarity_scores_mpn) / len(similarity_scores_mpn)
                }

        results.append(similarity_result)
        count += 1
        if count % 100 == 0:
            with open(output_file_name, 'w') as output_file:
                json.dump(results, output_file, indent=4)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(f"Finish the process of {count} samples. Process of last 100 samples consume {elapsed_time} seconds")
    with open(output_file_name, 'w') as output_file:
        json.dump(results, output_file, indent=4)

def main(args):
    similarity_with_repeating_generation(args.conversation_json_path, args.similarity_json_path, args.temperatures, args.repeating_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conversation_json_path', type=str, required = True)
    parser.add_argument('--similarity_json_path', type=str, required = True)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
    parser.add_argument("--repeating_num", type=int,  default=10)
    args = parser.parse_args()
    main(args)


import argparse
import json
import numpy as np
import random
from sklearn.metrics import roc_auc_score

def load_data(member_similarity_file, non_member_similarity_file, temperature, metric):
    with open(member_similarity_file, 'r') as file:
        member_data_all = json.load(file)
    with open(non_member_similarity_file, 'r') as file:
        non_member_data_all = json.load(file)
    member_data = [item[f'similarity_{temperature}'][metric] for item in member_data_all]
    non_member_data = [item[f'similarity_{temperature}'][metric] for item in non_member_data_all]
    return member_data, non_member_data

def image_only_inference(member_data, non_member_data, granularity):
    similarity_list = []
    label_list = []
    for _ in range(1000):
        samples_member = random.sample(member_data, granularity)
        samples_non_member = random.sample(non_member_data, granularity)
        similarity_list.append(np.mean(samples_member))
        label_list.append(1)
        similarity_list.append(np.mean(samples_non_member))
        label_list.append(0)

    auc = roc_auc_score(label_list, similarity_list)
    return auc

def main(args):
    member_data, non_member_data = load_data(args.member_similarity_file, args.non_member_similarity_file, args.temperature, args.similarity_metric)
    aucs = []

    for _ in range(5):
        auc = image_only_inference(member_data, non_member_data, args.granularity)
        aucs.append(auc)

    avg_auc = sum(aucs) / len(aucs)

    print(f'Accuracy: {avg_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member_similarity_file', type=str, required = True)
    parser.add_argument('--non_member_similarity_file', type=str, required = True)
    parser.add_argument('--granularity', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f')

    args = parser.parse_args()
    main(args)

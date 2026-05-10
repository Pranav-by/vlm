import json
import numpy as np
import os
import argparse
import random
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def load_data(member_similarity_file, non_member_similarity_file, temperature, metric):
    with open(member_similarity_file, 'r') as file:
        member_data_all = json.load(file)
    with open(non_member_similarity_file, 'r') as file:
        non_member_data_all = json.load(file)
    member_data = [item[f'similarity_{temperature}'][metric] for item in member_data_all]
    non_member_data = [item[f'similarity_{temperature}'][metric] for item in non_member_data_all]
    return member_data, non_member_data

def reference_non_member_inference(member_data, non_member_data, granularity):
    random.shuffle(non_member_data)
    half = len(non_member_data) // 2
    reference_non_member = non_member_data[:half]
    target_non_member = non_member_data[half:]
    target_member = member_data
    p_list = []
    label_list = []
    for _ in range(1000):
        samples_target_member = random.sample(target_member, granularity)
        samples_reference_non_member = random.sample(reference_non_member, granularity)
        samples_target_non_member = random.sample(target_non_member, granularity)

        mean_target_member, mean_reference_non_member, mean_target_non_member = np.mean(samples_target_member), np.mean(samples_reference_non_member), np.mean(samples_target_non_member)
        var_target_member, var_reference_non_member, var_target_non_member = np.var(samples_target_member, ddof=1), np.var(samples_reference_non_member, ddof=1), np.var(samples_target_non_member, ddof=1)
        
        z_member = (mean_target_member - mean_reference_non_member) / np.sqrt(var_target_member / len(samples_target_member) + var_reference_non_member / len(samples_reference_non_member))
        p_member = 1- norm.cdf(z_member)
        p_list.append(p_member)
        label_list.append(0)
        
        z_non_member = (mean_target_non_member - mean_reference_non_member) / np.sqrt(var_reference_non_member / len(samples_reference_non_member) + var_target_non_member / len(samples_target_non_member))
        p_non_member = 1- norm.cdf(z_non_member)
        p_list.append(p_non_member)
        label_list.append(1)

        # print(f'p_member:{p_member}, p_non_member:{p_non_member}')

    auc = roc_auc_score(label_list, p_list)
    return auc


def main(args):
    member_data, non_member_data = load_data(args.member_similarity_file, args.non_member_similarity_file, args.temperature, args.similarity_metric)
    aucs = []

    for _ in range(5):
        auc = reference_non_member_inference(member_data, non_member_data, args.granularity)
        aucs.append(auc)

    avg_auc = sum(aucs) / len(aucs)

    print(f'Accuracy: {avg_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member_similarity_file', type=str, required = True)
    parser.add_argument('--non_member_similarity_file', type=str, required = True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f')

    args = parser.parse_args()
    main(args)


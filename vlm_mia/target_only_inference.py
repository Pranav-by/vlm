import json
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from random import sample
import argparse

def load_data(member_similarity_file, non_member_similarity_file):
    with open(member_similarity_file, 'r') as file:
        member_data_all = json.load(file)
    with open(non_member_similarity_file, 'r') as file:
        non_member_data_all = json.load(file)
    return member_data_all, non_member_data_all

def perform_z_test(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_se = np.sqrt(std1**2/n1 + std2**2/n2)
    z = (mean1 - mean2) / pooled_se
    p_value = norm.sf(z)
    return p_value

def target_only_inference(member_data_all, non_member_data_all, granularity, temperature_low, temperature_high, metric):
    all_indices_member = range(len(member_data_all))
    all_indices_non_member = range(len(non_member_data_all))
    p_list = []
    label_list = []
    for _ in range(1000):
        member_sampled_indices = sample(all_indices_member, granularity)
        member_low = [member_data_all[index][f'similarity_{temperature_low}'][metric]  for index in member_sampled_indices]
        member_high = [member_data_all[index][f'similarity_{temperature_high}'][metric]  for index in member_sampled_indices]
        p_member = perform_z_test(member_low, member_high)
        p_list.append(p_member)
        label_list.append(0)

        non_member_sampled_indices = sample(all_indices_non_member, granularity)
        non_member_low = [non_member_data_all[index][f'similarity_{temperature_low}'][metric]  for index in non_member_sampled_indices]
        non_member_high = [non_member_data_all[index][f'similarity_{temperature_high}'][metric]  for index in non_member_sampled_indices]
        p_non_member = perform_z_test(non_member_low, non_member_high)
        p_list.append(p_non_member)
        label_list.append(1)
        # print(f'p_member:{p_member}, p_non_member:{p_non_member}')
    auc = roc_auc_score(label_list, p_list)
    return auc

def main(args):
    member_data_all, non_member_data_all = load_data(args.member_similarity_file, args.non_member_similarity_file)
    aucs = []

    for _ in range(5):
        auc = target_only_inference(member_data_all, non_member_data_all, args.granularity, args.temperature_low, args.temperature_high, args.similarity_metric)
        aucs.append(auc)

    avg_auc = sum(aucs) / len(aucs)

    print(f'Accuracy: {avg_auc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member_similarity_file', type=str, required = True)
    parser.add_argument('--non_member_similarity_file', type=str, required = True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperature_high', type=float, default=1.6)
    parser.add_argument('--temperature_low', type=float, default=0.1)
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f')

    args = parser.parse_args()
    main(args)

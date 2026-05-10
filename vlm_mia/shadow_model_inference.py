import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import numpy as np
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_data(data, test_size=0.2):
    train, validation = train_test_split(data, test_size=test_size, random_state=42)
    return train, validation

def sample_and_compute_metrics(data, sample_number, granularity, temperatures, metric, label, with_variance):
    results = []
    for _ in range(sample_number):
        sampled = np.random.choice(data, size=granularity, replace=False)
        mean_scores = [np.mean([item[f'similarity_{temp}'][metric] for item in sampled]) for temp in temperatures]
        if with_variance:
            variance_scores = [np.var([item[f'similarity_{temp}'][metric] for item in sampled]) for temp in temperatures]
            mean_scores.extend(variance_scores)
        mean_scores.append(label) 
        results.append(mean_scores)
    return results

def create_datasets(shadow_member_similarity_file, shadow_non_member_similarity_file, target_member_similarity_file, target_non_member_similarity_file, temperatures, granularity, metric, with_variance=True):
    member_train_val_data = load_json(shadow_member_similarity_file)
    non_member_train_val_data = load_json(shadow_non_member_similarity_file)
    member_test_data = load_json(target_member_similarity_file)
    non_member_test_data = load_json(target_non_member_similarity_file)

    member_train, member_val = split_data(member_train_val_data, test_size=0.2)
    non_member_train, non_member_val = split_data(non_member_train_val_data, test_size=0.2)

    train_dataset = sample_and_compute_metrics(member_train, 8000, granularity, temperatures, metric, label=1, with_variance=with_variance) + \
                sample_and_compute_metrics(non_member_train, 8000, granularity, temperatures, metric, label=0, with_variance=with_variance)
    val_dataset = sample_and_compute_metrics(member_val, 2000, granularity, temperatures, metric, label=1, with_variance=with_variance) + \
            sample_and_compute_metrics(non_member_val, 2000, granularity, temperatures, metric, label=0, with_variance=with_variance)
    test_dataset = sample_and_compute_metrics(member_test_data, 10000, granularity, temperatures, metric, label=1, with_variance=with_variance) + \
                sample_and_compute_metrics(non_member_test_data, 10000, granularity, temperatures, metric, label=0, with_variance=with_variance)

    return train_dataset, val_dataset, test_dataset

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def load_data(data):
    data = np.array(data)
    features = torch.tensor(data[:, :-1], dtype=torch.float32)
    labels = torch.tensor(data[:, -1], dtype=torch.float32)
    return TensorDataset(features, labels)


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None):
    best_loss = float('inf')
    best_model_wts = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(dim=1), target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output.squeeze(dim=1), target).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

        if scheduler:
            scheduler.step(val_loss)
            # current_lr = optimizer.param_groups[0]['lr']
            # print(f'Current Learning Rate: {current_lr}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    if best_model_wts:
        model.load_state_dict(best_model_wts)

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = (output.squeeze() > 0.5).int()
            predictions.extend(predicted.tolist())
            actuals.extend(target.tolist())
    
    accuracy = accuracy_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    return accuracy, recall, precision


def train_and_evaluate_model(train_dataset, val_dataset, test_dataset, epochs, learning_rate, schedule_factor, schedule_patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_data(train_dataset)
    val_data = load_data(val_dataset)
    test_data = load_data(test_dataset)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    input_dim = train_data.tensors[0].shape[1]
    model = BinaryClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=schedule_factor, patience=schedule_patience) 

    train_model(model, train_loader, val_loader, epochs=epochs, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    accuracy, recall, precision = evaluate_model(model, test_loader)
    return accuracy, recall, precision


def main(args):
    train_dataset, val_dataset, test_dataset = create_datasets(args.shadow_member_similarity_file, args.shadow_non_member_similarity_file, args.target_member_similarity_file, args.target_non_member_similarity_file, args.temperatures, args.granularity, args.similarity_metric, with_variance=args.with_variance)
    accuracies = []
    recalls = []
    precisions = []
    for _ in range(5):
        accuracy, recall, precision = train_and_evaluate_model(train_dataset, val_dataset, test_dataset, args.epochs, args.learning_rate, args.schedule_factor, args.schedule_patience)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)

    print(f'Accuracy: {avg_accuracy:.4f}, Recall: {avg_recall:.4f}, Precision: {avg_precision:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shadow_member_similarity_file', type=str, required = True)
    parser.add_argument('--shadow_non_member_similarity_file', type=str, required = True)
    parser.add_argument('--target_member_similarity_file', type=str, required = True)
    parser.add_argument('--target_non_member_similarity_file', type=str, required = True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f')
    parser.add_argument('--with_variance', action='store_true')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--schedule_factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--schedule_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced')
    args = parser.parse_args()
    main(args)
 
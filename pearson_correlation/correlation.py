import json
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations

# File paths
gearnet_path = '../workspace_data/similarity_scores_gearnet_gemma2.json'
gat_path = '../workspace_data/similarity_scores_gat_gemma2.json'
gvp_path = '../workspace_data/similarity_scores_gvp_gemma2.json'
scannet_path = '../workspace_data/similarity_scores_scannet_gemma2.json'
random_path = '../workspace_data/similarity_scores_gearnet_gemma2_random.json'

# Load all JSON files
with open(gearnet_path, 'r') as file1, open(gat_path, 'r') as file2, open(gvp_path, 'r') as file3, \
     open(scannet_path, 'r') as file4, open(random_path, 'r') as file5:
    data1 = json.load(file1)
    data2 = json.load(file2)
    data3 = json.load(file3)
    data4 = json.load(file4)
    data5 = json.load(file5)

# Get common proteins across all files
common_proteins = set(data1.keys()) & set(data2.keys()) & set(data3.keys()) & set(data4.keys()) & set(data5.keys())

# Prepare a dictionary to store similarities
similarities = {
    'gearnet': [],
    'gat': [],
    'gvp': [],
    'scannet': [],
    'random': []
}

# Collect similarity scores for the common proteins
for protein in common_proteins:
    similarities['gearnet'].append(data1[protein]["similarity_score"])
    similarities['gat'].append(data2[protein]["similarity_score"])
    similarities['gvp'].append(data3[protein]["similarity_score"])
    similarities['scannet'].append(data4[protein]["similarity_score"])
    similarities['random'].append(data5[protein]["similarity_score"])

print(len(common_proteins))

# List of model to compare
model_names = ['gearnet', 'gat', 'gvp', 'scannet', 'random']

# Calculate and print Pearson correlation for all combinations of model
for model1, model2 in combinations(model_names, 2):
    similarities_file1 = similarities[model1]
    similarities_file2 = similarities[model2]
    
    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(similarities_file1, similarities_file2)
    
    print(f"Pearson correlation between {model1} and {model2}: {pearson_corr:.4f}")

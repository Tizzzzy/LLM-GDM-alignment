import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the projection head for the protein representation
class ProteinProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProteinProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

class TokenProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TokenProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x): # (1, 4096)
        return F.normalize(self.fc(x), dim=-1)

seed = 42  # You can use any integer
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

with open("../workspace_data/protein_representations_gemma2.json", "r") as json_file:
    protein_representations_gemma2 = json.load(json_file)

with open("../GAT_new/protein_representations_gat.json", "r") as json_file:
    protein_representations_gat = json.load(json_file)

protein_ids = list(protein_representations_gemma2.keys())
random.shuffle(protein_ids)

train_split = int(0.8 * len(protein_ids))
val_split = int(0.9 * len(protein_ids))

train_ids = protein_ids[:train_split]
val_ids = protein_ids[train_split:val_split]
test_ids = protein_ids[val_split:]

def create_representation_lists(protein_ids, protein_representations_gat, protein_representations_gemma2):
    word_representation_list = []
    graph_feature_list = []
    for protein_id in protein_ids:
        if protein_id in protein_representations_gat:
            graph_feature = torch.tensor(protein_representations_gat[protein_id]["protein_representation_avg"]).to(device)
            word_representation = torch.tensor(protein_representations_gemma2[protein_id]["word_representation"]).to(device) # (2304, )
            word_representation = word_representation.unsqueeze(0)
            # word_representation = F.normalize(word_representation, dim=-1)
                
            word_representation_list.append(word_representation)
            graph_feature_list.append(graph_feature)
    print(graph_feature.shape, word_representation.shape)
    return word_representation_list, graph_feature_list

train_word_list, train_graph_list = create_representation_lists(train_ids, protein_representations_gat, protein_representations_gemma2)
val_word_list, val_graph_list = create_representation_lists(val_ids, protein_representations_gat, protein_representations_gemma2)
test_word_list, test_graph_list = create_representation_lists(test_ids, protein_representations_gat, protein_representations_gemma2)
print(f"length of train: {len(train_word_list)}, length of val: {len(val_word_list)}, length of test: {len(test_word_list)}")

# Initialize the projection heads
protein_proj_head = ProteinProjectionHead(input_dim=64, output_dim=2304).to(device)
token_proj_head = TokenProjectionHead(input_dim=2304, output_dim=2304).to(device)

protein_proj_head.load_state_dict(torch.load('../workspace_data/best_protein_proj_head_gat_gemma2.pth'))
token_proj_head.load_state_dict(torch.load('../workspace_data/best_token_proj_head_gat_gemma2.pth'))

protein_proj_head.eval()
token_proj_head.eval()

positive_score_list = []
with torch.no_grad():
    projected_proteins_test = [protein_proj_head(graph_feature) for graph_feature in test_graph_list]
    projected_tokens_test = [token_proj_head(word_representation) for word_representation in test_word_list]

for protein_rep, word_rep in zip(projected_proteins_test, projected_tokens_test):
    cosine_sim = F.cosine_similarity(protein_rep, word_rep, dim=-1)
    positive_score_list.append(cosine_sim)

# print(protein_rep, word_rep)

positive_score = torch.mean(torch.stack(positive_score_list))

negative_score_list = []
for i in range(len(projected_proteins_test)):
    available_negatives = [j for j in range(len(projected_tokens_test)) if j != i]
    for j in available_negatives:
        negative_sim = F.cosine_similarity(projected_proteins_test[i], projected_tokens_test[j], dim=-1)
        negative_score_list.append(torch.abs(negative_sim))
        
negative_score = torch.mean(torch.stack(negative_score_list))

print(f"positive_score: {positive_score}, negative_score: {negative_score}")

metric = positive_score - negative_score
print(metric)
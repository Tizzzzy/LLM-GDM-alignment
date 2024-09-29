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

with open("../workspace_data/protein_representations_scannet/protein_representations_scannet.json", "r") as json_file:
    protein_representations_scannet = json.load(json_file)

with open("../workspace_data/protein_representations_gemma2.json", "r") as json_file:
    protein_representations_gemma2 = json.load(json_file)

protein_ids = list(protein_representations_gemma2.keys())
random.shuffle(protein_ids)

train_split = int(0.8 * len(protein_ids))
val_split = int(0.9 * len(protein_ids))

train_ids = protein_ids[:train_split]
val_ids = protein_ids[train_split:val_split]
test_ids = protein_ids[val_split:]


def create_representation_lists(protein_ids, protein_representations_scannet, protein_representations_gemma2):
    word_representation_list = []
    graph_feature_list = []
    for protein_id in protein_ids:
        if protein_id in protein_representations_scannet:
            graph_feature = torch.tensor(protein_representations_scannet[protein_id]["average_pooled"]).to(device)
            word_representation = torch.tensor(protein_representations_gemma2[protein_id]["word_representation"]).to(device) # (2304, )
            word_representation = word_representation.unsqueeze(0)
            # word_representation = F.normalize(word_representation, dim=-1)
            # print(graph_feature.shape, word_representation.shape)
                
            word_representation_list.append(word_representation)
            graph_feature_list.append(graph_feature)
    print(graph_feature.shape, word_representation.shape)
    return word_representation_list, graph_feature_list

train_word_list, train_graph_list = create_representation_lists(train_ids, protein_representations_scannet, protein_representations_gemma2)
val_word_list, val_graph_list = create_representation_lists(val_ids, protein_representations_scannet, protein_representations_gemma2)
test_word_list, test_graph_list = create_representation_lists(test_ids, protein_representations_scannet, protein_representations_gemma2)
print(f"length of train: {len(train_word_list)}, length of val: {len(val_word_list)}, length of test: {len(test_word_list)}")

protein_proj_head = ProteinProjectionHead(input_dim=128, output_dim=2304).to(device)
token_proj_head = TokenProjectionHead(input_dim=2304, output_dim=2304).to(device)

class CosineContrastiveLoss(nn.Module):
    def __init__(self, temperature, num_negatives, lambdaa, margin):
        super(CosineContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.lambdaa = lambdaa
        self.margin = margin

# version 4 InfoNCE loss
    def forward(self, projected_proteins, projected_tokens):
        assert len(projected_proteins) == len(projected_tokens), "Mismatch in number of protein and token representations"

        total_loss = 0.0
        for i in range(len(projected_proteins)):
            # Cosine similarity for the positive pair (i == j)
            positive_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[i], dim=-1)
            positive_sim = (positive_sim + 1) / 2

            # Contrastive loss for negative pairs (i != j)
            available_negatives = [j for j in range(len(projected_tokens)) if j != i]
            negative_sims = []
            for j in available_negatives:
                negative_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[j], dim=-1)
                negative_sim = (negative_sim + 1) / 2
                negative_sims.append(negative_sim)
            
            # Convert the list of negative similarities to a tensor
            negative_sims = torch.stack(negative_sims)
            # print(negative_sims.shape) # negative_sims shape = torch.Size([batch_size-1, 1])
            
            # Compute the InfoNCE loss
            exp_positive_sim = torch.exp(positive_sim / self.temperature)
            exp_negative_sims = torch.exp(negative_sims / self.temperature)
            loss = -torch.log(exp_positive_sim / (exp_positive_sim + exp_negative_sims.sum()))

            # Accumulate the total loss
            total_loss += loss

        # print(projected_proteins[i].squeeze().shape, projected_tokens[i].shape)
        
        # Return the average loss over all pairs
        return total_loss / len(projected_proteins)


contrastive_loss_fn = CosineContrastiveLoss(temperature=0.2, num_negatives=64, lambdaa=0.5, margin=0.3).to(device)

# optimizer = torch.optim.Adam(list(protein_proj_head.parameters()), lr=2e-3)
optimizer = torch.optim.Adam(list(protein_proj_head.parameters()) + list(token_proj_head.parameters()), lr=1e-3)
num_epochs = 40
batch_size = 32
best_val_loss = float('inf')

# Training loop with batch
for epoch in range(num_epochs):
    protein_proj_head.train()
    token_proj_head.train()

    for i in range(0, len(train_word_list), batch_size):
        batch_word_list = train_word_list[i:i+batch_size]
        batch_graph_list = train_graph_list[i:i+batch_size]

        projected_proteins = [protein_proj_head(graph_feature) for graph_feature in batch_graph_list]
        projected_tokens = [token_proj_head(word_representation) for word_representation in batch_word_list] 

        train_loss = contrastive_loss_fn(projected_proteins, projected_tokens)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    protein_proj_head.eval()
    token_proj_head.eval()

    with torch.no_grad():

        projected_proteins_val = [protein_proj_head(graph_feature) for graph_feature in val_graph_list]
        projected_tokens_val = [token_proj_head(word_representation) for word_representation in val_word_list]
        
        val_loss = contrastive_loss_fn(projected_proteins_val, projected_tokens_val)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    if val_loss.item() < best_val_loss:
        torch.save(protein_proj_head.state_dict(), '../workspace_data/best_protein_proj_head_scannet_gemma2.pth')
        torch.save(token_proj_head.state_dict(), '../workspace_data/best_token_proj_head_scannet_gemma2.pth')
        best_val_loss = val_loss.item()
        print("best projection head saved")

# testing
protein_proj_head = ProteinProjectionHead(input_dim=128, output_dim=2304).to(device)
token_proj_head = TokenProjectionHead(input_dim=2304, output_dim=2304).to(device)

protein_proj_head.load_state_dict(torch.load('../workspace_data/best_protein_proj_head_scannet_gemma2.pth'))
token_proj_head.load_state_dict(torch.load('../workspace_data/best_token_proj_head_scannet_gemma2.pth'))

protein_proj_head.eval()
token_proj_head.eval()

with torch.no_grad():

    projected_proteins_test = [protein_proj_head(graph_feature) for graph_feature in test_graph_list]
    projected_tokens_test = [token_proj_head(word_representation) for word_representation in test_word_list]

for protein_rep, word_rep in zip(projected_proteins_test, projected_tokens_test):
    cosine_sim = F.cosine_similarity(protein_rep, word_rep, dim=-1)
    print(cosine_sim)

test_loss = contrastive_loss_fn(projected_proteins_test, projected_tokens_test)
print(f"Test Loss: {test_loss.item()}")


# test negative pairs
protein_proj_head.eval()
token_proj_head.eval()

length = min(len(val_graph_list), len(test_word_list))
val_graph_list = val_graph_list[:length]
test_word_list = test_word_list[:length]

with torch.no_grad():

    projected_proteins = [protein_proj_head(graph_feature) for graph_feature in val_graph_list]
    projected_tokens = [token_proj_head(word_representation) for word_representation in test_word_list]

for protein_rep, token_rep in zip(projected_proteins, projected_tokens):
    cosine_sim = F.cosine_similarity(protein_rep, token_rep, dim=-1)
    print(cosine_sim)
    break
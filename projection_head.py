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
        # return self.fc(x)

# Define the projection head for the token representation
class TokenProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TokenProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)
        # return self.fc(x)


# # Define the projection head for the protein representation
# class ProteinProjectionHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(ProteinProjectionHead, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  # First linear layer
#         self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second linear layer
#         self.activation = nn.ReLU()  # Non-linear activation function

#     def forward(self, x):
#         x = self.activation(self.fc1(x))  # Apply first linear layer and activation
#         x = self.fc2(x)  # Apply second linear layer
#         return F.normalize(x, dim=-1)

# # Define the projection head for the token representation
# class TokenProjectionHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(TokenProjectionHead, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  # First linear layer
#         self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second linear layer
#         self.activation = nn.ReLU()  # Non-linear activation function

#     def forward(self, x):
#         x = self.activation(self.fc1(x))  # Apply first linear layer and activation
#         x = self.fc2(x)  # Apply second linear layer
#         return F.normalize(x, dim=-1) 

seed = 42  # You can use any integer
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

with open("protein_representations.json", "r") as json_file:
    protein_representations = json.load(json_file)

protein_ids = list(protein_representations.keys())
random.shuffle(protein_ids)

train_split = int(0.8 * len(protein_ids))
val_split = int(0.9 * len(protein_ids))

train_ids = protein_ids[:train_split]
val_ids = protein_ids[train_split:val_split]
test_ids = protein_ids[val_split:]

def create_representation_lists(protein_ids, protein_representations):
    word_representation_list = []
    graph_feature_list = []
    for protein_id in protein_ids:
        word_representation = torch.tensor(protein_representations[protein_id]["word_representation"]).to(device)
        graph_feature = torch.tensor(protein_representations[protein_id]["graph_feature"]).to(device)
        word_representation_list.append(word_representation)
        graph_feature_list.append(graph_feature)
    return word_representation_list, graph_feature_list

train_word_list, train_graph_list = create_representation_lists(train_ids, protein_representations)
val_word_list, val_graph_list = create_representation_lists(val_ids, protein_representations)
test_word_list, test_graph_list = create_representation_lists(test_ids, protein_representations)
print(f"length of train: {len(train_word_list)}, length of val: {len(val_word_list)}, length of test: {len(test_word_list)}")

# Initialize the projection heads
protein_proj_head = ProteinProjectionHead(input_dim=3072, output_dim=2304).to(device)
token_proj_head = TokenProjectionHead(input_dim=2304, output_dim=2304).to(device) # projection head shape torch.Size([1, 256])

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

            # Contrastive loss for negative pairs (i != j)
            available_negatives = [j for j in range(len(projected_tokens)) if j != i]
            negative_sims = []
            for j in available_negatives:
                negative_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[j], dim=-1)
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
        
        # Return the average loss over all pairs
        return total_loss / len(projected_proteins)
        
# version 3 InfoNCE loss 有进展的
    # def forward(self, projected_proteins, projected_tokens):
    #     assert len(projected_proteins) == len(projected_tokens), "Mismatch in number of protein and token representations"

    #     total_loss = 0.0
    #     for i in range(len(projected_proteins)):
    #         # Cosine similarity for the positive pair (i == j)
    #         positive_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[i], dim=-1)
            
    #         # Contrastive loss for negative pairs (i != j)
    #         available_negatives = [j for j in range(len(projected_tokens)) if j != i]
    #         negative_sims = []
    #         for j in available_negatives:
    #             negative_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[j], dim=-1)
    #             negative_sims.append(negative_sim)

    #         negative_sims = torch.stack(negative_sims)
    #         all_sims = torch.cat([positive_sim.unsqueeze(0), negative_sims], dim=0)
    #         all_sims = all_sims / self.temperature

    #         log_sum_exp_sims = torch.logsumexp(all_sims, dim=0)

    #         # Calculate the loss
    #         loss = -positive_sim / self.temperature + log_sum_exp_sims
    #         total_loss += loss

    #     # Return the average loss over all samples
    #     return total_loss / len(projected_proteins)
            


# Version 2 from original paper
    # def forward(self, projected_proteins, projected_tokens):
    #     assert len(projected_proteins) == len(projected_tokens), "Mismatch in number of protein and token representations"

    #     total_loss = 0.0
    #     for i in range(len(projected_proteins)):
    #         # Cosine similarity for the positive pair (i == j)
    #         positive_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[i], dim=-1)
    #         # print(f"positive_loss: {positive_loss}")
            
    #         # Contrastive loss for negative pairs (i != j)
    #         available_negatives = [j for j in range(len(projected_tokens)) if j != i]
    #         negative_indices = random.sample(available_negatives, min(self.num_negatives, len(available_negatives)))
    #         # print(len(negative_indices))
    #         negative_sims = torch.tensor(
    #             [F.cosine_similarity(projected_proteins[i], projected_tokens[j], dim=-1) for j in negative_indices]
    #         )

    #         neg_exp_sum = torch.logsumexp(negative_sims / self.temperature, dim=0)

    #         loss = -positive_sim / self.temperature + neg_exp_sum
    #         total_loss += loss

    #     total_loss = total_loss / len(projected_proteins)
    #     return total_loss

# version 1 from mengnan du
    # def forward(self, projected_proteins, projected_tokens):
    #     assert len(projected_proteins) == len(projected_tokens), "Mismatch in number of protein and token representations"

    #     total_loss = 0.0
    #     for i in range(len(projected_proteins)):
    #         # Cosine similarity for the positive pair (i == j)
    #         positive_sim = F.cosine_similarity(projected_proteins[i], projected_tokens[i], dim=-1)
    #         positive_loss = (1 - positive_sim) ** 2
    #         # print(f"positive_loss: {positive_loss}")
            
    #         # Contrastive loss for negative pairs (i != j)
    #         available_negatives = [j for j in range(len(projected_tokens)) if j != i]
    #         negative_indices = random.sample(available_negatives, min(self.num_negatives, len(available_negatives)))
    #         negative_sims = torch.tensor(
    #             [F.cosine_similarity(projected_proteins[i], projected_tokens[j], dim=-1) for j in negative_indices]
    #         )
            
    #         negative_loss = (negative_sims ** 2).mean()
    #         negative_loss = self.lambdaa * negative_loss
    #         # print(f"negative_loss: {negative_loss}")

    #         loss = positive_loss + negative_loss
    #         total_loss += loss
            
    #     total_loss = total_loss / len(projected_proteins)
    #     # print(f"total_loss: {total_loss}")
    #     return total_loss


contrastive_loss_fn = CosineContrastiveLoss(temperature=0.2, num_negatives=64, lambdaa=10, margin=0.3).to(device)

optimizer = torch.optim.Adam(list(protein_proj_head.parameters()) + list(token_proj_head.parameters()), lr=1e-3)
num_epochs = 15
batch_size = 32
best_val_loss = float('inf')

# Training loop without batch
# for epoch in range(num_epochs):
#     protein_proj_head.train()
#     token_proj_head.train()
#     optimizer.zero_grad()

#     projected_proteins = [protein_proj_head(graph_feature) for graph_feature in train_graph_list]
#     projected_tokens = [token_proj_head(word_representation.unsqueeze(0)) for word_representation in train_word_list]

#     train_loss = contrastive_loss_fn(projected_proteins, projected_tokens)
#     train_loss.backward()
#     optimizer.step()

#     protein_proj_head.eval()
#     token_proj_head.eval()

#     with torch.no_grad():
#         projected_proteins_val = [protein_proj_head(graph_feature) for graph_feature in val_graph_list]
#         projected_tokens_val = [token_proj_head(word_representation.unsqueeze(0)) for word_representation in val_word_list]
        
#         val_loss = contrastive_loss_fn(projected_proteins_val, projected_tokens_val)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")


# Training loop with batch
for epoch in range(num_epochs):
    protein_proj_head.train()
    token_proj_head.train()

    for i in range(0, len(train_word_list), batch_size):
        batch_word_list = train_word_list[i:i+batch_size]
        batch_graph_list = train_graph_list[i:i+batch_size]

        projected_proteins = [protein_proj_head(graph_feature) for graph_feature in batch_graph_list]
        projected_tokens = [token_proj_head(word_representation.unsqueeze(0)) for word_representation in batch_word_list]

        train_loss = contrastive_loss_fn(projected_proteins, projected_tokens)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


    protein_proj_head.eval()
    token_proj_head.eval()

    with torch.no_grad():
        projected_proteins_val = [protein_proj_head(graph_feature) for graph_feature in val_graph_list]
        projected_tokens_val = [token_proj_head(word_representation.unsqueeze(0)) for word_representation in val_word_list]
        
        val_loss = contrastive_loss_fn(projected_proteins_val, projected_tokens_val)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    if val_loss.item() < best_val_loss:
        torch.save(protein_proj_head.state_dict(), 'best_protein_proj_head.pth')
        torch.save(token_proj_head.state_dict(), 'best_token_proj_head.pth')
        best_val_loss = val_loss.item()
        print("best projection head saved")


# testing
protein_proj_head = ProteinProjectionHead(input_dim=3072, output_dim=2304).to(device)
token_proj_head = TokenProjectionHead(input_dim=2304, output_dim=2304).to(device) 

protein_proj_head.load_state_dict(torch.load('best_protein_proj_head.pth'))
token_proj_head.load_state_dict(torch.load('best_token_proj_head.pth'))

protein_proj_head.eval()
token_proj_head.eval()

with torch.no_grad():
    projected_proteins_test = [protein_proj_head(graph_feature) for graph_feature in test_graph_list]
    projected_tokens_test = [token_proj_head(word_representation.unsqueeze(0)) for word_representation in test_word_list]

for protein_rep, token_rep in zip(projected_proteins_test, projected_tokens_test):
    cosine_sim = F.cosine_similarity(protein_rep, token_rep, dim=-1)
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
    projected_tokens = [token_proj_head(token.unsqueeze(0)) for token in test_word_list]

for protein_rep, token_rep in zip(projected_proteins, projected_tokens):
    cosine_sim = F.cosine_similarity(protein_rep, token_rep, dim=-1)
    print(cosine_sim)

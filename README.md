# LLM-GNN_alignment

This repo is about we want to find out if we can align the representation of a protein text description from LLMs and representation of a protein graph from GNNs, if the text description and graph are about the same protein. Therefore, in our experiments we trained two linear layer of projection heads, one for the GNNs and one for the LLMs. The 

Below code suppose to use one linear layer to convert the protein representation dimension (1, 3072) to the same as the word representation dimension (2304,), which becomes (1, 2304) after unsqueeze(0). 
The code will also train the linear layer with contrastive loss function, so that the cosine similarity of positive protein and word pair will maximize to 1, also the cosine similarity of negative protein and word pair will be close to 0 (not to -1 since I want negative pairs have no relation at all. ). 

## Research Question 1:


1. Use `rcsb_api.py` to download all protein pdb and fasta files from rcsb.
2. Use `summarize.py` to use GPT-4 to summarize each protein fasta information into text format. So we can get the protein representation from LLM later
3. Then we need to get the protein representation from two kinds of models:
   a. Text based model (LLMs):
     -- Feed the previous protein fasta summary into LLM and get the representation from the last layer
   b. GNN based model:
     -- Each GNN models requires different input format. But the main idea is that, we either directly feed the protein pdb files into the model or we first convert the pdb files into some sort of graph information data, and then feed into the GNN. Then we will get the representation of the protein from the model.
4. After we got the representation from both text based model and GNN. we need to train projection head to align the text representation with graph representation. The code is in `projection_head` folder.
5. AFter we trained the projection head, we need to use metric to calculate how well is our projection head. The metric is in `metric` folder.
6. After we got the performance, we can also calculate the pearson correlation of accross different projection head. The code is in `pearson_correlation` folder.

## Analyze the influence of different dimension size on the alignment.
1. We choose Gearnet and Gemma2
2. Originally Gearnet has dimension of `3072` since this GNN contains six 512 hidden layers.
3. Therefore, we need to train the Gearnet from the start. We choose the the retrain hidden layer size of `[512]`, `[512, 512]`, `[512, 512, 512]`, `[512, 512, 512, 512]`, `[512, 512, 512, 512, 512]`. These retrianed model will give us dimension of 512, 1024, 1536, 2048, 2560.
4. Then we use the same steps above to train the projection head of Gearnet and Gemma2. Then use `metric` to calculate how well does the projection head does.

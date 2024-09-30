# LLM-GNN_alignment

This repo is about we want to find out if we can align the representation of a protein text description from LLMs and representation of a protein graph from GNNs, if the text description and graph are about the same protein. Therefore, in our experiments we trained two linear layer of projection heads, one for the GNNs and one for the LLMs. Both of the projection heads will have the exact same structure, with one simple linear layer. The purpose of the projection layer is to align the GNNs representaion to the same dimension of LLMs representation. Then the process of training wil maximize the cosine similarity of two representations of the same protein from different modality to 1. At the same time minimize the cosine similarity between two representation that from different proteins to 0 (not to -1 since -1 means negative relationship between two representations).

## Requirement:


## Preprocess:

1. Use `rcsb_api.py` to download all protein pdb and fasta files from rcsb.
2. Use `summarize.py` to use GPT-4 to summarize each protein fasta information into text format. So we can get the protein representation from LLM later

## Representation:
1. From `Preprocess`, we have protein's pdb files and text descriptions. Then we will use pdb file for GNN, and text description for LLM.
2. For text representation from LLM:
   -- Feed the previous protein text descriptions into LLM and get the representation from the last layer.
3. For graph representation from GNN:
   -- Each GNN models requires different input format. But the main idea is that, we either directly feed the protein pdb files into the model or we first convert the pdb files into some sort of graph information data, and then feed into the GNN. Then we will get the representation of the protein from the model.
4. Both modality representation code is in `representation` folder. 

## Research Question 1:
What kind of model pairs (a LLM and a GNN), have better alignment:
1. After we got the representation from both LLM and GNN. we need to train projection head to align the graph representation with text representation. The code is in `projection_head` folder.
2. AFter we trained the projection head, we need to use metric to calculate how well is our projection head. The metric is in `metric` folder. The higher the score, means better the alignments

## Research Question 2:
We want to analyze the influence of different dimensional size of a fixed model pairs on the alignment.
1. We choose Gearnet and Gemma2
2. Originally Gearnet has dimension of `3072` since this GNN contains six 512 hidden layers.
3. Therefore, we need to train the Gearnet from the start. We choose the the retrain hidden layer size of `[512]`, `[512, 512]`, `[512, 512, 512]`, `[512, 512, 512, 512]`, `[512, 512, 512, 512, 512]`. These retrianed model will give us dimension of 512, 1024, 1536, 2048, 2560.
4. Then we use the same steps above to train the projection head of Gearnet and Gemma2. Then use `metric` to calculate how well does the projection head does.


## Research Question 3:

## Research Question 4:

1. After we got the performance, we can also calculate the pearson correlation of accross different projection head. The code is in `pearson_correlation` folder.



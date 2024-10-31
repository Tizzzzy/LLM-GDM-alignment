# LLM-GNN_alignment

This repo is about we want to find out if we can align the representation of a protein text description from LLMs and representation of a protein graph from GNNs, if the text description and graph are about the same protein. Therefore, in our experiments we trained two linear layer of projection heads, one for the GNNs and one for the LLMs. Both of the projection heads will have the exact same structure, with one simple linear layer. The purpose of the projection layer is to align the GNNs representaion to the same dimension of LLMs representation. Then the process of training wil maximize the cosine similarity of two representations of the same protein from different modality to 1. At the same time minimize the cosine similarity between two representation that from different proteins to 0 (not to -1 since -1 means negative relationship between two representations).

# Requesting model access from META
## 1. Requesting model access from Google
visit this [link](https://ai.google.dev/gemma) and request the access to the Gemma-7B model. 

## 2. Requesting model access from Hugging Face
Once request is approved, use the same email adrress to get the access of the model from HF [here](https://huggingface.co/google/gemma-7b).

Once both requests are approved, follow the below directions.

# Setup

## 1. Environment preparation
```python
git clone https://github.com/Tizzzzy/LLM-DGM-alignment.git

cd LLM-DGM-alignment

pip install git+https://github.com/huggingface/transformers

# python 3.10 or higher recommended
pip install -r requirements.txt

huggingface-cli login
```

## 2. Authorising HF token
Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

Run below code and enter your token. It will authenticate your HF account
```python
>>> huggingface-cli login

or

>>> from huggingface_hub import login
>>> login(YOUR_HF_TOKEN)
```

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
1. After we got the each model's pair alignment performance, we can also calculate the pearson correlation of accross different model pairs. The code is in `pearson_correlation` folder.
   -- First you have to use previously trained projection heads to calculate the each proteins alignment score using `rank_similarity_(model_pair).py`.
   -- After you get the alignment score for each protein, we can run the correlation using `correlation.py`

## Research Question 3:
1. One of the main reason why we are doing the alignment experiment is to analyze what kind of protein will have high alignment score between the model pair. Is there any feature in a protein will affect the alignment score?
2. We conduct experiments in three angle:
   -- Amino acids sequence length:
      -- Use the `sequence_length_check.ipynb` to analyze whether sequence length will affect the alignment score. Here we show the answer is NO.
   -- Protein's rareness:
      -- Use the `rareness_check.ipynb` to analyze whether protein's rareness will affect the alignment score. Here the answer is YES.
   -- Number of Chains:
      -- Use the `count_chains.ipynb` to analyze whether protein's number of chains will affect the alignment score. Here the answer is NO.

## Research Question 4:
We want to analyze the influence of different dimensional size of a fixed model pairs on the alignment.
1. We choose Gearnet and Gemma2
2. Originally Gearnet has dimension of `3072` since this GNN contains six 512 hidden layers.
3. Therefore, we need to train the Gearnet from the start. We choose the the retrain hidden layer size of `[64]`, `[128]`, `[256]`, `[512]`, `[512, 512]`. For a fair comparison, we also retrained the original hidden layer size `[512, 512, 512, 512, 512, 512]` using the same training data. These retrianed model will give us dimension of 64, 128, 256, 512, 1024, 3072. The training code is in `train_gearnet.py`.
4. Then we use the same steps above to train the projection head of Gearnet and Gemma2. Then use `metric` to calculate how well does the projection head does.

## Research Question 5:
In the previous experiment, for each projection head, we only used one linear layer. Therefore, we are wondering does multiple linear layers improve the alignment performance.
We conduct 2 layers and 3 layers, the code is in `projection_head` folder, and you need to run `multi_layer` version.

## Research Question 6:
Previously, the LLMs are directly load from huggingface. It is raw version without any finetuning. Therefore, we are wonding whether a protein version of LLMs will help the alignment score.
1. We choose `llama3.1-8B` to finetune. The training data is the same as the data `summarize.py` output. The finetune code is in `finetune_llama31.py`
2. After finetuned, you need to use the previous step to get representation of each protein. Then train the projection head, then use metric to check the performance.


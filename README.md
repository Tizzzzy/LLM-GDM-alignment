# LLM-GNN_alignment

This repo is about we want to find out if we can align the representation of a protein text description from LLMs and representation of a protein graph from GNNs, if the text description and graph are about the same protein. Therefore, in our experiments we trained two linear layer of projection heads, one for the GNNs and one for the LLMs. Both of the projection heads will have the exact same structure, with one simple linear layer. The purpose of the projection layer is to align the GNNs representaion to the same dimension of LLMs representation. Then the process of training wil maximize the cosine similarity of two representations of the same protein from different modality to 1. At the same time minimize the cosine similarity between two representation that from different proteins to 0 (not to -1 since -1 means negative relationship between two representations).

# Setup

## Requesting model access from Google / META
Visit this [link](https://blog.google/technology/developers/google-gemma-2/) and request the access to the Gemma2 model. 
Visit this [link](https://ai.meta.com/blog/meta-llama-3-1/) and request the access to the LLaMa3.1 model. 

## Requesting model access from Hugging Face
Once request is approved, use the same email adrress to get the access of the model from HF [Gemma2 2B](https://huggingface.co/google/gemma-2-2b), [LLaMa3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [LLaMa3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B).

Once both requests are approved, follow the below directions.

## Environment preparation
```python
git clone https://github.com/Tizzzzy/LLM-DGM-alignment.git

cd LLM-DGM-alignment

# create a conda environment with python 3.10 or higher recommended
conda create -n LLM-DGM-alignment python == 3.10.12

# based on your CUDA version, install pytorch from https://pytorch.org/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Authorising HF token
Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

Run below code and enter your token. It will authenticate your HF account
```python
>>> huggingface-cli login

or

>>> from huggingface_hub import login
>>> login(YOUR_HF_TOKEN)
```

## Preprocess:

1. **Download Protein Files:**
   - Use `rcsb_api.py` to download 20,000 (or more) protein PDB and FASTA files from RCSB.
   - The downloaded files will be saved in `content/protein_files`.

2. **Summarize Protein Information:**
   - Use `summarize.py` to generate text descriptions for each protein based on its FASTA file using regular expressions and GPT-4.
   - This will create a JSON file named `protein_summaries.json` containing summarized text descriptions.

---

## Representation Extraction:

### Overview
From the preprocessing step, we have protein PDB files and corresponding text descriptions. These will be used as follows:
- **DGM**: Feed PDB files to obtain graph-based representations.
- **LLM**: Feed Text descriptions to obtain text-based representations.

### 1. Text Representation using LLMs
- **Input**: `protein_summaries.json` containing text descriptions.
- **Process**: Feed these descriptions into an LLM and extract representations from the last layer.
- **Script**: Use the scripts in `LLM-DGM-alignment/representation` and run `representation_{LLM}.py`.

### 2. Graph Representation using DGMs
- Different DGM models require specific input formats. Below are the instructions for each model:

#### GearNet
- **Support**: Directly accepts PDB files.
- **Setup**: Create a conda environment and download the pre-trained `angle_gearnet_edge` model from the [GearNet GitHub repository](https://github.com/DeepGraphLearning/GearNet).
- **Script**: Run `representation_gearnet.py` to get graph representations.
- **Output**: `protein_representations_gearnet.json` with all the graph representations from GearNet.

#### ScanNet
- **Support**: Directly accepts PDB files.
- **Setup**: Clone and install dependencies from the [ScanNet GitHub repository](https://github.com/jertubiana/ScanNet). Replace `predict_features.py` in their repo with the version provided in `representation/ScanNet/predict_features.py`.
- **Script**: Run `predict_features.py` to generate representations.
- **Output**: `protein_representations_scannet.json` with all the graph representations from ScanNet.

#### GVP
- **Support**: Does not accept PDB files directly.
- **Setup**: Install dependencies from the [GVP GitHub repository](https://github.com/drorlab/gvp). The modified code is provided in `representation/GVP/`.
- **Preprocessing**: Use `pdb_to_json.py` to convert PDB files into JSON format.
  - **Output**: `pdb_json.json`.
- **Script**: Run `models.py` in `representation/GVP/src/` to obtain graph representations.
- **Output**: `representation_gvp.json` with all the graph representations from GVP.

#### GAT
- **Support**: Does not accept PDB files directly.
- **Setup**: Install dependencies from the [GAT GitHub repository](https://github.com/PetarV-/GAT/tree/master). The modified code is provided in `representation/GAT/`.
- **Script**: Run `representation_gat.py`. This script will automatically convert PDB files into node features and graph structures for the GAT model.
- **Output**: `protein_representations_gat.json` with all the graph representations from GAT.

## Representation Alignment:

### Overview
Once we have obtained the text and graph representations for proteins from the LLMs and DGMs, we can proceed to train the projection head. The purpose of the projection head is to map these representations, which may have different dimensions, into a unified representation space. For instance, if a graph representation has a dimension of [1, 3072] and a text representation has a dimension of [1, 2304], the projection head will align both into the same space, such as [1, 2304].

### Structure
All related code is organized in the `projection_head` folder, which contains subfolders named after each DGM. Each of these subfolders includes four scripts, each designed for a specific model pairing (an LLM and a DGM). The goal is to train two projection heads for each model pair: one for the LLM and another for the DGM. Initially, we use a single linear layer for the projection head design, but we also experiment with adding one or two additional linear layers to the DGM projection head.

### Code Details
- `projection_head_{GDM}_gemma2.py:` Trains projection heads for a model pair that includes Gemma2 2B as the LLM.
- `projection_head_{GDM}_llama31_8B.py:` Trains projection heads for a model pair with LLaMa3.1 8B as the LLM.
- `projection_head_{GDM}_llama31_70B.py:` Trains projection heads for a model pair with LLaMa3.1 70B as the LLM.
- `projection_head_{GDM}_gemma2_multilayer.py:` Trains projection heads for a model pair with Gemma2 2B as the LLM, incorporating one or two additional linear layers in the DGM projection head.

### Output
Running these scripts will generate a saved projection head model at `best_protein_proj_head_{DGM}_{LLM}.pth`. These projection heads are crucial for evaluating the alignment quality between the DGM and LLM representations.


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


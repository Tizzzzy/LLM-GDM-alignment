# LLM-GDM-alignment

This includes an original implementation of "[Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation](https://arxiv.org/pdf/2411.05316)" by [Dong Shu](https://scholar.google.com/citations?user=KfIlTroAAAAJ&hl=en), [Bingbing Duan](https://www.biology.pitt.edu/person/bingbing-duan), [Kai Guo](https://scholar.google.com/citations?user=v6jYru8AAAAJ), [Kaixiong Zhou](https://kaixiong-zhou.github.io/), [Jiliang Tang](https://www.cse.msu.edu/~tangjili/), [Mengnan Du](https://mengnandu.com/).

[[Paper](https://arxiv.org/pdf/2411.05316)] | [[Website](https://tizzzzy.github.io/LLM-GDM-alignment.github.io/)]

## Abstract

Latent representation alignment has become a foundational technique for constructing multimodal large language models (MLLM) by mapping embeddings from different modalities into a shared space, often aligned with the embedding space of large language models (LLMs) to enable effective cross-modal understanding. While preliminary protein-focused MLLMs have emerged, they have predominantly relied on heuristic approaches, lacking a fundamental understanding of optimal alignment practices across representations. In this study, we explore the alignment of multimodal representations between LLMs and Geometric Deep Models (GDMs) in the protein domain. We comprehensively evaluate three state-of-the-art LLMs (Gemma2-2B, LLaMa3.1-8B, and LLaMa3.1-70B) with four protein-specialized GDMs (GearNet, GVP, ScanNet, GAT). Our work examines alignment factors from both model and protein perspectives, identifying challenges in current alignment methodologies and proposing strategies to improve the alignment process. Our key findings reveal that GDMs incorporating both graph and 3D structural information align better with LLMs, larger LLMs demonstrate improved alignment capabilities, and protein rarity significantly impacts alignment performance. We also find that increasing GDM embedding dimensions, using two-layer projection heads, and fine-tuning LLMs on protein-specific data substantially enhance alignment quality. These strategies offer potential enhancements to the performance of protein-related multimodal models.

## Our Findings
1. We found that GDMs integrating both graph and 3D structural information of proteins tend to align better with LLMs. 
2. Larger LLMs with higher embedding dimensions showed improved alignment performance with the same GDM.
3. Our analysis revealed strong correlations between high-alignment model pairs and other high-alignment pairs.
4. Notably, the rarity of a protein significantly affects the model's alignment performance, with rare proteins posing greater challenges.
5. We highlight the challenges and limitations of current protein datasets for representation alignment, particularly due to unequal levels of study across proteins and the presence of homologous relationships among them.
6. In terms of model architecture, we discovered that retraining GDMs to have higher embedding dimensions enhances alignment with LLMs.
7. The complexity of the projection head also plays a crucial role, as increasing the number of layers improves alignment up to a certain threshold, beyond which benefits diminish.
8. Finally, we found that fine-tuning LLMs with protein-specific data can significantly improve alignment with GDMs.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper ✨:
```
@article{shu2024exploring,
  title={Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation},
  author={Shu, Dong and Duan, Bingbing and Guo, Kai and Zhou, Kaixiong and Tang, Jiliang and Du, Mengnan},
  journal={arXiv preprint arXiv:2411.05316},
  year={2024}
}
```

![image](https://github.com/user-attachments/assets/f785c1ea-e7ba-4d65-9cd8-77b135766ebe)

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
Running these scripts will generate a saved projection head model for GDM at `best_protein_proj_head_{DGM}_{LLM}.pth`, and a saved projection head model for LLM at `best_token_proj_head_{DGM}_{LLM}.pth`. These projection heads are crucial for evaluating the alignment quality between the DGM and LLM representations.


## Research Question 1:
Which LLM-GDM model pairs demonstrate the best alignment performance?

To evaluate the alignment performance of the LLM-DGM model pairs after training the projection heads, we use a dedicated metric. The code for this metric is located in the `metric` folder. A higher alignment score indicates better performance between a given model pair.

### Structure
The `metric` folder is organized similarly to the `projection_head` folder, with four subfolders named after each DGM. Each subfolder contains four scripts, each tailored to evaluate a specific model pairing (an LLM and a DGM).

### Evaluation
- Each script uses the projection heads trained in the `Representation Alignment` section.
- The output is a score ranging from [-1, 1], where a higher score indicates better alignment between the representations of the model pair.

## Research Question 2:
Is there a correlation between different model pairs?

To determine whether there is a relationship between the alignment performances of different model pairs, we calculate the Pearson correlation. The code for this analysis is located in the `pearson_correlation` folder. A higher correlation score indicates that the alignment performances of two model pairs are more similar.

### Steps to Calculate Correlation
1. Compute Alignment Scores:
   - Use the previously trained projection heads to calculate alignment scores for each protein’s graph and text representation.
   - Run `rank_similarity_(model_pair).py` for each model pair to generate these scores.
2. Calculate Correlation:
   - Once you have the alignment scores for all proteins, use `correlation.py` to compute the Pearson correlation between different model pairs.

## Research Question 3:
What types of proteins align well across all model pairs, and which do not?

A key objective of our alignment experiments is to identify the characteristics of proteins that result in high or low alignment scores across different model pairs. We explore whether specific protein features influence alignment performance.

### Experimental Analysis
We conduct our investigation from three perspectives:
1. Amino Acid Sequence Length:
   - Use `sequence_length_check.ipynb` to analyze the impact of sequence length on alignment scores.
   - Finding: Sequence length does not significantly affect alignment scores.

2. Protein Rarity:
   - Use `rareness_check.ipynb` to explore whether the rarity of a protein influences alignment performance.
   - Finding: Rarity has a notable impact on alignment scores, with rarer proteins generally aligning less well.

3. Number of Chains:
   - Use `count_chains.ipynb` to examine the effect of the number of protein chains on alignment scores.
   - Finding: The number of chains does not significantly affect alignment scores.

## Research Question 4:
Does increasing the GDM dimension improve alignment performance?

To investigate whether higher-dimensional graph representations improve alignment performance, we use the GearNet model as our GDM for evaluation.

### Experimental Setup
GearNet’s original architecture has a dimensionality of `3072`, achieved through six hidden layers of size `512`. For our study, we retrain GearNet from scratch with varying hidden layer sizes to explore different output dimensions:

- Hidden Layer Configurations: [64], [128], [256], [512], [512, 512], and [512, 512, 512, 512, 512, 512] (the original configuration).
- These configurations yield output dimensions of 64, 128, 256, 512, 1024, and 3072, respectively. We stop at [512, 512, 512, 512, 512, 512] as it matches GearNet’s original setup.

### Implementation
- The training code is in `train_gearnet.py`. By default, `hidden_dims` is set to [512, 512]. To modify the hidden layer sizes, adjust the `hidden_dims=[]` parameter as needed.
- After retraining GearNet with the new dimensions, please follow the same steps to generate protein graph representations, train the projection heads for the retrained GearNet and Gemma2, and then use the `metric` to evaluate alignment performance.

## Research Question 5:
Does adding layers to the GDM’s projection head enhance alignment performance?

Previously, we used a single linear layer for the projection head. This raises the question: does using multiple linear layers improve alignment performance between model pairs?

### Experiment Setup
We extend the projection head to include two-layer and three-layer configurations. The code for these multi-layer projection heads can be found in the `projection_head` folder. To run these experiments, use the `multilayer` versions of the scripts. By default, the LLM model used is Gemma2 2B, but you can easily switch to other LLM models as desired.

### Evaluation
After training the multi-layer projection heads, follow the same evaluation steps as before to measure alignment performance. This experiment helps us understand whether a more complex projection head architecture leads to better alignment results.


## Research Question 6:
Does fine-tuning an LLM on protein data enhance alignment performance?

Initially, the LLMs used in our experiments were loaded from Hugging Face in their raw, pre-trained versions without any fine-tuning. We hypothesize that fine-tuning these LLMs on protein-specific domain knowledge could improve alignment scores.

### Fine-Tuning Setup
We provide training code for fine-tuning `LLaMa3.1 8B` in the `finetune_llama31.py` script. The training data consists of the text descriptions from `protein_summaries.json`, generated using `summarize.py`. If you wish to fine-tune other LLMs, you can do so with simple modifications to the provided script.

### Evaluation
After fine-tuning, follow these steps:
1. Obtain protein text representations using the fine-tuned LLM.
2. Train the projection heads as before.
3. Use the metric to evaluate the alignment performance and analyze the impact of fine-tuning.

# Suggestions for Developing Multimodal LLMs
Our findings from `Research Questions 4-6` provide important insights for the design of future multimodal LLMs in the protein domain. We discovered that GDMs which incorporate both graph and 3D structural information of proteins demonstrate superior alignment with LLMs, suggesting future designs should prioritize models capable of capturing these multidimensional protein representations. The complex relationships between different proteins must be carefully considered during the alignment process to avoid oversimplified mappings that fail to capture subtle protein interactions. Our research also revealed that higher-dimensional embeddings in LLMs contribute to better alignment by capturing richer semantic information, indicating that increasing the capacity of the LLM's embedding space can enhance overall performance. When designing projection heads, we found that a two-layer architecture offers the optimal balance between simplicity and performance, as additional layers provide diminishing returns. Furthermore, fine-tuning LLMs with domain-specific data, such as protein descriptions, significantly improves alignment with GDMs, highlighting the importance of customizing LLMs to better understand the intricacies of the protein domain for more effective cross-modal integration.

### 📞 Contact

If you have any question or suggestion related to this project, feel free to open an issue or pull request.

### ✨ Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```
@article{shu2024exploring,
  title={Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation},
  author={Shu, Dong and Duan, Bingbing and Guo, Kai and Zhou, Kaixiong and Tang, Jiliang and Du, Mengnan},
  journal={arXiv preprint arXiv:2411.05316},
  year={2024}
}
```

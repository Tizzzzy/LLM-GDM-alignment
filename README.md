# LLM-GNN_alignment

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

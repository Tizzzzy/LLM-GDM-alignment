import huggingface_hub
huggingface_hub.login("hf_")
import sys
import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B").to('cuda')

json_file_path = "protein_summaries.json"
output_json_path = "protein_representations_llama31.json/protein_representations_llama31.json"


with open(json_file_path, 'r') as json_file:
    protein_datas = json.load(json_file)

protein_representations = {}
count = 0

if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        protein_representations = json.load(json_file)

print(len(protein_representations))

for protein_id, summary in protein_datas.items():

    if protein_id in protein_representations:
        print(f"Skipping {protein_id}, already get representation.")
        continue

    word = summary
    layer_num = -1 # 最后一层

    model.eval()
    input_ids = tokenizer(word, return_tensors="pt").input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    word_representation = hidden_states[layer_num].squeeze()[-1].cpu().numpy().tolist()
    print(len(word_representation))

    protein_representations[protein_id] = {
        "word_representation": word_representation,
    }

    count += 1
    if count % 10 == 0:
        with open(output_json_path, "w") as out_json_file:
            json.dump(protein_representations, out_json_file, indent=4)
        print(f"Updated {output_json_path} with {count} representations.")
        torch.cuda.empty_cache()
        # break

# Final save to ensure all data is stored
with open(output_json_path, "w") as out_json_file:
    json.dump(protein_representations, out_json_file, indent=4)
    torch.cuda.empty_cache()



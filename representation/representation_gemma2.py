import huggingface_hub
huggingface_hub.login("hf_")

import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to('cuda')

import os
import sys
import argparse
import torch
import json

json_file_path = "../workspace_data/protein_summaries.json"
output_json_path = "../workspace_data/protein_representations_gemma2.json"
temp_path = "../workspace_data/protein_representations.json"

with open(json_file_path, 'r') as json_file:
    protein_datas = json.load(json_file)

protein_representations = {}
count = 0

if os.path.exists(temp_path):
    with open(temp_path, 'r') as json_file:
        protein_representations = json.load(json_file)

    protein_representations = {
        key: {"word_representation": value['word_representation']} 
        for key, value in protein_representations.items() 
        if 'word_representation' in value
    }
    first_item = list(protein_representations.items())[0]
    print(first_item)

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

    # Periodically save the JSON file to avoid data loss
    count += 1
    if count % 10 == 0:
        with open(output_json_path, "w") as out_json_file:
            json.dump(protein_representations, out_json_file, indent=4)
        print(f"Updated {output_json_path} with {count} representations.")
        torch.cuda.empty_cache()
        # break

    # if count >= 500:
    #     break

# Final save to ensure all data is stored
with open(output_json_path, "w") as out_json_file:
    json.dump(protein_representations, out_json_file, indent=4)
    print(f"Final updated representations.")
    torch.cuda.empty_cache()



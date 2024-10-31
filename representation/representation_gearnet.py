import huggingface_hub
import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import os
import sys
import argparse
import torch
# from torchdrug import core
sys.path.append(os.path.dirname(os.path.dirname("./GearNet")))
# sys.path.insert(0, './GearNet')
# from gearnet.model import GearNetIEConv
from torchdrug.data import Protein
# from torchdrug.core import Registry as R
from torchdrug import data, utils
from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import models
import json


graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to('cuda')
pthfile = '../workspace_data/angle_gearnet_edge.pth'
net = torch.load(pthfile, map_location=torch.device('cuda'))
gearnet_edge.load_state_dict(net)

json_file_path = "../workspace_data/protein_summaries.json"
output_json_path = "../workspace_data/protein_representations_gearnet.json"

with open(json_file_path, 'r') as json_file:
    protein_datas = json.load(json_file)

protein_representations = {}
count = 0

for protein_id, summary in protein_datas.items():

    if protein_id in protein_representations:
        print(f"Skipping {protein_id}, already get representation.")
        continue

  # 把pdb representation搞到手
    pdb_file = f"../content/protein_files/pdb/{protein_id}.pdb"
    try:
        protein = Protein.from_pdb(pdb_file, atom_feature="position", bond_feature="length", residue_feature="symbol")
        _protein = Protein.pack([protein])
        protein_ = graph_construction_model(_protein)
        protein_.view = 'residue'
        protein_ = protein_.to('cuda')

        with torch.no_grad():
            gearnet_edge.eval()
            output = gearnet_edge(protein_, protein_.node_feature.float(), all_loss=None, metric=None)
            graph_feature = output['graph_feature'].cpu().numpy().tolist()
            if len(set(graph_feature[0])) < 2:
                continue
            print(len(graph_feature[0]))
            # print(node_feature.shape)
    except Exception as e:
        print(f"Error processing PDB file for {protein_id}: {e}")
        continue

    protein_representations[protein_id] = {
        "graph_feature": graph_feature
        # "node_feature": node_feature
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

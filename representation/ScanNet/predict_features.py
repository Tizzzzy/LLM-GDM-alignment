import predict_bindingsites
from preprocessing import PDBio
from utilities.paths import model_folder,structures_folder,MSA_folder
import numpy as np
import json
import os
# import torch

def predict_features(list_queries,layer='SCAN_filter_activity_aa',
                     model='ScanNet_PPI_noMSA',
                     output_format='numpy',
                     model_folder=model_folder,
                     biounit=False,
                     permissive=False):
    '''
    Usages:
     list_dictionary_features = predict_features(list_queries,output_format='dictionary')
     list_features, list_residueids = predict_features(list_queries,output_format='numpy')
    Example: 
    list_queries = ['1a3x_A','2p6b_AB','1a3y']
    list_dictionary_features = list of residues-level features, each element of the form Nresidues X Nfeatures.

    '''
    if not isinstance(list_queries,list):
        list_queries = [list_queries]
        return_one = True
        permissive = False
    else:
        return_one = False
    query_pdbs = []
    query_chain_ids = []
    nlayers = len(layer) if isinstance(layer,list) else 1

    for query in list_queries:
        pdb,chain_ids = PDBio.parse_str(query)
        query_pdbs.append(pdb)
        query_chain_ids.append(chain_ids)

    

    if 'noMSA' in model:
        pipeline = predict_bindingsites.pipeline_noMSA
        use_MSA = False
    else:
        pipeline = predict_bindingsites.pipeline_MSA
        use_MSA = True

    query_outputs = predict_bindingsites.predict_interface_residues(
    query_pdbs=query_pdbs,
    query_chain_ids=query_chain_ids,
    pipeline=pipeline,
    model=model,
    model_folder=model_folder,
    structures_folder=structures_folder,
    MSA_folder=MSA_folder,
    biounit=biounit,
    assembly=True,
    layer=layer,
    use_MSA=use_MSA,
    overwrite_MSA=False,
    Lmin=1,
    output_chimera=False,
    permissive=permissive,
    output_format = output_format
    )
    if output_format == 'numpy':
        try:
            query_pdbs, query_names, query_features, query_residue_ids, query_sequences = query_outputs
            # print(query_features)
            
            # query_features = np.vstack(query_features[0][0])
            # query_features = query_features.reshape(1, query_features.shape[0], query_features.shape[1])
            pooled_representation = np.mean(query_features, axis=1)
            pooled_representation = np.mean(pooled_representation, axis=0).reshape(1, -1)
            # max_pooled = np.max(query_features, axis=0)
            
            # print(f"numpy, average_pooled: {average_pooled}")
            print(f"numpy, average_pooled shape: {pooled_representation.shape}")
            # print(f"numpy, max_pooled: {max_pooled}")
            # print(f"numpy, max_pooled shape: {max_pooled.shape}")
    
            return pooled_representation
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None




if __name__ == '__main__':
    model = 'ScanNet_PPI_noMSA' # Protein-protein binding site prediction model without evolutionary information.

    layer_choices = [
        'SCAN_filters_atom_aggregated_activity', # Atomic Neighborhood Embedding Module, *after* pooling. Atomic neighborhoods have radius of about 5 Angstrom.  Size: [Naa,64].
        'all_embedded_attributes_aa', # Embedded residue type or PWM (first 32 channels) + Atomic Neighborhood Embedding Module, *after* pooling (last 64 channels). Size: [Naa,96].
        'SCAN_filter_activity_aa', # Amino Acid Neighborhood Embedding Module. Amino acid neighborhoods have radius of about 11 Angstrom. Size: [Naa,128].
        'SCAN_filters_aa_embedded_1', # Non-linear, 32-dimensional projection of Amino Acid Neighborhood Embedding Module output. Input to the neighborhood attention module. Size: [Naa,32].
        None, # The binding site probabilities Size: ([Naa,])
    ]

    output_format = 'numpy' #'dictionary' # 'numpy'


    layer = layer_choices[2]
    # layer = [layer_choices[1],layer_choices[2],layer_choices[4]] # Multiple layers are supported.

    json_file_path = "../workspace_data/protein_summaries.json"
    output_json_path = "../workspace_data/protein_representations_scannet.json"

    with open(json_file_path, 'r') as json_file:
        protein_datas = json.load(json_file)

    protein_representations = {}
    count = 0

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as json_file:
            protein_representations = json.load(json_file)

    protein_items = list(protein_datas.items())

    protein_items = protein_items

    # for protein_id, summary in protein_datas.items():
    for protein_id, summary in protein_items:

        if protein_id in protein_representations:
            print(f"Skipping {protein_id}, already get representation.")
            continue
            
        input_path = f"../content/protein_files/pdb/{protein_id}.pdb"

        pooled_representation = predict_features([input_path],layer=layer,model=model,output_format=output_format,permissive=True)
        if pooled_representation is None:
            continue

        protein_representations[protein_id] = {
            "average_pooled": pooled_representation.tolist(),
            # "max_pooled": max_pooled.tolist()
        }
        
        count += 1
        if count % 10 == 0:
            # print(protein_representations)
            with open(output_json_path, "w") as out_json_file:
                # print("here")
                json.dump(protein_representations, out_json_file, indent=4)
            print(f"Updated {output_json_path} with {count} representations.")
            # torch.cuda.empty_cache()
            # break
            
    with open(output_json_path, "w") as out_json_file:
        json.dump(protein_representations, out_json_file, indent=4)
        # torch.cuda.empty_cache()

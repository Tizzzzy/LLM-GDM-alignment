from Bio.PDB import PDBParser
import numpy as np
import networkx as nx
import tensorflow as tf
from utils import process
from models import GAT
import os
import json
from scipy.spatial import KDTree
import signal
import gc

json_file_path = "protein_summaries.json"
output_json_path = "protein_representations_gat.json"
checkpoint_file = "pre_trained/cora/mod_cora.ckpt"

with open(json_file_path, 'r') as json_file:
    protein_datas = json.load(json_file)
    print(len(protein_datas))

protein_representations = {}
count = 0

if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        protein_representations = json.load(json_file)

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)


def pdb_to_matrix(pdb_file_path):
    # Initialize PDB parser
    # print('here')
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    # print('here2')
    # Parameters
    distance_threshold = 5.0  # Threshold to define an edge based on distance between atoms
    
    # Extract atoms and create a graph
    G = nx.Graph()
    atom_list = []
    atom_coords = []
    # print('here3')
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_id = len(atom_list)
                    atom_list.append(atom)
                    atom_coords.append(atom.coord)
                    G.add_node(atom_id, element=atom.element, coord=atom.coord)
    # print(f"G: {G}")
    # print('here4')
    # Create edges based on distance threshold
    # Use KD-Tree for efficient neighbor search
    kd_tree = KDTree(atom_coords)
    pairs = kd_tree.query_pairs(r=distance_threshold)
    
    # Create edges based on distance threshold
    for i, j in pairs:
        G.add_edge(i, j)
        
    # print('here5')
    # Generate the adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    print(f"protein_adj: {adj_matrix.shape}")
    # print('here6')
    # Generate the feature matrix (example: using atom type and coordinates)
    feature_matrix = []
    for atom in atom_list:
        element_feature = [ord(atom.element[0])]  # Simple numerical encoding of the element type
        coord_feature = atom.coord.tolist()
        feature_matrix.append(element_feature + coord_feature)
    # print('here7')
    feature_matrix = np.array(feature_matrix)
    
    # Output the shapes of the matrices
    # print("Adjacency Matrix Shape:", adj_matrix.shape)
    # print("Feature Matrix Shape:", feature_matrix.shape)
    return adj_matrix, feature_matrix


# for protein_id, summary in protein_datas.items():
for protein_id, summary in reversed(list(protein_datas.items())):

#  or protein_id == "4TZ5" or protein_id == "5IVH"
    if protein_id in protein_representations:
        print(f"Skipping {protein_id}, already get representation.")
        continue

    # Path to the PDB file
    pdb_file_path = f"../content2/content2/protein_files/pdb/{protein_id}.pdb"
    if not os.path.exists(pdb_file_path):
        continue
    # pdb_file_path = f'../content/protein_files/pdb/{protein_id}.pdb'
    print(f"protein_id: {protein_id}")

    try:
        signal.alarm(500)
        protein_adj, protein_features = pdb_to_matrix(pdb_file_path)

        if protein_adj.shape[0] > 20000:
            print(f"Skipping {protein_id}, protein too big")
            continue

        nb_nodes = protein_features.shape[0]
        padding_size = 1433 - protein_features.shape[1]
        protein_features = np.pad(protein_features, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
        ft_size = protein_features.shape[1]
                
        # Model hyperparameters
        hid_units = [8]  # Number of hidden units per attention head in each layer
        n_heads = [8]  # Number of attention heads, no output layer needed
        
        # Prepare input data
        protein_features = protein_features[np.newaxis]  # Shape: (1, nb_nodes, ft_size)
        protein_adj = protein_adj[np.newaxis]  # Shape: (1, nb_nodes, nb_nodes)
        biases = process.adj_to_bias(protein_adj, [nb_nodes], nhood=1)
        
        # TensorFlow graph
        try:
            with tf.Graph().as_default(), tf.Session() as sess:
                # Placeholders
                ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))
                bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
                attn_drop = tf.placeholder(dtype=tf.float32, shape=())
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        
                # Instantiate GAT model
                latent_representation = GAT.inference(ftr_in, nb_classes=None, nb_nodes=nb_nodes, training=False, 
                                      attn_drop=attn_drop, ffd_drop=ffd_drop, 
                                      bias_mat=bias_in, hid_units=hid_units, n_heads=n_heads)
        
                # Example input (use the features and biases as input)
                feed_dict = {
                    ftr_in: protein_features,
                    bias_in: biases,
                    attn_drop: 0.0,
                    ffd_drop: 0.0
                }
                
                # Initialize variables
                sess.run(tf.global_variables_initializer())
                
                # Load the pretrained model
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_file)
                
                # Get the latent representation
                latent_output = sess.run(latent_representation, feed_dict=feed_dict)
                
                # Apply average pooling to remove the seq_num dimension
                latent_output_avg_pooled = tf.reduce_mean(latent_output, axis=1)  # Shape: [1, 64]
        
                # Apply max pooling to remove the seq_num dimension
                # latent_output_max_pooled = tf.reduce_max(latent_output, axis=1)  # Shape: [1, 64]
    
                # pooled_output_max = sess.run(latent_output_max_pooled)
                pooled_output_avg = sess.run(latent_output_avg_pooled)
                
                # latent_output_squeezed = np.squeeze(latent_output, axis=0)
                # print("Latent Representation Max Shape:", pooled_output_max.shape)
                # print("Latent Representation Max:", pooled_output_max)
                print("Latent Representation Avg Shape:", pooled_output_avg.shape)
                # print("Latent Representation Avg:", pooled_output_avg)
    
                protein_representations[protein_id] = {
                    # "protein_representation_max": pooled_output_max.tolist(),
                    "protein_representation_avg": pooled_output_avg.tolist()
                }
    
                count += 1
                if count % 10 == 0:
                    with open(output_json_path, "w") as out_json_file:
                        json.dump(protein_representations, out_json_file, indent=4)
                    print(f"Updated {output_json_path} with {count} representations.")
    
                # break
                signal.alarm(0)
        finally:
            sess.close()
            del sess
            tf.reset_default_graph()
            gc.collect()
            signal.alarm(0)
            
    except TimeoutException:
        print(f"Timeout: Skipping protein_id {protein_id} after 5 minutes of processing.")
        # sess.close()
        # del sess
        # tf.reset_default_graph()
        # gc.collect()
        # signal.alarm(0)
        continue
    except Exception as e:
        print(f"Error processing PDB file for {protein_id}: {e}")
        # sess.close()
        # del sess
        # tf.reset_default_graph()
        # gc.collect()
        # signal.alarm(0)
        continue
    # finally:
    #     # Clean up memory after each protein processing
    #     sess.close()
    #     tf.reset_default_graph()
    #     gc.collect()
    #     signal.alarm(0)
                
# Final save to ensure all data is stored
with open(output_json_path, "w") as out_json_file:
    json.dump(protein_representations, out_json_file, indent=4)
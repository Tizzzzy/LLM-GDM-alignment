import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.datasets import load_dataset
print("here1")

import numpy as np
print("here2")

import tensorflow as tf
print("here3")

from tensorflow.keras import Model, Sequential
print("here4")

from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization
print("here5")

from src.gvp import GVP, GVPDropout, GVPLayerNorm, vs_concat
print("here6")

from src.models import Encoder, Decoder, StructuralFeatures
print("here7")

# Specify the model checkpoint directory
model_dir = '/models/cath_pretrained'
print("here8")

# Load the pretrained model
model = CPDModel()  # Modify this line based on how the model is loaded
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(model_dir)).expect_partial()
print("here9")

latent_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)

# Load your protein data in JSON format
# protein_data_path = '../workspace_data/pdb_json.json'
protein_data_path = '/data/ts50.json'
dataset = load_dataset(protein_data_path, batch_size=1, shuffle=False)
print("here10")

# Iterate through the dataset and predict the design
for structure, seq, mask in dataset:
    # n = 1  # Number of sequences to sample

    latent_representation = latent_model([structure, seq, mask])
    latent_representation = tf.cast(latent_representation, tf.float32).numpy()

    print(f"Latent representation: {latent_representation.shape}")
    break

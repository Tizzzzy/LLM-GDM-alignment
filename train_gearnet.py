from torchdrug import datasets
from torchdrug import transforms
from torchdrug import models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

# Replace EnzymeCommissionToy with the full EnzymeCommission dataset
dataset = datasets.EnzymeCommission("~/protein-datasets/", transform=transform, atom_feature=None, bond_feature=None)

# Split the dataset into train, validation, and test sets
train_set, valid_set, test_set = dataset.split()

print("Shape of function labels for a protein: ", dataset[0]["targets"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

_protein = data.Protein.pack([dataset[0]["graph"]])
protein_ = graph_construction_model(_protein)

# train
gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512],
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

from torchdrug import tasks
task = tasks.MultipleBinaryClassification(
    gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
    task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"]
)

from torchdrug import core
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, batch_size=4)
solver.train(num_epoch=10)
results = solver.evaluate("valid")
print(f"Validation results: {results}")

torch.save(task.model.state_dict(), "gearnet_model_weights_2_512.pth")
import torch_geometric
from torch_geometric.data import GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='tmp/cora', name='cora')
data = dataset[0]
dataloader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=dataset.processed_dir,
                                     num_workers=4)
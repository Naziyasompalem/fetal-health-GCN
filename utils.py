import torch
from torch_geometric.data import Data


def create_graph(X, y):
# Convert features and labels to tensors
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y.values, dtype=torch.long)


# Create a simple fully connected edge_index
num_nodes = x.size(0)
row = torch.arange(num_nodes).repeat(num_nodes)
col = torch.arange(num_nodes).unsqueeze(1).repeat(1, num_nodes).view(-1)
edge_index = torch.stack([row, col], dim=0)


data = Data(x=x, edge_index=edge_index, y=y)
return data

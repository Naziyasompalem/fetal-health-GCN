import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from preprocess import load_data, preprocess_data, split_data
from model import GCN
from utils import create_graph


# Load and preprocess dataset
file_path = 'fetal_health.csv' # replace with your dataset path
data = load_data(file_path)
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)


# Create PyG graph objects
train_graph = create_graph(X_train, y_train)
test_graph = create_graph(X_test, y_test)


# Model, optimizer, loss
model = GCN(input_dim=X.shape[1], hidden_dim=16, output_dim=len(y.unique()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(50):
model.train()
optimizer.zero_grad()
out = model(train_graph)
loss = F.cross_entropy(out, train_graph.y)
loss.backward()
optimizer.step()
print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


# Testing
model.eval()
with torch.no_grad():
pred = model(test_graph).argmax(dim=1)
acc = (pred == test_graph.y).sum().item() / test_graph.y.size(0)
print(f'Test Accuracy: {acc:.4f}')

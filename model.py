import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) model for graph classification.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels) # -1 for inferred input features
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1) # Binary classification output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # Aggregate node features to graph features

        # 3. Apply a final classifier
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return torch.sigmoid(x) # Sigmoid for binary classification

if __name__ == "__main__":
    # Example usage:
    # This part will only run when model.py is executed directly
    # In a real scenario, you would instantiate this model in your training script.
    
    # Create a dummy Data object for demonstration
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 1) # 3 nodes, 1 feature per node
    batch = torch.tensor([0, 0, 0]) # All nodes belong to the same graph
    dummy_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    model = GCN(hidden_channels=64)
    print("Model architecture:")
    print(model)
    
    # Test forward pass
    output = model(dummy_data)
    print("\nOutput shape (should be [1, 1] for binary classification):")
    print(output.shape)
    print("Output value (probability):")
    print(output.item())
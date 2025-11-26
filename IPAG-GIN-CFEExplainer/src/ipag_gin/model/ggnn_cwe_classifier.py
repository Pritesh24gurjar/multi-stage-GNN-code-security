import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool


class GGNN_CWE_Classifier(nn.Module):
    """
    Gated Graph Neural Network for multi-class CWE classification.
    
    Architecture:
        1. Node Feature Embedding (3 separate embeddings for node_type, original_type, label)
        2. Multiple GGNN layers with gated message passing
        3. Graph-level pooling (mean + max)
        4. MLP classifier head
    
    Based on:
        - Devign architecture (GGNN for vulnerability detection)
        - Your IPAG representation with multi-type edges
    """
    
    def __init__(
        self,
        vocab_sizes,
        num_classes,
        embedding_dim=128,
        hidden_dim=200,
        num_ggnn_layers=3,
        num_steps=8,
        dropout=0.3,
        use_edge_weights=True
    ):
        """
        Args:
            vocab_sizes (dict): Dictionary with keys 'node_type', 'original_type', 'label'
                               containing vocabulary sizes for each feature
            num_classes (int): Number of CWE classes to predict
            embedding_dim (int): Dimension for each embedding (default: 128)
            hidden_dim (int): Hidden dimension for GGNN and MLP (default: 200)
            num_ggnn_layers (int): Number of GGNN layers (default: 3)
            num_steps (int): Number of propagation steps per GGNN layer (default: 8)
            dropout (float): Dropout rate (default: 0.3)
            use_edge_weights (bool): Whether to use edge weights in GGNN (for CFEExplainer)
        """
        super(GGNN_CWE_Classifier, self).__init__()
        
        self.vocab_sizes = vocab_sizes
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_ggnn_layers = num_ggnn_layers
        self.use_edge_weights = use_edge_weights
        
        # ============================================
        # 1. Node Feature Embeddings
        # ============================================
        # Three separate embeddings for the 3 index features
        self.node_type_embedding = nn.Embedding(
            vocab_sizes['node_type'], 
            embedding_dim,
            padding_idx=0
        )
        
        self.original_type_embedding = nn.Embedding(
            vocab_sizes['original_type'], 
            embedding_dim,
            padding_idx=0
        )
        
        self.label_embedding = nn.Embedding(
            vocab_sizes['label'], 
            embedding_dim,
            padding_idx=0
        )
        
        # Project concatenated embeddings to hidden_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # 2. GGNN Layers
        # ============================================
        self.ggnn_layers = nn.ModuleList([
            GatedGraphConv(hidden_dim, num_steps)
            for _ in range(num_ggnn_layers)
        ])
        
        self.ggnn_dropout = nn.Dropout(dropout)
        
        # ============================================
        # 3. Graph-Level Pooling
        # ============================================
        # We use both mean and max pooling and concatenate
        # This gives richer graph-level representation
        pooled_dim = hidden_dim * 2  # mean + max
        
        # ============================================
        # 4. Classification Head (MLP)
        # ============================================
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings and linear layers with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, 3] - indices for (node_type, original_type, label)
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector [num_nodes] - which graph each node belongs to
            edge_weight: Optional edge weights [num_edges] - for CFEExplainer
        
        Returns:
            logits: [batch_size, num_classes] - class predictions (raw logits)
        """
        # ============================================
        # 1. Embed Node Features
        # ============================================
        # x is [num_nodes, 3] with indices
        node_type_emb = self.node_type_embedding(x[:, 0])      # [num_nodes, embedding_dim]
        original_type_emb = self.original_type_embedding(x[:, 1])  # [num_nodes, embedding_dim]
        label_emb = self.label_embedding(x[:, 2])              # [num_nodes, embedding_dim]
        
        # Concatenate all embeddings
        h = torch.cat([node_type_emb, original_type_emb, label_emb], dim=-1)  # [num_nodes, 3*embedding_dim]
        
        # Project to hidden_dim
        h = self.feature_projection(h)  # [num_nodes, hidden_dim]
        
        # ============================================
        # 2. GGNN Message Passing
        # ============================================
        for ggnn_layer in self.ggnn_layers:
            if self.use_edge_weights and edge_weight is not None:
                # For CFEExplainer: use edge weights
                h_new = ggnn_layer(h, edge_index, edge_weight=edge_weight)
            else:
                h_new = ggnn_layer(h, edge_index)
            
            # Residual connection + activation + dropout
            h = F.relu(h_new + h)  # Residual
            h = self.ggnn_dropout(h)
        
        # ============================================
        # 3. Graph-Level Pooling
        # ============================================
        # Mean pooling
        h_mean = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        
        # Max pooling
        h_max = global_max_pool(h, batch)    # [batch_size, hidden_dim]
        
        # Concatenate both poolings
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # [batch_size, hidden_dim*2]
        
        # ============================================
        # 4. Classification
        # ============================================
        logits = self.classifier(h_graph)  # [batch_size, num_classes]
        
        return logits
    
    def predict(self, x, edge_index, batch, edge_weight=None):
        """
        Convenience method for inference.
        
        Returns:
            predictions: [batch_size] - predicted class indices
            probabilities: [batch_size, num_classes] - class probabilities
        """
        logits = self.forward(x, edge_index, batch, edge_weight)
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        return predictions, probabilities
    
    def get_node_embeddings(self, x, edge_index, edge_weight=None):
        """
        Get node embeddings after GGNN layers (for explainability/visualization).
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        # Embed features
        node_type_emb = self.node_type_embedding(x[:, 0])
        original_type_emb = self.original_type_embedding(x[:, 1])
        label_emb = self.label_embedding(x[:, 2])
        h = torch.cat([node_type_emb, original_type_emb, label_emb], dim=-1)
        h = self.feature_projection(h)
        
        # Pass through GGNN layers
        for ggnn_layer in self.ggnn_layers:
            if self.use_edge_weights and edge_weight is not None:
                h_new = ggnn_layer(h, edge_index, edge_weight=edge_weight)
            else:
                h_new = ggnn_layer(h, edge_index)
            h = F.relu(h_new + h)
            h = self.ggnn_dropout(h)
        
        return h



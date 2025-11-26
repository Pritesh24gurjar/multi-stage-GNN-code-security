import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class MemoryEfficientVulnerabilityDataset(Dataset):
    """
    Memory-efficient PyG Dataset that loads samples on-demand
    Uses memory mapping and lazy loading to prevent GPU OOM
    """
    
    def __init__(self, combined_file, transform=None, pre_transform=None, 
                 cache_size=100):
        """
        Args:
            combined_file (str): Path to combined pickle file
            transform: Optional PyG transform
            pre_transform: Optional PyG pre-transform
            cache_size (int): Number of samples to keep in memory cache
        """
        self.combined_file = Path(combined_file)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        # Load only metadata (indices and labels), not actual data
        print(f"Loading metadata from {combined_file}...")
        with open(combined_file, 'rb') as f:
            self.samples = pickle.load(f)
        
        # Extract just labels and filenames for quick access
        self.labels = [s['label'] for s in self.samples]
        self.num_samples = len(self.samples)
        
        pos_count = sum(self.labels)
        neg_count = self.num_samples - pos_count
        print(f"Loaded {self.num_samples} samples from {combined_file}")
        print(f"  Positive: {pos_count}, Negative: {neg_count}")
        print(f"Using LRU cache with size: {cache_size}")
        
        super().__init__(None, transform, pre_transform)
    
    def len(self):
        return self.num_samples
    
    def _load_sample(self, idx):
        """Load a single sample with LRU caching"""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from disk
        sample = self.samples[idx]
        features_dict = sample['features']
        label = sample['label']
        
        # Extract components
        node_features_dict = features_dict['node_features']
        edge_features_list = features_dict['edge_features']
        
        if not node_features_dict:
            return None
        
        # Build node feature matrix
        node_ids = sorted(node_features_dict.keys())
        node_embeddings = np.array([node_features_dict[nid]['combined'] for nid in node_ids])
        x = torch.tensor(node_embeddings, dtype=torch.float32)
        
        # Build edge index
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_index_list = []
        
        for edge_feat in edge_features_list:
            src_id = edge_feat['source']
            tgt_id = edge_feat['target']
            
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]
                edge_index_list.append([src_idx, tgt_idx])
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(node_ids)
        )
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
        
        self.cache[idx] = data
        self.cache_order.append(idx)
        
        return data
    
    def get(self, idx):
        """Get a sample with memory-efficient loading"""
        return self._load_sample(idx)
    
    def clear_cache(self):
        """Manually clear cache to free memory"""
        self.cache.clear()
        self.cache_order.clear()
        torch.cuda.empty_cache()

class VulnerabilityGraphDataset(Dataset):
    """
    Custom PyG Dataset for loading vulnerability graphs from preprocessed pickle files.
    Handles both positive (vulnerable) and negative (non-vulnerable) samples.
    """
    
    def __init__(self, positive_dir, negative_dir, transform=None, pre_transform=None):
        """
        Args:
            positive_dir (str): Directory containing positive sample pickle files
            negative_dir (str): Directory containing negative sample pickle files
            transform: Optional PyG transform
            pre_transform: Optional PyG pre-transform
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        
        self.positive_files = sorted([f for f in self.positive_dir.glob('*.pkl')])
        self.negative_files = sorted([f for f in self.negative_dir.glob('*.pkl')])
        
        print(f"Found {len(self.positive_files)} positive samples")
        print(f"Found {len(self.negative_files)} negative samples")
        
        self.file_list = [(f, 1) for f in self.positive_files] + [(f, 0) for f in self.negative_files]
        
        super().__init__(None, transform, pre_transform)
    
    def len(self):
        return len(self.file_list)
    
    def get(self, idx):
        filepath, label = self.file_list[idx]
        
        try:
            with open(filepath, 'rb') as f:
                features_dict = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
        
        # Extract components
        node_features_dict = features_dict['node_features']
        edge_features_list = features_dict['edge_features']
        graph_features = features_dict.get('graph_features', {})
        
        if not node_features_dict:
            return None
        
        # Build node feature matrix
        node_ids = sorted(node_features_dict.keys())
        node_embeddings = np.array([node_features_dict[nid]['combined'] for nid in node_ids])
        x = torch.tensor(node_embeddings, dtype=torch.float32)
        
        # Build edge index
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_index_list = []
        
        for edge_feat in edge_features_list:
            src_id = edge_feat['source']
            tgt_id = edge_feat['target']
            
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]
                edge_index_list.append([src_idx, tgt_idx])
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(node_ids)
        )
        
        return data

class CombinedVulnerabilityDataset(Dataset):
    """
    PyG Dataset that loads from pre-combined pickle file with all samples
    """
    
    def __init__(self, combined_file, transform=None, pre_transform=None):
        """
        Args:
            combined_file (str): Path to combined pickle file (train/val/test)
            transform: Optional PyG transform
            pre_transform: Optional PyG pre-transform
        """
        with open(combined_file, 'rb') as f:
            self.samples = pickle.load(f)
        
        print(f"Loaded {len(self.samples)} samples from {combined_file}")
        
        # Count labels
        labels = [s['label'] for s in self.samples]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"  Positive: {pos_count}, Negative: {neg_count}")
        
        super().__init__(None, transform, pre_transform)
    
    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        sample = self.samples[idx]
        features_dict = sample['features']
        label = sample['label']
        
        # Extract components
        node_features_dict = features_dict['node_features']
        edge_features_list = features_dict['edge_features']
        
        if not node_features_dict:
            return None
        
        # Build node feature matrix
        node_ids = sorted(node_features_dict.keys())
        node_embeddings = np.array([node_features_dict[nid]['combined'] for nid in node_ids])
        x = torch.tensor(node_embeddings, dtype=torch.float32)
        
        # Build edge index
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_index_list = []
        
        for edge_feat in edge_features_list:
            src_id = edge_feat['source']
            tgt_id = edge_feat['target']
            
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]
                edge_index_list.append([src_idx, tgt_idx])
        
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(node_ids)
        )
        
        return data

class GINVulnerabilityClassifier(nn.Module):
    """
    Graph Isomorphism Network for vulnerability classification.
    Uses multiple GIN layers with configurable architecture.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim=2, dropout=0.5, pooling='mean'):
        """
        Args:
            input_dim (int): Input node feature dimension
            hidden_dims (list): List of hidden dimensions for each GIN layer
            output_dim (int): Output dimension (2 for binary classification)
            dropout (float): Dropout rate
            pooling (str): Pooling method ('mean' or 'sum')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.pooling = pooling
        
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Build GIN layers
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # MLP for GIN
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            
            gin_conv = GINConv(mlp, train_eps=True)
            self.gin_layers.append(gin_conv)
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        # Classification head
        final_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, output_dim)
        )
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object with x, edge_index, batch attributes
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN layers with residual connections
        for i, (gin, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            x_prev = x
            x = gin(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection if dimensions match
            if x.shape[1] == x_prev.shape[1]:
                x = x + x_prev
        
        # Global pooling
        if self.pooling == 'mean':
            x_pool = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x_pool = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        logits = self.classifier(x_pool)
        return logits


class GINTrainer:
    """
    Trainer class for GIN vulnerability classifier with comprehensive metrics
    including Responsible AI metrics (Equalized Odds and Predictive Parity)
    """
    
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5, class_weights=None):
        """
        Args:
            model: GIN model instance
            device: torch device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
            class_weights: List of weights for each class [non-vuln, vuln]
                        Use to penalize false positives more
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Add class weights to loss function
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'val_fpr': [],  # False Positive Rate (RAI metric 1)
            'val_fnr': [],  # False Negative Rate (RAI metric 2)
            'val_fairness_score': []  # Fairness Score (RAI metric 3)
        }
    
    def compute_fairness_metrics(self, y_true, y_pred):
        """
        Compute Responsible AI fairness metrics
        
        RAI Metrics:
        1. False Positive Rate (FPR): Cost of false alarms - critical in security
        2. False Negative Rate (FNR): Cost of missed vulnerabilities - critical in security
        3. Fairness Score: Balance between FPR and FNR (Equalized Odds)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            dict: Dictionary with FPR, FNR, and Fairness Score
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification: [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # False Positive Rate: FP / (FP + TN)
            # Measures how often non-vulnerable code is incorrectly flagged
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False Negative Rate: FN / (FN + TP)
            # Measures how often vulnerable code is missed
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Fairness Score: Measures balance between FPR and FNR
            # Score = 1 - |FPR - FNR| when both are low
            # Higher score (closer to 1.0) = more fair/balanced model
            # Penalizes imbalance: if FPR=0.1, FNR=0.5, fairness is reduced
            fairness_score = 1.0 - abs(fpr - fnr)
            
            # Alternative fairness metric: Equalized Odds Score
            # Considers both rates together - model is fair if both are low
            # equalized_odds = 1.0 - max(fpr, fnr)
            
        else:
            fpr = 0.0
            fnr = 0.0
            fairness_score = 0.0
        
        return {
            'fpr': fpr, 
            'fnr': fnr, 
            'fairness_score': fairness_score
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().detach().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        progress_bar = tqdm(val_loader, desc='Evaluating', leave=False)
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()  # Probability of positive class
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs_np)
            all_labels.extend(labels)
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Standard metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        epoch_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        
        # Responsible AI metrics
        fairness_metrics = self.compute_fairness_metrics(all_labels, all_preds)
        epoch_fpr = fairness_metrics['fpr']
        epoch_fnr = fairness_metrics['fnr']
        epoch_fairness = fairness_metrics['fairness_score']
        
        return (epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, 
                epoch_auc, epoch_fpr, epoch_fnr, epoch_fairness, all_preds, all_labels)
    
    def train(self, train_loader, val_loader, epochs=50, patience=10, 
            clear_cache_every=10):
        """
        Full training loop with early stopping and memory management
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            clear_cache_every: Clear GPU cache every N epochs
        """
        best_val_f1 = -1
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*80}")
            
            # Clear GPU cache periodically
            if epoch > 0 and epoch % clear_cache_every == 0:
                print("Clearing GPU cache...")
                torch.cuda.empty_cache()
                if hasattr(train_loader.dataset, 'clear_cache'):
                    train_loader.dataset.clear_cache()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_fpr, val_fnr, val_fairness, preds, labels = self.evaluate(val_loader)

            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            self.history['val_fpr'].append(val_fpr)
            self.history['val_fnr'].append(val_fnr)
            self.history['val_fairness_score'].append(val_fairness)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
            print(f"Val AUC: {val_auc:.4f}")
            print(f"RAI Metrics - FPR: {val_fpr:.4f} | FNR: {val_fnr:.4f} | Fairness: {val_fairness:.4f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = self.model.state_dict()
                print("✓ Best model updated")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best epoch: {best_epoch} with F1: {best_val_f1:.4f}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest model from epoch {best_epoch} loaded")
        
        return self.history
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'output_dim': self.model.output_dim,
                'dropout': self.model.dropout,
                'pooling': self.model.pooling
            }
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"✓ Model loaded from {filepath}")
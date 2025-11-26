import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model classes
from ipag_gin.model.gin_classifier import GINVulnerabilityClassifier, GINTrainer


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


class MemoryEfficientVulnerabilityDataset(Dataset):
    """
    Memory-efficient PyG Dataset that loads samples on-demand
    Uses LRU cache and lazy loading to prevent GPU OOM
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
        
        # Extract just labels for quick access
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def setup_argument_parser():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Train GIN Vulnerability Classifier on BigVul Dataset')
    
    # Pre-combined data (REQUIRED)
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to preprocessed training data pickle file')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to preprocessed validation data pickle file')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Path to preprocessed test data pickle file')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='./gin_results',
                        help='Output directory for results and checkpoints')
    
    # Model architecture
    parser.add_argument('--input-dim', type=int, default=776,
                        help='Input feature dimension (768 embedding + 3 type + 5 structural)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden dimensions for GIN layers')
    parser.add_argument('--output-dim', type=int, default=2,
                        help='Output dimension (2 for binary classification)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sum'],
                        help='Pooling method')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--class-weights', type=float, nargs=2, default=None,
                        help='Class weights for loss function [non-vuln, vuln]. Example: --class-weights 1.0 1.5')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Memory management options
    parser.add_argument('--use-efficient-loader', action='store_true',
                        help='Use memory-efficient data loader (for large datasets)')
    parser.add_argument('--cache-size', type=int, default=100,
                        help='Number of samples to keep in memory cache')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Pin memory for faster GPU transfer (disable if OOM)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of batches to prefetch (reduce if OOM)')
    parser.add_argument('--clear-cache-every', type=int, default=10,
                        help='Clear GPU cache every N epochs')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). If None, auto-select')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for DataLoader')
    
    return parser


def print_config(config):
    """Pretty print configuration"""
    print("\n" + "="*80)
    print("Configuration")
    print("="*80)
    for key, value in config.items():
        print(f"  {key:<25} : {value}")
    print("="*80 + "\n")


def save_config(config, filepath):
    """Save configuration to JSON file"""
    # Convert non-serializable types
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, (list, dict, str, int, float, bool, type(None))):
            config_serializable[k] = v
        else:
            config_serializable[k] = str(v)
    
    with open(filepath, 'w') as f:
        json.dump(config_serializable, f, indent=4)


def plot_advanced_metrics(history, val_preds, val_labels, val_probs, save_dir):
    """Generate advanced evaluation plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Progression', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy Progression', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 and optionally precision/recall if available
    axes[1, 0].plot(history['val_f1'], label='F1', linewidth=2, color='blue')
    if 'val_precision' in history:
        axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2, color='green')
    if 'val_recall' in history:
        axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Validation Metrics', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['val_auc'], linewidth=2, color='green', label='AUC')
    if 'val_fairness_score' in history:
        axes[1, 1].plot(history['val_fairness_score'], linewidth=2, color='purple', label='Fairness Score')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('AUC & Fairness Score', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_dir / 'training_history.png'}")
    plt.close()
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-Vulnerable', 'Vulnerable'],
           yticklabels=['Non-Vulnerable', 'Vulnerable'],
           ylabel='True label',
           xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")
    plt.close()
    
    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_dir / 'roc_curve.png'}")
    plt.close()
    
    # Plot 4: RAI Metrics (FPR and FNR over epochs)
    if 'val_fpr' in history and 'val_fnr' in history:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(history['val_fpr'], linewidth=2, color='red', label='FPR (False Positive Rate)')
        ax.plot(history['val_fnr'], linewidth=2, color='darkred', label='FNR (False Negative Rate)')
        if 'val_fairness_score' in history:
            ax2 = ax.twinx()
            ax2.plot(history['val_fairness_score'], linewidth=2, color='green', 
                    label='Fairness Score', linestyle='--')
            ax2.set_ylabel('Fairness Score', fontsize=11, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=11)
        ax.set_title('Responsible AI Metrics Over Training', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'rai_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✓ RAI metrics plot saved to {save_dir / 'rai_metrics.png'}")
        plt.close()


def evaluate_final(model, data_loader, device, output_dir, split_name='Validation'):
    """Final evaluation on a dataset split with all metrics including RAI"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"\nFinal Evaluation on {split_name} Set...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating {split_name}', leave=False):
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs_np)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate standard metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    # Calculate RAI metrics
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fairness_score = 1.0 - abs(fpr - fnr)
    else:
        fpr = fnr = fairness_score = 0.0
    
    print("\n" + "="*80)
    print(f"{split_name} Evaluation Results")
    print("="*80)
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  AUC        : {auc_score:.4f}")
    print(f"\nResponsible AI Metrics:")
    print(f"  FPR (False Positive Rate) : {fpr:.4f}")
    print(f"  FNR (False Negative Rate) : {fnr:.4f}")
    print(f"  Fairness Score            : {fairness_score:.4f}")
    print("="*80 + "\n")
    
    print(f"{split_name} Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Non-Vulnerable', 'Vulnerable']))
    
    # Save metrics to file
    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc_score),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'fairness_score': float(fairness_score)
    }
    
    with open(output_dir / f'{split_name.lower()}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return all_preds, all_labels, all_probs, metrics


def main():
    """Main training script"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Select device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"GIN Vulnerability Classifier - BigVul Dataset")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    # Verify preprocessed files exist
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    test_file = Path(args.test_file) if args.test_file else None
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if test_file and not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Prepare config dictionary
    config = {
        'train_file': str(train_file),
        'val_file': str(val_file),
        'test_file': str(test_file) if test_file else 'None',
        'input_dim': args.input_dim,
        'hidden_dims': args.hidden_dims,
        'output_dim': args.output_dim,
        'dropout': args.dropout,
        'pooling': args.pooling,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'class_weights': args.class_weights,
        'epochs': args.epochs,
        'patience': args.patience,
        'use_efficient_loader': args.use_efficient_loader,
        'cache_size': args.cache_size,
        'clear_cache_every': args.clear_cache_every,
        'seed': args.seed,
        'device': str(device)
    }
    
    print_config(config)
    save_config(config, output_dir / 'config.json')
    
    # Load datasets with memory efficiency options
    print("Loading datasets...")
    if args.use_efficient_loader:
        print("Using memory-efficient data loader...")
        train_dataset = MemoryEfficientVulnerabilityDataset(
            train_file, 
            cache_size=args.cache_size
        )
        val_dataset = MemoryEfficientVulnerabilityDataset(
            val_file, 
            cache_size=args.cache_size // 2
        )
        if test_file:
            test_dataset = MemoryEfficientVulnerabilityDataset(
                test_file, 
                cache_size=args.cache_size // 2
            )
    else:
        train_dataset = CombinedVulnerabilityDataset(train_file)
        val_dataset = CombinedVulnerabilityDataset(val_file)
        if test_file:
            test_dataset = CombinedVulnerabilityDataset(test_file)
    
    # DataLoaders with memory management
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    if test_file:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
        )
    
    # Initialize model
    print(f"\nInitializing GIN model...")
    model = GINVulnerabilityClassifier(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        dropout=args.dropout,
        pooling=args.pooling
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate samples per parameter ratio
    num_train_samples = len(train_dataset)
    ratio = num_train_samples / trainable_params
    print(f"Samples/Parameter ratio: {ratio:.6f}")
    if ratio < 0.01:
        print("⚠ WARNING: Very low samples/parameter ratio - model may overfit")
        print("  Consider: reducing model size or collecting more data")
    print()
    
    # Initialize trainer with class weights
    trainer = GINTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=args.class_weights
    )
    
    # Train with memory management
    print("Starting training...\n")
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            clear_cache_every=args.clear_cache_every
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*80)
            print("ERROR: GPU Out of Memory!")
            print("="*80)
            print("Try these solutions:")
            print("  1. Reduce batch size: --batch-size 16")
            print("  2. Use efficient loader: --use-efficient-loader")
            print("  3. Reduce cache size: --cache-size 50")
            print("  4. Clear cache more often: --clear-cache-every 5")
            print("  5. Disable pin memory: remove --pin-memory")
            print("="*80 + "\n")
            raise
        else:
            raise
    
    # Final evaluation on validation set
    val_preds, val_labels, val_probs, val_metrics = evaluate_final(
        model, val_loader, device, output_dir, split_name='Validation'
    )
    
    # Generate plots
    plot_advanced_metrics(history, val_preds, val_labels, val_probs, output_dir)
    
    # Evaluate on test set if available
    if test_file:
        test_preds, test_labels, test_probs, test_metrics = evaluate_final(
            model, test_loader, device, output_dir, split_name='Test'
        )
        
        # Create test plots directory
        test_plots_dir = output_dir / 'test_plots'
        test_plots_dir.mkdir(exist_ok=True)
        plot_advanced_metrics(history, test_preds, test_labels, test_probs, test_plots_dir)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    trainer.save_model(final_model_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=4)
    print(f"✓ Training history saved to {history_path}")
    
    # Clear GPU cache one final time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - config.json              : Training configuration")
    print(f"  - final_model.pth          : Trained model checkpoint")
    print(f"  - training_history.json    : Per-epoch metrics")
    print(f"  - validation_metrics.json  : Final validation metrics")
    if test_file:
        print(f"  - test_metrics.json        : Final test metrics")
    print(f"  - training_history.png     : Training curves")
    print(f"  - confusion_matrix.png     : Confusion matrix")
    print(f"  - roc_curve.png           : ROC curve")
    print(f"  - rai_metrics.png         : RAI metrics over training")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
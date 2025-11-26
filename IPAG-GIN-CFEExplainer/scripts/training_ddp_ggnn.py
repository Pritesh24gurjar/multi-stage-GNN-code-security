#!/usr/bin/env python
"""
Distributed Data Parallel (DDP) training for GGNN CWE Classifier
Handles multi-GPU training without DataParallel issues
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from ipag_gin.model.ggnn_cwe_classifier import GGNN_CWE_Classifier


class CWEGraphDataset:
    """PyTorch Geometric Dataset wrapper for processed graphs."""
    
    def __init__(self, pkl_path, metadata_path=None):
        with open(pkl_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        print(f"[Rank {dist.get_rank()}] Loaded {len(self.graphs)} graphs")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        
        graph_dict = self.graphs[idx]
        x = torch.from_numpy(graph_dict['features']).long()
        
        edges = graph_dict['ipag_edges']
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            sources = [int(e['source']) for e in edges]
            targets = [int(e['target']) for e in edges]
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        y = torch.tensor([graph_dict['cwe_idx']], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=graph_dict['num_nodes'])
        data.cwe_id = graph_dict['cwe_id']
        data.lang = graph_dict['lang']
        
        return data


class MetricsCalculator:
    """Calculate comprehensive metrics."""
    
    @staticmethod
    def compute_all_metrics(labels, predictions, probabilities=None):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        auc_score = 0.0
        if probabilities is not None and len(np.unique(labels)) > 1:
            try:
                auc_score = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
            except:
                auc_score = 0.0
        
        fpr, fnr = MetricsCalculator._compute_error_rates(labels, predictions)
        fairness_score = MetricsCalculator._compute_fairness_score(labels, predictions)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc_score),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'fairness_score': float(fairness_score)
        }
    
    @staticmethod
    def _compute_error_rates(labels, predictions):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        unique_classes = np.unique(labels)
        fprs, fnrs = [], []
        
        for class_idx in unique_classes:
            y_true_binary = (labels == class_idx).astype(int)
            y_pred_binary = (predictions == class_idx).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fprs.append(fpr)
            fnrs.append(fnr)
        
        return np.mean(fprs), np.mean(fnrs)
    
    @staticmethod
    def _compute_fairness_score(labels, predictions):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        unique_classes = np.unique(labels)
        class_f1_scores = []
        
        for class_idx in unique_classes:
            y_true_binary = (labels == class_idx).astype(int)
            y_pred_binary = (predictions == class_idx).astype(int)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            class_f1_scores.append(f1)
        
        f1_std = np.std(class_f1_scores)
        fairness_score = 1.0 - min(f1_std / 0.5, 1.0)
        
        return float(fairness_score)


class DDPTrainer:
    """Multi-GPU trainer using DistributedDataParallel."""
    
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # Only rank 0 prints
        self.is_main = self.rank == 0
        
        self.exp_dir = Path(config['output_dir']) / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_main:
            print(f"\n{'='*70}")
            print(f"Multi-GPU Training: Rank {self.rank}/{self.world_size}")
            print(f"{'='*70}")
        
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Setup DDP-compatible data loaders."""
        if self.is_main:
            print("Loading dataset...")
        
        dataset = CWEGraphDataset(self.config['data_path'], self.config['metadata_path'])
        
        # Filter rare classes
        labels = [dataset[i].y.item() for i in range(len(dataset))]
        class_counts = Counter(labels)
        min_samples = self.config.get('min_samples_per_class', 100)
        valid_classes = {cls: cnt for cls, cnt in class_counts.items() if cnt >= min_samples}
        
        if self.is_main:
            print(f"Original classes: {len(class_counts)}")
            print(f"Classes after filtering (min_samples={min_samples}): {len(valid_classes)}")
            print(f"Removed: {len(class_counts) - len(valid_classes)} classes")
        
        valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
        filtered_labels = [labels[i] for i in valid_indices]
        
        if self.is_main:
            print(f"Original samples: {len(labels):,}")
            print(f"Samples after filtering: {len(valid_indices):,} ({len(valid_indices)/len(labels)*100:.1f}%)")
        
        old_to_new_class = {old_cls: new_cls for new_cls, old_cls in enumerate(sorted(valid_classes.keys()))}
        remapped_labels = [old_to_new_class[label] for label in filtered_labels]
        
        # Verify remapping
        assert min(remapped_labels) == 0, "Remapped labels should start at 0"
        assert max(remapped_labels) == len(valid_classes) - 1, "Remapped labels should end at num_classes-1"
        
        # Split dataset
        train_idx, test_idx = train_test_split(
            range(len(valid_indices)),
            test_size=self.config['test_size'],
            random_state=self.config['random_seed'],
            stratify=remapped_labels
        )
        
        train_labels_split = [remapped_labels[i] for i in train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.config['val_size'],
            random_state=self.config['random_seed'],
            stratify=train_labels_split
        )
        
        # Remap data objects
        def remap_data(data, old_to_new):
            from torch_geometric.data import Data
            new_data = Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes)
            old_label = data.y.item()
            new_data.y = torch.tensor([old_to_new[old_label]], dtype=torch.long)
            if hasattr(data, 'cwe_id'):
                new_data.cwe_id = data.cwe_id
            if hasattr(data, 'lang'):
                new_data.lang = data.lang
            return new_data
        
        self.train_data = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in train_idx]
        self.val_data = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in val_idx]
        self.test_data = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in test_idx]
        
        self.num_classes = len(valid_classes)
        self.dataset_metadata = dataset.metadata
        self.old_to_new_class = old_to_new_class
        
        # Verify labels
        all_train_labels = [d.y.item() for d in self.train_data]
        assert min(all_train_labels) == 0, f"Train labels min should be 0, got {min(all_train_labels)}"
        assert max(all_train_labels) == self.num_classes - 1, f"Train labels max should be {self.num_classes-1}, got {max(all_train_labels)}"
        
        if self.is_main:
            print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
            print(f"Num classes: {self.num_classes}")
            print(f"Train label range: {min(all_train_labels)}-{max(all_train_labels)} ✓")
            
        # DDP samplers
        self.train_sampler = DistributedSampler(
            self.train_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.config['random_seed']
        )
        
        self.val_sampler = DistributedSampler(
            self.val_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        self.test_sampler = DistributedSampler(
            self.test_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config['batch_size'],
            sampler=self.train_sampler,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.config['batch_size'],
            sampler=self.val_sampler,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.config['batch_size'],
            sampler=self.test_sampler,
            num_workers=0
        )
    
    def setup_model(self):
        """Setup DDP model."""
        vocab_sizes = self.dataset_metadata['vocab_sizes']
        
        model = GGNN_CWE_Classifier(
            vocab_sizes=vocab_sizes,
            num_classes=self.num_classes,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_ggnn_layers=self.config['num_ggnn_layers'],
            num_steps=self.config['num_steps'],
            dropout=self.config['dropout']
        )
        
        model = model.to(self.device)
        
        # Wrap with DDP
        self.model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
        
        # Loss - train_data is already remapped
        remapped_train_labels = [self.train_data[i].y.item() for i in range(len(self.train_data))]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(remapped_train_labels),
            y=remapped_train_labels
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model: {total_params:,} parameters")
            print(f"DDP World Size: {self.world_size}")
            print(f"Num classes (after filtering): {self.num_classes}")
            print(f"Class weights shape: {class_weights.shape}")

    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_sampler.set_epoch(self.current_epoch)  # Important for DDP
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training", disable=not self.is_main):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch.y.squeeze().detach().cpu().numpy())
            
            del batch, logits, loss
            
            # Periodic cache clear
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(loader, desc="Evaluating", disable=not self.is_main):
            batch = batch.to(self.device)
            
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y.squeeze())
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            del batch, logits, loss
        
        all_probs = np.vstack(all_probs) if all_probs else np.array([])
        avg_loss = total_loss / len(loader)
        
        metrics = MetricsCalculator.compute_all_metrics(all_labels, all_preds, all_probs)
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
        
        return metrics
    
    def train(self, epochs):
        """Main training loop."""
        best_val_f1 = 0
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 15)  # ← ADD THIS
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            if self.is_main:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val F1: {val_metrics['f1_score']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            
            self.scheduler.step(val_metrics['f1_score'])
            
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience_counter = 0
                if self.is_main:
                    print(f"✓ Best F1: {best_val_f1:.4f}")
                    self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:  # ← USE patience HERE
                    if self.is_main:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            torch.cuda.empty_cache()
        
        if self.is_main:
            with open(self.exp_dir / 'training_history.json', 'w') as f:
                json.dump(history, f)

    
    def test(self):
        """Test on test set."""
        test_metrics = self.evaluate(self.test_loader)
        
        if self.is_main:
            print("\n" + "="*70)
            print("TEST METRICS")
            print("="*70)
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"AUC: {test_metrics['auc']:.4f}")
            print(f"Fairness: {test_metrics['fairness_score']:.4f}")
            
            test_metrics_json = {
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score'],
                'auc': test_metrics['auc'],
                'fpr': test_metrics['fpr'],
                'fnr': test_metrics['fnr'],
                'fairness_score': test_metrics['fairness_score']
            }
            
            with open(self.exp_dir / 'test_metrics.json', 'w') as f:
                json.dump(test_metrics_json, f, indent=2)
    
    def save_checkpoint(self, filename):
        """Save checkpoint (only on rank 0)."""
        if self.rank == 0:
            checkpoint = {
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, self.exp_dir / filename)
    
    def cleanup(self):
        """Cleanup DDP."""
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_ggnn_layers', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--min_samples_per_class', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                   help='Early stopping patience')

    
    args = parser.parse_args()
    config = vars(args)
    
    # Setup DDP
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    try:
        trainer = DDPTrainer(config)
        trainer.train(config['epochs'])
        trainer.test()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

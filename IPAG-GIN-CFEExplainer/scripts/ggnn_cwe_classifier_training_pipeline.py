import os
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from ipag_gin.model.ggnn_cwe_classifier import GGNN_CWE_Classifier
from collections import Counter


class MemoryManager:
    """Proactive GPU memory management."""
    
    def __init__(self, device, memory_threshold=0.85):
        self.device = device
        self.memory_threshold = memory_threshold
        
    def get_memory_info(self):
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'total': 0, 'free': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free,
            'utilization': allocated / total if total > 0 else 0
        }
    
    def clear_memory(self):
        """Aggressively clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def check_memory_pressure(self):
        """Check if memory pressure is high."""
        info = self.get_memory_info()
        return info['utilization'] > self.memory_threshold
    
    def print_memory_stats(self, prefix=""):
        """Print formatted memory statistics."""
        info = self.get_memory_info()
        print(f"{prefix}GPU Memory: {info['allocated']:.2f}GB / {info['total']:.2f}GB "
              f"({info['utilization']*100:.1f}% used, {info['free']:.2f}GB free)")


class CWEGraphDataset:
    """PyTorch Geometric Dataset wrapper for processed graphs."""
    
    def __init__(self, pkl_path, metadata_path=None):
        with open(pkl_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.metadata = None
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        print(f"Loaded {len(self.graphs)} graphs from {pkl_path}")
        if self.metadata:
            print(f"  CWE classes: {self.metadata['num_classes']}")
            print(f"  Feature dim: {self.metadata['feature_dim']}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph_dict = self.graphs[idx]
        
        x = torch.from_numpy(graph_dict['features']).long()
        
        edges = graph_dict['ipag_edges']
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)
        else:
            sources = [int(e['source']) for e in edges]
            targets = [int(e['target']) for e in edges]
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            
            edge_type_map = {'CHILD': 0, 'CONTROL_FLOW': 1, 'NEXT_TOKEN': 2}
            edge_type = torch.tensor(
                [edge_type_map.get(e['type'], 0) for e in edges],
                dtype=torch.long
            )
        
        y = torch.tensor([graph_dict['cwe_idx']], dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            num_nodes=graph_dict['num_nodes']
        )
        
        data.cwe_id = graph_dict['cwe_id']
        data.lang = graph_dict['lang']
        
        return data


class MetricsCalculator:
    """Calculate comprehensive metrics including fairness metrics."""
    
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
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc_score),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'fairness_score': float(fairness_score)
        }
        
        return metrics
    
    @staticmethod
    def _compute_error_rates(labels, predictions):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        unique_classes = np.unique(labels)
        fprs = []
        fnrs = []
        
        for class_idx in unique_classes:
            y_true_binary = (labels == class_idx).astype(int)
            y_pred_binary = (predictions == class_idx).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs.append(fpr)
            
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fnrs.append(fnr)
        
        avg_fpr = np.mean(fprs) if fprs else 0
        avg_fnr = np.mean(fnrs) if fnrs else 0
        
        return avg_fpr, avg_fnr
    
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
    
    @staticmethod
    def compute_per_class_metrics(labels, predictions, num_classes, idx_to_cwe):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        per_class = {}
        
        for class_idx in range(num_classes):
            y_true_binary = (labels == class_idx).astype(int)
            y_pred_binary = (predictions == class_idx).astype(int)
            
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            support = np.sum(y_true_binary)
            
            per_class[idx_to_cwe[class_idx]] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(support)
            }
        
        return per_class


class Trainer:
    """Training manager with comprehensive OOM handling."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, scheduler, 
                 class_weights, device, exp_dir, num_classes, idx_to_cwe, 
                 gradient_accumulation_steps=4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_weights = class_weights
        self.device = device
        self.exp_dir = Path(exp_dir)
        self.num_classes = num_classes
        self.idx_to_cwe = idx_to_cwe
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Memory manager
        self.memory_manager = MemoryManager(device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.best_metrics = None
        
        print(f"\n{'='*70}")
        print("TRAINER CONFIGURATION")
        print(f"{'='*70}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {train_loader.batch_size * gradient_accumulation_steps}")
        self.memory_manager.print_memory_stats("Initial ")
        print(f"{'='*70}\n")
    
    def train_epoch(self):
        """Train for one epoch with gradient accumulation and memory management."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        accumulation_loss = 0
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    logits = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = self.criterion(logits, batch.y.squeeze())
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    accumulation_loss += loss.item()
                    
                    # Update weights every N steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        total_loss += accumulation_loss
                        accumulation_loss = 0
                    
                    # Collect predictions
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(batch.y.squeeze().detach().cpu().numpy())
                    
                    # Clean up
                    del batch, logits, loss, preds
                    
                    # Periodic memory cleanup
                    if (batch_idx + 1) % 100 == 0:
                        self.memory_manager.clear_memory()
                    
                    # Memory monitoring every 500 batches
                    if (batch_idx + 1) % 500 == 0:
                        pbar.write(f"  [Batch {batch_idx+1}] " + 
                                  f"Memory: {self.memory_manager.get_memory_info()['allocated']:.2f}GB")
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f"{total_loss/(batch_idx//self.gradient_accumulation_steps + 1):.4f}"})
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  OOM at batch {batch_idx}. Clearing memory and skipping batch...")
                        self.memory_manager.clear_memory()
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise e
        
        # Handle remaining gradients
        if accumulation_loss > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulation_loss
        
        avg_loss = total_loss / (len(self.train_loader) // self.gradient_accumulation_steps)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Final cleanup
        self.memory_manager.clear_memory()
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, split_name="Validation"):
        """Evaluate with chunked processing to avoid OOM."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        chunk_size = 50  # Process predictions in smaller chunks
        
        with tqdm(loader, desc=f"Evaluating {split_name}", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = batch.to(self.device)
                    
                    logits = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = self.criterion(logits, batch.y.squeeze())
                    
                    total_loss += loss.item()
                    
                    # Move to CPU immediately to free GPU memory
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()
                    labels = batch.y.squeeze().cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    all_probs.append(probs)
                    
                    # Clean up
                    del batch, logits, loss, probs, preds, labels
                    
                    # Periodic cleanup
                    if (batch_idx + 1) % chunk_size == 0:
                        self.memory_manager.clear_memory()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  OOM during evaluation at batch {batch_idx}. Clearing memory...")
                        self.memory_manager.clear_memory()
                        continue
                    else:
                        raise e
        
        # Concatenate probabilities efficiently
        all_probs = np.vstack(all_probs)
        
        avg_loss = total_loss / len(loader)
        
        metrics = MetricsCalculator.compute_all_metrics(all_labels, all_preds, all_probs)
        
        metrics['loss'] = avg_loss
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
        
        # Final cleanup
        self.memory_manager.clear_memory()
        
        return metrics
    
    def train(self, epochs, early_stopping_patience=15):
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            # Memory check before epoch
            self.memory_manager.print_memory_stats("Pre-epoch ")
            
            try:
                train_loss, train_acc = self.train_epoch()
                val_metrics = self.evaluate(self.val_loader, "Validation")
                
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1_score'])
                self.history['val_auc'].append(val_metrics['auc'])
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"Val F1: {val_metrics['f1_score']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
                print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
                print(f"FPR: {val_metrics['fpr']:.4f} | FNR: {val_metrics['fnr']:.4f} | Fairness: {val_metrics['fairness_score']:.4f}")
                
                self.memory_manager.print_memory_stats("Post-epoch ")
                
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1_score'])
                else:
                    self.scheduler.step()
                
                if val_metrics['f1_score'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1_score']
                    self.best_epoch = epoch
                    self.best_metrics = val_metrics.copy()
                    self.save_checkpoint('best_model.pth')
                    patience_counter = 0
                    print(f"✓ Best model saved (F1: {self.best_val_f1:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  OOM during epoch {epoch+1}. Clearing memory and continuing...")
                    self.memory_manager.clear_memory()
                    continue
                else:
                    raise e
        
        print("\n" + "="*70)
        print(f"Training completed. Best epoch: {self.best_epoch+1}, Best F1: {self.best_val_f1:.4f}")
        print("="*70)
        
        self.save_history()
        self.plot_training_history()
    
    def test(self):
        print("\n" + "="*70)
        print("TESTING ON HELD-OUT TEST SET")
        print("="*70)
        
        self.load_checkpoint('best_model.pth')
        
        test_metrics = self.evaluate(self.test_loader, "Test")
        
        print(f"\n{'='*70}")
        print("TEST METRICS")
        print("="*70)
        print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"Precision:      {test_metrics['precision']:.4f}")
        print(f"Recall:         {test_metrics['recall']:.4f}")
        print(f"F1-Score:       {test_metrics['f1_score']:.4f}")
        print(f"AUC:            {test_metrics['auc']:.4f}")
        print(f"FPR:            {test_metrics['fpr']:.4f}")
        print(f"FNR:            {test_metrics['fnr']:.4f}")
        print(f"Fairness Score: {test_metrics['fairness_score']:.4f}")
        print("="*70)
        
        self.save_test_metrics(test_metrics)
        
        test_plots_dir = self.exp_dir / 'test_plots'
        test_plots_dir.mkdir(exist_ok=True)
        
        self.plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], test_plots_dir / 'confusion_matrix.png')
        self.plot_roc_curves(test_metrics['labels'], test_metrics['probabilities'], test_plots_dir / 'roc_curve.png')
        self.plot_rai_metrics(test_metrics['labels'], test_metrics['predictions'], test_plots_dir / 'rai_metrics.png')
        self.plot_training_history_in_testplots(test_plots_dir)
        
        import shutil
        for plot_file in test_plots_dir.glob('*.png'):
            shutil.copy(plot_file, self.exp_dir / plot_file.name)
        
        print(f"\n✓ All test plots saved to {test_plots_dir}")
        
        return test_metrics
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'history': self.history
        }
        torch.save(checkpoint, self.exp_dir / filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(self.exp_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_epoch = checkpoint['best_epoch']
        self.best_metrics = checkpoint.get('best_metrics')
        self.history = checkpoint['history']
    
    def save_history(self):
        with open(self.exp_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_test_metrics(self, metrics):
        test_metrics_json = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc': metrics['auc'],
            'fpr': metrics['fpr'],
            'fnr': metrics['fnr'],
            'fairness_score': metrics['fairness_score']
        }
        
        with open(self.exp_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics_json, f, indent=2)
        
        per_class = MetricsCalculator.compute_per_class_metrics(metrics['labels'], metrics['predictions'], self.num_classes, self.idx_to_cwe)
        
        with open(self.exp_dir / 'per_class_metrics.json', 'w') as f:
            json.dump(per_class, f, indent=2)
        
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=[self.idx_to_cwe[i] for i in range(self.num_classes)],
            zero_division=0
        )
        
        with open(self.exp_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + report)
    
    def plot_confusion_matrix(self, labels, predictions, save_path):
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.idx_to_cwe[i] for i in range(self.num_classes)],
                    yticklabels=[self.idx_to_cwe[i] for i in range(self.num_classes)],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, labels, probabilities, save_path):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(self.num_classes):
            y_true = (np.array(labels) == i).astype(int)
            y_score = probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'{self.idx_to_cwe[i]} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rai_metrics(self, labels, predictions, save_path):
        precision = precision_score(labels, predictions, average=None, zero_division=0, labels=range(self.num_classes))
        recall = recall_score(labels, predictions, average=None, zero_division=0, labels=range(self.num_classes))
        f1 = f1_score(labels, predictions, average=None, zero_division=0, labels=range(self.num_classes))
        
        x = np.arange(self.num_classes)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('CWE Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics (Responsible AI)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self.idx_to_cwe[i] for i in range(self.num_classes)], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['val_f1'], label='F1-Score', linewidth=2)
        axes[1, 0].plot(self.history['val_auc'], label='AUC', linewidth=2)
        axes[1, 0].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation F1 and AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history['learning_rates'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history_in_testplots(self, test_plots_dir):
        import shutil
        shutil.copy(self.exp_dir / 'training_history.png', test_plots_dir / 'training_history.png')


class ExperimentManager:
    """Manages multiple experiments and summarizes results."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.results_summary = {}
    
    def add_result(self, exp_name, metrics, config):
        self.results_summary[exp_name] = {
            'metrics': metrics,
            'config': config
        }
    
    def save_summary(self):
        summary_path = self.output_dir / 'experiments_summary.json'
        
        summary_data = {}
        for exp_name, data in self.results_summary.items():
            summary_data[exp_name] = {
                'metrics': data['metrics'],
                'config': {
                    'embedding_dim': data['config'].get('embedding_dim'),
                    'hidden_dim': data['config'].get('hidden_dim'),
                    'num_ggnn_layers': data['config'].get('num_ggnn_layers'),
                    'num_steps': data['config'].get('num_steps'),
                    'dropout': data['config'].get('dropout'),
                    'learning_rate': data['config'].get('learning_rate'),
                    'batch_size': data['config'].get('batch_size')
                }
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n✓ Experiment summary saved to {summary_path}")
        
        self.print_comparison_table()
    
    def print_comparison_table(self):
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY")
        print("="*100)
        
        headers = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Fairness']
        print(f"{headers[0]:<20} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12} {headers[4]:>12} {headers[5]:>12} {headers[6]:>12}")
        print("-" * 100)
        
        for exp_name, data in sorted(self.results_summary.items()):
            metrics = data['metrics']
            print(f"{exp_name:<20} {metrics['accuracy']:>12.4f} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1_score']:>12.4f} {metrics['auc']:>12.4f} {metrics['fairness_score']:>12.4f}")
        
        print("="*100)


def create_model(vocab_sizes, num_classes, device='cuda', **kwargs):
    model = GGNN_CWE_Classifier(vocab_sizes=vocab_sizes, num_classes=num_classes, **kwargs)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


def run_experiment(config, exp_dir, exp_manager):
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_dir.name}")
    print("="*70)
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Initialize memory manager
    memory_manager = MemoryManager(device)
    memory_manager.print_memory_stats("Initial ")
    
    print("\nLoading dataset...")
    
    dataset = CWEGraphDataset(pkl_path=config['data_path'], metadata_path=config['metadata_path'])
    
    vocab_sizes = dataset.metadata['vocab_sizes']
    num_classes = dataset.metadata['num_classes']
    idx_to_cwe = dataset.metadata['idx_to_cwe']
    
    print("\nFiltering rare classes...")
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    
    # Count samples per class
    class_counts = Counter(labels)
    print(f"Original classes: {len(class_counts)}")
    print(f"Class distribution stats:")
    print(f"  Max samples: {max(class_counts.values()):,}")
    print(f"  Min samples: {min(class_counts.values()):,}")
    print(f"  Median samples: {np.median(list(class_counts.values())):.0f}")
    
    # Filter classes with less than min_samples_per_class
    min_samples_per_class = config.get('min_samples_per_class', 100)
    valid_classes = {cls: count for cls, count in class_counts.items() if count >= min_samples_per_class}
    
    print(f"\nFiltering classes with < {min_samples_per_class} samples...")
    print(f"Classes after filtering: {len(valid_classes)}")
    print(f"Classes removed: {len(class_counts) - len(valid_classes)}")
    
    # Keep only samples from valid classes
    valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
    filtered_labels = [labels[i] for i in valid_indices]
    
    print(f"Original samples: {len(labels):,}")
    print(f"Samples after filtering: {len(valid_indices):,}")
    print(f"Removed samples: {len(labels) - len(valid_indices):,}")
    
    # Remap class indices to be contiguous (0, 1, 2, ...)
    old_to_new_class = {old_cls: new_cls for new_cls, old_cls in enumerate(sorted(valid_classes.keys()))}
    remapped_labels = [old_to_new_class[label] for label in filtered_labels]
    actual_num_classes = len(valid_classes)
    
    print(f"Remapped to {actual_num_classes} classes (0-{actual_num_classes-1})")
    
    print("\nSplitting dataset...")
    
    # Split using remapped indices
    train_idx, test_idx = train_test_split(
        range(len(valid_indices)),
        test_size=config['test_size'],
        random_state=config['random_seed'],
        stratify=remapped_labels
    )
    
    train_labels = [remapped_labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=config['val_size'],
        random_state=config['random_seed'],
        stratify=train_labels
    )
    
    # Map back to original dataset indices AND remap labels
    def remap_data(data, old_to_new_class):
        """Create a copy of data with remapped label."""
        new_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type if hasattr(data, 'edge_type') else None,
            num_nodes=data.num_nodes
        )
        # Remap the label
        old_label = data.y.item()
        new_label = old_to_new_class[old_label]
        new_data.y = torch.tensor([new_label], dtype=torch.long)
        
        # Copy metadata
        if hasattr(data, 'cwe_id'):
            new_data.cwe_id = data.cwe_id
        if hasattr(data, 'lang'):
            new_data.lang = data.lang
            
        return new_data
    
    train_dataset = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in train_idx]
    val_dataset = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in val_idx]
    test_dataset = [remap_data(dataset[valid_indices[i]], old_to_new_class) for i in test_idx]
    
    # Verify label range
    print(f"\nVerifying labels...")
    train_labels_check = [d.y.item() for d in train_dataset]
    print(f"  Train label range: {min(train_labels_check)} to {max(train_labels_check)}")
    print(f"  Expected range: 0 to {actual_num_classes - 1}")
    assert max(train_labels_check) < actual_num_classes, "Label out of range!"
    print("  ✓ Labels verified")
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Use pin_memory=False and num_workers=0 for better memory management
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, pin_memory=False, num_workers=0)
    
    # Compute class weights using remapped labels
    class_weights = compute_class_weight('balanced', classes=np.unique(remapped_labels), y=remapped_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights computed for {len(class_weights)} classes")
    
    print("\nCreating model...")
    model = create_model(
        vocab_sizes=vocab_sizes,
        num_classes=actual_num_classes,
        device=device,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_ggnn_layers=config['num_ggnn_layers'],
        num_steps=config['num_steps'],
        dropout=config['dropout']
    )
    
    memory_manager.print_memory_stats("After model creation ")
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Create remapped idx_to_cwe dict
    remapped_idx_to_cwe = {new_idx: idx_to_cwe[old_cls] for old_cls, new_idx in old_to_new_class.items()}
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        class_weights=class_weights,
        device=device,
        exp_dir=exp_dir,
        num_classes=actual_num_classes,
        idx_to_cwe=remapped_idx_to_cwe,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4)
    )
    
    trainer.train(epochs=config['epochs'], early_stopping_patience=config['early_stopping_patience'])
    
    test_metrics = trainer.test()
    
    exp_manager.add_result(exp_dir.name, test_metrics, config)
    
    print(f"\n✓ Experiment completed: {exp_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train GGNN for CWE classification (OOM-resistant)")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed_graphs_c_only.pkl')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to dataset_metadata.pkl')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory for experiments')
    
    parser.add_argument('--run_all', action='store_true', help='Run all predefined experiments')
    parser.add_argument('--exp_name', type=str, default=None, help='Single experiment name')
    
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_ggnn_layers', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--batch_size', type=int, default=16, help='Physical batch size (reduced for 12GB GPU)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Accumulate gradients over N batches')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--min_samples_per_class', type=int, default=100, help='Minimum samples per class (filters out rare classes)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    exp_manager = ExperimentManager(output_dir)
    
    if args.run_all:
        experiments = {
            'exp1_baseline': {
                'embedding_dim': 128, 'hidden_dim': 200, 'num_ggnn_layers': 3, 
                'num_steps': 8, 'dropout': 0.3, 'learning_rate': 0.001,
                'batch_size': 16, 'gradient_accumulation_steps': 8
            },
            'exp2_deep': {
                'embedding_dim': 128, 'hidden_dim': 200, 'num_ggnn_layers': 5, 
                'num_steps': 8, 'dropout': 0.4, 'learning_rate': 0.001,
                'batch_size': 12, 'gradient_accumulation_steps': 10
            },
            'exp3_wide': {
                'embedding_dim': 256, 'hidden_dim': 300, 'num_ggnn_layers': 3, 
                'num_steps': 8, 'dropout': 0.3, 'learning_rate': 0.0005,
                'batch_size': 12, 'gradient_accumulation_steps': 10
            },
            'exp4_heavy_dropout': {
                'embedding_dim': 128, 'hidden_dim': 200, 'num_ggnn_layers': 3, 
                'num_steps': 8, 'dropout': 0.5, 'learning_rate': 0.001,
                'batch_size': 16, 'gradient_accumulation_steps': 8
            }
        }
        
        for exp_name, exp_config in experiments.items():
            config = {
                'data_path': args.data_path, 'metadata_path': args.metadata_path,
                'epochs': args.epochs, 'weight_decay': args.weight_decay,
                'early_stopping_patience': args.early_stopping_patience, 'test_size': args.test_size,
                'val_size': args.val_size, 'random_seed': args.random_seed,
                'min_samples_per_class': args.min_samples_per_class,
                **exp_config
            }
            
            exp_dir = output_dir / exp_name
            exp_dir.mkdir(exist_ok=True)
            
            run_experiment(config, exp_dir, exp_manager)
        
        exp_manager.save_summary()
    
    else:
        exp_name = args.exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = vars(args)
        config.pop('run_all')
        config.pop('exp_name')
        config.pop('output_dir')
        
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        run_experiment(config, exp_dir, exp_manager)
        exp_manager.save_summary()


if __name__ == "__main__":
    main()
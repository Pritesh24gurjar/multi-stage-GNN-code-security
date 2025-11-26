#!/usr/bin/env python3
"""
Find Optimal Decision Threshold for GIN Vulnerability Classifier
Balances FPR, FNR, and Fairness Score while maintaining minimum recall
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import argparse
import json
import sys

from ipag_gin.model.gin_classifier import GINVulnerabilityClassifier, CombinedVulnerabilityDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm


def find_optimal_threshold(model, val_loader, device, min_recall=0.75, min_precision=0.50):
    """
    Find threshold that balances FPR and FNR while maintaining minimum recall and precision
    
    Args:
        model: Trained GIN model
        val_loader: Validation DataLoader
        device: torch device
        min_recall: Minimum acceptable recall (default: 0.75)
        min_precision: Minimum acceptable precision (default: 0.50)
    
    Returns:
        tuple: (best_threshold, best_metrics, all_results)
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    print("\nCollecting predictions...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating', leave=False):
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics for range of thresholds
    print(f"\n{'='*100}")
    print(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
          f"{'F1':<10} {'FPR':<10} {'FNR':<10} {'Fairness':<10}")
    print(f"{'='*100}")
    
    best_threshold = 0.5
    best_fairness = -1.0
    best_metrics = {}
    all_results = []
    
    for threshold in np.arange(0.20, 0.90, 0.05):
        preds = (all_probs >= threshold).astype(int)
        cm = confusion_matrix(all_labels, preds)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fairness = 1.0 - abs(fpr - fnr)
            
            result = {
                'threshold': float(threshold),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'fairness_score': float(fairness),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
            all_results.append(result)
            
            # Highlight if meets constraints
            marker = ""
            if recall >= min_recall and precision >= min_precision:
                marker = " ✓"
            
            print(f"{threshold:<12.2f} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} "
                  f"{f1:<10.4f} {fpr:<10.4f} {fnr:<10.4f} {fairness:<10.4f}{marker}")
            
            # Select best threshold that maintains constraints
            if fairness > best_fairness and recall >= min_recall and precision >= min_precision:
                best_fairness = fairness
                best_threshold = threshold
                best_metrics = result.copy()
    
    print(f"{'='*100}")
    
    if not best_metrics:
        print("\n⚠ WARNING: No threshold meets the minimum recall and precision constraints!")
        print(f"  Constraints: Recall >= {min_recall}, Precision >= {min_precision}")
        print("\n  Relaxing constraints to find best threshold...")
        
        # Find best without precision constraint
        for result in all_results:
            if result['recall'] >= min_recall and result['fairness_score'] > best_fairness:
                best_fairness = result['fairness_score']
                best_threshold = result['threshold']
                best_metrics = result.copy()
        
        if not best_metrics:
            # Just pick highest fairness
            best_metrics = max(all_results, key=lambda x: x['fairness_score'])
            best_threshold = best_metrics['threshold']
            print(f"\n  Using threshold with highest fairness (no constraints)")
    
    print(f"\n{'='*100}")
    print(f"✓ Best Threshold: {best_threshold:.2f}")
    print(f"{'='*100}")
    print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"  Precision:      {best_metrics['precision']:.4f}")
    print(f"  Recall:         {best_metrics['recall']:.4f}")
    print(f"  F1 Score:       {best_metrics['f1']:.4f}")
    print(f"  FPR:            {best_metrics['fpr']:.4f}")
    print(f"  FNR:            {best_metrics['fnr']:.4f}")
    print(f"  Fairness:       {best_metrics['fairness_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {best_metrics['tn']:<6} FP: {best_metrics['fp']}")
    print(f"  FN: {best_metrics['fn']:<6} TP: {best_metrics['tp']}")
    print(f"{'='*100}")
    
    return best_threshold, best_metrics, all_results


def plot_threshold_curves(all_results, output_path):
    """
    Plot threshold vs metrics curves
    """
    try:
        import matplotlib.pyplot as plt
        
        thresholds = [r['threshold'] for r in all_results]
        accuracies = [r['accuracy'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1s = [r['f1'] for r in all_results]
        fprs = [r['fpr'] for r in all_results]
        fnrs = [r['fnr'] for r in all_results]
        fairness = [r['fairness_score'] for r in all_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy, Precision, Recall, F1
        axes[0, 0].plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        axes[0, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
        axes[0, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
        axes[0, 0].plot(thresholds, f1s, label='F1', linewidth=2)
        axes[0, 0].set_xlabel('Threshold', fontsize=11)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title('Performance Metrics vs Threshold', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: FPR and FNR
        axes[0, 1].plot(thresholds, fprs, label='FPR', linewidth=2, color='red')
        axes[0, 1].plot(thresholds, fnrs, label='FNR', linewidth=2, color='darkred')
        axes[0, 1].set_xlabel('Threshold', fontsize=11)
        axes[0, 1].set_ylabel('Error Rate', fontsize=11)
        axes[0, 1].set_title('Error Rates vs Threshold', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fairness Score
        axes[1, 0].plot(thresholds, fairness, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Threshold', fontsize=11)
        axes[1, 0].set_ylabel('Fairness Score', fontsize=11)
        axes[1, 0].set_title('Fairness Score vs Threshold', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: FPR vs FNR (trade-off)
        axes[1, 1].scatter(fprs, fnrs, c=thresholds, cmap='viridis', s=50, alpha=0.7)
        axes[1, 1].plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Perfect Balance')
        axes[1, 1].set_xlabel('FPR (False Positive Rate)', fontsize=11)
        axes[1, 1].set_ylabel('FNR (False Negative Rate)', fontsize=11)
        axes[1, 1].set_title('FPR vs FNR Trade-off', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Threshold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Threshold curves saved to {output_path}")
        plt.close()
        
    except ImportError:
        print("\n⚠ Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\n⚠ Could not create plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Find optimal decision threshold for GIN classifier')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to validation data pickle file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--min-recall', type=float, default=0.75,
                        help='Minimum acceptable recall (default: 0.75)')
    parser.add_argument('--min-precision', type=float, default=0.50,
                        help='Minimum acceptable precision (default: 0.50)')
    parser.add_argument('--output', type=str, default='optimal_threshold.json',
                        help='Output JSON file for threshold and metrics')
    parser.add_argument('--plot', action='store_true',
                        help='Generate threshold curves plot')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.model_path).exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.val_file).exists():
        print(f"ERROR: Validation file not found: {args.val_file}")
        sys.exit(1)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*100}")
    print(f"Optimal Threshold Finder")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Validation data: {args.val_file}")
    print(f"Constraints: Recall >= {args.min_recall}, Precision >= {args.min_precision}")
    print(f"{'='*100}\n")
    
    # Load checkpoint
    print(f"Loading model...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Recreate model
    model_config = checkpoint['model_config']
    model = GINVulnerabilityClassifier(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['output_dim'],
        dropout=model_config['dropout'],
        pooling=model_config['pooling']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully\n")
    
    # Load validation data
    print(f"Loading validation data...")
    try:
        val_dataset = CombinedVulnerabilityDataset(args.val_file)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=0)
        print(f"✓ Loaded {len(val_dataset)} validation samples\n")
    except Exception as e:
        print(f"ERROR: Failed to load validation data: {e}")
        sys.exit(1)
    
    # Find optimal threshold
    try:
        best_threshold, best_metrics, all_results = find_optimal_threshold(
            model, val_loader, device, args.min_recall, args.min_precision
        )
    except Exception as e:
        print(f"\nERROR: Threshold search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'best_threshold': best_metrics,
        'constraints': {
            'min_recall': args.min_recall,
            'min_precision': args.min_precision
        },
        'all_thresholds': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = output_path.parent / f"{output_path.stem}_curves.png"
        plot_threshold_curves(all_results, plot_path)
    
    print("\n✓ Threshold optimization complete!")


if __name__ == "__main__":
    main()
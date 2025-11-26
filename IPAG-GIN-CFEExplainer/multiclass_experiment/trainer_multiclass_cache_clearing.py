"""
End-to-end training pipeline for GGNN CWE multi-class classification.

Steps:
    1. Load preprocessed graphs, vocabularies, and metadata produced by
       `batch_data_preprocessing_multiclass.py`.
    2. Build PyTorch Geometric `Data` objects from the serialized graphs.
    3. Split into train / validation / test sets (stratified by CWE index).
    4. Train GGNN_CWE_Classifier with metrics logged per epoch.
    5. Evaluate best checkpoint on the held-out test set.
    6. Compute Responsible AI metrics (per-class performance + disparities).

This script is designed to work with large datasets (~200K graphs) and a
single 12GB GPU. Tune batch_size and num_workers as needed.

Example:
    python train_ggnn_multiclass.py \
        --data_dir data/processed \
        --batch_size 64 \
        --epochs 40 \
        --lr 1e-3 \
        --hidden_dim 200 \
        --embedding_dim 128
"""

import os
import argparse
import random
from pathlib import Path
import pickle
from collections import Counter
import gc               
import json             

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import from your project package
from ipag_gin.graph.vocabulary_builder import VocabularyBuilder
from ipag_gin.model.ggnn_cwe_classifier import GGNN_CWE_Classifier


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pyg_graph(graph_dict, add_reverse_edges: bool = True) -> Data | None:
    """
    Convert one preprocessed graph_dict into a torch_geometric.data.Data object.

    graph_dict keys (from batch_data_preprocessing_multiclass.py):
        - "features": np.ndarray [num_nodes, 3] (indices for 3 vocabularies)
        - "ipag_edges": list of {"source": int/str, "target": int/str, "type": str}
        - "cwe_idx": int (class index)
        - plus metadata: "cwe_id", "cve_id", "vul", "lang", "code", ...
    """
    features = graph_dict["features"]
    ipag_edges = graph_dict["ipag_edges"]
    y = graph_dict["cwe_idx"]

    # Handle empty graphs
    if features.size == 0:
        return None

    x = torch.from_numpy(features).long()  # [num_nodes, 3]
    num_nodes = x.shape[0]

    src, dst = [], []
    for e in ipag_edges:
        try:
            s = int(e["source"])
            t = int(e["target"])
        except (ValueError, TypeError):
            # Skip malformed edges
            continue

        # Optionally, drop edges pointing outside [0, num_nodes)
        if not (0 <= s < num_nodes and 0 <= t < num_nodes):
            continue

        src.append(s)
        dst.append(t)

    if add_reverse_edges:
        all_src = src + dst
        all_dst = dst + src
    else:
        all_src = src
        all_dst = dst

    if len(all_src) == 0:
        # no edges => isolated nodes
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.long),
        num_nodes=num_nodes,
    )

    # Optional small metadata
    data.cwe_id = graph_dict.get("cwe_id")
    data.cve_id = graph_dict.get("cve_id")
    data.lang = graph_dict.get("lang")

    return data


def stratified_split(graphs, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Simple stratified split by label indices.
    Returns lists of indices: train_idx, val_idx, test_idx
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    label_to_indices = {}
    for idx, y in enumerate(labels):
        label_to_indices.setdefault(int(y), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for y, idxs in label_to_indices.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Ensure at least 1 sample in each split when possible
        if n_train == 0 and n > 0:
            n_train = 1
        if n_val == 0 and n - n_train > 1:
            n_val = 1

        train_part = idxs[:n_train]
        val_part = idxs[n_train:n_train + n_val]
        test_part = idxs[n_train + n_val:]

        train_idx.extend(train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_part)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def create_dataloaders(
    graphs,
    train_idx,
    val_idx,
    test_idx,
    batch_size=64,
    num_workers=0,
):
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(num_classes, all_targets, all_preds):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        conf[int(t), int(p)] += 1
    return conf


def compute_metrics_from_confusion(conf):
    """
    Compute accuracy, macro-precision, macro-recall, macro-F1
    from a confusion matrix [num_classes, num_classes].
    """
    num_classes = conf.size(0)

    support_per_class = conf.sum(dim=1)
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    # Avoid division by zero
    precision_per_class = tp.float() / (tp.float() + fp.float() + 1e-8)
    recall_per_class = tp.float() / (tp.float() + fn.float() + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + 1e-8
    )

    accuracy = tp.sum().float() / conf.sum().float().clamp(min=1.0)
    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()

    return {
        "accuracy": accuracy.item(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# --- Responsible AI: per-class + disparity metrics ------------------------#

def compute_responsible_ai_report(conf, class_id_to_label=None):
    """
    Build a Responsible AI metrics report from a confusion matrix.

    Returns a dict with:
        - overall metrics (acc, macro precision/recall/F1)
        - per_class metrics: precision/recall/F1/support
        - disparity metrics across classes (min/max/ratio/diff)
    """
    num_classes = conf.size(0)

    support_per_class = conf.sum(dim=1)  # true counts per class
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    precision = tp.float() / (tp.float() + fp.float() + 1e-8)
    recall = tp.float() / (tp.float() + fn.float() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    overall = compute_metrics_from_confusion(conf)

    per_class = []
    for c in range(num_classes):
        label_name = (
            class_id_to_label.get(c, f"class_{c}")
            if class_id_to_label is not None
            else f"class_{c}"
        )
        per_class.append(
            {
                "class_id": int(c),
                "label": label_name,
                "support": int(support_per_class[c].item()),
                "precision": float(precision[c].item()),
                "recall": float(recall[c].item()),
                "f1": float(f1[c].item()),
            }
        )

    # Disparity metrics across classes (for transparency)
    f1_values = f1.detach().cpu().numpy()
    recall_values = recall.detach().cpu().numpy()

    f1_min = float(np.min(f1_values))
    f1_max = float(np.max(f1_values))
    f1_diff = f1_max - f1_min
    f1_ratio = float(f1_max / (f1_min + 1e-8))

    recall_min = float(np.min(recall_values))
    recall_max = float(np.max(recall_values))
    recall_diff = recall_max - recall_min
    recall_ratio = float(recall_max / (recall_min + 1e-8))

    disparity = {
        "f1_min": f1_min,
        "f1_max": f1_max,
        "f1_diff": f1_diff,
        "f1_ratio": f1_ratio,
        "recall_min": recall_min,
        "recall_max": recall_max,
        "recall_diff": recall_diff,
        "recall_ratio": recall_ratio,
    }

    report = {
        "overall": overall,
        "per_class": per_class,
        "disparity": disparity,
    }
    return report


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_targets, all_preds = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        preds = out.argmax(dim=-1).detach().cpu()
        targets = batch.y.view(-1).detach().cpu()

        all_preds.append(preds)
        all_targets.append(targets)

        # Explicitly drop references to batch tensors before next iteration
        del batch, out, loss, preds, targets    # <<< NEW

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    num_classes = int(all_targets.max().item() + 1)

    conf = compute_confusion_matrix(num_classes, all_targets, all_preds)
    metrics = compute_metrics_from_confusion(conf)

    avg_loss = total_loss / max(total_examples, 1)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets, all_preds = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        preds = out.argmax(dim=-1).detach().cpu()
        targets = batch.y.view(-1).detach().cpu()

        all_preds.append(preds)
        all_targets.append(targets)

        # Drop references asap
        del batch, out, loss, preds, targets    # <<< NEW

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_precision": 0.0,
                "macro_recall": 0.0, "macro_f1": 0.0}

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    conf = compute_confusion_matrix(num_classes, all_targets, all_preds)
    metrics = compute_metrics_from_confusion(conf)

    avg_loss = total_loss / max(total_examples, 1)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate_with_confusion(model, loader, criterion, device, num_classes):
    """
    Detailed evaluation used for final test/RAI report.
    Returns (metrics_dict, confusion_matrix_tensor).
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets, all_preds = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        preds = out.argmax(dim=-1).detach().cpu()
        targets = batch.y.view(-1).detach().cpu()

        all_preds.append(preds)
        all_targets.append(targets)

        del batch, out, loss, preds, targets    # <<< NEW

    if total_examples == 0:
        conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
        metrics = {"loss": 0.0, "accuracy": 0.0, "macro_precision": 0.0,
                   "macro_recall": 0.0, "macro_f1": 0.0}
        return metrics, conf

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    conf = compute_confusion_matrix(num_classes, all_targets, all_preds)
    metrics = compute_metrics_from_confusion(conf)

    avg_loss = total_loss / max(total_examples, 1)
    metrics["loss"] = avg_loss
    return metrics, conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GGNN CWE multi-class classifier (train/val/test)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing processed_graphs_multi_class.pkl, "
             "vocabularies.pkl, and dataset_metadata.pkl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (tune for 12GB GPU; 32–64 is usually safe).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension for each of the 3 node feature types.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=200,
        help="Hidden dimension for GGNN and MLP.",
    )
    parser.add_argument(
        "--num_ggnn_layers",
        type=int,
        default=3,
        help="Number of GGNN layers.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=8,
        help="Propagation steps per GGNN layer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of data used for training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Proportion of data used for validation.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Proportion of data used for testing.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 is safest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save best model checkpoint.",
    )
    args = parser.parse_args()

    # Basic sanity check on splits
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, got {total_ratio}"
        )

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    graphs_path = data_dir / "processed_graphs_multi_class.pkl"
    vocab_path = data_dir / "vocabularies.pkl"
    metadata_path = data_dir / "dataset_metadata.pkl"

    if not graphs_path.exists():
        raise FileNotFoundError(f"Processed graphs not found at {graphs_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabularies not found at {vocab_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    # -----------------------------------------------------------------------
    # 1. Load processed graphs and metadata
    # -----------------------------------------------------------------------
    print("\nLoading processed graphs...")
    with open(graphs_path, "rb") as f:
        all_graphs_raw = pickle.load(f)
    print(f"Loaded {len(all_graphs_raw)} graphs from {graphs_path}")

    print("Loading metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    vocab_sizes = metadata["vocab_sizes"]
    total_samples = metadata["total_samples"]
    idx_to_cwe = metadata.get("idx_to_cwe", {})  # <-- NEW
    print(f"  vocab_sizes: {vocab_sizes}")
    print(f"  total_samples (metadata): {total_samples}")
    print(f"  idx_to_cwe entries: {len(idx_to_cwe)}")  # optional debug


    # -------------------------------------------------------------------
    # Keep only top-30 classes by frequency and remap labels to [0, 29]
    # -------------------------------------------------------------------
    print("\nFiltering to top 30 CWE classes by frequency...")

    # Count original labels
    label_counts = Counter(int(g["cwe_idx"]) for g in all_graphs_raw)
    most_common_30 = label_counts.most_common(30)  # [(cwe_idx, count), ...]
    top_classes = [c for c, _ in most_common_30]
    top_class_set = set(top_classes)

    print("Top 30 classes (original indices and counts):")
    for c, cnt in most_common_30:
        print(f"  CWE_idx {c}: {cnt} samples")

    # Map original class indices -> new compact indices 0..29
    old_to_new = {old: new for new, (old, _) in enumerate(most_common_30)}
    # Map new compact indices 0..29 back to original CWE index
    new_to_old = {new: old for old, new in old_to_new.items()}  # <-- NEW


    filtered_graphs_raw = []
    for g in all_graphs_raw:
        orig_y = int(g["cwe_idx"])
        if orig_y not in old_to_new:
            continue  # drop rare classes

        new_g = dict(g)               # shallow copy to avoid mutating original
        new_g["cwe_idx"] = old_to_new[orig_y]
        filtered_graphs_raw.append(new_g)

    print(
        f"Kept {len(filtered_graphs_raw)} graphs "
        f"out of {len(all_graphs_raw)} after top-30 filtering."
    )

    all_graphs_raw = filtered_graphs_raw
    num_classes = len(old_to_new)    # should be 30 now
    print(f"New num_classes (top-30 only): {num_classes}")

    # -----------------------------------------------------------------------
    # 2. Build PyG Data objects
    # -----------------------------------------------------------------------
    print("\nConverting graphs to PyG Data objects...")
    pyg_graphs = []
    labels = []
    skipped = 0
    for g in all_graphs_raw:
        data = build_pyg_graph(g)
        if data is None:
            skipped += 1
            continue
        pyg_graphs.append(data)
        labels.append(int(g["cwe_idx"]))
    print(f"Converted {len(pyg_graphs)} graphs, skipped {skipped} empty graphs.")

    # -----------------------------------------------------------------------
    # 3. Stratified train/val/test split
    # -----------------------------------------------------------------------
    print("\nCreating stratified train/val/test split...")
    train_idx, val_idx, test_idx = stratified_split(
        pyg_graphs,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"Split sizes -> train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        pyg_graphs,
        train_idx,
        val_idx,
        test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # -----------------------------------------------------------------------
    # 4. Build model, optimizer, and loss
    # -----------------------------------------------------------------------
    print("\nInitializing GGNN model...")
    model = GGNN_CWE_Classifier(
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_ggnn_layers=args.num_ggnn_layers,
        num_steps=args.num_steps,
        dropout=args.dropout,
        use_edge_weights=False, 
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    rai_report_path = os.path.join(args.checkpoint_dir, "responsible_ai_report.json")  # <<< NEW

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    best_val_f1 = 0.0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Train F1 (macro): {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1 (macro): {val_metrics['macro_f1']:.4f}"
        )

        # Save best checkpoint by validation macro F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "vocab_sizes": vocab_sizes,
                    "num_classes": num_classes,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  --> New best model saved to {best_ckpt_path}")

        # ---------------------------------------------------------------
        # Cache clearing to avoid GPU/CPU RAM buildup across epochs
        # ---------------------------------------------------------------
        gc.collect()                       # free Python-side objects
        if device.type == "cuda":
            torch.cuda.empty_cache()       # release cached CUDA memory
            torch.cuda.ipc_collect()       # clean up inter-process memory

    # -----------------------------------------------------------------------
    # 6. Final test evaluation using best checkpoint
    # -----------------------------------------------------------------------
    print("\nLoading best checkpoint for final test evaluation...")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    else:
        print("WARNING: No checkpoint found; using last epoch model for test.")

    # Use detailed evaluation to get confusion matrix for RAI metrics
    test_metrics, test_conf = evaluate_with_confusion(
        model, test_loader, criterion, device, num_classes
    )

    print("\n=== FINAL TEST METRICS ===")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Acc:  {test_metrics['accuracy']:.4f}")
    print(f"Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Test Macro F1:        {test_metrics['macro_f1']:.4f}")

    # -----------------------------------------------------------------------
    # 7. Responsible AI metrics report
    # -----------------------------------------------------------------------
    print("\n=== RESPONSIBLE AI METRICS (PER-CLASS & DISPARITIES) ===")

    # Map compact class IDs (0..29) back to original CWE IDs (e.g., "CWE-79")
    if idx_to_cwe and 'new_to_old' in locals():
        class_id_to_label = {
            new_id: str(idx_to_cwe.get(old_id, f"orig_CWE_{old_id}"))
            for new_id, old_id in new_to_old.items()
        }
    else:
        class_id_to_label = None

    rai_report = compute_responsible_ai_report(
        test_conf, class_id_to_label=class_id_to_label
    )

    # Print some high-level fairness-ish stats
    print("Overall:", rai_report["overall"])
    print("Disparity:", rai_report["disparity"])

    # Show worst 5 classes by F1 (so you can inspect tail failures)
    sorted_by_f1 = sorted(
        rai_report["per_class"], key=lambda x: x["f1"]
    )
    print("\nLowest 5 classes by F1:")
    for item in sorted_by_f1[:5]:
        print(
            f"  {item['label']} (id={item['class_id']}): "
            f"support={item['support']}, "
            f"precision={item['precision']:.3f}, "
            f"recall={item['recall']:.3f}, "
            f"f1={item['f1']:.3f}"
        )

    print("\nHighest 5 classes by F1:")
    for item in sorted_by_f1[-5:]:
        print(
            f"  {item['label']} (id={item['class_id']}): "
            f"support={item['support']}, "
            f"precision={item['precision']:.3f}, "
            f"recall={item['recall']:.3f}, "
            f"f1={item['f1']:.3f}"
        )

    # Save full RAI report to JSON for later analysis
    with open(rai_report_path, "w", encoding="utf-8") as f:
        json.dump(rai_report, f, indent=2)
    print(f"\nResponsible AI report saved to: {rai_report_path}")


if __name__ == "__main__":
    main()

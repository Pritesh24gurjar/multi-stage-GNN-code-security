#!/usr/bin/env python
"""
Model comparison + Responsible AI analysis for GGNN CWE classifiers.

Usage example:

python analyze_models.py \
    --data_dir data/splits \
    --checkpoints \
        path/to/initial_best_model.pt \
        path/to/exp1_baseline_best_model.pt \
        path/to/exp6_more_steps_best_model.pt \
    --analysis_dir multiclass_experiment/analysis \
    --batch_size 64 \
    --device cuda
"""

from __future__ import annotations

import os
import argparse
import random
from pathlib import Path
import pickle
from collections import Counter
import json
import gc

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

# Import your project model
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
    """
    features = graph_dict["features"]
    ipag_edges = graph_dict["ipag_edges"]
    y = graph_dict["cwe_idx"]

    if features.size == 0:
        return None

    x = torch.from_numpy(features).long()
    num_nodes = x.shape[0]

    src, dst = [], []
    for e in ipag_edges:
        try:
            s = int(e["source"])
            t = int(e["target"])
        except (ValueError, TypeError):
            continue

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
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.long),
        num_nodes=num_nodes,
    )

    data.cwe_id = graph_dict.get("cwe_id")
    data.cve_id = graph_dict.get("cve_id")
    data.lang = graph_dict.get("lang")

    return data


def stratified_split(graphs, labels, train_ratio=0.8, val_ratio=0.1,
                     test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    label_to_indices = {}
    for idx, y in enumerate(labels):
        label_to_indices.setdefault(int(y), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for _, idxs in label_to_indices.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
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


def create_dataloaders(graphs, train_idx, val_idx, test_idx,
                       batch_size=64, num_workers=0):
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
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
    num_classes = conf.size(0)

    support_per_class = conf.sum(dim=1)
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

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


def compute_responsible_ai_report(conf, class_id_to_label=None):
    num_classes = conf.size(0)

    support_per_class = conf.sum(dim=1)
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

    return {
        "overall": overall,
        "per_class": per_class,
        "disparity": disparity,
    }


@torch.no_grad()
def evaluate_with_confusion(model, loader, criterion, device, num_classes):
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

        del batch, out, loss, preds, targets

    if total_examples == 0:
        conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
        metrics = {"loss": 0.0, "accuracy": 0.0,
                   "macro_precision": 0.0,
                   "macro_recall": 0.0,
                   "macro_f1": 0.0}
        return metrics, conf

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    conf = compute_confusion_matrix(num_classes, all_targets, all_preds)
    metrics = compute_metrics_from_confusion(conf)
    metrics["loss"] = total_loss / max(total_examples, 1)

    return metrics, conf


# ---------------------------------------------------------------------------
# Plotting helpers (existing)
# ---------------------------------------------------------------------------

def plot_overall_metrics(results, analysis_dir):
    model_names = list(results.keys())
    macro_f1 = [results[m]["test_metrics"]["macro_f1"] for m in model_names]
    accuracy = [results[m]["test_metrics"]["accuracy"] for m in model_names]
    macro_prec = [results[m]["test_metrics"]["macro_precision"] for m in model_names]
    macro_rec = [results[m]["test_metrics"]["macro_recall"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    ax.bar(x - 0.5 * width, macro_prec, width, label="Macro Precision")
    ax.bar(x + 0.5 * width, macro_rec, width, label="Macro Recall")
    ax.bar(x + 1.5 * width, macro_f1, width, label="Macro F1")

    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(analysis_dir, "overall_metrics_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rai_disparities(results, analysis_dir):
    model_names = list(results.keys())
    f1_diff = [results[m]["rai_report"]["disparity"]["f1_diff"] for m in model_names]
    recall_diff = [results[m]["rai_report"]["disparity"]["recall_diff"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, f1_diff, width)
    ax1.set_ylabel("F1 max - F1 min")
    ax1.set_title("RAI Disparity: F1 Spread Across Classes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "rai_f1_disparity.png"), dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, recall_diff, width)
    ax2.set_ylabel("Recall max - Recall min")
    ax2.set_title("RAI Disparity: Recall Spread Across Classes")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "rai_recall_disparity.png"), dpi=200)
    plt.close(fig2)


def plot_per_class_f1(results, analysis_dir, top_k=10):
    for model_name, res in results.items():
        per_class = res["rai_report"]["per_class"]
        per_class_sorted = sorted(per_class, key=lambda x: x["f1"])

        bottom = per_class_sorted[:top_k]
        top = per_class_sorted[-top_k:]

        fig_b, ax_b = plt.subplots(figsize=(10, 5))
        labels_b = [c["label"] for c in bottom]
        f1_b = [c["f1"] for c in bottom]
        x_b = np.arange(len(bottom))
        ax_b.bar(x_b, f1_b)
        ax_b.set_ylabel("F1")
        ax_b.set_title(f"{model_name}: Lowest {top_k} Classes by F1")
        ax_b.set_xticks(x_b)
        ax_b.set_xticklabels(labels_b, rotation=45, ha="right")
        ax_b.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(analysis_dir, f"{model_name}_lowest_{top_k}_f1.png"),
            dpi=200,
        )
        plt.close(fig_b)

        fig_t, ax_t = plt.subplots(figsize=(10, 5))
        labels_t = [c["label"] for c in top]
        f1_t = [c["f1"] for c in top]
        x_t = np.arange(len(top))
        ax_t.bar(x_t, f1_t)
        ax_t.set_ylabel("F1")
        ax_t.set_title(f"{model_name}: Highest {top_k} Classes by F1")
        ax_t.set_xticks(x_t)
        ax_t.set_xticklabels(labels_t, rotation=45, ha="right")
        ax_t.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(analysis_dir, f"{model_name}_highest_{top_k}_f1.png"),
            dpi=200,
        )
        plt.close(fig_t)


# ---------------------------------------------------------------------------
# NEW Plotting helpers for richer analysis
# ---------------------------------------------------------------------------

def _get_sorted_per_class(results):
    """
    Return:
      classes: list of class_ids
      per_model_per_class: dict[model_name] -> list of per_class items sorted by class_id
    """
    model_names = list(results.keys())
    per_model = {}
    for name in model_names:
        pcs = results[name]["rai_report"]["per_class"]
        pcs_sorted = sorted(pcs, key=lambda x: x["class_id"])
        per_model[name] = pcs_sorted
    classes = [c["class_id"] for c in next(iter(per_model.values()))]
    return classes, per_model


def plot_per_class_f1_comparison(results, analysis_dir):
    classes, per_model = _get_sorted_per_class(results)
    model_names = list(results.keys())
    num_models = len(model_names)

    x = np.arange(len(classes))
    total_width = 0.8
    width = total_width / max(num_models, 1)
    offsets = np.linspace(-total_width/2 + width/2, total_width/2 - width/2, num_models)

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, name in enumerate(model_names):
        f1s = [c["f1"] for c in per_model[name]]
        ax.bar(x + offsets[i], f1s, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1 Comparison Across Models")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "per_class_f1_comparison.png"), dpi=200)
    plt.close(fig)


def _get_best_and_worst_models(results):
    """
    Returns (worst_model_name, best_model_name) by macro_f1.
    """
    items = list(results.items())
    items_sorted = sorted(items, key=lambda kv: kv[1]["test_metrics"]["macro_f1"])
    worst = items_sorted[0][0]
    best = items_sorted[-1][0]
    return worst, best


def plot_per_class_f1_delta(results, analysis_dir):
    classes, per_model = _get_sorted_per_class(results)
    worst, best = _get_best_and_worst_models(results)

    baseline_f1 = np.array([c["f1"] for c in per_model[worst]])
    best_f1 = np.array([c["f1"] for c in per_model[best]])
    delta = best_f1 - baseline_f1

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(classes, delta)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Class ID")
    ax.set_ylabel(f"F1 Improvement ({best} - {worst})")
    ax.set_title("Per-Class F1 Improvement (Best vs Worst Model)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out_path = os.path.join(analysis_dir, "per_class_f1_delta_best_vs_worst.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rai_radar_chart(results, analysis_dir):
    labels = ["F1 Min", "F1 Max", "F1 Diff", "F1 Ratio",
              "Recall Min", "Recall Max", "Recall Diff", "Recall Ratio"]
    num_vars = len(labels)

    model_names = list(results.keys())
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for name in model_names:
        d = results[name]["rai_report"]["disparity"]
        vals = [
            d["f1_min"], d["f1_max"], d["f1_diff"], d["f1_ratio"],
            d["recall_min"], d["recall_max"], d["recall_diff"], d["recall_ratio"],
        ]
        vals += vals[:1]
        ax.plot(angles, vals, label=name)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("RAI Disparity Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out_path = os.path.join(analysis_dir, "rai_radar_chart.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overall_metrics_trend(results, analysis_dir):
    model_names = list(results.keys())
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in metrics:
        series = [results[name]["test_metrics"][m] for name in model_names]
        ax.plot(model_names, series, marker="o", label=m)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics Trend Across Models")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(analysis_dir, "overall_metrics_trend.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_fnr_comparison(results, analysis_dir):
    classes, per_model = _get_sorted_per_class(results)
    worst, best = _get_best_and_worst_models(results)

    worst_rec = np.array([c["recall"] for c in per_model[worst]])
    best_rec = np.array([c["recall"] for c in per_model[best]])

    worst_fnr = 1.0 - worst_rec
    best_fnr = 1.0 - best_rec

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(classes, worst_fnr, label=f"{worst} FNR")
    ax.plot(classes, best_fnr, label=f"{best} FNR")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("False Negative Rate (1 - recall)")
    ax.set_title("Per-Class False Negative Rate: Best vs Worst Model")
    plt.xticks(rotation=90)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(analysis_dir, "per_class_fnr_comparison_best_vs_worst.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare GGNN CWE classifiers and plot RAI metrics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with processed_graphs_multi_class.pkl and dataset_metadata.pkl",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of .pt checkpoint files to compare.",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="analysis",
        help="Directory to save comparison graphs and JSON summaries.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Num DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for stratified split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    os.makedirs(args.analysis_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    graphs_path = data_dir / "processed_graphs_multi_class.pkl"
    metadata_path = data_dir / "dataset_metadata.pkl"

    if not graphs_path.exists():
        raise FileNotFoundError(f"Processed graphs not found at {graphs_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    print("\nLoading processed graphs...")
    with open(graphs_path, "rb") as f:
        all_graphs_raw = pickle.load(f)
    print(f"Loaded {len(all_graphs_raw)} graphs")

    print("Loading metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    vocab_sizes = metadata["vocab_sizes"]
    idx_to_cwe = metadata.get("idx_to_cwe", {})

    print("\nFiltering to top 30 CWE classes by frequency...")
    label_counts = Counter(int(g["cwe_idx"]) for g in all_graphs_raw)
    most_common_30 = label_counts.most_common(30)
    old_to_new = {old: new for new, (old, _) in enumerate(most_common_30)}
    new_to_old = {new: old for old, new in old_to_new.items()}

    filtered_graphs_raw = []
    for g in all_graphs_raw:
        orig_y = int(g["cwe_idx"])
        if orig_y not in old_to_new:
            continue
        new_g = dict(g)
        new_g["cwe_idx"] = old_to_new[orig_y]
        filtered_graphs_raw.append(new_g)

    print(f"Kept {len(filtered_graphs_raw)} graphs after top-30 filtering.")
    num_classes = len(old_to_new)
    print(f"num_classes (top-30): {num_classes}")

    pyg_graphs, labels = [], []
    skipped = 0
    for g in filtered_graphs_raw:
        data = build_pyg_graph(g)
        if data is None:
            skipped += 1
            continue
        pyg_graphs.append(data)
        labels.append(int(g["cwe_idx"]))
    print(f"Converted {len(pyg_graphs)} graphs, skipped {skipped} empty graphs.")

    train_idx, val_idx, test_idx = stratified_split(
        pyg_graphs,
        labels,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
    )
    print(
        f"Split sizes -> train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    _, _, test_loader = create_dataloaders(
        pyg_graphs,
        train_idx,
        val_idx,
        test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if idx_to_cwe:
        class_id_to_label = {
            new_id: str(idx_to_cwe.get(old_id, f"orig_CWE_{old_id}"))
            for new_id, old_id in new_to_old.items()
        }
    else:
        class_id_to_label = None

    results = {}
    criterion = nn.CrossEntropyLoss()

    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        model_name = ckpt_path.stem

        print(f"\n=== Evaluating checkpoint: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=device)

        ckpt_num_classes = ckpt["num_classes"]
        if ckpt_num_classes != num_classes:
            raise ValueError(
                f"Checkpoint num_classes={ckpt_num_classes} "
                f"does not match dataset num_classes={num_classes}"
            )

        ckpt_args = ckpt["args"]
        embedding_dim = ckpt_args["embedding_dim"]
        hidden_dim = ckpt_args["hidden_dim"]
        num_ggnn_layers = ckpt_args["num_ggnn_layers"]
        num_steps = ckpt_args["num_steps"]
        dropout = ckpt_args["dropout"]

        model = GGNN_CWE_Classifier(
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_ggnn_layers=num_ggnn_layers,
            num_steps=num_steps,
            dropout=dropout,
            use_edge_weights=False,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics, test_conf = evaluate_with_confusion(
            model, test_loader, criterion, device, num_classes
        )
        rai_report = compute_responsible_ai_report(
            test_conf, class_id_to_label=class_id_to_label
        )

        print("Test metrics:", test_metrics)
        print("RAI disparity:", rai_report["disparity"])

        results[model_name] = {
            "checkpoint_path": str(ckpt_path),
            "test_metrics": test_metrics,
            "rai_report": rai_report,
        }

        with open(
            os.path.join(args.analysis_dir, f"{model_name}_rai_report.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(rai_report, f, indent=2)

        del model, ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    summary_path = os.path.join(args.analysis_dir, "models_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined comparison summary saved to: {summary_path}")

    print("\nGenerating comparison graphs...")

    # Original plots
    plot_overall_metrics(results, args.analysis_dir)
    plot_rai_disparities(results, args.analysis_dir)
    plot_per_class_f1(results, args.analysis_dir, top_k=10)

    # New richer analysis plots
    plot_per_class_f1_comparison(results, args.analysis_dir)
    plot_per_class_f1_delta(results, args.analysis_dir)
    plot_rai_radar_chart(results, args.analysis_dir)
    plot_overall_metrics_trend(results, args.analysis_dir)
    plot_per_class_fnr_comparison(results, args.analysis_dir)

    print(f"Graphs saved under: {args.analysis_dir}")


if __name__ == "__main__":
    main()

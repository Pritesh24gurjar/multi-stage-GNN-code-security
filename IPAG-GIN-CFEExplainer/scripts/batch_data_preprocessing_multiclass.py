import os
import pandas as pd
from pathlib import Path
import pickle
from ipag_gin.graph.build_language import LanguageBuilder
from ipag_gin.graph.ipag_builder import IPAGBuilder
from ipag_gin.graph.vocabulary_builder import VocabularyBuilder, NodeFeatureEncoder
from tqdm import tqdm
import itertools
import argparse
import numpy as np


def discover_data_files(src_root, cwes=None, lang_filter=None):
    """
    Recursively find all data.csv files along with their CWE/CVE/label.
    Returns a list of dicts with complete metadata.
    
    Args:
        src_root (str): Root directory containing CWE folders
        cwes (list): List of specific CWE folder patterns to process (e.g., ['CWE-119', 'CWE-20'])
                     If None, process all CWE-* folders
    
    Returns:
        list: List of dicts with code samples and metadata
    """
    data_list = []
    root = Path(src_root)
    
    # Get CWE directories
    if cwes:
        cwe_dirs = itertools.chain.from_iterable(root.glob(folder) for folder in cwes)
    else:
        cwe_dirs = root.glob("CWE-*")
    
    cwe_dirs = list(cwe_dirs)  # Convert to list to iterate multiple times if needed
    print(f"Found {len(cwe_dirs)} CWE directories to process")
    
    for cwe_dir in cwe_dirs:
        if not cwe_dir.is_dir():
            continue
        
        cwe_id = cwe_dir.name
        cve_count = 0
        
        for cve_dir in cwe_dir.iterdir():
            if not cve_dir.is_dir():
                continue
                
            cve_id = cve_dir.name
            
            for label in ["0", "1"]:
                label_dir = cve_dir / label
                if not label_dir.is_dir():
                    continue
                
                data_path = label_dir / "data.csv"
                if not data_path.exists():
                    continue
                
                try:
                    df = pd.read_csv(data_path)
                    
                    # Normalize language names
                    df['lang'] = df['lang'].replace("c++", "cpp").str.strip().str.lower()
                    
                    for idx, row in df.iterrows():
                        lang = row["lang"].replace("c++", "cpp").strip().lower()
                        # Filter by language if specified
                        if lang_filter:
                            if isinstance(lang_filter, str):
                                lang_filter = [lang_filter]
                            if lang not in lang_filter:
                                continue  # Skip this sample
                        data_list.append({
                            "cwe_id": cwe_id,
                            "cve_id": cve_id,
                            "vul": int(row["vul"]),
                            "lang": row["lang"],
                            "code": str(row["code"]),  # Ensure string
                            "csv_path": str(data_path),
                            "row_idx": idx
                        })
                    
                    cve_count += 1
                    
                except Exception as e:
                    print(f"Warning: Error reading {data_path}: {e}")
                    continue
        
        print(f"  {cwe_id}: {cve_count} CVEs processed")
    
    return data_list


def create_cwe_mapping(data_list):
    """
    Create CWE ID to integer index mapping for classification.
    
    Args:
        data_list: List of data dicts with 'cwe_id' field
    
    Returns:
        tuple: (cwe_to_idx, idx_to_cwe) dictionaries
    """
    unique_cwes = sorted(set(item['cwe_id'] for item in data_list))
    cwe_to_idx = {cwe: idx for idx, cwe in enumerate(unique_cwes)}
    idx_to_cwe = {idx: cwe for cwe, idx in cwe_to_idx.items()}
    
    print(f"\nCWE Mapping created:")
    print(f"  Total CWE classes: {len(cwe_to_idx)}")
    for cwe, idx in sorted(cwe_to_idx.items(), key=lambda x: x[1]):
        print(f"    {idx}: {cwe}")
    
    return cwe_to_idx, idx_to_cwe


def print_dataset_statistics(df_all, cwe_to_idx):
    """Print comprehensive dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal samples: {len(df_all)}")
    print(f"Total CWE classes: {len(cwe_to_idx)}")
    
    # Vulnerability distribution
    vul_counts = df_all['vul'].value_counts()
    print(f"\nVulnerability Distribution:")
    print(f"  Vulnerable (1): {vul_counts.get(1, 0)} ({vul_counts.get(1, 0)/len(df_all)*100:.1f}%)")
    print(f"  Non-vulnerable (0): {vul_counts.get(0, 0)} ({vul_counts.get(0, 0)/len(df_all)*100:.1f}%)")
    
    # CWE distribution
    print(f"\nCWE Class Distribution:")
    cwe_counts = df_all['cwe_id'].value_counts()
    for cwe_id, count in cwe_counts.items():
        print(f"  {cwe_id}: {count} samples ({count/len(df_all)*100:.1f}%)")
    
    # Language distribution
    print(f"\nLanguage Distribution:")
    lang_counts = df_all['lang'].value_counts()
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count} samples ({count/len(df_all)*100:.1f}%)")
    
    print("="*70)


def main(src_root, chunk_size, cwes=None, output_dir=".", min_label_freq=2):
    """
    Main preprocessing pipeline.
    
    Args:
        src_root (str): Root directory containing CWE folders
        chunk_size (int): Number of samples per processing chunk
        cwes (list): List of specific CWE patterns to process (optional)
        output_dir (str): Directory to save output files
        min_label_freq (int): Minimum frequency for labels in vocabulary
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # =======================
    # 1. Discover and Load
    # =======================
    print("="*70)
    print("STEP 1: DISCOVERING CODE SAMPLES")
    print("="*70)
    
    all_entries = discover_data_files(src_root, cwes=cwes, lang_filter=args.lang_filter)
    
    if not all_entries:
        print("ERROR: No code samples found!")
        return
    
    print(f"\nTotal code samples discovered: {len(all_entries)}")
    
    # Create CWE mapping
    cwe_to_idx, idx_to_cwe = create_cwe_mapping(all_entries)
    
    # Convert to DataFrame
    df_all = pd.DataFrame(all_entries)
    
    # Add CWE index for classification
    df_all['cwe_idx'] = df_all['cwe_id'].map(cwe_to_idx)
    
    # Print statistics
    print_dataset_statistics(df_all, cwe_to_idx)
    
    # Calculate chunks
    num_samples = len(df_all)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    print(f"\nProcessing in {num_chunks} chunks of {chunk_size} samples")
    
    # =======================
    # 2. First Pass: Build Vocabulary and Cache IPAGs
    # =======================
    print("\n" + "="*70)
    print("STEP 2: BUILDING VOCABULARY AND EXTRACTING IPAGs")
    print("="*70)
    
    vocab_builder = VocabularyBuilder()
    ipag_cache_dir = output_dir / "ipag_cache"
    ipag_cache_dir.mkdir(exist_ok=True)
    
    for chunk_idx in tqdm(range(num_chunks), desc="Pass 1 - Vocabulary Building"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)
        chunk = df_all.iloc[start_idx:end_idx].copy()
        
        # Get unique languages in this chunk
        chunk_langs = chunk["lang"].unique().tolist()
        
        try:
            # Build language map
            lang_builder = LanguageBuilder(chunk_langs)
            lang_map = lang_builder.build()
            
            # Build IPAGs for this chunk
            ipag_builder = IPAGBuilder(
                source=chunk["code"].reset_index(drop=True),
                language=chunk["lang"].reset_index(drop=True),
                lang_map=lang_map
            )
            
            ipag_nodes, ipag_edges = ipag_builder.build()
            
            # Update vocabulary builder
            vocab_builder.update(ipag_nodes)
            
            # Save IPAGs to cache
            cache_file = ipag_cache_dir / f"ipag_chunk_{chunk_idx}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "ipag_nodes": ipag_nodes,
                    "ipag_edges": ipag_edges,
                    "chunk_start": start_idx,
                    "chunk_end": end_idx
                }, f)
                
        except Exception as e:
            print(f"\nError processing chunk {chunk_idx}: {e}")
            # Save empty cache to maintain chunk numbering
            with open(ipag_cache_dir / f"ipag_chunk_{chunk_idx}.pkl", "wb") as f:
                pickle.dump({
                    "ipag_nodes": [[] for _ in range(len(chunk))],
                    "ipag_edges": [[] for _ in range(len(chunk))],
                    "chunk_start": start_idx,
                    "chunk_end": end_idx
                }, f)
            continue
    
    # Finalize vocabulary
    print("\nFinalizing vocabulary...")
    vocab_builder.finalize(min_freq=min_label_freq)
    
    vocab_path = output_dir / "vocabularies.pkl"
    vocab_builder.save(vocab_path)
    print(f"Vocabularies saved to {vocab_path}")
    
    # =======================
    # 3. Second Pass: Encode Features and Save Final Dataset
    # =======================
    print("\n" + "="*70)
    print("STEP 3: ENCODING NODE FEATURES")
    print("="*70)
    
    # Reload vocabulary
    vocab_builder = VocabularyBuilder.load(vocab_path)
    encoder = NodeFeatureEncoder(vocab_builder)
    
    print(f"\nFeature dimension: {encoder.get_feature_dim(use_one_hot=False)}")
    print(f"Vocabulary sizes: {encoder.get_vocab_sizes()}")
    
    final_graphs = []
    skipped_count = 0
    
    for chunk_idx in tqdm(range(num_chunks), desc="Pass 2 - Feature Encoding"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_samples)
        chunk = df_all.iloc[start_idx:end_idx]
        
        # Load cached IPAGs
        cache_file = ipag_cache_dir / f"ipag_chunk_{chunk_idx}.pkl"
        
        try:
            with open(cache_file, "rb") as f:
                ipag_chunk = pickle.load(f)
            
            ipag_nodes = ipag_chunk["ipag_nodes"]
            ipag_edges = ipag_chunk["ipag_edges"]
            
            # Encode each graph
            for i, (nodes, edges) in enumerate(zip(ipag_nodes, ipag_edges)):
                if not nodes:  # Skip empty graphs
                    skipped_count += 1
                    continue
                
                features = encoder.encode(nodes, use_one_hot=False)
                
                graph_dict = {
                    "ipag_nodes": nodes,
                    "ipag_edges": edges,
                    "features": features,  # [num_nodes, 3]
                    "num_nodes": len(nodes),
                    "num_edges": len(edges),
                    "cwe_id": chunk.iloc[i]["cwe_id"],
                    "cwe_idx": chunk.iloc[i]["cwe_idx"],  # For classification
                    "cve_id": chunk.iloc[i]["cve_id"],
                    "vul": chunk.iloc[i]["vul"],
                    "lang": chunk.iloc[i]["lang"],
                    "code": chunk.iloc[i]["code"],
                }
                final_graphs.append(graph_dict)
                
        except Exception as e:
            print(f"\nError encoding chunk {chunk_idx}: {e}")
            continue
    
    # =======================
    # 4. Save Final Dataset
    # =======================
    print("\n" + "="*70)
    print("STEP 4: SAVING PROCESSED DATASET")
    print("="*70)
    
    # Save processed graphs
    graphs_path = output_dir / "processed_graphs_multi_class.pkl"
    with open(graphs_path, "wb") as f:
        pickle.dump(final_graphs, f)
    
    print(f"\nProcessed graphs saved to {graphs_path}")
    print(f"  Total graphs: {len(final_graphs)}")
    print(f"  Skipped (empty): {skipped_count}")
    
    # Save metadata
    metadata = {
        "cwe_to_idx": cwe_to_idx,
        "idx_to_cwe": idx_to_cwe,
        "num_classes": len(cwe_to_idx),
        "vocab_sizes": encoder.get_vocab_sizes(),
        "feature_dim": encoder.get_feature_dim(use_one_hot=False),
        "total_samples": len(final_graphs),
        "skipped_samples": skipped_count,
        "language_distribution": df_all['lang'].value_counts().to_dict(),
        "cwe_distribution": df_all['cwe_id'].value_counts().to_dict(),
    }
    
    metadata_path = output_dir / "dataset_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\nMetadata saved to {metadata_path}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {graphs_path}")
    print(f"  - {vocab_path}")
    print(f"  - {metadata_path}")
    print(f"  - {ipag_cache_dir}/ (intermediate cache)")
    
    print(f"\nDataset summary:")
    print(f"  Total graphs: {len(final_graphs)}")
    print(f"  CWE classes: {len(cwe_to_idx)}")
    print(f"  Feature dimension: {metadata['feature_dim']}")
    print(f"  Vocabulary sizes: {metadata['vocab_sizes']}")
    
    print("\n✓ Ready for model training!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BigVul dataset for multi-class CWE classification"
    )
    
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Root directory containing CWE-* folders"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Number of samples to process per chunk (default: 500)"
    )
    
    parser.add_argument(
        "--cwes",
        type=str,
        nargs='+',
        default=None,
        help="Specific CWE folders to process (e.g., CWE-119 CWE-20). If not provided, process all."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output files (default: current directory)"
    )
    
    parser.add_argument(
        "--min_label_freq",
        type=int,
        default=2,
        help="Minimum frequency for labels in vocabulary (default: 2)"
    )
    parser.add_argument(
    "--lang_filter",
    type=str,
    nargs='+',
    default=None,
    help="Only include these languages (e.g., --lang_filter c)"
    )
    
    args = parser.parse_args()
    
    main(
        src_root=args.src_root,
        chunk_size=args.chunk_size,
        cwes=args.cwes,
        output_dir=args.output_dir,
        min_label_freq=args.min_label_freq
    )

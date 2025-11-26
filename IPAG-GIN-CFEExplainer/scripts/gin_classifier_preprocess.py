import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import gc
import sys


def get_file_count(cwe, cve_prefix, src_base):
    """
    Count total files without loading them
    """
    src_path = Path(src_base) / cwe
    total_files = 0
    
    cve_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name.startswith(cve_prefix)]
    
    for subdir in cve_dirs:
        label_dir_negative = subdir / "0"
        if label_dir_negative.exists():
            total_files += len(list(label_dir_negative.glob("*.pkl")))
        
        label_dir_positive = subdir / "1"
        if label_dir_positive.exists():
            total_files += len(list(label_dir_positive.glob("*.pkl")))
    
    return total_files, len(cve_dirs)


def load_sample_metadata(pkl_file, cve_id, label):
    """
    Load only metadata (not full graph data) to save memory
    """
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            # Just check if valid, don't store the full data
            if isinstance(data, list):
                data = data[0]
            
            # Store only path and label, not the actual features
            return {
                'filepath': str(pkl_file),
                'label': label,
                'filename': f"{cve_id}_{pkl_file.name}"
            }
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        return None


def create_metadata_index(cwe, cve_prefix, src_base):
    """
    Create lightweight metadata index instead of loading all data
    """
    src_path = Path(src_base) / cwe
    
    print(f"\nCreating metadata index...")
    print(f"Source directory: {src_path}\n")
    
    all_samples = []
    positive_count = 0
    negative_count = 0
    
    # Iterate through all CVE subdirectories
    cve_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name.startswith(cve_prefix)]
    
    for subdir in tqdm(cve_dirs, desc='Indexing CVE directories'):
        cve_id = subdir.name
        
        # Process negative samples (label 0)
        label_dir_negative = subdir / "0"
        if label_dir_negative.exists():
            for pkl_file in label_dir_negative.glob("*.pkl"):
                sample_meta = load_sample_metadata(pkl_file, cve_id, 0)
                if sample_meta:
                    all_samples.append(sample_meta)
                    negative_count += 1
        
        # Process positive samples (label 1)
        label_dir_positive = subdir / "1"
        if label_dir_positive.exists():
            for pkl_file in label_dir_positive.glob("*.pkl"):
                sample_meta = load_sample_metadata(pkl_file, cve_id, 1)
                if sample_meta:
                    all_samples.append(sample_meta)
                    positive_count += 1
        
        # Force garbage collection periodically
        if len(all_samples) % 1000 == 0:
            gc.collect()
    
    print(f"\n✓ Indexed {positive_count} positive samples")
    print(f"✓ Indexed {negative_count} negative samples")
    print(f"✓ Total samples: {len(all_samples)}")
    
    return all_samples


def load_features_from_file(filepath):
    """
    Load actual features from file on-demand
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            features_dict = data[0] if isinstance(data, list) else data
            return features_dict
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_split_in_chunks(samples, output_file, chunk_size=100):
    """
    Save dataset in memory-efficient way by processing in chunks
    """
    print(f"\nSaving {len(samples)} samples to {output_file}...")
    print(f"Processing in chunks of {chunk_size}...")
    
    processed_samples = []
    
    for i in tqdm(range(0, len(samples), chunk_size), desc='Processing chunks'):
        chunk = samples[i:i+chunk_size]
        
        # Load actual features for this chunk
        for sample_meta in chunk:
            features_dict = load_features_from_file(sample_meta['filepath'])
            if features_dict:
                processed_samples.append({
                    'features': features_dict,
                    'label': sample_meta['label'],
                    'filename': sample_meta['filename']
                })
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Save all at once
    print(f"Writing to disk...")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_samples, f, protocol=4)
    
    print(f"✓ Saved {len(processed_samples)} samples")
    
    # Clear memory
    del processed_samples
    gc.collect()


def load_and_combine_data_efficient(cwe, cve_prefix, src_base, output_dir,
                                    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                    seed=42, chunk_size=100):
    """
    Memory-efficient version: Create metadata index first, then load in chunks
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Get file count estimate
    total_files, num_cves = get_file_count(cwe, cve_prefix, src_base)
    print(f"Found approximately {total_files} files across {num_cves} CVE directories")
    
    # Estimate memory requirement
    estimated_memory_gb = (total_files * 0.5) / 1024  # Rough estimate: 0.5 MB per file
    print(f"Estimated memory requirement: ~{estimated_memory_gb:.2f} GB")
    
    if estimated_memory_gb > 25:
        print("⚠ WARNING: Large dataset detected. Using chunked processing...")
        chunk_size = min(chunk_size, 50)  # Reduce chunk size for very large datasets
    
    # Step 2: Create lightweight metadata index
    all_samples = create_metadata_index(cwe, cve_prefix, src_base)
    
    if not all_samples:
        raise ValueError("No samples found!")
    
    # Step 3: Stratified split on metadata (lightweight)
    labels = [s['label'] for s in all_samples]
    
    print(f"\nSplitting data (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})...")
    
    train_meta, temp_meta, train_labels, temp_labels = train_test_split(
        all_samples, labels, 
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )
    
    val_meta, test_meta, val_labels, test_labels = train_test_split(
        temp_meta, temp_labels,
        test_size=test_ratio/(val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_meta)} samples (Pos: {sum(train_labels)}, Neg: {len(train_labels)-sum(train_labels)})")
    print(f"  Val:   {len(val_meta)} samples (Pos: {sum(val_labels)}, Neg: {len(val_labels)-sum(val_labels)})")
    print(f"  Test:  {len(test_meta)} samples (Pos: {sum(test_labels)}, Neg: {len(test_labels)-sum(test_labels)})")
    
    # Step 4: Save splits by loading features in chunks
    train_file = output_dir / 'train_data.pkl'
    val_file = output_dir / 'val_data.pkl'
    test_file = output_dir / 'test_data.pkl'
    
    print(f"\n{'='*80}")
    print("Processing Training Set")
    print(f"{'='*80}")
    save_split_in_chunks(train_meta, train_file, chunk_size)
    
    print(f"\n{'='*80}")
    print("Processing Validation Set")
    print(f"{'='*80}")
    save_split_in_chunks(val_meta, val_file, chunk_size)
    
    print(f"\n{'='*80}")
    print("Processing Test Set")
    print(f"{'='*80}")
    save_split_in_chunks(test_meta, test_file, chunk_size)
    
    print(f"\n{'='*80}")
    print("✓ All data preprocessed successfully!")
    print(f"{'='*80}")
    print(f"Saved to: {output_dir}")
    print(f"  - {train_file.name}")
    print(f"  - {val_file.name}")
    print(f"  - {test_file.name}\n")
    
    return str(train_file), str(val_file), str(test_file)


def main():
    parser = argparse.ArgumentParser(description='Memory-Efficient Preprocessing for BigVul Data')
    
    parser.add_argument('--src-base', type=str, required=True,
                        help='Source base directory (e.g., data/graphs)')
    parser.add_argument('--cwe', type=str, required=True,
                        help='CWE identifier (e.g., CWE-119)')
    parser.add_argument('--cve-prefix', type=str, required=True,
                        help='CVE prefix to filter (e.g., CVE-2019-)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Chunk size for memory-efficient processing')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Print system info
    print(f"\n{'='*80}")
    print("Memory-Efficient Data Preprocessing")
    print(f"{'='*80}")
    print(f"Python version: {sys.version}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"{'='*80}\n")
    
    # Run preprocessing
    try:
        train_file, val_file, test_file = load_and_combine_data_efficient(
            cwe=args.cwe,
            cve_prefix=args.cve_prefix,
            src_base=args.src_base,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            chunk_size=args.chunk_size
        )
        
        print(f"\n{'='*80}")
        print("✓ Preprocessing Complete!")
        print(f"{'='*80}")
        print(f"\nUse these files for training:")
        print(f"  --train-file {train_file}")
        print(f"  --val-file {val_file}")
        print(f"  --test-file {test_file}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Preprocessing failed!")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
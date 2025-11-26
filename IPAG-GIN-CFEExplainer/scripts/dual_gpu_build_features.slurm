#!/bin/bash
#SBATCH --job-name=ipag-feat-chunks
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --partition=markov_gpu
#SBATCH --time=12:00:00
#SBATCH --output=ipag_feat_chunks_%A.out
#SBATCH --error=ipag_feat_chunks_%A.err

module load Python/3.11.3-GCCcore-12.3.0
source /mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/venv/bin/activate

CHUNKS_ROOT="/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/graphs"
SCRIPT_PATH="/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/scripts/batch_feature_build.py"
BATCH_SIZE=512

# Detect number of GPUs available for the job
N_GPUS=$(nvidia-smi -L | wc -l)

# Recursively find all chunk pairs
mapfile -t NODES_PATHS < <(find "$CHUNKS_ROOT" -type f -name 'ipag_nodes_chunk_*.joblib' | sort)
mapfile -t EDGES_PATHS < <(find "$CHUNKS_ROOT" -type f -name 'ipag_edges_chunk_*.joblib' | sort)

if (( ${#NODES_PATHS[@]} != ${#EDGES_PATHS[@]} )); then
    echo "Mismatched number of node and edge chunk files!"
    exit 1
fi

PIDS=()
for idx in "${!NODES_PATHS[@]}"; do
    nodes_path="${NODES_PATHS[$idx]}"
    edges_path="${EDGES_PATHS[$idx]}"
    label_folder="$(dirname "$nodes_path")"
    chunk_suffix="$(basename "$nodes_path" .joblib | sed 's/ipag_nodes_chunk_//')"
    save_path="$label_folder/features_chunk_${chunk_suffix}.pkl"

    # === Only process if feature file doesn't already exist ===
    if [[ -f "$save_path" ]]; then
        echo "Skipping already-processed chunk: $save_path"
        continue
    fi

    # Assign GPU index in a round-robin fashion
    gpu_idx=$((idx%N_GPUS))

    CUDA_VISIBLE_DEVICES=$gpu_idx python $SCRIPT_PATH \
        --nodes_path "$nodes_path" \
        --edges_path "$edges_path" \
        --save_path "$save_path" \
        --cuda_device_idx $gpu_idx \
        --batch_size $BATCH_SIZE &

    PIDS+=($!)

    # Wait if N_GPUS processes are active
    if (( ${#PIDS[@]} == N_GPUS )); then
        wait "${PIDS[@]}"
        PIDS=()
    fi
done

# Wait for any remaining jobs
wait
echo "Feature extraction complete for all (new) chunks!"

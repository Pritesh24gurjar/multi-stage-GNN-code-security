#!/bin/bash
#SBATCH --job-name=gin-exp5
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=markov_gpu
#SBATCH --time=04:00:00
#SBATCH --output=gin_exp5_%A.out
#SBATCH --error=gin_exp5_%A.err

module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate

echo "=========================================="
echo "Experiment 5: Tiny Model (64→32)"
echo "Goal: Better samples/parameter ratio (0.049 vs 0.006)"
echo "Started at: $(date)"
echo "=========================================="
echo ""
echo "GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
echo ""

EXPERIMENT_SRC="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/outputs"
DATA_DIR="${EXPERIMENT_SRC}/preprocessed_data"

# Verify preprocessed data exists
if [ ! -f "${DATA_DIR}/train_data.pkl" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run preprocess_data.slurm first"
    exit 1
fi

echo "Using preprocessed data from: ${DATA_DIR}"
echo ""

mkdir -p "${EXPERIMENT_SRC}/experiments"

# Run training with tiny architecture
python /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/scripts/trainer.py \
  --train-file "${DATA_DIR}/train_data.pkl" \
  --val-file "${DATA_DIR}/val_data.pkl" \
  --test-file "${DATA_DIR}/test_data.pkl" \
  --output-dir "${EXPERIMENT_SRC}/experiments/exp5_tiny_model" \
  --hidden-dims 64 32 \
  --dropout 0.3 \
  --learning-rate 5e-4 \
  --class-weights 1.0 1.5 \
  --batch-size 32 \
  --epochs 50 \
  --patience 10 \
  --num-workers 0 \
  --use-efficient-loader \
  --cache-size 100 \
  --clear-cache-every 5 \
  --pin-memory

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 137 ]; then
    echo "=========================================="
    echo "WARNING: Job killed (likely OOM)"
    echo "=========================================="
    echo "Try reducing:"
    echo "  - batch size (--batch-size 16)"
    echo "  - cache size (--cache-size 50)"
    echo "=========================================="
elif [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

# Run threshold tuning if model exists
if [ -f "${EXPERIMENT_SRC}/experiments/exp5_tiny_model/final_model.pth" ]; then
    echo ""
    echo "=========================================="
    echo "Starting Threshold Tuning"
    echo "=========================================="
    python /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/scripts/gin_find_optimal_threshold.py \
      --model-path "${EXPERIMENT_SRC}/experiments/exp5_tiny_model/final_model.pth" \
      --val-file "${DATA_DIR}/val_data.pkl" \
      --output "${EXPERIMENT_SRC}/experiments/exp5_tiny_model/optimal_threshold.json"\
      --plot
    echo "✓ Threshold tuning complete"
else
    echo "⚠ Model not found, skipping threshold tuning"
fi

echo ""
echo "=========================================="
echo "Final GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv
echo ""
echo "Completed at: $(date)"
echo "=========================================="

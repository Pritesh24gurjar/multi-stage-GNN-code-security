#!/bin/bash
#SBATCH --job-name=ggnn-cwe-exp
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2080
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --time=12:00:00
#SBATCH --output=ggnn_cwe_exp1_%A.out
#SBATCH --error=ggnn_cwe_exp1_%A.err

# ============================================
# GGNN CWE Classification - Single Experiment
# ============================================

echo "=========================================="
echo "GGNN CWE Classification Experiment"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# ============================================
# Environment Setup
# ============================================

module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate

echo "GPU Memory Info:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
echo ""

# ============================================
# Paths Configuration
# ============================================

PROJECT_ROOT="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer"
DATA_DIR="${PROJECT_ROOT}/data/splits"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments"

# Input data files
DATA_PATH="${DATA_DIR}/processed_graphs_multi_class.pkl"
METADATA_PATH="${DATA_DIR}/dataset_metadata.pkl"

# Verify data exists
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Processed graphs not found at ${DATA_PATH}"
    echo "Please run preprocessing script first"
    exit 1
fi

if [ ! -f "${METADATA_PATH}" ]; then
    echo "ERROR: Metadata not found at ${METADATA_PATH}"
    echo "Please run preprocessing script first"
    exit 1
fi

echo "Using data from:"
echo "  Graphs: ${DATA_PATH}"
echo "  Metadata: ${METADATA_PATH}"
echo ""

# ============================================
# Experiment Configuration
# ============================================

# Experiment name (change for different experiments)
EXP_NAME="exp1_baseline"

# Model hyperparameters
EMBEDDING_DIM=128
HIDDEN_DIM=200
NUM_GGNN_LAYERS=3
NUM_STEPS=8
DROPOUT=0.3

# Training hyperparameters
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=5e-4
EARLY_STOPPING_PATIENCE=15

# Data split
TEST_SIZE=0.2
VAL_SIZE=0.1
RANDOM_SEED=42

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "=========================================="
echo "Model Configuration:"
echo "  Embedding Dim: ${EMBEDDING_DIM}"
echo "  Hidden Dim: ${HIDDEN_DIM}"
echo "  GGNN Layers: ${NUM_GGNN_LAYERS}"
echo "  Propagation Steps: ${NUM_STEPS}"
echo "  Dropout: ${DROPOUT}"
echo ""
echo "Training Configuration:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Early Stopping Patience: ${EARLY_STOPPING_PATIENCE}"
echo ""

# ============================================
# Create Output Directory
# ============================================

EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

echo "Output directory: ${EXP_DIR}"
echo ""

# ============================================
# Run Training
# ============================================

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

python ${PROJECT_ROOT}/scripts/ggnn_cwe_classifier_training_pipeline.py \
    --data_path "${DATA_PATH}" \
    --metadata_path "${METADATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --embedding_dim ${EMBEDDING_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_ggnn_layers ${NUM_GGNN_LAYERS} \
    --num_steps ${NUM_STEPS} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --test_size ${TEST_SIZE} \
    --val_size ${VAL_SIZE} \
    --random_seed ${RANDOM_SEED} \
    --min_samples_per_class 100 \
    --gradient_accumulation_steps 8

EXIT_CODE=$?

echo ""
echo "=========================================="

# ============================================
# Check Training Status
# ============================================

if [ $EXIT_CODE -eq 137 ]; then
    echo "WARNING: Job killed (likely OOM)"
    echo "=========================================="
    echo "Try reducing:"
    echo "  - Batch size: --batch_size 16"
    echo "  - Hidden dimensions: --hidden_dim 128"
    echo "  - GGNN layers: --num_ggnn_layers 2"
    echo "=========================================="
    exit 137
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    exit ${EXIT_CODE}
else
    echo "✓ Training completed successfully"
    echo "=========================================="
fi

# ============================================
# Verify Outputs
# ============================================

echo ""
echo "Verifying outputs..."

REQUIRED_FILES=(
    "best_model.pth"
    "config.json"
    "test_metrics.json"
    "training_history.json"
    "training_history.png"
    "confusion_matrix.png"
)

MISSING_FILES=0
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ -f "${EXP_DIR}/${FILE}" ]; then
        echo "  ✓ ${FILE}"
    else
        echo "  ✗ ${FILE} (missing)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

echo ""

if [ ${MISSING_FILES} -eq 0 ]; then
    echo "✓ All output files generated successfully"
else
    echo "⚠ ${MISSING_FILES} output file(s) missing"
fi

# ============================================
# Display Results Summary
# ============================================

if [ -f "${EXP_DIR}/test_metrics.json" ]; then
    echo ""
    echo "=========================================="
    echo "Test Metrics Summary"
    echo "=========================================="
    python3 -c "
import json
with open('${EXP_DIR}/test_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f\"Accuracy:       {metrics['accuracy']:.4f}\")
    print(f\"Precision:      {metrics['precision']:.4f}\")
    print(f\"Recall:         {metrics['recall']:.4f}\")
    print(f\"F1-Score:       {metrics['f1_score']:.4f}\")
    print(f\"AUC:            {metrics['auc']:.4f}\")
    print(f\"FPR:            {metrics['fpr']:.4f}\")
    print(f\"FNR:            {metrics['fnr']:.4f}\")
    print(f\"Fairness Score: {metrics['fairness_score']:.4f}\")
"
    echo "=========================================="
fi

# ============================================
# Final GPU Memory Status
# ============================================

echo ""
echo "=========================================="
echo "Final GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""
echo "Completed at: $(date)"
echo "=========================================="

# ============================================
# Save Job Info
# ============================================

cat > "${EXP_DIR}/job_info.txt" << EOF
Job Name: ${SLURM_JOB_NAME}
Job ID: ${SLURM_JOB_ID}
Node: ${SLURM_NODELIST}
Started: $(date)
Experiment: ${EXP_NAME}
Output Directory: ${EXP_DIR}
Exit Code: ${EXIT_CODE}
EOF

echo ""
echo "Job information saved to: ${EXP_DIR}/job_info.txt"
echo ""

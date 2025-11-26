#!/bin/bash
#SBATCH --job-name=ggnn-cwe-top30
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2080
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --time=12:00:00
#SBATCH --output=ggnn_cwe_top30_%A.out
#SBATCH --error=ggnn_cwe_top30_%A.err

# ============================================
# GGNN CWE Top-30 Classification - Experiment
# ============================================

echo "=========================================="
echo "GGNN CWE Top-30 Classification Experiment"
echo "Started at: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_NODELIST}"
echo "=========================================="
echo ""

# ============================================
# Environment Setup
# ============================================

module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate

echo "Python version:"
python -V
echo ""

echo "GPU Memory Info (initial):"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
echo ""

# ============================================
# Paths Configuration
# ============================================

PROJECT_ROOT="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer"
DATA_DIR="${PROJECT_ROOT}/data/splits"
CHECKPOINT_DIR="${PROJECT_ROOT}/multiclass_experiment/checkpoints"

TRAIN_SCRIPT="${PROJECT_ROOT}/multiclass_experiment/trainer_multiclass_cache_clearing.py"  # <- current pipeline

GRAPHS_PATH="${DATA_DIR}/processed_graphs_multi_class.pkl"
METADATA_PATH="${DATA_DIR}/dataset_metadata.pkl"

# Sanity check inputs
if [ ! -f "${GRAPHS_PATH}" ]; then
    echo "ERROR: Processed graphs not found at ${GRAPHS_PATH}"
    echo "Please run preprocessing script first."
    exit 1
fi

if [ ! -f "${METADATA_PATH}" ]; then
    echo "ERROR: Metadata not found at ${METADATA_PATH}"
    echo "Please run preprocessing script first."
    exit 1
fi

echo "Using data from:"
echo "  Graphs:   ${GRAPHS_PATH}"
echo "  Metadata: ${METADATA_PATH}"
echo ""





# ============================================
# Experiment Configuration
# ============================================

EXP_NAME="exp1_top30_baseline"

# Model hyperparameters
EMBEDDING_DIM=64
HIDDEN_DIM=128
NUM_GGNN_LAYERS=3
NUM_STEPS=4
DROPOUT=0.3

# Training hyperparameters
BATCH_SIZE=8
EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-5
NUM_WORKERS=4
SEED=42

# Data split ratios
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1

DEVICE="cuda"

echo "=========================================="
echo "Experiment: ${EXP_NAME}"
echo "=========================================="
echo "Experiment 1"
echo "Approximate parameter count: Core GGNN ≈ 0.31M parameters (3·6·128² + readout). Total model comfortably below several million parameters and far under the 33M limit."
echo "Expected behaviour: This baseline provides moderate width and depth with stable optimization. It should offer reliable convergence and define the reference F1/AUC and fairness profile for all other experiments."

echo ""
echo "Model Configuration:"
echo "  Embedding Dim:   ${EMBEDDING_DIM}"
echo "  Hidden Dim:      ${HIDDEN_DIM}"
echo "  GGNN Layers:     ${NUM_GGNN_LAYERS}"
echo "  Propagation Steps: ${NUM_STEPS}"
echo "  Dropout:         ${DROPOUT}"
echo ""
echo "Training Configuration:"
echo "  Batch Size:      ${BATCH_SIZE}"
echo "  Epochs:          ${EPOCHS}"
echo "  Learning Rate:   ${LEARNING_RATE}"
echo "  Weight Decay:    ${WEIGHT_DECAY}"
echo "  Num Workers:     ${NUM_WORKERS}"
echo "  Seed:            ${SEED}"
echo ""
echo "Data Splits:"
echo "  Train Ratio:     ${TRAIN_RATIO}"
echo "  Val Ratio:       ${VAL_RATIO}"
echo "  Test Ratio:      ${TEST_RATIO}"
echo ""

# ============================================
# Output / Checkpoint Directory
# ============================================

EXP_DIR="${CHECKPOINT_DIR}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

echo "Checkpoint directory: ${EXP_DIR}"
echo ""

# Set checkpoint dir env or pass directly
CHECKPOINT_ARG="${EXP_DIR}"

# ============================================
# Run Training
# ============================================

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

python "${TRAIN_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --embedding_dim ${EMBEDDING_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_ggnn_layers ${NUM_GGNN_LAYERS} \
    --num_steps ${NUM_STEPS} \
    --dropout ${DROPOUT} \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio ${VAL_RATIO} \
    --test_ratio ${TEST_RATIO} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --checkpoint_dir "${EXP_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training script exited with code: ${EXIT_CODE}"
echo "=========================================="
echo ""

# ============================================
# Check Training Status (simple)
# ============================================

if [ ${EXIT_CODE} -eq 137 ]; then
    echo "WARNING: Job killed (likely OOM)."
    echo "Try reducing:"
    echo "  - Batch size (e.g., 16 -> 8 or 4)"
    echo "  - Hidden dim (e.g., 200 -> 128)"
    echo "  - GGNN layers (e.g., 3 -> 2)"
elif [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
else
    echo "✓ Training completed successfully"
fi

# ============================================
# Final GPU Memory Status
# ============================================

echo ""
echo "Final GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo ""

echo "Completed at: $(date)"
echo "=========================================="
echo ""

# ============================================
# Save Job Info (minimal)
# ============================================

JOB_INFO_FILE="${EXP_DIR}/job_info.txt"

cat > "${JOB_INFO_FILE}" << EOF
Job Name:          ${SLURM_JOB_NAME}
Job ID:            ${SLURM_JOB_ID}
Node:              ${SLURM_NODELIST}
Started:           $(date)
Experiment:        ${EXP_NAME}
Checkpoint Dir:    ${EXP_DIR}
Exit Code:         ${EXIT_CODE}
Train Script:      ${TRAIN_SCRIPT}
Data Dir:          ${DATA_DIR}
Batch Size:        ${BATCH_SIZE}
Epochs:            ${EPOCHS}
Learning Rate:     ${LEARNING_RATE}
Hidden Dim:        ${HIDDEN_DIM}
GGNN Layers:       ${NUM_GGNN_LAYERS}
Propagation Steps: ${NUM_STEPS}
EOF

echo "Job information saved to: ${JOB_INFO_FILE}"
echo ""

#!/bin/bash
#SBATCH --job-name=ggnn-cwe-analysis
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2080
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --time=04:00:00
#SBATCH --output=ggnn_cwe_analysis_%A.out
#SBATCH --error=ggnn_cwe_analysis_%A.err

# ============================================
# GGNN CWE Top-30 - Model Comparison + RAI
# ============================================

echo "=========================================="
echo "GGNN CWE Top-30 Model Analysis & RAI"
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
ANALYSIS_DIR="${PROJECT_ROOT}/multiclass_experiment/analysis"

ANALYSIS_SCRIPT="${PROJECT_ROOT}/multiclass_experiment/analyze_models.py"

GRAPHS_PATH="${DATA_DIR}/processed_graphs_multi_class.pkl"
METADATA_PATH="${DATA_DIR}/dataset_metadata.pkl"

# Example checkpoints to compare (edit these as needed)
CHECKPOINTS=(
    "${PROJECT_ROOT}/multiclass_experiment/checkpoints/exp1_top30_baseline/exp1_baseline_best_model.pt"
    "${PROJECT_ROOT}/multiclass_experiment/checkpoints/exp6_top30_more_steps/exp6_more_steps_best_model.pt"
    "${PROJECT_ROOT}/notebooks/checkpoints/initial_best_model.pt"
)

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

MISSING_CKPT=0
for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "${ckpt}" ]; then
        echo "ERROR: Checkpoint not found: ${ckpt}"
        MISSING_CKPT=1
    fi
done

if [ ${MISSING_CKPT} -ne 0 ]; then
    echo "One or more checkpoints are missing. Aborting."
    exit 1
fi

echo "Using data from:"
echo "  Graphs:    ${GRAPHS_PATH}"
echo "  Metadata:  ${METADATA_PATH}"
echo ""
echo "Comparing checkpoints:"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  - ${ckpt}"
done
echo ""

# ============================================
# Analysis Configuration
# ============================================

BATCH_SIZE=4
NUM_WORKERS=1
SEED=42
DEVICE="cpu"

echo "=========================================="
echo "Analysis Configuration"
echo "=========================================="
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Num Workers:  ${NUM_WORKERS}"
echo "  Seed:         ${SEED}"
echo "  Device:       ${DEVICE}"
echo "  Analysis Dir: ${ANALYSIS_DIR}"
echo ""

mkdir -p "${ANALYSIS_DIR}"

# ============================================
# Run Analysis
# ============================================

echo "=========================================="
echo "Starting Analysis"
echo "=========================================="
echo ""

python "${ANALYSIS_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --analysis_dir "${ANALYSIS_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --device ${DEVICE}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Analysis script exited with code: ${EXIT_CODE}"
echo "=========================================="
echo ""

# ============================================
# Simple Status Check
# ============================================

if [ ${EXIT_CODE} -eq 137 ]; then
    echo "WARNING: Job killed (likely OOM or time limit)."
    echo "Try reducing:"
    echo "  - Batch size"
    echo "  - Num workers"
elif [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Analysis failed with exit code ${EXIT_CODE}"
else
    echo "✓ Analysis completed successfully"
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

JOB_INFO_FILE="${ANALYSIS_DIR}/analysis_job_${SLURM_JOB_ID}.txt"

cat > "${JOB_INFO_FILE}" << EOF
Job Name:         ${SLURM_JOB_NAME}
Job ID:           ${SLURM_JOB_ID}
Node:             ${SLURM_NODELIST}
Started:          $(date)
Analysis Dir:     ${ANALYSIS_DIR}
Exit Code:        ${EXIT_CODE}
Analysis Script:  ${ANALYSIS_SCRIPT}
Data Dir:         ${DATA_DIR}
Batch Size:       ${BATCH_SIZE}
Num Workers:      ${NUM_WORKERS}
Device:           ${DEVICE}
Checkpoints:
$(printf '  - %s\n' "${CHECKPOINTS[@]}")
EOF

echo "Job information saved to: ${JOB_INFO_FILE}"
echo ""

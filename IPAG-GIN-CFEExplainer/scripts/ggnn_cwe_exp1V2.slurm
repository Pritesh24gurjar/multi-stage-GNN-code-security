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
# NO BATCH SKIPPING MODE
# ============================================


echo "=========================================="
echo "GGNN CWE Classification Experiment"
echo "NO BATCH SKIPPING - Improved Memory Management"
echo "Started at: $(date)"
echo "=========================================="
echo ""


# ============================================
# Environment Setup
# ============================================


module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate


echo "Python Environment:"
which python
python --version
echo ""


echo "PyTorch Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""


echo "GPU Memory Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
echo ""


# ============================================
# Paths Configuration
# ============================================


PROJECT_ROOT="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer"
DATA_DIR="${PROJECT_ROOT}/data/splits"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"


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


# Check file sizes
DATA_SIZE=$(du -h "${DATA_PATH}" | cut -f1)
META_SIZE=$(du -h "${METADATA_PATH}" | cut -f1)
echo "Data file sizes:"
echo "  Graphs: ${DATA_SIZE}"
echo "  Metadata: ${META_SIZE}"
echo ""


# ============================================
# Experiment Configuration
# ============================================


# Experiment name (change for different experiments)
EXP_NAME="exp1_baselinev2"


# Model hyperparameters
EMBEDDING_DIM=128
HIDDEN_DIM=200
NUM_GGNN_LAYERS=3
NUM_STEPS=8
DROPOUT=0.3


# Training hyperparameters (optimized for 12GB GPU)
BATCH_SIZE=16                      # Physical batch size
GRADIENT_ACCUMULATION_STEPS=8      # Effective batch size = 16 * 8 = 128
EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=5e-4
EARLY_STOPPING_PATIENCE=15


# Data split
TEST_SIZE=0.2
VAL_SIZE=0.1
RANDOM_SEED=42
MIN_SAMPLES_PER_CLASS=100


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
echo "  Physical Batch Size: ${BATCH_SIZE}"
echo "  Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Early Stopping Patience: ${EARLY_STOPPING_PATIENCE}"
echo ""
echo "Data Configuration:"
echo "  Test Size: ${TEST_SIZE}"
echo "  Val Size: ${VAL_SIZE}"
echo "  Min Samples per Class: ${MIN_SAMPLES_PER_CLASS}"
echo "  Random Seed: ${RANDOM_SEED}"
echo ""


# ============================================
# Create Output Directory
# ============================================


EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"
mkdir -p "${EXP_DIR}/logs"


echo "Output directory: ${EXP_DIR}"
echo ""


# Copy config to experiment directory for reproducibility
cat > "${EXP_DIR}/experiment_config.txt" << EOF
Experiment Configuration
========================

Started: $(date)
Node: ${SLURM_NODELIST}
Job ID: ${SLURM_JOB_ID}

Model:
  - Embedding Dim: ${EMBEDDING_DIM}
  - Hidden Dim: ${HIDDEN_DIM}
  - GGNN Layers: ${NUM_GGNN_LAYERS}
  - Propagation Steps: ${NUM_STEPS}
  - Dropout: ${DROPOUT}

Training:
  - Physical Batch Size: ${BATCH_SIZE}
  - Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}
  - Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
  - Epochs: ${EPOCHS}
  - Learning Rate: ${LEARNING_RATE}
  - Weight Decay: ${WEIGHT_DECAY}
  - Early Stopping Patience: ${EARLY_STOPPING_PATIENCE}

Data:
  - Test Size: ${TEST_SIZE}
  - Val Size: ${VAL_SIZE}
  - Min Samples per Class: ${MIN_SAMPLES_PER_CLASS}
  - Random Seed: ${RANDOM_SEED}

Files:
  - Data: ${DATA_PATH}
  - Metadata: ${METADATA_PATH}
EOF


# ============================================
# Setup GPU Monitoring
# ============================================


# Start GPU monitoring in background
GPU_LOG="${EXP_DIR}/logs/gpu_usage.log"
echo "timestamp,gpu_util,mem_used_mb,mem_free_mb,temperature" > "${GPU_LOG}"


monitor_gpu() {
    while true; do
        TIMESTAMP=$(date +%s)
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits)
        echo "${TIMESTAMP},${GPU_INFO}" >> "${GPU_LOG}"
        sleep 30
    done
}


# Start monitoring
monitor_gpu &
GPU_MONITOR_PID=$!
echo "GPU monitoring started (PID: ${GPU_MONITOR_PID})"
echo "GPU logs: ${GPU_LOG}"
echo ""


# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${GPU_MONITOR_PID}" ]; then
        kill ${GPU_MONITOR_PID} 2>/dev/null || true
        echo "GPU monitoring stopped"
    fi
}


# Register cleanup on exit
trap cleanup EXIT


# ============================================
# Run Training
# ============================================


echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""


START_TIME=$(date +%s)


# Use the improved training pipeline
python ${SCRIPT_DIR}/ggnn_cwe_classifier_training_pipelineV2.py \
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
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --test_size ${TEST_SIZE} \
    --val_size ${VAL_SIZE} \
    --random_seed ${RANDOM_SEED} \
    --min_samples_per_class ${MIN_SAMPLES_PER_CLASS} 2>&1 | tee "${EXP_DIR}/logs/training.log"


EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))


echo ""
echo "=========================================="


# ============================================
# Check Training Status
# ============================================


if [ $EXIT_CODE -eq 137 ]; then
    echo "WARNING: Job killed (likely OOM)"
    echo "=========================================="
    echo "Memory Optimization Suggestions:"
    echo "  1. Reduce batch size: --batch_size 12"
    echo "  2. Increase gradient accumulation: --gradient_accumulation_steps 10"
    echo "  3. Reduce hidden dimensions: --hidden_dim 150"
    echo "  4. Reduce GGNN layers: --num_ggnn_layers 2"
    echo "  5. Filter larger graphs during preprocessing"
    echo "=========================================="
    exit 137
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo "Check logs for details:"
    echo "  ${EXP_DIR}/logs/training.log"
    echo "=========================================="
    exit ${EXIT_CODE}
else
    echo "✓ Training completed successfully"
    echo "Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
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
    "per_class_metrics.json"
    "classification_report.txt"
    "training_history.json"
    "training_history.png"
    "confusion_matrix.png"
    "roc_curve.png"
    "rai_metrics.png"
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


# Show per-class metrics summary
if [ -f "${EXP_DIR}/per_class_metrics.json" ]; then
    echo ""
    echo "=========================================="
    echo "Per-Class Metrics (Top 5 by F1-Score)"
    echo "=========================================="
    python3 -c "
import json
with open('${EXP_DIR}/per_class_metrics.json', 'r') as f:
    per_class = json.load(f)
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    print(f\"{'CWE':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\")
    print('-' * 60)
    for cwe, metrics in sorted_classes[:5]:
        print(f\"{cwe:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['support']:>10}\")
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


# ============================================
# Analyze GPU Usage
# ============================================


if [ -f "${GPU_LOG}" ]; then
    echo "=========================================="
    echo "GPU Usage Statistics"
    echo "=========================================="
    python3 -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${GPU_LOG}')
    if len(df) > 0:
        print(f\"Average GPU Utilization: {df['gpu_util'].mean():.1f}%\")
        print(f\"Peak GPU Utilization: {df['gpu_util'].max():.1f}%\")
        print(f\"Average Memory Used: {df['mem_used_mb'].mean():.0f} MB\")
        print(f\"Peak Memory Used: {df['mem_used_mb'].max():.0f} MB\")
        print(f\"Average Temperature: {df['temperature'].mean():.1f}°C\")
        print(f\"Peak Temperature: {df['temperature'].max():.1f}°C\")
        print(f\"Total Monitoring Points: {len(df)}\")
except Exception as e:
    print(f\"Could not analyze GPU logs: {e}\")
"
    echo "=========================================="
fi


echo ""
echo "Completed at: $(date)"
echo "=========================================="


# ============================================
# Save Job Info
# ============================================


cat > "${EXP_DIR}/job_info.txt" << EOF
Job Information
===============

Job Name: ${SLURM_JOB_NAME}
Job ID: ${SLURM_JOB_ID}
Node: ${SLURM_NODELIST}
Account: ${SLURM_JOB_ACCOUNT}
Partition: ${SLURM_JOB_PARTITION}

Started: $(date -d @${START_TIME})
Ended: $(date -d @${END_TIME})
Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s

Experiment: ${EXP_NAME}
Output Directory: ${EXP_DIR}
Exit Code: ${EXIT_CODE}

Configuration:
  - Physical Batch Size: ${BATCH_SIZE}
  - Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}
  - Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
  - Epochs: ${EPOCHS}
  - Learning Rate: ${LEARNING_RATE}

Files Generated:
$(ls -lh "${EXP_DIR}" | grep -v "^d" | awk '{print "  - " $9 " (" $5 ")"}')

Logs:
  - Training: ${EXP_DIR}/logs/training.log
  - GPU Usage: ${GPU_LOG}
EOF


echo ""
echo "Job information saved to: ${EXP_DIR}/job_info.txt"
echo ""


# ============================================
# Create Results Summary HTML (Optional)
# ============================================


if [ -f "${EXP_DIR}/test_metrics.json" ]; then
    python3 -c "
import json
import os

exp_dir = '${EXP_DIR}'
exp_name = '${EXP_NAME}'

# Load metrics
with open(os.path.join(exp_dir, 'test_metrics.json'), 'r') as f:
    metrics = json.load(f)

# Create simple HTML report
html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>{exp_name} - Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 50%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .metric {{ font-weight: bold; }}
        img {{ max-width: 100%; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{exp_name}</h1>
    <h2>Test Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td class=\"metric\">Accuracy</td><td>{metrics['accuracy']:.4f}</td></tr>
        <tr><td class=\"metric\">Precision</td><td>{metrics['precision']:.4f}</td></tr>
        <tr><td class=\"metric\">Recall</td><td>{metrics['recall']:.4f}</td></tr>
        <tr><td class=\"metric\">F1-Score</td><td>{metrics['f1_score']:.4f}</td></tr>
        <tr><td class=\"metric\">AUC</td><td>{metrics['auc']:.4f}</td></tr>
        <tr><td class=\"metric\">FPR</td><td>{metrics['fpr']:.4f}</td></tr>
        <tr><td class=\"metric\">FNR</td><td>{metrics['fnr']:.4f}</td></tr>
        <tr><td class=\"metric\">Fairness Score</td><td>{metrics['fairness_score']:.4f}</td></tr>
    </table>
    
    <h2>Visualizations</h2>
    <h3>Training History</h3>
    <img src=\"training_history.png\" alt=\"Training History\">
    
    <h3>Confusion Matrix</h3>
    <img src=\"confusion_matrix.png\" alt=\"Confusion Matrix\">
    
    <h3>ROC Curves</h3>
    <img src=\"roc_curve.png\" alt=\"ROC Curves\">
    
    <h3>Per-Class Metrics</h3>
    <img src=\"rai_metrics.png\" alt=\"Per-Class Metrics\">
</body>
</html>
'''

with open(os.path.join(exp_dir, 'results.html'), 'w') as f:
    f.write(html)

print('HTML report generated: ${EXP_DIR}/results.html')
" 2>/dev/null || true
fi


echo ""
echo "=========================================="
echo "All outputs saved to: ${EXP_DIR}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review results: ${EXP_DIR}/results.html"
echo "  2. Check training log: ${EXP_DIR}/logs/training.log"
echo "  3. Analyze GPU usage: ${GPU_LOG}"
echo "  4. Compare with other experiments"
echo ""
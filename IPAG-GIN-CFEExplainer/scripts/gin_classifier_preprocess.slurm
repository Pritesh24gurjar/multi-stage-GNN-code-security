#!/bin/bash
#SBATCH --job-name=gin-preprocess
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --time=02:00:00
#SBATCH --output=gin_classifier_preprocess_%A.out
#SBATCH --error=gin_classifier_preprocess_%A.err

module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate

echo "=========================================="
echo "Memory-Efficient Data Preprocessing"
echo "Started at: $(date)"
echo "=========================================="
echo ""
echo "Memory Info:"
free -h
echo ""

EXPERIMENT_SRC="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/outputs"
DATA_DIR="${EXPERIMENT_SRC}/preprocessed_data_final"

# Create output directory
mkdir -p "${DATA_DIR}"

# Run data preprocessing with memory-efficient chunking
python /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/scripts/gin_classifier_preprocess.py \
  --src-base "/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/graphs" \
  --cwe "CWE-119" \
  --cve-prefix "CVE" \
  --output-dir "${DATA_DIR}" \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --chunk-size 25

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 137 ]; then
    echo "=========================================="
    echo "ERROR: Job killed (OOM)"
    echo "=========================================="
    echo "The dataset is too large for available memory."
    echo ""
    echo "Solutions:"
    echo "  1. Increase memory: #SBATCH --mem=128G"
    echo "  2. Reduce chunk size: --chunk-size 25"
    echo "  3. Process one CWE at a time"
    echo "=========================================="
    exit 137
elif [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Preprocessing failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Verifying Output Files"
echo "=========================================="
ls -lh "${DATA_DIR}"/*.pkl

echo ""
echo "Dataset Statistics:"
python -c "
import pickle
import sys

try:
    with open('${DATA_DIR}/train_data.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('${DATA_DIR}/val_data.pkl', 'rb') as f:
        val = pickle.load(f)
    with open('${DATA_DIR}/test_data.pkl', 'rb') as f:
        test = pickle.load(f)
    
    print(f'✓ Train samples: {len(train)}')
    print(f'✓ Val samples: {len(val)}')
    print(f'✓ Test samples: {len(test)}')
    print(f'✓ Total: {len(train) + len(val) + len(test)}')
except Exception as e:
    print(f'✗ Error verifying files: {e}')
    sys.exit(1)
"

VERIFY_CODE=$?

if [ $VERIFY_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Output verification failed"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "Completed at: $(date)"
echo "=========================================="
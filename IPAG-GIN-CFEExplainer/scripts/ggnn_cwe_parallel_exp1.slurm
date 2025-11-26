#!/bin/bash
#SBATCH --job-name=ggnn-cwe-ddp
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --gpus=2                 # 1 GPU per task
#SBATCH --cpus-per-task=4
#SBATCH --constraint=gpu2080
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --time=24:00:00
#SBATCH --output=ggnn_cwe_ddp_%A.out
#SBATCH --error=ggnn_cwe_ddp_%A.err

######## ENVIRONMENT SETUP ########

module load Python/3.11.5-GCCcore-13.2.0
source /home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/venv-2.4/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

######## PATHS ########

PROJECT_ROOT="/home/sxi219/RAI-Account/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer"
DATA_DIR="${PROJECT_ROOT}/data/splits"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments"
DATA_PATH="${DATA_DIR}/processed_graphs_multi_class.pkl"
METADATA_PATH="${DATA_DIR}/dataset_metadata.pkl"

######## MODEL & TRAINING CONFIG ########

EXP_NAME="exp1_ddp_bs16"
EMBEDDING_DIM=128
HIDDEN_DIM=200
NUM_GGNN_LAYERS=3
NUM_STEPS=8
DROPOUT=0.3

BATCH_SIZE=16   # Each GPU will independently use this batch size
EPOCHS=100
LEARNING_RATE=0.001
WEIGHT_DECAY=5e-4
EARLY_STOPPING_PATIENCE=15

TEST_SIZE=0.2
VAL_SIZE=0.1
RANDOM_SEED=42

EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

######## LAUNCH ########

echo "=========================================="
echo "Running Multi-GPU DDP: $SLURM_JOBID"
echo "Started at: $(date)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ${PROJECT_ROOT}/scripts/training_ddp_ggnn.py \
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
    --min_samples_per_class 100

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Multi-GPU DDP Job completed successfully"
else
    echo "X DDP Job failed with exit code $EXIT_CODE"
fi
echo "Completed at: $(date)"
echo "=========================================="

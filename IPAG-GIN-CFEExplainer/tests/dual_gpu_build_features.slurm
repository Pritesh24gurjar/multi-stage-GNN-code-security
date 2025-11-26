#!/bin/bash
#SBATCH --job-name=ipag-gpu
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --error=ipag_job.err  # Log file for output
#SBATCH --output=ipag_job.out

# Setup environment
source /mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/venv/bin/activate

SCRIPT_PATH="/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/tests/test_ipag_builder.py"
BATCH0_PATH="/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/processed/batch0.pkl"
BATCH1_PATH="/mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/data/processed/batch1.pkl"

# Run first half in one background process on GPU 0 (MIG device 0)
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --chunk_idx 0 --batch_size 1280 --save_path $BATCH0_PATH &

# Run second half in a second background process on GPU 1 (MIG device 1)
CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH --chunk_idx 1 --batch_size 1280 --save_path $BATCH1_PATH &

wait # Wait for both processes to finish

echo "All jobs finished! Features for both GPUs saved."

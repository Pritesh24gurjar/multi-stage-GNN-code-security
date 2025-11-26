#!/bin/bash
#SBATCH --job-name=ipag-data-preprocess
#SBATCH --account=csds447
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --error=ipag_data_preprocess.err  # Log file for output
#SBATCH --output=ipag_data_preprocess.out

module load Python/3.11.3-GCCcore-12.3.0

source /mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/venv/bin/activate

python /mnt/vstor/courses/csds447/sxi219/multi-stage-GNN-code-security/IPAG-GIN-CFEExplainer/scripts/batch_data_preprocess.py
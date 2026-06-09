#!/bin/bash
#SBATCH --job-name=train_fpdh_pixelshuffle8
#SBATCH --partition=accelerated-h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new/logs/NAFNet_width32_fpdh_pixelshuffle8_%J.out
#SBATCH --error=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new/logs/NAFNet_width32_fpdh_pixelshuffle8_%J.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new
mkdir -p logs

source /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/etc/profile.d/conda.sh
conda activate /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/envs/env2


PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_fpdh_pixelshuffle8.yml --fpdh_drop_prob 0.8
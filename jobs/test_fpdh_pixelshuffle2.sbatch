#!/bin/bash
#SBATCH --job-name=test_fpdh_pixelshuffle2
#SBATCH --partition=accelerated-h100
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new/logs/test_fpdh_pixelshuffle2_%J.out
#SBATCH --error=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new/logs/test_fpdh_pixelshuffle2_%J.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new
mkdir -p logs

source /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/etc/profile.d/conda.sh
conda activate /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/envs/env2

PYTHONPATH=. python basicsr/test.py -opt /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/nafnet_new/options/test/GoPro/NAFNet_width32_fpdh_pixelshuffle2.yml
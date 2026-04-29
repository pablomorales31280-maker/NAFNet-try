#!/bin/bash
#SBATCH --job-name=train_fp_lctc
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_fp_lctc.out
#SBATCH --error=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_fp_lctc.err

set -euo pipefail

cd /home/pmorales/Shashank_code_BoA/freq-nafnet
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate env

PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_fp_lctc.yml
#!/bin/bash
#SBATCH --job-name=train_conv_freqavgup
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_conv_freqavgup.out
#SBATCH --error=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_conv_freqavgup.err

set -euo pipefail

cd /home/pmorales/Shashank_code_BoA/freq-nafnet
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate env

PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_conv_freqavgup.yml
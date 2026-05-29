#!/bin/bash
#SBATCH --job-name=train_pixelunshuffle_freqavgup
#SBATCH --partition=gpu-vram-94gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_pixelunshuffle_freqavgup_%J.out
#SBATCH --error=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/NAFNet_width32_pixelunshuffle_freqavgup_%J.err

set -euo pipefail

cd /home/pmorales/Shashank_code_BoA/freq-nafnet
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate env

PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_pixelunshuffle_freqavgup.yml
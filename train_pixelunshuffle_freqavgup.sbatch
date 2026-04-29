#!/bin/bash
#SBATCH --job-name=train_pixelunshuffle_freqavgup
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/pmorales/Shashank code BoA/freq-nafnet/logs/NAFNet_width32_pixelunshuffle_freqavgup.out
#SBATCH --error=/home/pmorales/Shashank code BoA/freq-nafnet/logs/NAFNet_width32_pixelunshuffle_freqavgup.err

set -euo pipefail

cd /home/pmorales/Shashank code BoA/freq-nafnet
mkdir -p logs

conda activate env

PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_pixelunshuffle_freqavgup.yml
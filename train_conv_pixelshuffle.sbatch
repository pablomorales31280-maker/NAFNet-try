#!/bin/bash
#SBATCH --job-name=train__conv_pixelshuffle
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/pmorales/Shashank code BoA/freq-nafnet/logs/NAFNet_width32_conv_pixelshuffle.out
#SBATCH --error=/home/pmorales/Shashank code BoA/freq-nafnet/logs/NAFNet_width32_conv_pixelshuffle.err

set -euo pipefail

cd /home/pmorales/Shashank code BoA/freq-nafnet
mkdir -p logs

conda activate env

PYTHONPATH=. python basicsr/train.py -opt options/train/GoPro/NAFNet_width32_conv_pixelshuffle.yml
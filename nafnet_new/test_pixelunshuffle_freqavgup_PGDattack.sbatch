#!/bin/bash
#SBATCH --job-name=test_pixelunshuffle_freqavgup_PGDattack
#SBATCH --partition=gpu-vram-94gb
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/test_pixelunshuffle_freqavgup_PGDattack_%J.out
#SBATCH --error=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/test_pixelunshuffle_freqavgup_PGDattack_%J.err

set -euo pipefail

cd /home/pmorales/Shashank_code_BoA/freq-nafnet
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate env

PYTHONPATH=. python basicsr/test.py -opt /home/pmorales/Shashank_code_BoA/freq-nafnet/options/adv_attack/GoPro/NAFNet_width32_pixelunshuffle_freqavgup.yml --attacking
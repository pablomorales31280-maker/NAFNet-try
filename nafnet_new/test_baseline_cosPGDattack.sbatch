#!/bin/bash
#SBATCH --job-name=test_baseline_cosPGDattack
#SBATCH --partition=gpu-vram-48gb
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/test_baseline_cosPGDattack_%J.out
#SBATCH --error=/home/pmorales/Shashank_code_BoA/freq-nafnet/logs/test_baseline_cosPGDattack_%J.err

set -euo pipefail

cd /home/pmorales/Shashank_code_BoA/freq-nafnet
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate env

PYTHONPATH=. python basicsr/test.py -opt /home/pmorales/Shashank_code_BoA/freq-nafnet/options/adv_attack/GoPro/NAFNet_width32_baseline_true_cos.yml --attacking
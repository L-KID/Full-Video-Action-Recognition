#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:pascal:2
#SBATCH --cpus-per-task=2

module use /opt/insy/modulefiles
module load cuda/10.0

srun python3 main.py movingmnist RGB \
     --arch resnet18 --num_segments 16 \
     --gd 20 --lr 0.001 --epochs 40 \
     --batch-size 16 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb

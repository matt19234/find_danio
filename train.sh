#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=18:00:00

#SBATCH --mail-user=m3regehr@uwaterloo.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=train
#SBATCH -o logs/train_%j.out

source activate find_danio

time python train.py "$1" "$2" "$3"

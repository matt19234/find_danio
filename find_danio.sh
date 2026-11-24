#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6GB
#SBATCH --time=1:00:00

#SBATCH --mail-user=m3regehr@uwaterloo.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=test
#SBATCH -o logs/test_%j.out

source activate find_danio

time python find_danio.py "$@"

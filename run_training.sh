#!/bin/bash
#SBATCH --job-name=drugsformer
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --partition=student

cd $HOME/DrugRepositioning
source activate repositioning

python -u train.py
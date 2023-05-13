#!/bin/bash
#SBATCH --job-name=drugsformer
#SBATCH --qos=test
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=student

cd $HOME/DrugRepositioning

conda env create -f environment.yml

# source activate repositioning
# conda install ...
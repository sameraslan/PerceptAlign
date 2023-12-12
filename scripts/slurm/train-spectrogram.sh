#!/bin/bash

## Example SBATCH Parameters Below
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00



module load anaconda
module load cuda/11.6.0

conda activate PerceptAlign

# python train.py --base configs/drum_codebook.yaml -t True --gpus 0,

# Finetuning with one class
python train.py --resume logs/2023-11-28T18-35-42_drum_codebook/checkpoints/last.ckpt --base configs/drum_codebook.yaml -t True --gpus 0,
# From scratch: python train.py --base configs/music_codebook.yaml -t True --gpus 0,
## For finteuning from checkpoint: python train.py --resume logs/2021-06-06T19-42-53_vas_codebook/checkpoints/last.ckpt --base configs/music_codebook.yaml -t True --gpus 0,
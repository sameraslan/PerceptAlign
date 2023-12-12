#!/bin/bash

## Example SBATCH Parameters Below
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00

# Load necessary modules
# Add any necessary modules you need to load here
module load anaconda
module load cuda/11.6.0

conda activate denseflowenv

cd dataset_prep
denseflow ./dot-videos-1k-extracted/video_names.txt -b=2 -a=tvl1 -s=1 -v -o=./dot-videos-1k-extracted/flowb2
# denseflow ./feature_extraction/continue_test.txt -b=20 -a=farn -s=1 -v -o=./urmp_dataset/test
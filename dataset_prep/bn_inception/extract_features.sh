#!/bin/bash

## Example SBATCH Parameters Below
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00

module restore

# Load necessary modules
# Add any necessary modules you need to load here
module load anaconda
module load cuda/11.6.0

conda activate PerceptAlign

cd dataset_prep/bn_inception
# Flow: python extract_feature.py -i ../dot-videos-1k-extracted/flow -o ../dot-videos-1k-extracted/feature_flow_bninception_dim1024_21.5fps -m Flow -t ../../data/dot1k_all.txt
# RGB: python extract_feature.py -i ../dot-videos-1k-extracted/frames -o ../dot-videos-1k-extracted/feature_rgb_bninception_dim1024_21.5fps -m RGB -t ../../data/dot1k_all.txt
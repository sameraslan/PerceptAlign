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

cd dataset_prep
python video_to_frames.py -i ./dot-videos-1k-extracted/videos_10s_21.5fps -o ./dot-videos-1k-extracted/frames
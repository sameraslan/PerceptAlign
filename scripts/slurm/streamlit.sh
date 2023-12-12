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

conda activate PerceptAlign
streamlit run --server.port 5554 --server.address 0.0.0.0 ./sample_visualization.py

# Then in a separate terminal, run: "ssh -N -L {PORT}:{GPU# (ex. "gpu06")}:{PORT} {username}@login.rockfish.jhu.edu"
# You can find the GPU number by doing squeue and viewing the nodelist
# Make sure you're connected to school VPN when doing this
# Do not change 0.0.0.0 to localhost, it won't work. If having trouble reference R-Studio-Server.slurm.script
# Wait actually it seems that it works on some gpus and not others

# ssh -N -L 5554:gpu09:5554 saslan1@login.rockfish.jhu.edu
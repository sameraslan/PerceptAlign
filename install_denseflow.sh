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

export ZZROOT=/scratch4/jfisch20/saslan1/SpecVQGAN/SpecVQGAN/denseflow_app
export PATH=$ZZROOT/bin:$PATH
export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH

# Fetch install scripts
echo "Fetching install scripts..."
git clone https://github.com/innerlee/setup.git
cd setup

# Installing ffmpeg dependencies
echo "Installing nasm..."
./zznasm.sh

echo "Installing yasm..."
./zzyasm.sh

echo "Installing libx264..."
./zzlibx264.sh

echo "Installing libx265..."
./zzlibx265.sh

echo "Installing libvpx..."
./zzlibvpx.sh

# Finally install ffmpeg
echo "Installing ffmpeg..."
./zzffmpeg.sh

# Installing opencv 4.3.0
echo "Installing OpenCV 4.3.0..."
./zzopencv.sh
export OpenCV_DIR=$ZZROOT

# Installing boost
echo "Installing Boost..."
./zzboost.sh
export BOOST_ROOT=$ZZROOT

# Installing hdf5
echo "Installing HDF5..."
./zzhdf5.sh

# Finally, install denseflow
echo "Installing DenseFlow..."
./zzdenseflow.sh

echo "Installation completed!"

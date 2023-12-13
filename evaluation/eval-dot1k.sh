#!/bin/bash

## Example SBATCH Parameters Below
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --job-name="eval_spec"
#SBATCH --output="./evaluation/outputs/dot1k/output%j.txt"

module load anaconda
module load cuda/11.6.0
conda activate PerceptAlign

# Environment Variables
# Modify this experiment path to point to the directory of the experiment in the logs directory you want to evaluate
EXPERIMENT_PATH=./logs/2023-12-10T07-20-27_dot1k_transformer  

SPEC_DIR_PATH="./data/dot1k/features/*/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="./data/dot1k/features/*/feature_rgb_bninception_dim1024_21.5fps/"
FLOW_FEATS_DIR_PATH="./data/dot1k/features/*/feature_flow_bninception_dim1024_21.5fps/"
SAMPLES_FOLDER="dot1k_validation"
SPLITS="[\"validation\"]"
SAMPLER_BATCHSIZE=32
SAMPLER_NUMWORKERS=16
SAMPLES_PER_VIDEO=10
DATASET="dot1k"
TOP_K=64 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook
NOW=`date +"%Y-%m-%dT%H-%M-%S"`


# Running evaluation
echo "Starting sampling..."
python evaluation/generate_samples.py \
    sampler.config_sampler=evaluation/configs/sampler.yaml \
    sampler.model_logdir=$EXPERIMENT_PATH \
    sampler.splits=$SPLITS \
    sampler.samples_per_video=$SAMPLES_PER_VIDEO \
    sampler.batch_size=$SAMPLER_BATCHSIZE \
    sampler.num_workers=$SAMPLER_NUMWORKERS\
    sampler.top_k=$TOP_K \
    data.params.spec_dir_path=$SPEC_DIR_PATH \
    data.params.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
    data.params.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
    sampler.now=$NOW

echo "Starting evaluation..."


python -u evaluate.py \
    config=./evaluation/configs/eval_melception_${DATASET,,}.yaml \
    input2.path_to_exp=$EXPERIMENT_PATH \
    patch.specs_dir=$SPEC_DIR_PATH \
    patch.spec_dir_path=$SPEC_DIR_PATH \
    patch.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
    patch.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
    input1.params.root=$EXPERIMENT_PATH/samples_$NOW/$SAMPLES_FOLDER 

echo "Evaluation completed."

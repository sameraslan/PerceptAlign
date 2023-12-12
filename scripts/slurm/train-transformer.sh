#!/bin/bash

## Example SBATCH Parameters Below
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00


# Hyperparameters
K=10
Offset=3
Segments=256

module load anaconda
module load cuda/11.6.0

conda activate PerceptAlign



# For training from scratch
python -u train.py --base configs/dot1k_transformer.yaml -t True --gpus 0, model.params.first_stage_config.params.ckpt_path=./logs/2023-11-20T15-47-04_dot1k_codebook/checkpoints/best.ckpt model.params.hyperparameters.K=${K} model.params.hyperparameters.offset=${Offset} model.params.hyperparameters.segments=${Segments}

# Below script works for fine-tuning tranformer from checkpoint
# python train.py --resume logs/fine-tuning_vas_transformer_oneclass/checkpoints/epoch=000015-v1.ckpt --base configs/music_transformer.yaml -t True --gpus 0, --no-test True\
#     model.params.first_stage_config.params.ckpt_path=./logs/good_codebook_firstvas_lastmusiconeclass/checkpoints/last.ckpt

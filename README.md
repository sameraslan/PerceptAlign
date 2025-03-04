# PerceptAlign: Real-Time Temporal Alignment for Audio Generation from Silent Videos

<div align="center">
   <img src="https://github.com/sameraslan/PerceptAlign/assets/82460915/93385dac-6ef6-433d-9c9a-b192657a67b0" width="100%">
   <br>
    <p>Figure 1. Our perceptual alignment loss pipeline for improved real-time temporal alignment!</p> 
</div>

## Overview
PerceptAlign addresses the challenging task of generating temporally aligned audio from silent videos. Leveraging transformer models to predict spectrogram slices, this project introduces a novel approach to improving the synchronization between video events and their corresponding audio outputs. Central to the method is the Perceptual Alignment Loss (PAL) function, designed to align predicted audio with visual cues by drawing inspiration from human auditory-visual synchrony.

This repository provides a comprehensive guide to running the PerceptAlign pipeline, which includes setting up the environment, generating the FlashDot dataset for evaluation, training the models, and performing real-time inference. Whether you're looking to reproduce the results, fine-tune the models, or explore audio-visual temporal alignment, this guide covers everything you need to get started.

Learn more with the [PerceptAlign Paper](https://github.com/user-attachments/files/19041265/Perceptual-Alignment-Paper.pdf).

## Table of Contents

1. [Setting Up Your Environment](#setting-up-your-environment)
2. [Creating the FlashDot (Dot1k) Dataset](#creating-the-flashdot-dataset)
3. [Extracting Features](#extracting-features)
4. [Preparing the Dataset for Training](#preparing-the-dataset-for-training)
5. [Setting Up for Training](#setting-up-for-training)
6. [Training the Codebook](#training-the-codebook)
7. [Training the Transformer](#training-the-transformer)
8. [Running Evaluation](#running-evaluation)
9. [Running Inference on Streamlit](#running-inference-on-streamlit)

## Setting Up Your Environment

### a. Download All the Below Models and Unzip

| Model                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Dot1K Codebook](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/EdHVO0DlYENNvAeDXx0Jh60BNnWXLLjmOkQt6QIESVh7Pg?e=8CPT4Z)    |
| [Dot1K Transformer](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/EV8Uxe9QQG1Hl3NZMtpAb24Btkl228SbJV6z-Jn0FZgLsQ?e=vrhuD3) |
| [VAS Codebook](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/EcqGmkL54_lBu1-axIbwCBUB7CrkJSUspQX1zFyFzwz83Q?e=jL9IXJ)      |
| [VAS Transformer](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/EVRDGBDavWdKlzxx-eyasbUB4q1reVuGLfSdRmg6Ymc9oQ?e=JONDQv)   |
| [VGGishish16](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/EfZ9Hta6n6hEu3hMJp1LwkkBtKInT1w2wXIXJmuo8VGz5g?e=UEUVxH)       |
| [Melception Logs](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/saslan1_jh_edu/Eab6VbHZHTlNkoBpuqVo2voBtltdQZL9O9qUW_VPG0VZ3w?e=LUC5g5)   |

To unzip the files from the command line, use `tar -xzvf file.tar.gz`

- Put the unzipped transformer and codebook models in the log folder.
- Put the vggishish16 model checkpoint put in modules/autoencoder/lpaps
- Put the melception logs folder in evaluation/. If an empty log file already exists, replace it with this one.

### b. Setup All Dependencies

```bash
# Make sure anaconda is installed then run
conda env create -f PerceptAlign_env  # Creating env for PerceptAlign
```

To activate the environment, run

```bash
conda activate PerceptAlign_env
```

*Maybe provide my model for download such that they can use for fine-tuning. Might need to download some other stuff for specvqgan*

### c. Setup DenseFlow

```bash
# Make sure anaconda is installed then run
git clone https://github.com/open-mmlab/denseflow.git
cd denseflow
conda env create -f denseflow_env.yml  # Creating env for denseflow
```

To activate the environment, run

```bash
conda activate denseflow_env
```

To install denseflow (imported the denseflow setup repo):

#### i. Install CUDA

Make sure that the driver version is >400.
Otherwise, the speed would be painfully slow, although it compiles and runs.

#### ii. Install OpenCV with CUDA support

We need OpenCV with CUDA enabled.
An install script can be found [Here](https://github.com/innerlee/setup/blob/master/zzopencv.sh).
Since OpenCV itself has many dependencies,
you can follow the example below to compile.

```bash
# ZZROOT is the root dir of all the installation
# you may put these lines into your `.bashrc` or `.zshrc`.
export ZZROOT=$HOME/app
export PATH=$ZZROOT/bin:$PATH
export LD_LIBRARY_PATH=$ZZROOT/lib:$ZZROOT/lib64:$LD_LIBRARY_PATH

# fetch install scripts
git clone https://github.com/innerlee/setup.git
cd setup

# opencv depends on ffmpeg for video decoding
# ffmpeg depends on nasm, yasm, libx264, libx265, libvpx
./zznasm.sh
./zzyasm.sh
./zzlibx264.sh
./zzlibx265.sh
./zzlibvpx.sh
# finally install ffmpeg
./zzffmpeg.sh

# install opencv 4.3.0
./zzopencv.sh
# you may put this line into your .bashrc
export OpenCV_DIR=$ZZROOT

```

#### iii. Install Boost Library

```bash
# install boost
./zzboost.sh
# you may put this line into your .bashrc
export BOOST_ROOT=$ZZROOT
```

#### iv. Install HDF5 Library (Optional)

```bash
# install hdf5
./zzhdf5.sh
```

#### v. Install denseflow

```bash
# finally, install denseflow
./zzdenseflow.sh
```

If you run into any issues running denseflow, you may need to add specific libraries env variables to the ~/bashrc file

## Creating the FlashDot Dataset
We outline how to create the FlashDot (or Dot1k) dataset—creative name, I know. Exhibiting videos of a red dot flashing with sound at varying frequencies and durations, this dataset boils down audio-visual temporal alignment to the most simple case and can be used for benchmarking any model of this sort.

<div align="center">
   <img src="https://github.com/sameraslan/PerceptAlign/assets/82460915/c1828d6e-1586-4027-84c1-437c66bdbb24" height="300">
   <br>
    <p>Figure 2. Set of frames from an example FlashDot video</p>
</div>


1. **Activate Environment:**
   - Ensure the necessary environment is activated ex. `conda activate PerceptAlign_env`.

2. **Run `create_dot_dataset.py`:**
   - Navigate to the root directory where `create_dot_dataset.py` is located.
   - Use the command line to run this script. The script offers several arguments to customize the dataset:
     - `--num_videos`: Specify the number of videos to generate.
     - `--random`: Add this flag to enable random variations within each video.
     - `--total_frames`: Set the total number of frames in each video (default is 220 frames).
     - `--min_frames_per_segment`: Define the minimum number of frames per segment for random videos (default is 10 frames).
     - `--max_frames_per_segment`: Define the maximum number of frames per segment for random videos (default is 50 frames).
     - `--flash_frequency_max`: Set the maximum frequency of flash for fixed pattern videos (default is 90).
     - `--fps`: Frames per second (default is 24 fps).

3. **Example Command:**
   - To generate a dataset with 10 random videos, each having a total of 300 frames, a minimum of 15 frames per segment, a maximum of 60 frames per segment, and a flash frequency maximum of 100, use the following command:
     
     ```bash
     python create_dot_dataset.py --num_videos 10 --random --total_frames 300 --min_frames_per_segment 15 --max_frames_per_segment 60 --flash_frequency_max 100 --fps 24
     ```

   - This command will create 10 videos with the specified parameters, saving them in the 'dot-videos' directory.

4. **Output:**
   - The script will output videos in the 'dot-videos' directory within the root folder.
   - Each video will be named either `random_dot_X.mp4` or `fixed_dot_X.mp4` depending on the mode selected, where `X` is the video number.
   - An accompanying `.wav` audio file for each video will also be generated and subsequently deleted after being integrated into the video file.

Note that the script automatically handles the creation of necessary directories and the cleanup of intermediate files.

## Extracting Features

Start by organizing your video files for processing:

1. **Prepare Your Videos:**
   - Navigate to the `dataset_prep` folder.
   - Place your folder of videos in this `dataset_prep` folder. Ensure all videos are in `.mp4` format.

### a. Video to Audio and Cleanup

To extract audio and split videos into frames, follow these steps:

1. **Run the Extraction Script:**

   - Use the `extract_audio_and_video_split.py` script.
   - Execute the script with the required arguments. Replace `{}` with your specific values:

     ```bash
     python extract_audio_and_video_split.py -i <input_dir> -o <output_dir> -d <duration> -a <audio_sample_rate> -v <video_fps> -n <num_workers>
     ```
   - **Parameters:**

     - `-i, --input_dir`: Path to the directory containing your videos. Default is `data/VAS/dog/videos`.
     - `-o, --output_dir`: Path to the directory where the extracted features will be stored. Default is `data/features/dog`.
     - `-d, --duration`: Duration of each video chunk in seconds. Default is `10`.
     - `-a, --audio_sample_rate`: Desired sample rate for the audio in Hz. Default is `22050`.
     - `-v, --video_fps`: Desired frames per second for the output video. Default is `21.5`.
     - `-n, --num_workers`: Number of worker processes to use. Default is `32`.

   If you're unsure about which values to put, keep the default, as this is compatible with the training pipeline.
2. **Script Execution:**

   - The script processes each video file in the input directory.
   - It splits each video into chunks of specified duration (`-d`).
   - For each chunk, the script:
     - Extracts the original audio segment.
     - Aligns and cuts the video according to the audio duration.
     - Converts the audio to the specified sample rate (`-a`).
     - Adjusts the video to the specified frames per second (`-v`).
   - All outputs are saved in the specified output directory (`-o`), organized into subfolders for easy access.
3. **Output Structure:**

   - The script generates several subdirectories in the output directory:
     - `audio_ori`: Contains the original audio segments.
     - `audio_<duration>s`: Contains audio files cut according to the specified duration.
     - `audio_<duration>s_<sample_rate>hz`: Contains audio files with the adjusted sample rate.
     - `videos_align`: Contains video files aligned with the audio segments.
     - `videos_<duration>s_<fps>fps`: Contains video files with adjusted frame rate.

### b. Extract Mel Spectrograms

After running the `extract_audio_and_video_split.py` script, use the `audio_10s_22050hz` subfolder from its output as your input for now running `extract_mel_spectrogram.py`.

```python
# Command to execute extract_mel_spectrogram.py
python extract_mel_spectrogram.py -i <input_dir> -o <output_dir> -l <length> -n <num_workers>
```

Make sure the output folder has name melspec_10s_22050hz, otherwise this won't work.

**Parameters:**

- `-i, --input_dir`: Path to the directory containing your audio files. Default is `data/features/dog/audio_10s_22050hz`.
- `-o, --output_dir`: Path to the directory where the Mel spectrograms will be stored. Default is `data/features/dog/melspec_10s_22050hz`.
- `-l, --length`: Length of the audio in samples. Default is `22050`.
- `-n, --num_worker`: Number of worker processes to use. Default is `32`.

**About Mel Spectrograms:**
Mel Spectrograms are a critical feature representation in audio processing, especially in the context of machine learning for sound analysis. They represent the power spectrum of sound in a way that is more closely aligned with human auditory perception. The Mel scale more closely mimics the human ear's response to different frequencies, emphasizing the nuances in lower frequencies more than higher ones.

By converting audio signals into Mel spectrograms, our system can more effectively learn patterns and characteristics of sound that are relevant to human listeners. This conversion is especially important for tasks like sound classification, music generation, or any application where understanding the content of audio in a human-like way is crucial.

### c. Video to Frames

After extracting audio and video with `extract_audio_and_video_split.py`, use the resulting `videos_10s_21.5fps` subfolder as your input for running `video_to_frames.py`.

```bash
# Command to execute video_to_frames.py
python video_to_frames.py -i <input_dir> -o <output_dir> -f <fps>
```

**Parameters:**

- `-i, --input_dir`: Path to the directory containing your video files. This should be the output directory of the `extract_audio_and_video_split.py` script, typically `videos_10s_21.5fps`.
- `-o, --output_dir`: Path to the directory where extracted frames will be stored.
- `-f, --fps`: Frames per second rate for frame extraction. Default is `21.5`.

### d. Video to Optical Flow Images

To extract optical flow images from videos using DenseFlow, follow these steps:

1. **Environment Setup:**

   - Ensure that the `denseflow_env` environment is activated. Use the following command:
     ```bash
     conda activate denseflow_env
     ```
   - Make sure you have a GPU available as DenseFlow requires it for processing.
2. **Prepare Video File Paths:**

   - Create a text file containing the path of each video file, with each path on a new line.
   - To generate this list automatically, use the `get_video_paths.py` file, with input directory as your videos directory and output as the desired text file path
3. **Run DenseFlow:**

   - Provide the generated text file as input to DenseFlow. Here is an example command:

     ```bash
     denseflow ./feature_extraction/dot1k_optical_flow_paths.txt -b=10 -a=tvl1 -s=1 -v -o=./dot1k/flow
     ```

     Keep an eye on the output. Some videos may fail if they're corrupted, in which case you'll have to modify the text file, removing the corrupted file and files with flow already extracted, and rerun for the remainder of the videos.

**DenseFlow Parameters:**

- `-a, --algorithm`: Optical flow algorithm (nv/tvl1/farn/brox). Default is `tvl1`.
- `-b, --bound`: Maximum of optical flow. Default is `32`.
- `--cf, --classFolder`: Output in `outputDir/class/video/flow.jpg` format.
- `-f, --force`: Process regardless of the marked `.done` file.
- `--if, --inputFrames`: Inputs are frames.
- `--newHeight, --nh`: New height. Default is `0`.
- `--newShort, --ns`: Short side length. Default is `0`.
- `--newWidth, --nw`: New width. Default is `0`.
- `-o, --outputDir`: Root directory of output. Default is `.` (current directory).
- `-s, --step`: Right - left (0 for img, non-0 for flow). Default is `0`.
- `--saveType, --st`: Save format type (png/h5/jpg). Default is `jpg`.
- `-v, --verbose`: Enable verbose output.

**Common Issues and Troubleshooting:**

- Ensure that the GPU is properly configured and recognized by DenseFlow.
- Verify that all paths in the input text file are correct and accessible.
- If DenseFlow fails to process a video, check its format and encoding to ensure compatibility.

### e. Extract RGB Features

After setting up your environment for GPU usage, change directory to `bn_inception` and run the `extract_feature.py` script to extract RGB features. Ensure that your input directory is the output directory from the `extract_audio_and_video_split.py` script.

```bash
# Command to execute extract_feature.py for RGB features
python extract_feature.py -i <input_dir> -o <output_dir> -m RGB -t <test_list> --input_size <input_size> --crop_fusion_type <crop_fusion_type> --dropout <dropout> -j <workers> --flow_prefix <flow_prefix>
```

**Parameters:**

- `-i, --input_dir`: Path to the directory containing your frame images.
- `-o, --output_dir`: Path to the directory where the RGB features will be stored.
- `-m, --modality`: Set this to `RGB` for extracting RGB features.
- `-t, --test_list`: Path to the test list file.
- `--input_size`: The input size for the network.
- `--crop_fusion_type`: Type of crop fusion (avg, max, topk).
- `--dropout`: Dropout ratio.
- `-j, --workers`: Number of data loading workers.
- `--flow_prefix`: Prefix for flow images.

**Instructions:**

- This script processes each frame image in the input directory and uses a pre-trained model to extract RGB features.
- The extracted features are stored in the specified output directory.
- This step is crucial for tasks that require understanding visual content at the frame level, such as video classification or activity recognition.

### f. Extract Optical Flow Features

Optical Flow features are essential for understanding the motion between video frames. They are particularly relevant in systems that analyze movements or changes in scene dynamics.

To extract Optical Flow features, run the `extract_feature.py` script in the `bn_inception` directory with the `Flow` argument:

```bash
# Command to execute extract_feature.py for Optical Flow features
python extract_feature.py -i <input_dir> -o <output_dir> -m Flow -t <test_list> --input_size <input_size> --crop_fusion_type <crop_fusion_type> --dropout <dropout> -j <workers> --flow_prefix <flow_prefix>
```

**Parameters:**

- `-i, --input_dir`: Path to the directory containing your optical flow images.
- `-o, --output_dir`: Path to the directory where the Optical Flow features will be stored.
- `-m, --modality`: Set this to `Flow` for extracting Optical Flow features.
- `-t, --test_list`: Path to the test list file.
- `--input_size`: The input size for the network.
- `--crop_fusion_type`: Type of crop fusion (avg, max, topk).
- `--dropout`: Dropout ratio.
- `-j, --workers`: Number of data loading workers.
- `--flow_prefix`: Prefix for flow images.

## Preparing the Dataset for Training

### a. Organize Dataset Videos and Features

Organize your dataset videos and corresponding extracted features (e.g., `mel_spec_10s_22050hz`, `feature_flow_bninception_dim1024_21.5fps`, `feature_rgb_bninception_dim1024_21.5fps`) into separate folders (if they aren't already this way from the previous step).

### b. Setup Training Dataset Structure

1. Create a new directory in your `data` folder, named after your training dataset (e.g., `dot1k` or `vas`).
2. Inside `data/{dataset_name}`, create two subfolders: `features` and `videos`.
3. Move the feature subfolders (`melspec_10s_22050hz`, `feature_flow_bninception_dim1024_21.5fps`, `feature_rgb_bninception_dim1024_21.5fps`) into `data/{dataset_name}/features`.
4. Move the `videos_10s_21.5fps` subfolder into `data/{dataset_name}/videos`.

### c. Create Train/Test Split Files

If your video folder already has class subfolders, use these to create train/test split `.txt` files. Otherwise, manually create a `.txt` file listing desired classes and file names (without extension), each on a new line.

```python
# Command to execute create_train_test_split_txts.py
python create_train_test_split_txts.py --base_folder <path_to_video_folder> --train_percentage <train_percent> --valid_percentage <valid_percent> --prefix <prefix_name>
```

### d. Synchronize Feature Classes

Use `sync_feature_classes.py` to ensure that your feature folders and class categories are aligned. This script helps in copying the correct feature files to the respective class subfolders in your dataset. Provide the `_all.txt` file and the feature folder as inputs. There's an additional `--mel` flag which should be used if you're working with the melspectrograms feature folder.

#### Basic Usage

For standard features like flow or RGB, use the command as follows:

```python
# Synchronizing non-melspectrogram features
python sync_feature_classes.py <path_to_all_txt_file> <path_to_feature_folder> --subfolder_name <subfolder_name> --destination_folder <destination_folder>
```

#### Synchronizing Melspectrograms

If you are working with the mel spectrograms feature folder, use the `--mel` flag to ensure the script looks for files with a `_mel` suffix. This is important to correctly match melspectrogram files which are named with this suffix.

```python
# Synchronizing melspectrogram features
python sync_feature_classes.py <path_to_all_txt_file> <path_to_feature_folder> --subfolder_name <subfolder_name> --destination_folder <destination_folder> --mel
```

Make sure to replace `<path_to_all_txt_file>`, `<path_to_feature_folder>`, `<subfolder_name>`, and `<destination_folder>` with the appropriate paths and names relevant to your dataset structure.

### e. Final Folder Structure Check

Verify that your folder structure follows this format:

- `data/{dataset_name}/`
  - `features/`
    - `{class_name}/`
      - `mel_spec_10s_22050hz/`
      - `feature_flow_bninception_dim1024_21.5fps/`
      - `feature_rgb_bninception_dim1024_21.5fps/`
  - `videos/`
    - `{class_name}/`
      - `videos_10s_21.5fps/`

Additionally, ensure you have `{dataset_name}_train.txt`, `{dataset_name}_valid.txt`, `{dataset_name}_test.txt`, and `{dataset_name}_all.txt` in your `data` folder. This structure is essential for the upcoming training steps.

## Setting Up for Training

### a. Create Configuration Files

1. **Run setup_training_configs.py:**

   - cd into the configs folder and find this script
   - This script will create YAML files and a Python file for your specific dataset.
   - Execute the script with the provided YAML files (`dot1k_codebook.yaml` and `dot1k_transformer.yaml` found in `configs`) and the Python file (`dot1k.py` in `specvqgan/data`) as inputs.

   ```bash
   # Command to execute setup_training_configs.py
   python setup_training_configs.py --yaml_files <path_to_yaml_file1> <path_to_yaml_file2> --python_file <path_to_python_file> --dataset_name <dataset_name>
   ```

   **Parameters:**

   - `--yaml_files`: Paths to the YAML files.
   - `--python_file`: Path to the Python file.
   - `--dataset_name`: Name of your dataset.
2. **Outputs:**

   - The script will generate new YAML files and a Python file, replacing references to 'dot1k' with your `{dataset_name}`.
   - Place the new YAML files in the `configs` folder (if not already there).
   - Place the new `{dataset_name}.py` file in `specvqgan/data/` (if not already there).

### b. Update Training Script

1. **Modify train.py:**
   - Locate the line containing `"split.target.split('.')[-1].startswith('VASSpecsCondOnFeats')"` in `train.py`.
   - Replace `'VASSpecsCondOnFeats'` with `'{dataset_name}SpecsCondOnFeats'` to match your dataset name.

### c. Customize Configuration Parameters

1. **Review and Edit YAML Files:**
   - Examine the generated YAML files.
   - Feel free to adjust parameters according to your needs (e.g., early stopping, learning rate, etc.).

### d. Ready for Training

With the configuration files set up and the training script updated, your setup is now ready for training.

## Training the Codebook

### a. Start Training

1. **Activate Environment:**

   - Ensure the necessary environment is activated ex. `conda activate PerceptAlign_env`.
2. **Run Training:**
   i. **From Scratch:**

   - Use a GPU to start the training process.
   - Basic training command:
     ```bash
     python train.py --base configs/{dataset_name}_codebook.yaml -t True --gpus 0,
     ```

or

ii. **Fine-Tuning from a Checkpoint:**

- To resume training from a checkpoint, use the `--resume` flag.
- Example fine-tuning command:
  ```bash
  python train.py --resume logs/2023-10-22T17-21-50_dot1k_codebook/checkpoints/last.ckpt --base configs/dot1k_codebook.yaml -t True --gpus 0,
  ```
- Checkpoints can be found in the `logs` directory.

3. **Monitor Training:**
   - Review the `logs` directory to see original and reconstructed images/audio.

## Training the Transformer

### a. Start Training

1. **Activate Environment:**

   - Ensure the necessary environment is activated ex. `conda activate PerceptAlign_env`
2. **Run Training:**
   i. **From Scratch:**

   - Use a GPU to start training the transformer.
   - Basic training command:
     ```bash
     python train.py --base configs/{dataset_name}_transformer.yaml -t True --gpus 0, model.params.first_stage_config.params.ckpt_path={trained codebook checkpoint}
     ```

or

ii. **Fine-Tuning from a Checkpoint:**

- To resume training from a checkpoint, use the `--resume` flag.
- Example fine-tuning command:
  ```bash
  python train.py --resume logs/fine-tuning_vas_transformer_oneclass/checkpoints/epoch=000015-v1.ckpt --base configs/dot1k_transformer.yaml -t True --gpus 0, --no-test True model.params.first_stage_config.params.ckpt_path=./logs/2023-10-22T17-21-50_dot1k_transformer/checkpoints/last.ckpt
  ```

3. **Monitor Training:**
   - Check the `logs` directory for attention, reconstruction, and sampled images/audio, and additional training information.

## Running Evaluation

1. **Activate Environment:**

   - Make sure the necessary environment is activated.
2. **Modify Evaluation File**

   - Open evaluation/eval-dot1k.sh or evaluation/eval-vas.sh
   - If you're running evaluation on your own dataset, copy one of the files and replace all occurences of 'vas' or 'dot1k' with your {dataset-name}
   - Modify the EXPERIMENT_PATH to the path of your experiment (likely in the logs file)
3. **Modify Config File**

   - Open evaluation/configs/eval_melception_dot1k.yaml or evaluation/configs/eval_melception_vas.yaml
   - If you're running evaluation on your own dataset, copy one of the files and replace all occurences of 'vas' or 'dot1k' with your {dataset-name}
   - Make sure all occurences of 'classes' correspond to your own classes
     - Eg. replace classes:['baby', 'dog'] with classes:['violin', 'cello']
   - If you don't want to evaluate on all metrics, set have{metric} to False
4. **Run**

   - Run your eval-{dataset-name}.sh file using a GPU and observe the output
   - Depending on the size of your dataset, GPU setup, etc., this may take a few hours due to feature extraction and dynamic time warping calculations.


## Running Inference On Streamlit

### Generalizable Streamlit Script Setup

1. **Activate Environment:**

   - Make sure the necessary environment is activated.
2. **Modify File**

   - Open sample_visualization.py.
   - Locate the sections with (there should be two)

     ```python
     if 'vas.VAS' in config.data.params.train.target:
     ```
   - Add an `elif` with your own dataset name. Ex. replacing

     ```python
     elif 'dot1k.dot1k' in config.data.params.train.target:
        datapath = './data/dot1k/'
        raw_vids_dir = os.path.join(datapath, 'video')
     ```

     with

     ```python
     elif '{datset_name}.{datset_name}' in config.data.params.train.target:
        datapath = './data/{datset_name}/'
        raw_vids_dir = os.path.join(datapath, 'video')
     ```

     and add

     ```python
     or '{datset_name}.{datset_name}' in config.data.params.train.target
     ```

     to the line:

     ```python
     if 'vas.VAS' in config.data.params.train.target or 'dot1k.dot1k' in config.data.params.train.target:
     ```
3. **Launch Streamlit:**

   - From a GPU, run the Streamlit script.
   - Command to start Streamlit:
     ```bash
     streamlit run --server.port 5554 --server.address 0.0.0.0 ./sample_visualization.py
     ```
   - You can choose any port you like.
4. **HPC Connection (skip if not using):**

   - If using an HPC, open a separate terminal and run:
     ```bash
     ssh -N -L {PORT}:{GPU#}:{PORT} {username}@{hpc}
     ```
   - Ensure you are connected to the HPC network or using its VPN.
5. **Access Streamlit App:**

   - Navigate to `localhost:{PORT_CHOSEN}` in your browser to run inference.
6. **Rename Models in Streamlit:**

   - To rename your models in the Streamlit app, modify the `name2type` dictionary in the `rename_models()` function in `sample_visualization.py`.
7. **Rename Checkpoint**

   - Make sure to name the checkpoint you want to use `best.ckpt`. Streamlit will select this one by default.

---

### Temporally Aligned Video to Audio! 🫨🫨🫨


import os
import argparse
import random

def split_videos(base_folder, train_percentage, valid_percentage, prefix="music"):
    """
    Split video files into train, validation, and test sets and write their names into text files.

    Args:
        base_folder (str): Path to the folder containing class subfolders with video files.
        train_percentage (float): Percentage of files for the training set.
        valid_percentage (float): Percentage of files for the validation set.
        prefix (str): Prefix for the output text files. Default is "music".
    """
    if train_percentage > 1 or valid_percentage > 1 or train_percentage < 0 or valid_percentage < 0:
        raise ValueError("train_percentage and valid_percentage must be between 0 and 1.")
    
    if train_percentage + valid_percentage >= 1:
        raise ValueError("The sum of train_percentage and valid_percentage must be less than 1.")

    video_files = []

    # Get all the class subfolders from the base_folder
    class_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    for class_folder in class_folders:
        folder_path = os.path.join(base_folder, class_folder)
        for video in os.listdir(folder_path):
            if video.endswith('.mp4'):
                video_name_without_ext = os.path.splitext(video)[0]
                video_files.append(f"{class_folder}/{video_name_without_ext}")

    # Shuffle video names for randomness
    random.shuffle(video_files)

    num_total = len(video_files)
    num_train = int(num_total * train_percentage)
    num_valid = int(num_total * valid_percentage)
    
    train_files = video_files[:num_train]
    valid_files = video_files[num_train:num_train + num_valid]
    test_files = video_files[num_train + num_valid:]

    output_folder = os.path.join(base_folder, "../video_names")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(os.path.join(output_folder, f"{prefix}_train.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(output_folder, f"{prefix}_valid.txt"), "w") as f:
        f.write("\n".join(valid_files))
    with open(os.path.join(output_folder, f"{prefix}_test.txt"), "w") as f:
        f.write("\n".join(test_files))
    with open(os.path.join(output_folder, f"{prefix}_all.txt"), "w") as f:
        f.write("\n".join(video_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split video files into train, validation, and test sets.")
    
    parser.add_argument("--base_folder", type=str, help="Path to the folder containing class subfolders with video files.", required=True)
    parser.add_argument("--train_percentage", type=float, help="Percentage of files for the training set.", default=.75)
    parser.add_argument("--valid_percentage", type=float, help="Percentage of files for the validation set.", default=.15)
    parser.add_argument("--prefix", type=str, help="Prefix name for the output text files.", default="music")
    
    args = parser.parse_args()
    split_videos(args.base_folder, args.train_percentage, args.valid_percentage, args.prefix)

# Usage: python ./dataset_prep/create_train_test_split_txts.py --base_folder ./data/dot1k/videos --train_percentage .7 --valid_percentage .2 --prefix dot1k



import os
import shutil
import argparse
import glob

def read_input_file(input_file_path):
    """ Read the input file and return a list of (class_name, file_name) tuples. """
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split('/') for line in lines]

def find_file_by_prefix(source_folder, file_prefix, mel=False):
    """ Find a file in the source folder that starts with the given prefix. """
    # Append '_mel' to the file prefix if mel is True
    if mel:
        file_prefix += '_mel'
    
    # Use glob to find matching files
    matching_files = glob.glob(os.path.join(source_folder, file_prefix + '.*'))
    return matching_files[0] if matching_files else None

def copy_files(class_file_pairs, source_folder, subfolder_name, destination_folder, overwrite, mel):
    """ Copy files from the source to the destination directory within the class subfolders. """
    for class_name, file_prefix in class_file_pairs:
        source_file_path = find_file_by_prefix(source_folder, file_prefix, mel)
        
        if source_file_path is None:
            print(f"Warning: No file starting with {file_prefix} found in source folder.")
            continue
        
        # Extract the full filename from the path
        file_name = os.path.basename(source_file_path)
        
        # Create the class directory and subdirectory if they don't exist
        class_dir_path = os.path.join(destination_folder, class_name)
        subfolder_path = os.path.join(class_dir_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        destination_file_path = os.path.join(subfolder_path, file_name)
        
        # Copy the file if it doesn't exist or if overwrite is True
        if overwrite or not os.path.exists(destination_file_path):
            shutil.copy2(source_file_path, destination_file_path)
            print(f"File {file_name} copied to {destination_file_path}.")
        else:
            print(f"File {file_name} already exists at destination and will not be overwritten.")

def main(args):
    class_file_pairs = read_input_file(args.input_file_path)
    
    overwrite = args.overwrite
    if overwrite is None:  # If the overwrite argument is not set, ask the user
        while True:
            user_input = input("Do you want to overwrite existing files? (yes/no): ").lower()
            if user_input in ['yes', 'no']:
                overwrite = user_input == 'yes'
                break
            else:
                print("Please enter 'yes' or 'no'.")

    # Copy the files
    copy_files(class_file_pairs, args.source_folder, args.subfolder_name, args.destination_folder, overwrite, args.mel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy files to a structured directory based on class and subfolder specifications.')
    
    parser.add_argument('input_file_path', type=str, help='Path to the input text file containing class/filename pairs.')
    parser.add_argument('source_folder', type=str, help='Path to the source folder where files are currently located.')
    parser.add_argument('--subfolder_name', type=str, help='Name of the subfolder to create within each class folder.')
    parser.add_argument('--destination_folder', type=str, help='Path to the destination folder where files should be copied.')
    parser.add_argument('--mel', action='store_true', help='Look for files with a "_mel" suffix after the filename.')
    parser.add_argument('-o', '--overwrite', type=lambda x: (str(x).lower() == 'true'), nargs='?', const=True, default=None,
                        help='Overwrite files at the destination if they already exist. If not specified, the script will ask.')
    
    args = parser.parse_args()
    main(args)

# Usage:
# rgb: python3 dataset_prep/sync_feature_classes.py data/music_all.txt urmp_dataset/feature_rgb_bninception_dim1024_21.5fps/ feature_rgb_bninception_dim1024_21.5fps data/music/features
# flow: python3 dataset_prep/sync_feature_classes.py data/music_all.txt urmp_dataset/feature_flow_bninception_dim1024_21.5fps/ feature_flow_bninception_dim1024_21.5fps data/music/features
# python3 dataset_prep/sync_feature_classes.py data/dot1k_all.txt dataset_prep/dot-videos-1k-extracted/melspec_10s_22050hz --subfolder_name melspec_10s_22050hz --destination_folder data/dot1k/features
import os
import shutil

# Define the source directory containing the 8 class subfolders
source_dir = 'data/music/features_8'

# Define the target directory for all classes
target_dir = 'data/music/features/all'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Define the list of subfolder names to be copied
subfolders_to_copy = ['feature_flow_bninception_dim1024_21.5fps',
                      'feature_rgb_bninception_dim1024_21.5fps',
                      'melspec_10s_22050hz']

# Loop through the 8 class subfolders
for class_folder in range(1, 9):
    class_folder_name = f'class_{class_folder}'
    class_folder_path = os.path.join(source_dir, class_folder_name)
    
    # Loop through the subfolders to copy
    for subfolder_name in subfolders_to_copy:
        subfolder_path = os.path.join(class_folder_path, subfolder_name)
        target_subfolder = os.path.join(target_dir, subfolder_name)

        # Create the target subdirectory if it doesn't exist
        os.makedirs(target_subfolder, exist_ok=True)
        
        # Loop through the files in the subfolder and copy them to the target directory
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, subfolder_name, file)
                shutil.copy2(source_file, target_file)

print('Files from class 1-8 have been copied to the "all" directory.')
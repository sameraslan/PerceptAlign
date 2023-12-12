import torch
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize, StandardScaler


def path_to_sharedkey(path, dataset_name, classes=None):
    if dataset_name.lower() == 'vggsound':
        # a generic oneliner which extracts the unique filename for the dataset.
        # Works on both FakeFolder and VGGSound* datasets
        sharedkey = Path(path).stem.replace('_mel', '').split('_sample_')[0]
    else:
        # in the case of vas the procedure is a bit more tricky and involves relying on the premise that
        # the folder names (.../VAS_validation/cls_0, .../cls_1 etc) are made after enumerating sorted list
        # of classes.
        classes = sorted(classes)
        target_to_label = {f'cls_{i}': c for i, c in enumerate(classes)}
        # replacing class folder with the name of the class to match the original dataset (cls_2 -> dog)
        for folder_cls_name, label in target_to_label.items():
            path = path.replace(folder_cls_name, label).replace('melspec_10s_22050hz/', '')
        # merging video name with class name to make a unique shared key
        sharedkey = Path(path).parent.stem + '_' + Path(path).stem.replace('_mel', '').split('_sample_')[0]
    # else:
    #     raise NotImplementedError
    return sharedkey

def combined_distance(u, v):
        return 0.5 * euclidean(u, v) + 0.5 * cosine(u, v)

def calculate_dtw(mel_dict_1, mel_dict_2, dataset_name, classes=None):
    # Define a combined Euclidean and Cosine distance function
    mel_1 = mel_dict_1["mel_spectrograms"]
    mel_2 = mel_dict_2["mel_spectrograms"]

    paths_1 = mel_dict_1['file_path_']
    paths_2 = mel_dict_2['file_path_']

    path_to_mel_1 = {p: m for p, m in zip(paths_1, mel_1)}
    path_to_mel_2 = {p: m for p, m in zip(paths_2, mel_2)}

    sharedkey_to_mel_1 = {path_to_sharedkey(p, dataset_name, classes): [] for p in paths_1}
    sharedkey_to_mel_2 = {path_to_sharedkey(p, dataset_name, classes): path_to_mel_2[p] for p in paths_2}

    # Grouping mel spectrograms by shared key
    for path, mel in path_to_mel_1.items():
        sharedkey = path_to_sharedkey(path, dataset_name, classes)
        sharedkey_to_mel_1[sharedkey].append(mel)

    # Initialize lists
    euclidean_dtw_list, cosine_dtw_list, combined_dtw_list = [], [], []

    # Prepare a list of all combinations
    combinations = []
    for sharedkey, mel_2 in sharedkey_to_mel_2.items():
        if sharedkey in sharedkey_to_mel_1:
            for mel_1 in sharedkey_to_mel_1[sharedkey]:
                combinations.append((sharedkey, mel_1, mel_2))

    # Iterate over combinations with tqdm
    for sharedkey, mel_1, mel_2 in tqdm(combinations, desc="Computing DTW"):
        # Normalize the mel spectrograms
        scaler = StandardScaler()
        norm_mel_1 = scaler.fit_transform(mel_1.T)
        norm_mel_2 = scaler.fit_transform(mel_2.T)

        # Calculate Euclidean DTW
        distance_euclidean, _ = fastdtw(norm_mel_1, norm_mel_2, dist=euclidean)
        euclidean_dtw_list.append(distance_euclidean)

        # Calculate Cosine DTW
        distance_cosine, _ = fastdtw(norm_mel_1.T, norm_mel_2.T, dist=cosine)
        cosine_dtw_list.append(distance_cosine)

        # Calculate Combined DTW
        distance_combined, _ = fastdtw(norm_mel_1.T, norm_mel_2.T, dist=combined_distance)
        combined_dtw_list.append(distance_combined)

    # Calculate overall averages
    overall_dtw_euclidean = np.mean(euclidean_dtw_list) if euclidean_dtw_list else float('nan')
    overall_dtw_cosine = np.mean(cosine_dtw_list) if cosine_dtw_list else float('nan')
    overall_dtw_combined = np.mean(combined_dtw_list) if combined_dtw_list else float('nan')

    return {
        'euclidean_dynamic_time_warping_distance': overall_dtw_euclidean,
        'cosine_dynamic_time_warping_distance': overall_dtw_cosine,
        'combined_dynamic_time_warping_distance': overall_dtw_combined
    }
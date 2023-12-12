import torch
import numpy as np
from pathlib import Path

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

def calculate_pc(mel_dict_1, mel_dict_2, dataset_name, classes=None):
    # Crop GT based on size of generation
    min_length = min(mel_dict_1["mel_spectrograms"][0].shape[-1], mel_dict_2["mel_spectrograms"][0].shape[-1])

    # Crop all spectrograms to the minimum length
    mel_dict_1["mel_spectrograms"] = [m[:, :min_length] for m in mel_dict_1["mel_spectrograms"]]
    mel_dict_2["mel_spectrograms"] = [m[:, :min_length] for m in mel_dict_2["mel_spectrograms"]]

    mel_1 = mel_dict_1["mel_spectrograms"]
    mel_2 = mel_dict_2["mel_spectrograms"]

    paths_1 = mel_dict_1['file_path_']
    paths_2 = mel_dict_2['file_path_']

    path_to_mel_1 = {p: m for p, m in zip(paths_1, mel_1)}
    path_to_mel_2 = {p: m for p, m in zip(paths_2, mel_2)}

    sharedkey_to_mel_1 = {path_to_sharedkey(p, dataset_name, classes): [] for p in paths_1}
    sharedkey_to_mel_2 = {path_to_sharedkey(p, dataset_name, classes): path_to_mel_2[p] for p in paths_2}

    # Grouping samples by shared key
    for path, mel in path_to_mel_1.items():
        sharedkey = path_to_sharedkey(path, dataset_name, classes)
        sharedkey_to_mel_1[sharedkey].append(mel)

    pc_list = []

    # Calculate PC for each shared key
    for sharedkey, mel_2 in sharedkey_to_mel_2.items():
        gen_mels = torch.stack(sharedkey_to_mel_1[sharedkey])
        gt_mel = mel_2.unsqueeze(0).expand_as(gen_mels)

        # Flatten tensors for correlation calculation
        gen_mels_np = gen_mels.view(gen_mels.size(0), -1).numpy()
        gt_mel_np = gt_mel.view(gt_mel.size(0), -1).numpy()

        # Calculate PC for this set of generated samples
        correlation = np.corrcoef(gen_mels_np, gt_mel_np)[0, 1]
        pc_list.append(correlation)

    # Calculate overall average PC
    overall_pc = np.mean(pc_list)

    return {
        'pearson_correlation': overall_pc
    }

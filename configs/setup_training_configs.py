import argparse
import os
import re


def process_file(file_path, dataset_name):
    with open(file_path, 'r') as file:
        content = file.read()

    # Using re.sub() for case-insensitive replacement
    modified_content = re.sub(r'dot1k', dataset_name, content, flags=re.IGNORECASE)

    # Determine the new file name (assuming case-insensitivity isn't needed here)
    dir_name, file_name = os.path.split(file_path)
    new_file_name = file_name.replace('dot1k', dataset_name)
    new_file_path = os.path.join(dir_name, new_file_name)

    with open(new_file_path, 'w') as file:
        file.write(modified_content)

def main():
    parser = argparse.ArgumentParser(description="Modify files by replacing 'dot1k' with a specified dataset name.")
    parser.add_argument('--yaml_files', nargs='+', help='List of YAML files to be processed', required=True)
    parser.add_argument('--python_file', help='Python file to be processed', required=True)
    parser.add_argument('--dataset_name', help='The dataset name to replace "dot1k" with', required=True)

    args = parser.parse_args()

    for yaml_file in args.yaml_files:
        process_file(yaml_file, args.dataset_name)

    process_file(args.python_file, args.dataset_name)

if __name__ == '__main__':
    main()


# Example execution
# python ./configs/setup_training_configs.py --yaml_files ./configs/dot1k_codebook.yaml ./configs/dot1k_transformer.yaml --python_file ./specvqgan/data/dot1k.py --dataset_name dot1k

tar -czvf packed_models/2023-12-10T01-45-35_dot1k_transformer.tar.gz logs/2023-12-10T01-45-35_dot1k_transformer/

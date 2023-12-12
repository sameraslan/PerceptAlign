import os
import argparse

def generate_video_paths(input_dir, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.mp4'):
                    f.write(os.path.join(root, file) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate a list of video paths.')
    parser.add_argument("-i", '--input_dir', type=str, help='Input directory containing videos')
    parser.add_argument("-o", '--output_file', type=str, help='Output file to write the list of video paths')

    args = parser.parse_args()
    generate_video_paths(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
import os
import subprocess
import argparse

def extract_frames(input_dir, output_dir, fps=21.5):
    """
    Extract frames from videos in the input directory and save them in the output directory.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .mp4 files in the input directory
    videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    for video in videos:
        video_path = os.path.join(input_dir, video)
        
        # Create a subdirectory for each video's frames
        video_name = os.path.splitext(video)[0]
        frame_dir = os.path.join(output_dir, video_name)
        
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        
        # Construct the ffmpeg command
        cmd = [
            'ffmpeg', 
            '-i', video_path, 
            '-vf', f'fps={fps}', 
            os.path.join(frame_dir, 'img_%05d.jpg')
        ]
        
        # Execute the command
        subprocess.run(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing .mp4 videos")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save extracted frames")
    parser.add_argument("-f", "--fps", type=float, default=21.5, help="Frames per second rate for extraction")
    
    args = parser.parse_args()
    
    extract_frames(args.input_dir, args.output_dir, args.fps)

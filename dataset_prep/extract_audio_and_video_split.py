'''
Adapted from `https://github.com/v-iashin/SpecVQGAN`.
Modified by Samer Aslan, 2023.
'''

import argparse
import os
import os.path as P
from functools import partial
from glob import glob
from multiprocessing import Pool


def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def safe_duration_extraction(video_path):
    """
    Safely extract video duration using ffmpeg.
    Returns the duration as a string or None if extraction fails.
    """
    duration = execCmd(f"ffmpeg -i {video_path}  2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//")
    duration = duration.strip()
    if len(duration.split(":")) == 3:
        return duration
    return None

def pipeline(video_path, output_dir, fps, sr, duration_target=10):
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")

    # Get the video's total duration safely
    duration = safe_duration_extraction(video_path)
    print(duration)
    if not duration:
        print(f"Failed to extract duration for video: {video_path}")
        return

    hour, min, sec = [float(_) for _ in duration.split(":")]
    duration_second = 3600*hour + 60*min + sec
    
    # Calculate the number of 10-second chunks
    n_chunks = int(duration_second // duration_target)
    
    for chunk_idx in range(n_chunks):
        # Define the start time and end time for the chunk
        start_time = chunk_idx * duration_target
        end_time = (chunk_idx + 1) * duration_target
        
        # Create chunk-specific names
        chunk_video_name = f"chunk_{chunk_idx}_{video_name}"
        chunk_audio_name = f"chunk_{chunk_idx}_{audio_name}"

        # Extract Original Audio for the chunk
        ori_audio_dir = P.join(output_dir, "audio_ori")
        os.makedirs(ori_audio_dir, exist_ok=True)
        os.system(f"ffmpeg -ss {start_time} -t {duration_target} -i {video_path} -loglevel error -f wav -vn -y {P.join(ori_audio_dir, chunk_audio_name)}")

        # Cut Video According to Audio
        align_video_dir = P.join(output_dir, "videos_align")
        os.makedirs(align_video_dir, exist_ok=True)
        os.system("ffmpeg -ss {} -t {} -i {} -loglevel error -c:v libx264 -c:a aac -strict experimental -b:a 98k -y {}".format(
                start_time, duration_target, video_path, P.join(align_video_dir, chunk_video_name)))

        # Extract Audio
        cut_audio_dir = P.join(output_dir, f"audio_{duration_target}s")
        os.makedirs(cut_audio_dir, exist_ok=True)
        os.system("ffmpeg -i {} -loglevel error -f wav -vn -y {}".format(
                P.join(align_video_dir, chunk_video_name), P.join(cut_audio_dir, chunk_audio_name)))

        # Change audio sample rate
        sr_audio_dir = P.join(output_dir, f"audio_{duration_target}s_{sr}hz")
        os.makedirs(sr_audio_dir, exist_ok=True)
        os.system("ffmpeg -i {} -loglevel error -ac 1 -ab 16k -ar {} -y {}".format(
                P.join(cut_audio_dir, chunk_audio_name), sr, P.join(sr_audio_dir, chunk_audio_name)))

        # Change video fps
        fps_audio_dir = P.join(output_dir, f"videos_{duration_target}s_{fps}fps")
        os.makedirs(fps_audio_dir, exist_ok=True)
        os.system("ffmpeg -y -i {} -loglevel error -r {} -c:v libx264 -strict -2 {}".format(
                  P.join(align_video_dir, chunk_video_name), fps, P.join(fps_audio_dir, chunk_video_name)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="data/VAS/dog/videos")
    parser.add_argument("-o", "--output_dir", default="data/features/dog")
    parser.add_argument("-d", "--duration", type=int, default=10)
    parser.add_argument("-a", '--audio_sample_rate', default='22050')
    parser.add_argument("-v", '--video_fps', default='21.5')
    parser.add_argument("-n", '--num_worker', type=int, default=32)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    duration_target = args.duration
    sr = args.audio_sample_rate
    fps = args.video_fps

    video_paths = [P.join(subdir, video) for subdir, _, videos in os.walk(input_dir) for video in videos if video.endswith('.mp4')]
    video_paths.sort()

    with Pool(args.num_worker) as p:
        p.map(partial(pipeline, output_dir=output_dir,
        sr=sr, fps=fps, duration_target=duration_target), video_paths)

import cv2
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import os
import random
import sys
import argparse

def create_frame(dot_present, frame_size=(1024, 1024)):
    frame = np.ones((frame_size[0], frame_size[1], 3), dtype='uint8') * 255
    if dot_present:
        cv2.circle(frame, (frame_size[0] // 2, frame_size[1] // 2), 50, (0, 0, 255), -1)
    return frame

def generate_sound(duration_ms):
    beep = Sine(1000).to_audio_segment(duration=duration_ms)
    silence = AudioSegment.silent(duration=1000-duration_ms)
    return beep + silence

def split_frames(remaining_frames, min_frames_per_segment, max_frames_per_segment):
    if remaining_frames <= min_frames_per_segment * 2:
        return [remaining_frames]
    else:
        split_point = random.randint(min_frames_per_segment, min(max_frames_per_segment, remaining_frames - min_frames_per_segment))
        return [split_point] + split_frames(remaining_frames - split_point, min_frames_per_segment, max_frames_per_segment)

def create_random_video(filename, total_frames, min_frames_per_segment, max_frames_per_segment, fps):
    audio = AudioSegment.silent(duration=total_frames * (1000 // fps))
    frames_folder = 'dot-videos/frames/'
    os.makedirs(frames_folder, exist_ok=True)

    frame_filenames = []  # List to store frame filenames

    assert (min_frames_per_segment < max_frames_per_segment)
    
    # Split the total frames into segments
    segments = split_frames(total_frames, min_frames_per_segment, max_frames_per_segment)

    # Print the final splits
    print("\n\nSplitting into bins of:", segments)

    frame_counter = 0
    dot_present = random.choice([True, False])  # Randomly decide if the first bin has a dot or not

    for segment_length in segments:
        for _ in range(segment_length):
            frame = create_frame(dot_present)
            frame_filename = os.path.join(frames_folder, f'frame_{frame_counter}.png')
            cv2.imwrite(frame_filename, frame)
            frame_filenames.append(frame_filename)

            if dot_present and frame_counter % segment_length == 0:
                audio_overlay = generate_sound(segment_length * (1000 // fps))
                audio = audio.overlay(audio_overlay, position=frame_counter * (1000 // fps))

            frame_counter += 1
        dot_present = not dot_present  # Toggle the dot's presence

    # Create and save the video
    clip = ImageSequenceClip(frame_filenames, fps=fps)
    audio_filename = filename.replace('.mp4', '.wav')
    audio.export(audio_filename, format='wav')
    audio_clip = AudioFileClip(audio_filename)
    final_clip = clip.set_audio(audio_clip)
    final_clip.write_videofile(filename, codec='libx264', fps=fps)

    # Clean up
    os.remove(audio_filename)
    for frame_file in frame_filenames:
        os.remove(frame_file)

def create_fixed_video(filename, total_frames=240, flash_duration_frames=10, flash_frequency_frames=90, fps=24):
    audio = AudioSegment.silent(duration=total_frames * (1000 // fps))
    frames_folder = 'dot-videos/frames/'
    os.makedirs(frames_folder, exist_ok=True)

    frame_filenames = []  # List to store frame filenames

    for i in range(total_frames):
        dot_present = i % flash_frequency_frames < flash_duration_frames
        frame = create_frame(dot_present)
        frame_filename = os.path.join(frames_folder, f'frame_{i}.png')
        cv2.imwrite(frame_filename, frame)
        frame_filenames.append(frame_filename)

        if dot_present and i % flash_frequency_frames == 0:
            audio_overlay = generate_sound(flash_duration_frames * (1000 // fps))
            audio = audio.overlay(audio_overlay, position=i * (1000 // fps))

    # Create video from frames
    clip = ImageSequenceClip(frame_filenames, fps=fps)

    # Add audio to the video
    audio_filename = filename.replace('.mp4', '.wav')
    audio.export(audio_filename, format='wav')
    audio_clip = AudioFileClip(audio_filename)
    final_clip = clip.set_audio(audio_clip)
    final_clip.write_videofile(filename, codec='libx264', fps=fps)

    # Clean up
    os.remove(audio_filename)
    for frame_file in frame_filenames:
        os.remove(frame_file)

def main(args):
    if not os.path.exists('dot-videos'):
        os.makedirs('dot-videos')

    for i in range(args.num_videos):
        if args.random:
            create_random_video(f'dot-videos/random_dot_{i}.mp4', args.total_frames, args.min_frames_per_segment, args.max_frames_per_segment, args.fps)
        else:
            flash_frequency_frames = random.randint(2, args.flash_frequency_max)
            flash_duration_frames = random.randint(1, flash_frequency_frames)
            create_fixed_video(f'dot-videos/fixed_dot_{i}.mp4', args.total_frames, flash_duration_frames, args.flash_frequency_frames, args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A highly customizable timing dataset creator')
    parser.add_argument('--num_videos', type=int, help='Number of videos to generate')
    parser.add_argument('--random', action='store_true', help='Enable intra-video randomness')
    parser.add_argument('--total_frames', type=int, default=220, help='Total number of frames in each video')
    parser.add_argument('--min_frames_per_segment', type=int, default=10, help='Minimum frames per segment for random videos')
    parser.add_argument('--max_frames_per_segment', type=int, default=50, help='Maximum frames per segment for random videos')
    parser.add_argument('--flash_frequency_max', type=int, default=90, help='Maximum frequency of flash for fixed pattern videos')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second')

    args = parser.parse_args()
    main(args)

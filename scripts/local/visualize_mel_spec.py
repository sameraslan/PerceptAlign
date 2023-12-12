import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import librosa
import os
import sys

def visualize_mel(mel_path):
    # Ensure the output directory exists
    output_dir = 'spec2imgwav_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the mel spectrogram from the provided file path
    mel = np.load(mel_path)

    print("Mel spectrogram shape:", mel.shape)
    print("Mel spectrogram min:", mel.min())
    print("Mel spectrogram max:", mel.max())
    print("Mel spectrogram mean:", mel.mean())
    print("Mel spectrogram std:", mel.std())

    # Display and save the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, y_axis='mel', x_axis='time', sr=22050, hop_length=512)
    plt.colorbar(format='%+2.2f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    basename = os.path.splitext(os.path.basename(mel_path))[0]
    filename = f"mel_spectrogram_{basename}.png"
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Mel spectrogram saved to {os.path.join(output_dir, 'mel_spectrogram.png')}")
    plt.close()


    clip_length_in_seconds = 10  # Adjust based on your actual clip length
    desired_time_frames = 860

    hop_length = (clip_length_in_seconds * 22050 - 2048) // (desired_time_frames - 1)
    print(hop_length)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a filepath to the mel.npy file as a command-line argument.")
        sys.exit(1)
    mel_path = sys.argv[1]
    visualize_mel(mel_path)

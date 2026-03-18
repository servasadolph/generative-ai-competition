#!/usr/bin/env python3
# load_audio.py

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Dataset paths
audio_dataset = "/datasets/competition/Deepfake Generation Audio"

print("🎵 Loading Audio Files...")

# Find all audio files
audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')
audio_files = []

for root, dirs, files in os.walk(audio_dataset):
    for file in files:
        if file.lower().endswith(audio_extensions):
            audio_files.append(os.path.join(root, file))

print(f"✅ Found {len(audio_files)} audio files")

# Load and analyze first audio file
if audio_files:
    audio_path = audio_files[0]
    print(f"\n📄 Loading: {os.path.basename(audio_path)}")
    
    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {len(y)/sr:.2f} seconds")
    print(f"   Samples: {len(y)}")
    print(f"   Dtype: {y.dtype}")
    
    # Display waveform
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
    plt.title(f"Waveform: {os.path.basename(audio_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Display spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title("Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    # Extract features (for model input)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print(f"\n✅ MFCC shape: {mfccs.shape}")  # Expected: (40, n_frames)
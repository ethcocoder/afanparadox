"""
Audio Preprocessing for Ethiopian Speech Data
Handles normalization, noise reduction, and formatting
"""

import librosa
import numpy as np
import torch
import torchaudio


class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_and_preprocess(self, audio_path):
        """
        Load audio and apply standard preprocessing
        """
        # Load audio (mono, specific sample rate)
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # Normalize volume
        audio = librosa.util.normalize(audio)
        
        return audio

    def extract_features(self, audio, feature_type="mel"):
        """
        Extract features (Mel spectrogram or similar)
        """
        if feature_type == "mel":
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=80, n_fft=1024, hop_length=256
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            return mel
        
        return audio

    def augment(self, audio):
        """
        Apply data augmentation (time stretching, pitch shifting)
        """
        # Time stretch
        rate = np.random.uniform(0.8, 1.2)
        audio_stretched = librosa.effects.time_stretch(y=audio, rate=rate)
        
        # Pitch shift
        n_steps = np.random.randint(-2, 3)
        audio_shifted = librosa.effects.pitch_shift(y=audio_stretched, sr=self.sample_rate, n_steps=n_steps)
        
        return audio_shifted


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_manifest, processor, preprocessor):
        """
        data_manifest: List of dicts with {"audio_path": "...", "text": "..."}
        processor: Wav2Vec2Processor
        preprocessor: AudioPreprocessor
        """
        self.data_manifest = data_manifest
        self.processor = processor
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data_manifest)

    def __getitem__(self, idx):
        item = self.data_manifest[idx]
        audio = self.preprocessor.load_and_preprocess(item["audio_path"])
        
        # Process audio to Wav2Vec2 inputs
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Tokenize labels
        with self.processor.as_target_processor():
            labels = self.processor(item["text"], return_tensors="pt").input_ids
            
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": labels.squeeze(0)
        }

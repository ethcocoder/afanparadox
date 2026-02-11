"""ASR (Automatic Speech Recognition) module for Ethiopian languages"""

from afanparadox.asr.model import Wav2Vec2ASR
from afanparadox.asr.trainer import ASRTrainer
from afanparadox.asr.inference import ASRInference

__all__ = ["Wav2Vec2ASR", "ASRTrainer", "ASRInference"]

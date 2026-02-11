"""ASR (Speech Recognition) module"""

from afanparadox.asr.model import Wav2Vec2ASR
from afanparadox.asr.trainer import ASRTrainer

__all__ = ["Wav2Vec2ASR", "ASRTrainer"]

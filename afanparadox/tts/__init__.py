"""TTS (Text-to-Speech) module"""

from afanparadox.tts.model import AfanTTS
from afanparadox.tts.vocoder import HiFiGANVocoder

__all__ = ["AfanTTS", "HiFiGANVocoder"]

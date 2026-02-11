"""
AfanParadox Voice Intelligence System

An indigenous Ethiopian voice assistant with:
- ASR (Automatic Speech Recognition) for Amharic, Oromo, Tigrinya
- Cultural Language Model with reasoning
- TTS (Text-to-Speech) synthesis

Built for offline-first deployment and linguistic sovereignty.
"""

__version__ = "0.1.0"
__author__ = "AfanParadox Team"

from afanparadox import asr, llm, tts, data, integration

__all__ = ["asr", "llm", "tts", "data", "integration"]

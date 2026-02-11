#!/usr/bin/env python3
"""
Demo script for AfanParadox Voice Assistant
Run with: python scripts/demo_voice.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from afanparadox.integration.voice_assistant import VoiceAssistant, demo


def main():
    """Main demo entry point"""
    demo()


if __name__ == "__main__":
    main()

"""
AfanParadox Voice Intelligence System - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="afanparadox",
    version="0.1.0",
    author="AfanParadox Team",
    description="Indigenous Ethiopian Voice Intelligence System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ethcocoder/afanparadox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core ML
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        
        # Audio processing
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pyaudio>=0.2.13",
        "audioread>=3.0.0",
        
        # ASR specific
        "datasets>=2.12.0",
        "jiwer>=3.0.0",  # WER/CER metrics
        
        # TTS specific
        "phonemizer>=3.2.0",
        "gruut>=2.3.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        
        # Utilities
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        
        # Training monitoring
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        
        # Web/API
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "gradio>=3.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.0",
        ],
        "deployment": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "optimum>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "afan-train-asr=afanparadox.scripts.train_asr:main",
            "afan-train-llm=afanparadox.scripts.train_llm:main",
            "afan-train-tts=afanparadox.scripts.train_tts:main",
            "afan-demo=afanparadox.scripts.demo_voice:main",
            "afan-record=afanparadox.scripts.record_audio:main",
        ],
    },
)

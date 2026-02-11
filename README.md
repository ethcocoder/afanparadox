# 🇪🇹 AfanParadox Voice Intelligence System

**Indigenous Ethiopian voice assistant with human-like intelligence**

AfanParadox is an end-to-end voice AI system designed for linguistic sovereignty, cultural intelligence, and offline-first deployment in Ethiopian languages (Amharic, Oromo, Tigrinya).

## 🎯 Features

- **🎙️ Speech Recognition (ASR)** - Transcribe Ethiopian speech to text
- **🧠 Cultural Language Model** - Understand context, proverbs, and cultural nuances
- **🔊 Natural Speech Synthesis (TTS)** - Human-like voice output with emotion
- **📡 Offline-First** - Works without internet connection
- **⚡ Edge-Optimized** - Runs on low-end devices (<300MB compressed)

## 🏗️ Architecture

```
Speech Input → ASR Model → Text → Language Model → Response → TTS Model → Speech Output
   (audio)      (Wav2Vec2)        (Transformer)              (FastSpeech2)    (audio)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/afanparadox/afanparadox.git
cd afanparadox

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Usage

**Voice Assistant Demo**:
```bash
afan-demo
```

**Train ASR Model**:
```bash
afan-train-asr --config configs/asr_config.yaml
```

**Train Language Model**:
```bash
afan-train-llm --config configs/llm_config.yaml
```

**Train TTS Model**:
```bash
afan-train-tts --config configs/tts_config.yaml
```

## 📁 Project Structure

```
afanparadox/
├── asr/                    # Speech Recognition
│   ├── model/              # Wav2Vec2 encoder + CTC decoder
│   ├── training/           # Training loops
│   └── evaluation/         # WER/CER metrics
├── llm/                    # Language Model
│   ├── architecture/       # Transformer model
│   ├── tokenizer/          # Morphology-aware tokenizer
│   └── cognitive/          # Cultural shaping
├── tts/                    # Text-to-Speech
│   ├── model/              # Acoustic model + Vocoder
│   ├── training/           # Training loops
│   └── evaluation/         # MOS evaluation
├── integration/            # End-to-end pipeline
├── deployment/             # Edge deployment
├── data/                   # Data collection & processing
└── scripts/                # Training & demo scripts
```

## 🎓 Core Components

### 1. ASR (Automatic Speech Recognition)
- Based on Wav2Vec2-XLS-R pre-trained model
- Fine-tuned for Amharic, Oromo, and Tigrinya
- Target Word Error Rate (WER): <15%

### 2. Language Model
- Morphology-aware tokenizer for agglutinative languages
- Cultural reasoning and proverb understanding
- 200M-400M parameters

### 3. TTS (Text-to-Speech)
- FastSpeech2 acoustic model
- HiFi-GAN neural vocoder
- Multi-speaker, emotional prosody
- Target Mean Opinion Score (MOS): >4.0/5.0

## 📊 Performance Targets

| Component | Metric | Target |
|-----------|--------|--------|
| ASR | Word Error Rate (WER) | <15% |
| ASR | Character Error Rate (CER) | <5% |
| LLM | Perplexity | <50 |
| LLM | Cultural Accuracy | >75% |
| TTS | Mean Opinion Score (MOS) | >4.0/5.0 |
| End-to-End | Latency | <2 seconds |
| Deployment | Total Size (compressed) | <300MB |

## 🗂️ Data Requirements

- **ASR**: 500-1000 hours of transcribed Ethiopian speech
- **LLM**: 2-5B tokens of Ethiopian text
- **TTS**: 20-40 hours of studio-quality voice recordings

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black afanparadox/
flake8 afanparadox/
```

## 📝 License

Apache 2.0 License

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions and support, open an issue or contact the team.

---

**Built for Ethiopian linguistic sovereignty 🇪🇹**

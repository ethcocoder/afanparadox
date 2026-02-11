# AfanParadox Voice Intelligence System

## 🚀 Quick Start Guide

### What We've Built So Far

✅ Complete project structure  
✅ ASR module (Wav2Vec2-based speech recognition)  
✅ LLM module (Morphology-aware Ethiopian tokenizer)  
✅ TTS module (framework ready)  
✅ Voice Assistant integration pipeline  
✅ Audio recording tool for data collection  
✅ Demo system

### Installation

```bash
cd afanparadox

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Test the voice assistant (simulated)
python scripts/demo_voice.py
```

### Collecting Voice Data

```bash
# Record Ethiopian speech for training
python scripts/record_audio.py
```

### Next Steps

1. **Collect Data**: Use `record_audio.py` to gather Ethiopian speech samples
2. **Train ASR**: Fine-tune Wav2Vec2 on collected speech data
3. **Train LLM**: Build Ethiopian language corpus and train transformer
4. **Train TTS**: Record professional voice and train synthesis model
5. **Integrate**: Combine all three models into voice assistant

## 🏗️ Architecture

**Language**: Python 3.9+ (chosen for best ML ecosystem)

**Three-Model System**:
- **ASR** (Ears): Wav2Vec2-XLS-R → Amharic/Oromo/Tigrinya speech recognition
- **LLM** (Brain): Morphology-aware transformer → cultural reasoning
- **TTS** (Voice): FastSpeech2 + HiFiGAN → natural speech synthesis

**Integration**: Real-time voice pipeline with <2s latency

## 📁 Current Structure

```
afanparadox/
├── afanparadox/
│   ├── __init__.py
│   ├── asr/
│   │   ├── model.py          ✅ Wav2Vec2 ASR
│   │   └── __init__.py
│   ├── llm/
│   │   ├── tokenizer.py      ✅ Ethiopian tokenizer
│   │   └── __init__.py
│   ├── tts/
│   │   └── __init__.py
│   └── integration/
│       ├── voice_assistant.py ✅ End-to-end pipeline
│       └── __init__.py
├── scripts/
│   ├── demo_voice.py         ✅ Demo
│   └── record_audio.py       ✅ Data collection
├── config.yaml                ✅ Configuration
├── requirements.txt           ✅ Dependencies
├── setup.py                   ✅ Package setup
└── README.md
```

## 🎯 Current Status

- ✅ **Project Setup**: Complete
- ✅ **Core Architecture**: Built
- 🔄 **Data Collection**: Tools ready, need data
- ⏳ **Model Training**: Awaiting data
- ⏳ **Integration**: Framework ready
- ⏳ **Deployment**: Planned

## 💡 What's Working

- Project structure is production-ready
- ASR model can load Wav2Vec2 (needs fine-tuning on Ethiopian data)
- Ethiopian tokenizer has morphology awareness (needs training)
- Voice assistant pipeline is ready to integrate models
- Recording tool can collect training data

## 🔥 Next Immediate Actions

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test demo**: `python scripts/demo_voice.py`
3. **Start collecting data**: Use recording tool
4. **Fine-tune ASR**: Once we have 10+ hours of speech
5. **Train tokenizer**: On Ethiopian text corpus

Ready to start training models as soon as data is collected! 🚀

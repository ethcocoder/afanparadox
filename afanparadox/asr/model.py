"""
Wav2Vec2-based ASR Model for Ethiopian Languages
Fine-tuned for Amharic, Oromo, and Tigrinya
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2ASR(nn.Module):
    """
    Wav2Vec2 ASR model with CTC decoder for Ethiopian languages
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        vocab_size: int = 128,  # Will be determined by language
        language: str = "amharic"
    ):
        super().__init__()
        
        self.language = language
        self.model_name = model_name
        
        # Load pre-trained Wav2Vec2 model
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            vocab_size=vocab_size,
        )
        
        # Load processor (feature extractor + tokenizer)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Freeze feature extractor initially (can unfreeze later for fine-tuning)
        self.model.freeze_feature_encoder()
    
    def forward(self, input_values, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            input_values: Raw audio waveform [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            labels: Target transcriptions (for training)
            
        Returns:
            Model outputs with logits and loss (if labels provided)
        """
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def transcribe(self, audio_array, sampling_rate=16000):
        """
        Transcribe audio to text
        
        Args:
            audio_array: Audio waveform as numpy array
            sampling_rate: Sampling rate (default 16kHz)
            
        Returns:
            Transcribed text
        """
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to same device as model
        input_values = inputs.input_values.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def unfreeze_feature_encoder(self):
        """Unfreeze feature encoder for full fine-tuning"""
        self.model.unfreeze_feature_encoder()
    
    def save_pretrained(self, save_directory):
        """Save model and processor"""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, load_directory, language="amharic"):
        """Load pre-trained model"""
        model = cls(model_name=load_directory, language=language)
        return model


if __name__ == "__main__":
    # Quick test
    print("Initializing Wav2Vec2 ASR model...")
    model = Wav2Vec2ASR(language="amharic")
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print("Feature encoder frozen:", model.model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.requires_grad == False)

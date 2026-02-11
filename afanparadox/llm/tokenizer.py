"""
Ethiopian Morphology-Aware Tokenizer
Optimized for Amharic, Oromo, and Tigrinya
"""

from typing import List, Dict
from transformers import PreTrainedTokenizer
import re


class EthiopianTokenizer(PreTrainedTokenizer):
    """
    Morphology-aware tokenizer for Ethiopian languages
    
    Combines:
    - Root detection for Semitic languages (Amharic, Tigrinya)
    - Morpheme segmentation for agglutinative features (Oromo)
    - BPE fallback for unknown words
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        language: str = "amharic",
        vocab_size: int = 50000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.language = language
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        # Initialize vocabulary (will be trained later)
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        
        # Morphological patterns (simplified - will be expanded)
        self.amharic_prefixes = ["እ", "ን", "ት", "ይ", "አ"]
        self.amharic_suffixes = ["ኝ", "ሽ", "ው", "ች", "ች", "ን", "ሁ", "ችሁ", "አችሁ"]
        
        self.oromo_suffixes = ["tti", "ni", "tu", "tan", "ti"]
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using morphology-aware segmentation
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Basic word tokenization
        words = text.split()
        tokens = []
        
        for word in words:
            # Try morphological segmentation for Amharic
            if self.language == "amharic":
                morphemes = self._segment_amharic(word)
                tokens.extend(morphemes)
            elif self.language == "oromo":
                morphemes = self._segment_oromo(word)
                tokens.extend(morphemes)
            else:
                # Fallback to character-level for unknown
                tokens.append(word)
        
        return tokens
    
    def _segment_amharic(self, word: str) -> List[str]:
        """
        Segment Amharic word into morphemes
        (Simplified - real implementation needs linguistic rules)
        """
        morphemes = []
        
        # Check for common prefixes
        for prefix in self.amharic_prefixes:
            if word.startswith(prefix) and len(word) > 2:
                morphemes.append(prefix)
                word = word[len(prefix):]
                break
        
        # Check for common suffixes
        for suffix in self.amharic_suffixes:
            if word.endswith(suffix) and len(word) > 2:
                morphemes.append(word[:-len(suffix)])
                morphemes.append(suffix)
                return morphemes
        
        # If no affix found, return whole word
        morphemes.append(word)
        return morphemes
    
    def _segment_oromo(self, word: str) -> List[str]:
        """Segment Oromo word (agglutinative)"""
        # Simplified implementation
        morphemes = [word]
        return morphemes
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        return self.unk_token
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary"""
        return self.vocab
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
        """Save vocabulary"""
        # Will be implemented when training tokenizer
        pass


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = EthiopianTokenizer(language="amharic")
    
    text = "እንኳን ደህና መጣችሁ"
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")

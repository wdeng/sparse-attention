import sentencepiece as spm
from typing import List, Optional, Dict
import os
import torch
import json

class Tokenizer:
    """Custom tokenizer with support for long contexts and special tokens."""
    
    def __init__(
        self,
        vocab_size: int = 128000,
        model_path: Optional[str] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        
        # Default special tokens
        self.special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "fill_token": "[FILL]",  # For long document handling
            **(special_tokens or {})
        }
        
        # Initialize tokenizer
        if model_path and os.path.exists(model_path):
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_path)
        else:
            self.sp_model = None
            
        # Create special token mappings
        self._create_special_token_mappings()
        
    def _create_special_token_mappings(self):
        """Create mappings for special tokens."""
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        
        # Reserve first tokens for special tokens
        for i, (name, token) in enumerate(self.special_tokens.items()):
            self.special_token_to_id[token] = i
            self.id_to_special_token[i] = token
            setattr(self, f"{name}_id", i)
            
    def train(
        self,
        texts: List[str],
        output_dir: str,
        vocab_size: Optional[int] = None,
    ):
        """Train SentencePiece tokenizer on texts."""
        vocab_size = vocab_size or self.vocab_size
        model_prefix = os.path.join(output_dir, f"tokenizer_{vocab_size}")
        
        # Write texts to temporary file
        tmp_file = os.path.join(output_dir, "train_texts.txt")
        with open(tmp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
                
        # Train SentencePiece model
        spm.SentencePieceTrainer.Train(
            input=tmp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size - len(self.special_tokens),
            pad_id=self.pad_token_id,
            unk_id=self.unk_token_id,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            user_defined_symbols=[self.special_tokens["fill_token"]],
            character_coverage=0.99999,
            model_type="bpe",
            num_threads=os.cpu_count(),
        )
        
        # Load trained model
        self.model_path = f"{model_prefix}.model"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.model_path)
        
        # Save special tokens
        with open(f"{model_prefix}.json", "w") as f:
            json.dump(self.special_tokens, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, model_path: str) -> "Tokenizer":
        """Load tokenizer from pretrained model."""
        # Load special tokens
        json_path = model_path.replace(".model", ".json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                special_tokens = json.load(f)
        else:
            special_tokens = None
            
        return cls(model_path=model_path, special_tokens=special_tokens)
        
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token ids."""
        if not self.sp_model:
            raise ValueError("Tokenizer model not loaded or trained")
            
        # Encode with SentencePiece
        ids = self.sp_model.EncodeAsIds(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
        # Truncate if needed
        if max_length:
            ids = ids[:max_length]
            
        return ids
        
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text."""
        if not self.sp_model:
            raise ValueError("Tokenizer model not loaded or trained")
            
        # Filter special tokens if requested
        if skip_special_tokens:
            ids = [id for id in ids if id not in self.id_to_special_token]
            
        # Decode with SentencePiece
        text = self.sp_model.DecodeIds(ids)
        return text
        
    def encode_long_document(
        self,
        text: str,
        chunk_size: int = 8192,
        overlap: int = 512,
    ) -> List[int]:
        """Encode long document with overlapping chunks and fill tokens."""
        # Encode full document
        ids = self.encode(text)
        
        # Split into chunks with overlap
        chunks = []
        for i in range(0, len(ids), chunk_size - overlap):
            chunk = ids[i:i + chunk_size]
            if i > 0:
                # Add fill token at start of non-first chunks
                chunk = [self.fill_token_id] + chunk
            if i + chunk_size < len(ids):
                # Add fill token at end of non-last chunks
                chunk = chunk + [self.fill_token_id]
            chunks.append(chunk)
            
        return chunks
        
    def save(self, output_dir: str):
        """Save tokenizer files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SentencePiece model
        if self.sp_model:
            model_path = os.path.join(output_dir, "tokenizer.model")
            self.sp_model.Save(model_path)
            
        # Save special tokens
        json_path = os.path.join(output_dir, "tokenizer.json")
        with open(json_path, "w") as f:
            json.dump(self.special_tokens, f, indent=2)
            
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size 
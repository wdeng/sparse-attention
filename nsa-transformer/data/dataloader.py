import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import numpy as np
from .tokenizer import Tokenizer

class StreamingDataset(IterableDataset):
    """Streaming dataset that efficiently handles long sequences with packing."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: Tokenizer,
        max_length: int = 8192,
        target_length: Optional[int] = None,
        buffer_size: int = 1000,
        shuffle: bool = True,
    ):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_length = target_length or max_length
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
        # Buffer for sequence packing
        self.token_buffer: List[int] = []
        
    def state_dict(self) -> Dict:
        """Save dataset state for resuming."""
        return {
            "token_buffer": self.token_buffer.copy(),
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load dataset state."""
        self.token_buffer = state_dict["token_buffer"].copy()
        
    def _fill_buffer(self) -> None:
        """Fill token buffer from streaming dataset."""
        while len(self.token_buffer) < self.buffer_size:
            try:
                example = next(iter(self.dataset))
                tokens = self.tokenizer.encode(example["text"])
                self.token_buffer.extend(tokens)
            except StopIteration:
                break
                
    def _get_chunk(self, length: int) -> List[int]:
        """Get a chunk of tokens of specified length."""
        if len(self.token_buffer) < length:
            self._fill_buffer()
            
        if len(self.token_buffer) < length:
            # End of dataset reached
            chunk = self.token_buffer
            self.token_buffer = []
            return chunk
            
        chunk = self.token_buffer[:length]
        self.token_buffer = self.token_buffer[length:]
        return chunk
        
    def _prepare_sample(self, tokens: List[int]) -> Dict[str, torch.Tensor]:
        """Prepare a training sample from tokens."""
        # Add special tokens
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
    def __iter__(self):
        while True:
            # Get sequence of target length
            tokens = self._get_chunk(self.target_length)
            if not tokens:
                break
                
            # Prepare training sample
            sample = self._prepare_sample(tokens)
            yield sample

class PackedDataLoader:
    """DataLoader that packs sequences efficiently for training."""
    
    def __init__(
        self,
        datasets: List[Tuple[StreamingDataset, float]],
        batch_size: int,
        max_tokens: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.datasets = datasets  # List of (dataset, weight) tuples
        self.batch_size = batch_size
        self.max_tokens = max_tokens or (batch_size * 8192)  # Default to 8k tokens per sequence
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create individual dataloaders
        self.dataloaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=max(1, int(num_workers * weight)),
                pin_memory=pin_memory,
            )
            for dataset, weight in datasets
        ]
        
        # Initialize iterators
        self.iterators = [iter(dl) for dl in self.dataloaders]
        
    def _get_next_batch(self, loader_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get next batch from specified dataloader."""
        try:
            batch = next(self.iterators[loader_idx])
        except StopIteration:
            # Reinitialize iterator
            self.iterators[loader_idx] = iter(self.dataloaders[loader_idx])
            batch = next(self.iterators[loader_idx])
        return batch
        
    def __iter__(self):
        while True:
            # Sample dataloader according to weights
            weights = np.array([weight for _, weight in self.datasets])
            loader_idx = np.random.choice(len(self.dataloaders), p=weights/weights.sum())
            
            # Get batch and yield
            try:
                batch = self._get_next_batch(loader_idx)
                if batch is None:
                    continue
                yield batch
            except StopIteration:
                break 
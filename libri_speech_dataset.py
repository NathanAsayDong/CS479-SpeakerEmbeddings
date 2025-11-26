import torch
import torchaudio
import sounddevice as sd
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader, IterableDataset

class LibriSpeechDataset(IterableDataset): # Inherit from IterableDataset to signal to DataLoader
    def __init__(self, split: str = "validation.clean", streaming: bool = False):
        """
        Args:
            split (str): The dataset split to load. 
                         Options: 'train.100', 'train.360', 'train.500', 
                                  'validation', 'test'
            streaming (bool): If True, streams the data instead of downloading the full dataset.
                              Useful for large datasets or quick testing.
        """
        super().__init__()
        self.split = split
        print(f"Loading LibriSpeech dataset (split='{split}')... this might take a while if not cached.")
        
        # Load the dataset from Hugging Face
        # "clean" config contains clean splits
        # "other" config contains noisy splits
        # We auto-detect config based on split name
        config_name = "clean" if "clean" in split or split in ['train.100', 'train.360', 'validation', 'test'] else "other"
        
        self.hf_dataset = load_dataset("librispeech_asr", config_name, split=split, streaming=streaming)
        
        # If not streaming, we can get length efficiently
        self.streaming = streaming
        if not streaming:
            self.dataset_len = len(self.hf_dataset)
        else:
            self.dataset_len = None

    def __len__(self):
        if self.streaming:
            # IterableDataset should NOT have __len__ if streaming
            # But we might need it for non-streaming mode if we want to use it as a map-style dataset wrapper
            # For this hybrid class, we can leave it, but we MUST inherit from IterableDataset
            # to prevent DataLoader from auto-checking len() before iteration in some contexts.
            raise NotImplementedError("Length is not available in streaming mode.")
        return self.dataset_len

    def __getitem__(self, idx):
        if self.streaming:
            # If streaming, we shouldn't really use __getitem__ with integer index
            # But for hybrid class support, we can try to support it if not streaming
            raise NotImplementedError("Cannot use index access on streaming dataset.")
            
        # Hugging Face dataset items are dicts:
        # {'file': 'path/to/file', 'audio': {'path': '...', 'array': array(...), 'sampling_rate': 16000}, 'text': '...'}
        item = self.hf_dataset[idx]
        return self._process_item(item)

    def __iter__(self):
        """Allow iteration for streaming datasets."""
        if not self.streaming:
            # If not streaming, use default iterator
            for i in range(len(self)):
                yield self.__getitem__(i)
        else:
            # If streaming, yield from hf_dataset
            for item in self.hf_dataset:
                yield self._process_item(item)

    def _process_item(self, item):
        """Helper to process raw HF item into our format."""
        audio_data = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        text = item['text']
        file_id = item['file'] 
        
        # Convert to torch tensor
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        return {
            "file_id": file_id,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "transcript": text
        }

    def show_sample(self, idx: int = None):
        """
        Plays the audio and prints the transcript for a sample.
        If idx is None, picks a random sample.
        """
        if self.streaming:
             print("Cannot picking random index in streaming mode. Showing first sample...")
             # Get the first item from the iterator
             sample = next(iter(self))
             idx = 0
        else:
            if idx is None:
                idx = random.randint(0, len(self) - 1)
            sample = self.__getitem__(idx)

        print(f"\n--- Sample {idx} ---")
        print(f"Transcript: {sample['transcript']}")
        print(f"Playing audio ({sample['sample_rate']} Hz)...")
        
        # Play audio
        # waveform is (time,) for HF datasets usually mono 1D array
        audio_data = sample['waveform'].numpy()
        sd.play(audio_data, sample['sample_rate'])
        sd.wait()
        print("Playback finished.")

def collate_audio_batch(batch):
    """
    Custom collate function to handle variable length audio.
    Pads waveforms to the length of the longest clip in the batch.
    """
    # Extract items
    file_ids = [item['file_id'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    sample_rates = [item['sample_rate'] for item in batch]
    
    # Pad waveforms
    # waveforms are (T,) tensors
    waveforms = [item['waveform'] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    
    return {
        "file_ids": file_ids,
        "waveforms": padded_waveforms,
        "sample_rates": sample_rates,
        "transcripts": transcripts
    }

if __name__ == "__main__":
    # Example Usage
    try:
        # Using streaming=True for instant start without waiting for 50GB download
        # Using 'validation' which corresponds to 'dev-clean' in the 'clean' config
        dataset = LibriSpeechDataset(split="validation", streaming=True)
        
        # Show a sample
        dataset.show_sample()
        
        # Test DataLoader
        if not dataset.streaming:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_audio_batch)
            for batch in dataloader:
                print(f"\nBatch keys: {batch.keys()}")
                print(f"Waveform batch shape: {batch['waveforms'].shape}")
                break
        else:
             print("\nSkipping DataLoader shuffle test in streaming mode.")
             # We can still test iteration with DataLoader, but shuffle must be False or buffer_size used
             # And batching works if we use the iterable dataset
             dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_audio_batch)
             print("Testing streaming DataLoader (batch_size=4)...")
             for batch in dataloader:
                print(f"Batch keys: {batch.keys()}")
                print(f"Waveform batch shape: {batch['waveforms'].shape}")
                print(f"Transcripts: {batch['transcripts']}")
                break
            
    except Exception as e:
        print(f"An error occurred: {e}")
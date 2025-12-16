import os
import torch
import soundfile as sf
import tempfile
from datasets import Dataset, Audio
from asr_service import ASRService
from translation_service import TranslationService
from tts_service import TTSService
from enums import Language

class SyntheticDataService:
    def __init__(self):
        self.translator = TranslationService()
        self.tts = TTSService()
        
    def generate_synthetic_data(self, dataset: Dataset, output_dir: str = "synthetic_data") -> Dataset:
        """
        Receives a HF Dataset (e.g. LibriSpeech) and generates the target Spanish audio 
        for each item, preserving the speaker style.
        
        Args:
            dataset: The input dataset (must contain 'audio' and 'transcript' columns usually)
            output_dir: Directory to save generated audio files.
            
        Returns:
            A new Dataset with an added 'synthetic_audio' column.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # We'll use the dataset.map function to process items
        # Note: If dataset is streaming, map works differently (returns iterable).
        # Ideally, this should be used with a non-streaming dataset for easier column addition,
        # or we accept that it yields a new iterable.
        
        def process_item(item):
            # 1. Get Source Info
            # HF Audio feature is usually decoded. We need the path or the bytes/array.
            # SpeechBrain (used in TTS for embedding) often likes a file path.
            # If the dataset provides a path, great. If not (bytes), we might need to save a temp file.
            
            # Accessing the original file path if available
            source_audio_path = item['file'] if 'file' in item else None
            english_text = item['text'] if 'text' in item else "" # LibriSpeech uses 'text'
            
            # If no path (e.g. purely in-memory or computed), save temp
            temp_source_file = None
            if not source_audio_path:
                # Reconstruct audio file from array for the embedding extractor
                audio_array = item['audio']['array']
                sample_rate = item['audio']['sampling_rate']
                
                temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(temp_fd)
                sf.write(temp_path, audio_array, sample_rate)
                source_audio_path = temp_path
                temp_source_file = temp_path

            try:
                # 2. Translate English -> Spanish
                spanish_text = self.translator.translate(english_text, target_language=Language.SPANISH)
                
                # 3. Synthesize Spanish Audio with Voice Cloning
                # Define output filename
                file_id = item['id'] if 'id' in item else os.path.basename(source_audio_path)
                output_filename = f"syn_{file_id}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                self.tts.synthesize(spanish_text, output_path, source_audio_path)
                
                # Return info to be added to the dataset item
                return {
                    "source_transcript": english_text,
                    "target_transcript": spanish_text,
                    "synthetic_audio_path": output_path
                }
                
            except Exception as e:
                print(f"Error processing item {english_text[:20]}...: {e}")
                return {
                    "source_transcript": english_text,
                    "target_transcript": "",
                    "synthetic_audio_path": None
                }
            finally:
                # Cleanup temp file if we created one
                if temp_source_file and os.path.exists(temp_source_file):
                    os.remove(temp_source_file)

        # Apply mapping
        # If streaming, this returns an IterableDataset
        synthetic_dataset = dataset.map(process_item)
        
        return synthetic_dataset

    def generate_synthetic_data_item(self, audio_path: str, transcript: str = None) -> str:
        """
        Receives an audio path (and optional transcript), translates it, 
        and generates the target audio for the item.
        Returns the path to the generated audio.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Source audio not found: {audio_path}")

        # If no transcript provided, we would ideally need ASR.
        # Instantiating ASR here just in case, or we could require transcript.
        if not transcript:
             asr = ASRService()
             transcript = asr.transcribe(audio_path)
        
        # Translate
        spanish_text = self.translator.translate(transcript, target_language=Language.SPANISH)
        
        # Synthesize
        directory = os.path.dirname(audio_path)
        filename = os.path.basename(audio_path)
        output_path = os.path.join(directory, f"synthetic_{filename}")
        
        self.tts.synthesize(spanish_text, output_path, audio_path)
        
        return output_path

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np
from enums import Language
from embedding_service import EmbeddingService

class TTSService:
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load TTS models
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Initialize EmbeddingService
        self.embedding_service = EmbeddingService()

    def synthesize(self, text: str, output_file: str, speaker_audio_path: str) -> str:
        """
        Synthesizes text to speech using the speaker style from speaker_audio_path.
        Saves output to output_file.
        """
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Get speaker embedding via EmbeddingService
        # Ensure embedding is on the same device as the model
        speaker_embeddings = self.embedding_service.extract_embedding(speaker_audio_path).to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=self.vocoder
            )
            
        # Save to file
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
        return output_file

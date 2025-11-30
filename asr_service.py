import whisper
try:
    import sounddevice as sd
except OSError:
    print("Warning: sounddevice/PortAudio not available. Microphone recording will fail.")
    sd = None
import numpy as np
import os
import torch

class ASRService:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.samplerate = 16000 # Whisper expects 16kHz

    def transcribe(self, audio_path: str, language: str = None) -> str:
        """Transcribes audio file to text."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # If language is specified, use it (Whisper supports direct transcription or translation, 
        # but for just transcribing non-English, 'language' arg helps)
        if language:
            result = self.model.transcribe(audio_path, language=language)
        else:
            result = self.model.transcribe(audio_path)
            
        return result["text"].strip()

    def record_audio(self, duration: int = 10, file_path: str = "enrollment.wav"):
        """Records audio for a specific duration to a file."""
        if sd is None:
            raise RuntimeError("SoundDevice not available. Cannot record audio.")
            
        print(f"Recording enrollment audio for {duration} seconds...")
        
        audio_data = sd.rec(int(duration * self.samplerate), 
                          samplerate=self.samplerate, 
                          channels=1)
        sd.wait()
        
        import soundfile as sf
        sf.write(file_path, audio_data, self.samplerate)
        print(f"Audio saved to {file_path}")
        return file_path

    def listen_transcribe(self, duration: int = 5) -> tuple[str, str]:
        """
        Records audio from microphone and transcribes it.
        Returns tuple of (transcribed_text, audio_file_path)
        """
        if sd is None:
            raise RuntimeError("SoundDevice not available. Cannot record audio.")

        print(f"Listening for {duration} seconds...")
        
        # Record audio
        audio_data = sd.rec(int(duration * self.samplerate), 
                          samplerate=self.samplerate, 
                          channels=1)
        sd.wait()
        
        # Flatten and convert to float32
        audio_flat = audio_data.flatten().astype(np.float32)
        
        # Transcribe directly from numpy array
        result = self.model.transcribe(audio_flat)
        text = result["text"].strip()
        
        # Save audio to a temporary file for embedding extraction later
        import soundfile as sf
        import tempfile
        
        # specific temporary file to persist across the session if needed, 
        # but for now a generic temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, self.samplerate)
        
        return text, temp_file.name

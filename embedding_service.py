import torchaudio

# Explicitly set the backend to "soundfile" if possible to avoid torchcodec issues if it persists,
# though we just installed torchcodec. But standard behavior for speechbrain is often soundfile.
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# Monkey-patch torchaudio.list_audio_backends if it's missing (removed in torchaudio 2.1+)
# SpeechBrain 1.0.3 might still depend on it.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
import torch

class EmbeddingService:
    def __init__(self):
        # Using cpu for speechbrain to avoid potential conflicts or VRAM issues if limited
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="tmp_model",
            run_opts={"device": "cpu"} 
        )

    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """Extracts speaker embedding from audio file."""
        signal = self.speaker_model.load_audio(audio_path)
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        embedding = self.speaker_model.encode_batch(signal)
        # SpeechT5 expects (1, 512) tensor
        # embedding is usually (Batch, Time, Channels) or similar from SpeechBrain
        # e.g. (1, 1, 512) for xvector
        
        # We want to return a clean (512,) or (1, 512) tensor
        # Squeeze batch and time dims if they exist
        emb_tensor = torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding
        
        # Squeeze to (512,) then unsqueeze if needed downstream, or standardize on (512,)
        # SpeechBrain embeddings are often (1, 1, 512)
        emb_tensor = emb_tensor.squeeze() # Should become (512,)
        
        return emb_tensor.to(self.device)

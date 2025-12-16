from transformers import MarianMTModel, MarianTokenizer
from enums import Language
import torch

class TranslationService:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate(self, text: str, target_language: Language = Language.SPANISH) -> str:
        """Translates text to target language."""
        # For now, we assume the model handles the direction (En -> Es)
        # If dynamic direction is needed, we'd need to load different models.
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            translated = self.model.generate(**inputs)
            
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

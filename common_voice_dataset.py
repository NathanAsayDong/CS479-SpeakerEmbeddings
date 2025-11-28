import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

class CommonVoiceDataset:
    def __init__(self, language_code: str = "es", split: str = "train"):
        """
        Args:
            language_code (str): The language code for the dataset (e.g., "es" for Spanish, "en" for English).
            split (str): The split to load ('train', 'dev', 'test').
        """
        self.language_code = language_code
        self.split = split
        
        # Download/Load the dataset path using kagglehub
        # Note: This downloads the *entire* Common Voice dataset structure if not cached.
        # It's huge. Be careful.
        print(f"Loading Common Voice dataset for language '{language_code}'...")
        self.dataset_path = kagglehub.dataset_download("mozillaorg/common-voice")
        
        # Construct path to the specific language TSV file
        # Common Voice structure on Kaggle usually: cv-corpus-X.X-20XX-XX-XX/LANGUAGE/
        # We need to find the correct folder.
        
        # Let's verify the structure by walking it if needed, but standard CV structure is:
        # root/cv-corpus-version/lang/
        
        # We will look for the language directory
        self.lang_path = self._find_language_dir(self.dataset_path, language_code)
        if not self.lang_path:
             raise FileNotFoundError(f"Could not find directory for language '{language_code}' in {self.dataset_path}")

        self.tsv_path = os.path.join(self.lang_path, f"{split}.tsv")
        if not os.path.exists(self.tsv_path):
             raise FileNotFoundError(f"TSV file not found: {self.tsv_path}")
             
        # Load metadata
        self.df = pd.read_csv(self.tsv_path, sep='\t')
        print(f"Loaded {len(self.df)} records for {language_code}/{split}")

    def _find_language_dir(self, root_path: str, lang_code: str):
        """Recursively finds the directory for the specific language code."""
        for root, dirs, files in os.walk(root_path):
            if lang_code in dirs:
                return os.path.join(root, lang_code)
        return None

    def get_audio_path(self, filename: str) -> str:
        """Returns the full path to an audio file."""
        # Audio clips are usually in a 'clips' subdirectory within the language folder
        return os.path.join(self.lang_path, "clips", filename)

    def get_samples_by_speaker(self, client_id: str):
        """Returns all samples for a specific speaker."""
        return self.df[self.df['client_id'] == client_id]

if __name__ == "__main__":
    # Example Usage
    try:
        # Note: This might download a massive dataset. 
        # Ideally, we would select a specific file from the dataset handle if KaggleHub supported partial downloads well,
        # but dataset_download gets the whole version.
        
        # For demonstration, we'll try to just print info if it was already downloaded or small enough.
        # Changing to 'en' for English samples.
        dataset = CommonVoiceDataset(language_code="en", split="dev") # 'dev' is smaller than train
        
        print("First 5 records:")
        print(dataset.df.head())
        
        # Check first audio file existence
        first_file = dataset.df.iloc[0]['path']
        full_path = dataset.get_audio_path(first_file)
        print(f"\nFirst audio file path: {full_path}")
        print(f"Exists: {os.path.exists(full_path)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")



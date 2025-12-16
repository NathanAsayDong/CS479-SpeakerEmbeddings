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
        print(f"Dataset path: {self.dataset_path}")
        
        # Construct path to the specific language TSV file
        # Common Voice structure on Kaggle usually: cv-corpus-X.X-20XX-XX-XX/LANGUAGE/
        # We need to find the correct folder.
        
        # Let's verify the structure by walking it if needed, but standard CV structure is:
        # root/cv-corpus-version/lang/
        
        # We will look for the language directory
        self.lang_path = self._find_language_dir(self.dataset_path, language_code)
        if not self.lang_path:
             raise FileNotFoundError(f"Could not find directory for language '{language_code}' in {self.dataset_path}")

        # Adjust for flattened structure:
        # If split is 'dev', look for 'cv-valid-dev.csv'
        # If split is 'train', look for 'cv-valid-train.csv'
        
        # Standard Common Voice name map
        # "train" -> "cv-valid-train"
        # "dev" -> "cv-valid-dev"
        # "test" -> "cv-valid-test"
        
        split_map = {
            "train": "cv-valid-train",
            "dev": "cv-valid-dev",
            "test": "cv-valid-test"
        }
        
        target_name = split_map.get(split, split)
        
        # Try finding CSV (Kaggle version often uses CSV instead of TSV)
        csv_path = os.path.join(self.lang_path, f"{target_name}.csv")
        tsv_path = os.path.join(self.lang_path, f"{target_name}.tsv")
        
        if os.path.exists(csv_path):
            self.metadata_path = csv_path
            sep = ','
        elif os.path.exists(tsv_path):
            self.metadata_path = tsv_path
            sep = '\t'
        else:
             # Try original simple name just in case
             simple_tsv = os.path.join(self.lang_path, f"{split}.tsv")
             if os.path.exists(simple_tsv):
                 self.metadata_path = simple_tsv
                 sep = '\t'
             else:
                 raise FileNotFoundError(f"Metadata file not found for split '{split}' in {self.lang_path}")
             
        # Load metadata
        self.df = pd.read_csv(self.metadata_path, sep=sep)
        
        # Normalize column names if needed
        # Kaggle CSV might have 'filename' instead of 'path'
        if 'filename' in self.df.columns and 'path' not in self.df.columns:
            self.df['path'] = self.df['filename']
            
        # Kaggle CSV has 'text' instead of 'sentence'
        if 'text' in self.df.columns and 'sentence' not in self.df.columns:
            self.df['sentence'] = self.df['text']
            
        print(f"Loaded {len(self.df)} records for {language_code}/{split}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        if 'client_id' not in self.df.columns:
            print("Warning: 'client_id' not found in metadata. Using dummy IDs or skipping speaker grouping.")
            # If no client_id, we can't really group by speaker.
            # Maybe use 'id' or index? But this defeats the purpose of "same speaker".
            # Or maybe it's called 'client' or 'speaker_id'?
            # Let's try to find a suitable column
            potential_ids = [c for c in self.df.columns if 'id' in c.lower() or 'client' in c.lower() or 'speaker' in c.lower()]
            if potential_ids:
                print(f"Potential ID columns found: {potential_ids}. Using '{potential_ids[0]}' as client_id.")
                self.df['client_id'] = self.df[potential_ids[0]]
            else:
                # If truly no ID, we can't do speaker-based experiments.
                pass

    def _find_language_dir(self, root_path: str, lang_code: str):
        """Recursively finds the directory for the specific language code."""
        print(f"Searching for language '{lang_code}' in {root_path}")
        
        # Kaggle Common Voice structure is confusing.
        # It seems the dataset is already split into 'cv-valid-train', 'cv-valid-dev', etc. at the root.
        # It might NOT have language folders if this is the English-only release or structured differently.
        # The user provided log shows folders like: 'cv-valid-dev', 'cv-valid-train'.
        # These likely contain the audio/metadata directly for the downloaded language.
        
        # If we see these folders, and we assume we downloaded the correct language dataset,
        # then the "language directory" is effectively the root_path itself, 
        # or we need to adjust how we find the TSV and Clips.
        
        # Let's verify if 'cv-valid-dev' exists. If so, return root_path and handle paths downstream?
        # OR: Return root_path and let the caller find the specific CSV/TSV.
        
        # The logs show 'cv-valid-dev.csv' exists at root.
        # This implies the structure is:
        # /root/
        #   cv-valid-train.csv
        #   cv-valid-train/ (folder with clips?)
        #   cv-valid-dev.csv
        #   cv-valid-dev/
        
        if os.path.exists(os.path.join(root_path, "cv-valid-dev.csv")):
            print(f"Found flattened dataset structure at {root_path}")
            return root_path
            
        # Fallback to original recursive search
        for root, dirs, files in os.walk(root_path):
            if lang_code in dirs:
                candidate = os.path.join(root, lang_code)
                if os.path.exists(os.path.join(candidate, 'clips')) or \
                   any(f.endswith('.tsv') for f in os.listdir(candidate)):
                    print(f"Found language directory: {candidate}")
                    return candidate
        
        print("Could not find exact language directory. Listing root directories for debugging:")
        try:
            for d in os.listdir(root_path):
                print(f" - {d}")
        except Exception as e:
            print(f"Error listing root: {e}")
            
        return None

    def get_audio_path(self, filename: str) -> str:
        """Returns the full path to an audio file."""
        # Audio clips are usually in a 'clips' subdirectory within the language folder
        # Sometimes filename includes 'clips/' prefix or .mp3 extension
        
        # Ensure filename ends with mp3 (Common Voice usually distributes mp3)
        if not filename.endswith(".mp3"):
            filename = filename + ".mp3"
            
        # Kaggle flattened structure often puts clips in a folder named after the split
        # e.g. "cv-valid-dev/sample-00000.mp3"
        # Or sometimes just a "clips" folder.
        
        # Check standard "clips" folder
        clips_path = os.path.join(self.lang_path, "clips", filename)
        if os.path.exists(clips_path):
            return clips_path
            
        # Check split-named folder (e.g. cv-valid-dev/filename)
        # We need to know which split this file belongs to, but self.split is general.
        # However, usually files are unique enough or we check multiple places.
        
        # Try folders present in root
        possible_folders = [d for d in os.listdir(self.lang_path) if os.path.isdir(os.path.join(self.lang_path, d))]
        for folder in possible_folders:
            candidate = os.path.join(self.lang_path, folder, filename)
            if os.path.exists(candidate):
                return candidate
                
        # If not found, return the standard clips path so error message is consistent
        return clips_path

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



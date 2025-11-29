import os
import random
import soundfile as sf
import numpy as np
import pandas as pd
from enums import Language
from common_voice_dataset import CommonVoiceDataset

class ExperimentSetup:
    def __init__(self, source_language: Language, target_language: Language, reference_durations: list[float] = [5.0, 10.0, 20.0], seed: int = 42):
        self.source_language = source_language
        self.target_language = target_language
        self.reference_durations = reference_durations
        self.seed = seed
        self.output_dir = "experiment_data"
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self, num_samples_per_duration: int = 5):
        """
        Prepares experiment data.
        For each target duration (e.g. 5s, 10s, 20s), selects `num_samples_per_duration` 
        audio clips from the dataset.
        Ensures each selected clip is long enough (or looped to be long enough) to serve as a reference.
        
        Returns a manifest list where each item represents one test case:
        {
            "sample_id": ...,
            "test_input_path": ..., (original audio, used for translation source)
            "test_input_text": ...,
            "reference_path": ..., (the cut/looped audio of specific duration)
            "target_duration": ...
        }
        """
        print(f"Preparing experiment data: {num_samples_per_duration} samples for each of {self.reference_durations}s durations...")
        
        cv_lang_code = "en" if self.source_language == Language.ENGLISH else "es"
        dataset = CommonVoiceDataset(language_code=cv_lang_code, split="dev")
        
        experiment_manifest = []
        
        # We need a total of (num_samples * num_durations) unique items to avoid reusing same clip for everything?
        # Or we can reuse the same source clip but with different reference durations to compare effect on SAME content.
        # Let's do the latter: For 5 source clips, generate ALL target duration references for each.
        
        # Shuffle entire dataset once
        shuffled_indices = dataset.df.sample(frac=1, random_state=self.seed).index
        
        valid_samples_found = 0
        
        for idx in shuffled_indices:
            if valid_samples_found >= num_samples_per_duration:
                break
                
            row = dataset.df.loc[idx]
            audio_path = dataset.get_audio_path(row['path'])
            
            if not os.path.exists(audio_path):
                continue
            
            # Check basic validity (e.g. > 1s)
            try:
                info = sf.info(audio_path)
                if info.duration < 1.0:
                    continue
            except Exception:
                continue
                
            # Found a valid source sample
            valid_samples_found += 1
            sample_id = f"sample_{valid_samples_found}"
            sample_dir = os.path.join(self.output_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            original_audio, sr = sf.read(audio_path)
            
            # For this single source sample, create a reference file for EACH target duration
            for duration in self.reference_durations:
                target_samples = int(duration * sr)
                
                # Create looped/cut audio
                if len(original_audio) >= target_samples:
                    # Cut
                    new_audio = original_audio[:target_samples]
                else:
                    # Loop
                    repeats = int(np.ceil(target_samples / len(original_audio)))
                    new_audio = np.tile(original_audio, repeats)[:target_samples]
                
                ref_filename = f"ref_{int(duration)}s.wav"
                ref_path = os.path.join(sample_dir, ref_filename)
                sf.write(ref_path, new_audio, sr)
                
                # Add to manifest
                experiment_manifest.append({
                    "sample_id": sample_id,
                    "test_input_path": audio_path,
                    "test_input_text": row['sentence'],
                    "reference_path": ref_path,
                    "target_duration": duration
                })
                
        return experiment_manifest

    def _create_concatenated_audio(self, dataset, sample_rows, target_duration_sec, output_path):
        # Deprecated logic since we can't group by speaker
        pass

if __name__ == "__main__":
    setup = ExperimentSetup(Language.ENGLISH, Language.SPANISH)
    manifest = setup.prepare_data(num_speakers=2)
    print("Manifest created:", len(manifest))

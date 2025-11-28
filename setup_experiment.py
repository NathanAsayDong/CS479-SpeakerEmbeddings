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

    def prepare_data(self, num_speakers: int = 5):
        """
        Prepares experiment data by selecting speakers and creating reference audio files 
        of specified durations, ensuring separation between reference and test data.
        """
        print(f"Preparing experiment data for {num_speakers} speakers...")
        
        # 1. Load Source Dataset (Common Voice)
        # We need a language code for Common Voice (e.g., 'en' for English)
        cv_lang_code = "en" if self.source_language == Language.ENGLISH else "es"
        dataset = CommonVoiceDataset(language_code=cv_lang_code, split="dev") # Using dev for faster loading/experimentation
        
        # 2. Select Speakers with Enough Data
        # Group by client_id to find speakers with many clips
        speaker_counts = dataset.df['client_id'].value_counts()
        eligible_speakers = speaker_counts[speaker_counts > 10].index.tolist() # Arbitrary threshold
        
        if len(eligible_speakers) < num_speakers:
            print(f"Warning: Only found {len(eligible_speakers)} eligible speakers.")
            selected_speakers = eligible_speakers
        else:
            selected_speakers = random.sample(eligible_speakers, num_speakers)
            
        experiment_manifest = []

        for speaker_id in selected_speakers:
            print(f"Processing speaker: {speaker_id[:10]}...")
            speaker_samples = dataset.get_samples_by_speaker(speaker_id)
            
            # Shuffle samples deterministically
            speaker_samples = speaker_samples.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
            # We need to reserve some samples for "Translation Source" (Test Input)
            # and use others to build "Reference Audio" (Embedding Source)
            
            # Let's say:
            # - 1 sample for Test Input (to be translated)
            # - Remaining samples concatenated to form References of varying lengths
            
            if len(speaker_samples) < 2:
                continue
                
            test_sample_row = speaker_samples.iloc[0]
            test_audio_path = dataset.get_audio_path(test_sample_row['path'])
            test_sentence = test_sample_row['sentence']
            
            # Remaining pool for references
            reference_pool = speaker_samples.iloc[1:]
            
            # Create Reference Files for each target duration
            speaker_dir = os.path.join(self.output_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)
            
            ref_paths = {}
            
            for duration in self.reference_durations:
                ref_filename = f"ref_{int(duration)}s.wav"
                ref_path = os.path.join(speaker_dir, ref_filename)
                
                # Concatenate audio until we hit the duration
                self._create_concatenated_audio(dataset, reference_pool, duration, ref_path)
                ref_paths[duration] = ref_path
            
            # Record in manifest
            experiment_manifest.append({
                "speaker_id": speaker_id,
                "test_input_path": test_audio_path,
                "test_input_text": test_sentence,
                "reference_paths": ref_paths # Dict {5.0: path, 10.0: path}
            })
            
        return experiment_manifest

    def _create_concatenated_audio(self, dataset, sample_rows, target_duration_sec, output_path):
        """Concatenates audio clips from the rows until target duration is reached."""
        audio_buffer = []
        current_duration = 0.0
        samplerate = 16000 # Standard for CV/SpeechBrain, but we should check
        
        for _, row in sample_rows.iterrows():
            if current_duration >= target_duration_sec:
                break
                
            path = dataset.get_audio_path(row['path'])
            if not os.path.exists(path):
                continue
                
            data, fs = sf.read(path)
            
            # Resample if needed? Assuming CV is usually 48k or 32k, often mp3. 
            # soundfile reads into float32 numpy array.
            # Ideally we ensure consistent samplerate. 
            # For simplicity, we trust sf.read or would need resampling logic.
            # Let's assume we just append raw data if FS matches, else skip or naive append (bad practice but simple for now)
            
            if samplerate is None:
                samplerate = fs
            elif fs != samplerate:
                # Simple skip for now if mismatch to avoid resampling complexity in setup
                continue
                
            audio_buffer.append(data)
            current_duration += len(data) / fs
            
        if not audio_buffer:
            # If empty, just create silence or copy one file
            print(f"Warning: Could not fill buffer for {output_path}")
            return

        # Concatenate
        combined_audio = np.concatenate(audio_buffer)
        
        # Trim to exact duration if slightly over
        target_samples = int(target_duration_sec * samplerate)
        if len(combined_audio) > target_samples:
            combined_audio = combined_audio[:target_samples]
            
        sf.write(output_path, combined_audio, samplerate)

if __name__ == "__main__":
    setup = ExperimentSetup(Language.ENGLISH, Language.SPANISH)
    manifest = setup.prepare_data(num_speakers=2)
    print("Manifest created:", len(manifest))

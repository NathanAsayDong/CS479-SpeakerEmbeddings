import json
import os
import pandas as pd
from enums import Language
from setup_experiment import ExperimentSetup
from synthetic_data_service import SyntheticDataService
# We might need embedding service if we want to compute similarity metrics here, 
# or we just rely on the synthetic generation.
from embedding_service import EmbeddingService
import torch
import torch.nn.functional as F

class ExperimentRunner:
    def __init__(self):
        self.synth_service = SyntheticDataService()
        self.embedding_service = EmbeddingService()
        self.results = []

    def run(self, manifest):
        """
        Runs the experiment for each item in the manifest.
        Manifest is a list of dicts:
        {
            "speaker_id": ...,
            "test_input_path": ..., 
            "test_input_text": ...,
            "reference_paths": {5.0: path, ...}
        }
        """
        print("Starting experiment run...")
        
        for item in manifest:
            speaker_id = item['speaker_id']
            test_input_path = item['test_input_path']
            # test_input_text = item['test_input_text'] # We rely on translation service handling text or ASR
            
            # Ground Truth Embedding (from the test input itself? Or effectively we want to see if Ref matches Test style)
            # Actually, we want to see if the OUTPUT audio matches the REFERENCE style (or original speaker style).
            # Let's verify similarity between:
            # 1. Original Speaker (Test Input) vs Output Audio
            # 2. Reference Audio vs Output Audio
            
            # Extract embedding of the original test input (Ground Truth Style)
            gt_embedding = self.embedding_service.extract_embedding(test_input_path)
            
            for duration, ref_path in item['reference_paths'].items():
                print(f"Running: Speaker {speaker_id[:8]} | Ref Duration: {duration}s")
                
                if not os.path.exists(ref_path):
                    print(f"Skipping missing ref: {ref_path}")
                    continue

                # 1. Generate Synthetic Output using this Reference
                # generate_synthetic_data_item uses the ref_path for style
                # We need to pass the reference path as the source for embedding, 
                # but the test_input_path as the content to translate.
                
                # Currently SyntheticDataService.generate_synthetic_data_item takes ONE audio path
                # and uses it for BOTH content and style.
                # We need to modify it or call TTS directly to separate them.
                
                # Let's call components directly for fine-grained control:
                try:
                    # A. Transcribe (if needed) & Translate
                    # We have test_input_text, so let's translate that directly
                    english_text = item['test_input_text']
                    spanish_text = self.synth_service.translator.translate(english_text, target_language=Language.SPANISH)
                    
                    # B. Synthesize using Reference Path for Style
                    output_filename = f"out_{speaker_id}_{duration}s.wav"
                    output_path = os.path.join(os.path.dirname(ref_path), output_filename)
                    
                    self.synth_service.tts.synthesize(spanish_text, output_path, ref_path)
                    
                    # 2. Evaluate
                    # Extract embedding from generated output
                    output_embedding = self.embedding_service.extract_embedding(output_path)
                    
                    # Cosine Similarity between Output and Ground Truth (Original Speaker)
                    similarity = F.cosine_similarity(gt_embedding.unsqueeze(0), output_embedding.unsqueeze(0))
                    score = similarity.item()
                    
                    print(f"  -> Similarity Score: {score:.4f}")
                    
                    self.results.append({
                        "speaker_id": speaker_id,
                        "duration": duration,
                        "similarity_score": score,
                        "original_text": english_text,
                        "translated_text": spanish_text,
                        "output_path": output_path
                    })
                    
                except Exception as e:
                    print(f"  -> Error: {e}")

    def save_results(self, filename="experiment_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    # 1. Setup
    setup = ExperimentSetup(Language.ENGLISH, Language.SPANISH)
    # Using small num_speakers for test run
    manifest = setup.prepare_data(num_speakers=2)
    
    # 2. Run
    runner = ExperimentRunner()
    runner.run(manifest)
    
    # 3. Save
    runner.save_results()


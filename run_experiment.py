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
            "sample_id": ...,
            "test_input_path": ..., 
            "test_input_text": ...,
            "reference_path": ...,
            "target_duration": ...
        }
        """
        print("Starting experiment run...")
        
        for item in manifest:
            sample_id = item['sample_id']
            test_input_path = item['test_input_path']
            ref_path = item['reference_path']
            duration = item['target_duration']
            
            print(f"Running: Sample {sample_id} | Ref Duration: {duration}s")
            
            if not os.path.exists(ref_path):
                print(f"Skipping missing ref: {ref_path}")
                continue

            try:
                # Extract embedding of the original test input (Ground Truth Style)
                gt_embedding = self.embedding_service.extract_embedding(test_input_path)
                
                # A. Translate
                english_text = item['test_input_text']
                spanish_text = self.synth_service.translator.translate(english_text, target_language=Language.SPANISH)
                
                # B. Synthesize using Reference Path for Style
                output_filename = f"out_{sample_id}_{duration}s.wav"
                output_path = os.path.join(os.path.dirname(ref_path), output_filename)
                
                self.synth_service.tts.synthesize(spanish_text, output_path, ref_path)
                
                    # 2. Evaluate
                    # A. Style Similarity (Cosine Similarity)
                    output_embedding = self.embedding_service.extract_embedding(output_path)
                    
                    # Ensure embeddings are tensors and handle dimensions
                    gt_embedding_t = gt_embedding
                    output_embedding_t = output_embedding
                    
                    # If they are already tensors, unsqueeze to add batch dim if needed (SpeechBrain/PyTorch expects BxN)
                    # SpeechT5 output is typically (1, 512) or (512,)
                    if gt_embedding_t.dim() == 1:
                        gt_embedding_t = gt_embedding_t.unsqueeze(0)
                    if output_embedding_t.dim() == 1:
                        output_embedding_t = output_embedding_t.unsqueeze(0)
                        
                    similarity = F.cosine_similarity(gt_embedding_t, output_embedding_t)
                    
                    # Extract scalar value safely
                    # .item() works on a 0-dim or 1-dim tensor with 1 element
                    if similarity.numel() == 1:
                        similarity_score = similarity.item()
                    else:
                        # If batched or unexpected shape, take mean or first
                        similarity_score = similarity.mean().item()
                    
                    # B. Translation Quality (BLEU proxy via ASR)
                    if not hasattr(self, 'asr'):
                        from asr_service import ASRService
                        self.asr = ASRService() 
                        
                    transcribed_spanish = self.asr.transcribe(output_path)
                    
                    print(f"  -> Similarity: {similarity_score:.4f} | Text match len: {len(transcribed_spanish)}/{len(spanish_text)}")
                
                self.results.append({
                    "sample_id": sample_id,
                    "duration": duration,
                    "similarity_score": similarity_score,
                    "original_text": english_text,
                    "target_text": spanish_text,
                    "transcribed_text": transcribed_spanish,
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


# Project Outline: Speaker Adaptation for English → Spanish Speech Generation

## Goal
Implement and evaluate a speaker-adapted speech generation pipeline that produces **Spanish speech** while matching a target speaker’s voice. In the final code, this became a comparison of:

- **Zero-shot speaker conditioning**: pretrained SpeechT5 TTS conditioned on a **SpeechBrain x-vector** extracted from a short speaker reference clip.
- **Full fine-tuning**: SpeechT5 TTS weights fine-tuned on a single VoxPopuli speaker’s English speech + transcripts (standard supervised TTS), then used to synthesize Spanish text with the same x-vector conditioning.

## Research Question
**Does full fine-tuning of SpeechT5 on a target speaker improve speaker similarity over zero-shot conditioning, without significantly harming intelligibility of the Spanish output?**

We additionally tested how **reference-audio duration** impacts the zero-shot conditioning baseline.

## Hypotheses (aligned to what we ran)
- **H1 (zero-shot duration effect)**: Longer reference clips (2–10s) will improve speaker similarity for zero-shot conditioning.
- **H2 (fine-tune vs zero-shot)**: Fine-tuned SpeechT5 will improve speaker similarity compared to the baseline for the same reference clip, but may trade off intelligibility on some sentences.

## Implemented Systems
### Models / components used in code
- **TTS**: `microsoft/speecht5_tts` + `microsoft/speecht5_hifigan`
- **Speaker embedding**: `speechbrain/spkrec-xvect-voxceleb` (x-vectors), used for both conditioning and evaluation
- **ASR (used as an evaluation proxy)**: Whisper `large-v3-turbo` (transcribe synthesized Spanish audio)
- **Text prompts**: EN/ES sentence pairs from `Helsinki-NLP/opus-100` (`en-es`, validation split)
- **Auxiliary (used in one loop)**: a text translation model (`google-t5/t5-large`) for backtranslation-style checks in the zero-shot experiment loop

### Speaker adaptation variants (actual)
- **Zero-shot**: no weight updates; condition SpeechT5 on x-vector from a speaker reference clip
- **Full fine-tune**: fine-tune SpeechT5 on VoxPopuli samples for a single speaker (text→speech), save the model, then use it for synthesis

> Note: **PEFT/LoRA/adapters were planned but not implemented** in the final code.

## Datasets (actual)
### Speaker data (voice / adaptation)
- **VoxPopuli (English)**: `facebook/voxpopuli`, used to obtain per-speaker audio + transcripts
- Experiments focused on **3 speakers**: `1055`, `28165`, `124992`
- A small subset for those speakers was saved to disk at `data/voxpopuli_en_validation_3speakers` for fine-tuning.

### Spanish text targets
- **Opus-100 EN/ES**: `Helsinki-NLP/opus-100`, `en-es`, validation split
- Sentences were filtered to be short (to reduce SpeechT5 failures on very long inputs).

## Metrics (actual)
### Speaker similarity (voice preservation)
- **Cosine similarity** between x-vectors:
  - x-vector(reference speaker audio) vs x-vector(synthesized Spanish audio)

### Intelligibility proxy (Spanish audio)
- **Sentence BLEU** between:
  - reference Spanish text (gold ES sentence) and
  - **Whisper transcript** of synthesized Spanish audio

## Experiments & Repo Artifacts
### Experiment A: Zero-shot conditioning vs reference duration sweep
For each speaker and duration \(d \in \{2,4,6,8,10\}\) seconds:
- Trim VoxPopuli speaker audio to \(d\) seconds and use it as the reference clip
- Synthesize Spanish for 5 fixed EN/ES pairs
- Compute mean BLEU + mean cosine similarity
- Saved qualitative example per (speaker, duration):
  - `results/zero_shot/speaker_{id}/duration_{d}/source_audio.wav`
  - `results/zero_shot/speaker_{id}/duration_{d}/output_spanish_audio.wav`
  - `results/zero_shot/speaker_{id}/duration_{d}/sample_input.json`
- Saved summary CSVs:
  - `results/zero_shot/results_{speaker_id}.csv`

### Experiment B: Full fine-tuning SpeechT5 per speaker
- Fine-tune SpeechT5 on a single speaker’s VoxPopuli samples (train/test split)
- Saved:
  - training outputs/checkpoints: `speecht5_finetuned_voxpopuli_speaker_{speaker_id}/...`
  - exported models: `models/finetuned_models/speecht5_finetuned_voxpopuli_speaker_{speaker_id}/`
- Fine-tuned experiment outputs:
  - `results/fine_tune/` (including `results_{speaker_id}.csv` and sample folders)

### Experiment C: Head-to-head comparison (zero-shot vs fine-tuned)
For each speaker (fixed reference clip from `results/zero_shot/speaker_{id}/duration_4/source_audio.wav`):
- Generate Spanish audio for 20 random EN/ES pairs
- Compute BLEU + cosine similarity for both methods
- Saved per-speaker CSVs:
  - `results/comparison/speaker_{id}/results.csv`

## Human Evaluation (Streamlit UI)
To complement automatic metrics, a Streamlit app collected human ratings comparing **zero-shot** vs **fine-tuned** outputs:

- Notebook-generated audio:
  - `data/audio/speaker_{ID}/zero_shot_{i}.wav`
  - `data/audio/speaker_{ID}/fine_tuned_{i}.wav`
  - prompts: `data/audio/five_random_sentence_pairs.json`
- UI: `human_evaluation/app.py`
- Ratings:
  - translation_accuracy, speaker_persona_match, naturalness, overall
- Saved to:
  - `results/human-evalution/evalator-{number}/human_eval_{timestamp}.json`
  - `results/human-evalution/evalator-{number}/responses.csv`

## Limitations / What changed from the original plan
- No PEFT/LoRA experiments (only zero-shot vs full fine-tune)
- No end-to-end “speech input → ASR → MT → TTS” dataset evaluation (e.g., CoVoST2); instead, Spanish targets came from Opus-100 text pairs and ASR was used on the synthesized output as an intelligibility proxy
- Automatic metric used was sentence BLEU (not chrF/WER), and speaker similarity was x-vector cosine similarity (no EER/SV classifier)

## Future Work
- Add a true speech-input benchmark and run the full cascaded ASR→MT→TTS pipeline end-to-end
- Implement PEFT (LoRA/adapters) on SpeechT5 and compare vs full fine-tuning
- Add WER/chrF and a speaker verification-style metric (EER) using x-vectors
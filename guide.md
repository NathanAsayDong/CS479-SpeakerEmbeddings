## CS479 Project Guide

This project is primarily driven through the Jupyter notebook: `cs479_project.ipynb`.

Most of the implementation (data loading, ASR, translation, speaker embeddings, TTS, fine-tuning, and evaluation) lives **directly inside the notebook cells**, so the best “how it works” documentation is to follow the notebook top-to-bottom.

---

## Running `cs479_project.ipynb`

### Prerequisites

- **Python environment**: this repo already includes a local `venv/` (recommended to use it).
- **Disk space / downloads**: running the notebook will download models + datasets (e.g., Hugging Face datasets and SpeechT5/Whisper components). This can take time and storage.

### Install dependencies (if needed)

The dependency list used during development is in `depricated/requirements.txt`.

From the project root:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Launch Jupyter and run the notebook

From the project root:

```bash
source venv/bin/activate
jupyter notebook
```

Then open `cs479_project.ipynb` and run the cells in order. The notebook:

- downloads/loads datasets into `data/`
- writes audio outputs into `output/` and `results/`
- saves fine-tuned models into `models/finetuned_models/`

### Where models are saved

When you fine-tune SpeechT5 in the notebook, the final exported model artifacts are saved here:

- `models/finetuned_models/speecht5_finetuned_voxpopuli_speaker_{speaker_id}/`

During training, Hugging Face Trainer logs/checkpoints are also written to a run directory like:

- `speecht5_finetuned_voxpopuli_speaker_{speaker_id}/` (TensorBoard logs, checkpoints, etc.)

---

## Using the TTS service (`TTSService`)

The notebook defines a `TTSService` class for synthesis. The key API is:

- `synthesize(text: str, output_file_path: str, speaker_audio_path: str) -> str`
  - Generates speech for `text` using the **speaker style** extracted from `speaker_audio_path`.
  - Writes the result to `output_file_path` and returns the path.

It also supports loading a fine-tuned model:

- `load_model_finetuned(model_path: str, processor_path: str)`

### Base (zero-shot) synthesis

In the notebook (after the `TTSService` cell has been run):

```python
tts = TTSService()

reference_audio = "results/zero_shot/speaker_1055/duration_10/source_audio.wav"
out_wav = tts.synthesize(
    text="Hola, ¿cómo estás?",
    output_file_path="output/temp_zero_shot.wav",
    speaker_audio_path=reference_audio,
)
print("Wrote:", out_wav)
```

### Fine-tuned synthesis

```python
tts_ft = TTSService()
tts_ft.load_model_finetuned(
    model_path="models/finetuned_models/speecht5_finetuned_voxpopuli_speaker_1055",
    processor_path="models/finetuned_models/speecht5_finetuned_voxpopuli_speaker_1055",
)

reference_audio = "results/zero_shot/speaker_1055/duration_10/source_audio.wav"
out_wav = tts_ft.synthesize(
    text="Hola, ¿cómo estás?",
    output_file_path="output/temp_fine_tuned.wav",
    speaker_audio_path=reference_audio,
)
print("Wrote:", out_wav)
```

If you re-run experiments, outputs will be overwritten in some “temp” files (e.g., `output/temp*.wav`) and appended/new files may be created under `results/`.

---

## Running the Human Evaluator App (Streamlit)

The human evaluation UI lives in `human_evaluation/app.py` and is documented in `human_evaluation/README.md`.

### What the app expects

Before running the UI, generate the evaluation audio from the notebook (the notebook has a “Qualitative Comparison (Human Evaluator)” section that writes these files).

Expected files:

- **Notebook-generated evaluation audio**
  - `data/audio/speaker_{ID}/zero_shot_{i}.wav`
  - `data/audio/speaker_{ID}/fine_tuned_{i}.wav` (optional)
- **Sentence pairs JSON** (written by the notebook)
  - `data/audio/five_random_sentence_pairs.json` (format: `[[en, es], ...]`)
- **Speaker reference audio** (used as the “target voice” reference)
  - `results/zero_shot/speaker_{ID}/duration_4/source_audio.wav`
  - falls back to `duration_10` if `duration_4` is missing

### Run the Streamlit app

From the project root:

```bash
source venv/bin/activate
streamlit run human_evaluation/app.py
```

### Where evaluation results are saved

When you click **Export results to disk** in the UI, outputs are written to:

- `results/human-evalution/evalator-{number}/human_eval_{timestamp}.json`
- `results/human-evalution/evalator-{number}/responses.csv`

(The directory name uses the repo’s existing spelling: `human-evalution/evalator-{number}`.)


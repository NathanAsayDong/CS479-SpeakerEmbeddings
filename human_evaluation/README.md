# Human Evaluation UI (Streamlit)

This folder contains a small Streamlit app to collect **human ratings** comparing **zero-shot** vs **fine-tuned** Spanish synthesis outputs.

## What it expects in your repo

- Notebook-generated audio (from your qualitative evaluation cell):
  - `data/audio/speaker_{ID}/zero_shot_{i}.wav`
  - `data/audio/speaker_{ID}/fine_tuned_{i}.wav` (optional)
- Text prompts (so the UI can display EN/ES):
  - `data/audio/five_random_sentence_pairs.json` (your notebook writes this)
    - format: `[[en, es], [en, es], ...]`

In this UI, the **speaker reference** is loaded from:

- `results/zero_shot/speaker_{ID}/duration_4/source_audio.wav` (falls back to `duration_10` if missing)

## How to run

From the project root (recommended: inside your venv):

```bash
streamlit run human_evaluation/app.py
```

## Where results are saved

When you click **Export results to disk**, outputs are written to:

- `results/human-evalution/evalator-{number}/human_eval_{timestamp}.json`
- `results/human-evalution/evalator-{number}/responses.csv`

Notes:
- The directory name intentionally matches your requested spelling: `human-evalution/evalator-{number}`.
- You can export multiple times; each export writes a new JSON and appends to `responses.csv`.



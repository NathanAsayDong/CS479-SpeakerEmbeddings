from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ZERO_SHOT_DIR = RESULTS_DIR / "zero_shot"
FINE_TUNE_DIR = RESULTS_DIR / "fine_tune"
GENERATED_AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

# NOTE: You asked for this exact (typo'd) output directory name.
HUMAN_EVAL_OUTPUT_DIR = RESULTS_DIR / "human-evalution"


METRICS: List[Tuple[str, str]] = [
    ("translation_accuracy", "Translation accuracy (meaning preserved from English → Spanish)"),
    ("speaker_persona_match", "Persona / speaker match (does it sound like the same speaker?)"),
    ("naturalness", "Naturalness (human-like, not robotic/glitchy)"),
    ("overall", "Overall quality"),
]


@dataclass
class Sample:
    speaker_id: str
    pair_index: int
    source_text_en: str
    target_text_es: str
    reference_audio_wav: Path
    zero_shot_audio_wav: Path
    fine_tune_audio_wav: Optional[Path]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_sentence_pairs(speaker_dir: Path) -> Dict[int, Dict[str, str]]:
    """
    Loads EN/ES sentence pairs so the UI can show the text prompts.

    Supported files (first one found wins):
    - Per-speaker (optional): `data/audio/speaker_{id}/sentence_pairs.json`
        - format: [{"en": "...", "es": "..."}, ...]
    - Global (from your notebook): `data/audio/five_random_sentence_pairs.json`
        - format: [[en, es], [en, es], ...]
    """
    candidates = [
        speaker_dir / "sentence_pairs.json",
        GENERATED_AUDIO_DIR / "five_random_sentence_pairs.json",
    ]
    pairs_path = next((p for p in candidates if p.exists()), None)
    if pairs_path is None:
        return {}
    try:
        payload = _read_json(pairs_path)
        out: Dict[int, Dict[str, str]] = {}
        if isinstance(payload, list):
            for i, item in enumerate(payload):
                if isinstance(item, dict):
                    en = str(item.get("en", "")).strip()
                    es = str(item.get("es", "")).strip()
                    out[i] = {"en": en, "es": es}
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    en = str(item[0]).strip()
                    es = str(item[1]).strip()
                    out[i] = {"en": en, "es": es}
        return out
    except Exception:
        return {}


def _pick_reference_audio(speaker_id: str) -> Path:
    """
    Matches the notebook cell: `results/zero_shot/speaker_{id}/duration_4/source_audio.wav`.
    Falls back to duration_10 if duration_4 is missing.
    """
    d4 = ZERO_SHOT_DIR / f"speaker_{speaker_id}" / "duration_4" / "source_audio.wav"
    if d4.exists():
        return d4
    d10 = ZERO_SHOT_DIR / f"speaker_{speaker_id}" / "duration_10" / "source_audio.wav"
    return d10


def discover_samples(
    speaker_ids: List[str],
    durations: Optional[List[str]] = None,  # kept for backwards-compat; unused in generated-audio mode
    max_per_speaker: int = 5,
) -> List[Sample]:
    """
    Discover samples from the notebook-generated audio:
      data/audio/speaker_{ID}/zero_shot_{i}.wav
      data/audio/speaker_{ID}/fine_tuned_{i}.wav
    """
    samples: List[Sample] = []
    pat = re.compile(r"^zero_shot_(\d+)\.wav$")

    for spk in speaker_ids:
        speaker_dir = GENERATED_AUDIO_DIR / f"speaker_{spk}"
        if not speaker_dir.exists():
            continue

        pair_text = _load_sentence_pairs(speaker_dir)
        reference_audio = _pick_reference_audio(spk)

        indices: List[int] = []
        for p in speaker_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                indices.append(int(m.group(1)))
        indices = sorted(indices)[:max_per_speaker]

        for i in indices:
            zero_audio = speaker_dir / f"zero_shot_{i}.wav"
            fine_audio = speaker_dir / f"fine_tuned_{i}.wav"
            if not fine_audio.exists():
                fine_audio = None

            source_text = pair_text.get(i, {}).get("en", "").strip()
            target_text = pair_text.get(i, {}).get("es", "").strip()
            samples.append(
                Sample(
                    speaker_id=str(spk),
                    pair_index=int(i),
                    source_text_en=source_text,
                    target_text_es=target_text,
                    reference_audio_wav=reference_audio,
                    zero_shot_audio_wav=zero_audio,
                    fine_tune_audio_wav=fine_audio,
                )
            )

    return samples


def _sample_to_jsonable(s: Sample) -> Dict[str, Any]:
    d = asdict(s)
    d["reference_audio_wav"] = str(s.reference_audio_wav)
    d["zero_shot_audio_wav"] = str(s.zero_shot_audio_wav)
    d["fine_tune_audio_wav"] = str(s.fine_tune_audio_wav) if s.fine_tune_audio_wav else None
    return d



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    # File-safe timestamp
    return time.strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    file_exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _method_block(
    *,
    method_key: str,
    method_label: str,
    audio_path: Optional[Path],
    state_key_prefix: str,
) -> Dict[str, Any]:
    st.subheader(method_label)
    if audio_path is None or not audio_path.exists():
        st.warning(f"Missing audio for **{method_label}** (file not found). You can still rate if you want, or leave comments.")
    else:
        st.audio(str(audio_path), format="audio/wav")

    metrics_out: Dict[str, Any] = {}
    cols = st.columns(2)
    for i, (metric_key, metric_label) in enumerate(METRICS):
        with cols[i % 2]:
            metrics_out[metric_key] = st.slider(
                metric_label,
                min_value=1,
                max_value=5,
                value=int(st.session_state.get(f"{state_key_prefix}:{method_key}:{metric_key}", 3)),
                step=1,
                key=f"{state_key_prefix}:{method_key}:{metric_key}",
            )

    metrics_out["comments"] = st.text_area(
        f"Comments for {method_label} (optional)",
        value=str(st.session_state.get(f"{state_key_prefix}:{method_key}:comments", "")),
        key=f"{state_key_prefix}:{method_key}:comments",
        height=80,
    )

    return metrics_out


def main() -> None:
    st.set_page_config(page_title="Human Evaluation – S2ST", layout="wide")

    st.title("Human Evaluation: English → Spanish Speech-to-Speech Translation")
    st.caption(
        "For each item: listen to a speaker reference, read the English source sentence, then listen to the Spanish outputs. "
        "Rate each output on 1–5 scales."
    )

    with st.sidebar:
        st.header("Evaluator")
        evaluator_number = st.number_input("evaluator_number", min_value=1, value=1, step=1)
        evaluator_name = st.text_input("Name/ID (optional)", value="")
        st.divider()
        st.header("Study setup (fixed)")
        st.write("- 3 speakers: 1055, 124992, 28165")
        st.write("- 5 items per speaker: pair indices 0–4 (from `data/audio/speaker_{id}/*.wav`)")
        st.write("- Speaker reference: `results/zero_shot/speaker_{id}/duration_4/source_audio.wav`")
        randomize = st.checkbox("Randomize order", value=False)
        show_gold_spanish = st.checkbox("Show gold Spanish text (can bias; usually keep OFF)", value=False)

    speakers = ["1055", "124992", "28165"]

    samples = discover_samples(speaker_ids=speakers, durations=None, max_per_speaker=5)
    if randomize:
        # deterministic shuffle per evaluator number, so re-runs don’t scramble progress
        import random

        rng = random.Random(int(evaluator_number))
        rng.shuffle(samples)
    else:
        samples = sorted(
            samples,
            key=lambda s: (
                int(s.speaker_id) if s.speaker_id.isdigit() else s.speaker_id,
                int(s.pair_index),
            ),
        )

    if not samples:
        st.error(
            "No samples discovered. Check that you generated audio files like "
            "`data/audio/speaker_1055/zero_shot_0.wav` (and optionally `fine_tuned_0.wav`)."
        )
        return

    # Session state for progress
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("responses", {})  # key -> response dict

    idx = int(st.session_state["idx"])
    idx = max(0, min(idx, len(samples) - 1))
    st.session_state["idx"] = idx

    sample = samples[idx]
    item_key = f"spk={sample.speaker_id}|pair={sample.pair_index}"
    state_prefix = f"item:{item_key}"

    st.progress((idx + 1) / len(samples), text=f"Item {idx + 1} / {len(samples)} — {item_key}")

    # Header block (text + reference audio)
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.subheader("Speaker reference (listen first)")
        if sample.reference_audio_wav.exists():
            st.audio(str(sample.reference_audio_wav), format="audio/wav")
            st.caption(f"Reference file: `{sample.reference_audio_wav.relative_to(PROJECT_ROOT)}`")
        else:
            st.error(f"Missing reference audio: `{sample.reference_audio_wav}`")

        st.subheader("English source text")
        st.write(sample.source_text_en if sample.source_text_en else "_(missing source text; ensure `data/audio/five_random_sentence_pairs.json` exists)_")

        st.subheader("Spanish target text")
        st.write(sample.target_text_es if sample.target_text_es else "_(missing target text; ensure `data/audio/five_random_sentence_pairs.json` exists)_")

        if show_gold_spanish:
            st.subheader("Gold Spanish text (reference, optional)")
            st.write(sample.target_text_es if sample.target_text_es else "_Gold Spanish not available._")

    with right:
        st.subheader("Rate the Spanish outputs (1–5)")
        z_metrics = _method_block(
            method_key="zero_shot",
            method_label="Zero-shot (Spanish audio)",
            audio_path=sample.zero_shot_audio_wav,
            state_key_prefix=state_prefix,
        )
        st.divider()
        ft_metrics = _method_block(
            method_key="fine_tune",
            method_label="Fine-tuned (Spanish audio)",
            audio_path=sample.fine_tune_audio_wav,
            state_key_prefix=state_prefix,
        )

        preference = st.radio(
            "Which output do you prefer overall?",
            options=["no_preference", "zero_shot", "fine_tune"],
            index=int(
                ["no_preference", "zero_shot", "fine_tune"].index(
                    st.session_state.get(f"{state_prefix}:preference", "no_preference")
                )
            ),
            key=f"{state_prefix}:preference",
            horizontal=True,
        )

        general_notes = st.text_area(
            "General notes for this item (optional)",
            value=str(st.session_state.get(f"{state_prefix}:general_notes", "")),
            key=f"{state_prefix}:general_notes",
            height=80,
        )

    # Navigation + save
    nav_cols = st.columns([1, 1, 1, 2])
    with nav_cols[0]:
        if st.button("← Prev", use_container_width=True, disabled=(idx == 0)):
            st.session_state["idx"] = max(0, idx - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("Save", use_container_width=True):
            st.session_state["responses"][item_key] = {
                "speaker_id": sample.speaker_id,
                "pair_index": sample.pair_index,
                "source_text_en": sample.source_text_en,
                "target_text_es": sample.target_text_es,
                "paths": {
                    "reference_audio_wav": str(sample.reference_audio_wav),
                    "zero_shot_audio_wav": str(sample.zero_shot_audio_wav),
                    "fine_tune_audio_wav": str(sample.fine_tune_audio_wav) if sample.fine_tune_audio_wav else None,
                },
                "ratings": {
                    "zero_shot": z_metrics,
                    "fine_tune": ft_metrics,
                    "preference": preference,
                },
                "general_notes": general_notes,
                "saved_at_unix": time.time(),
            }
            st.success("Saved in session. Use “Export results” when finished (or anytime).")
    with nav_cols[2]:
        if st.button("Next →", use_container_width=True, disabled=(idx >= len(samples) - 1)):
            st.session_state["idx"] = min(len(samples) - 1, idx + 1)
            st.rerun()

    with nav_cols[3]:
        export = st.button("Export results to disk", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Completion")
    completed = len(st.session_state["responses"])
    st.write(f"Completed items: **{completed} / {len(samples)}**")

    if export:
        out_dir = HUMAN_EVAL_OUTPUT_DIR / f"evalator-{int(evaluator_number)}"
        _ensure_dir(out_dir)

        payload = {
            "evaluator_number": int(evaluator_number),
            "evaluator_name": evaluator_name,
            "exported_at_unix": time.time(),
            "exported_at": _now_ts(),
            "n_items_total": len(samples),
            "n_items_completed": completed,
            "samples_discovered": [_sample_to_jsonable(s) for s in samples],
            "responses": st.session_state["responses"],
        }

        json_path = out_dir / f"human_eval_{_now_ts()}.json"
        _write_json(json_path, payload)

        # Also write a flat CSV for easy analysis.
        flat_rows: List[Dict[str, Any]] = []
        for k, r in st.session_state["responses"].items():
            for method in ["zero_shot", "fine_tune"]:
                row = {
                    "evaluator_number": int(evaluator_number),
                    "evaluator_name": evaluator_name,
                    "item_key": k,
                    "speaker_id": r["speaker_id"],
                    "pair_index": r.get("pair_index", None),
                    "method": method,
                    "preference": r["ratings"].get("preference", "no_preference"),
                    "source_text_en": r.get("source_text_en", ""),
                    "target_text_es": r.get("target_text_es", ""),
                    "general_notes": r.get("general_notes", ""),
                    "method_comments": r["ratings"].get(method, {}).get("comments", ""),
                    "saved_at_unix": r.get("saved_at_unix", None),
                }
                for metric_key, _metric_label in METRICS:
                    row[metric_key] = r["ratings"].get(method, {}).get(metric_key, None)
                flat_rows.append(row)

        csv_path = out_dir / "responses.csv"
        _append_csv(csv_path, flat_rows)

        st.success(f"Exported:\n- `{json_path.relative_to(PROJECT_ROOT)}`\n- `{csv_path.relative_to(PROJECT_ROOT)}`")


if __name__ == "__main__":
    main()



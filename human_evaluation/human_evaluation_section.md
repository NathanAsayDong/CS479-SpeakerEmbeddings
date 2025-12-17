# Human Evaluation

Because automatic metrics do not fully capture whether synthesized translations both **preserve meaning** and **sound like the intended speaker**, we performed a human evaluation with bilingual listeners.

## Evaluators

We recruited evaluators who are **native Spanish speakers** and also **highly proficient in English**, enabling them to judge meaning preservation from English into Spanish by listening to the synthesized Spanish output.

## Stimuli and conditions

We evaluated three target speakers (Speaker 1–3). For each speaker, we prepared **five English sentences** (the same sentence set across speakers), and synthesized Spanish speech under two adaptation conditions:

- **Zero-shot (conditioning-only)**: a pretrained TTS model conditioned on a speaker embedding, without updating TTS weights.
- **Fine-tuned**: the TTS model fine-tuned on speaker-specific data, then used to synthesize the same Spanish translations.

For every evaluation item, the evaluator was shown the **English source text** and listened to:

1. A short **speaker reference** clip (to establish the target voice identity).
2. The **English source audio** (spoken by the target speaker).
3. The **Spanish synthesized output** for each condition (zero-shot and fine-tuned).

## Rating rubric (quantitative)

Evaluators rated each synthesized Spanish output on 5-point Likert scales (1 = poor, 5 = excellent):

- **Translation accuracy**: how well the Spanish audio preserves the meaning of the English source.
- **Speaker/persona match**: whether the Spanish output sounds like the same speaker as the reference (identity/timbre).
- **Tone/prosody match**: whether emotion, emphasis, and speaking style match the reference speaker.
- **Naturalness**: perceived human-likeness (absence of robotic artifacts, glitches, or discontinuities).
- **Spanish pronunciation & intelligibility**: clarity and correctness of Spanish phonetics and overall understandability.
- **Overall quality**: holistic judgment considering all factors above.

Evaluators also indicated an **overall preference** between the two conditions (zero-shot vs fine-tuned) and could provide free-form comments.

## Procedure

The evaluation was administered through a lightweight **Streamlit** interface to standardize presentation and capture responses. Each evaluator completed ratings for all combinations of:

- 3 speakers × 5 sentences × 2 systems = 30 system outputs

The interface presented each item with embedded audio playback for the reference, English source, and both Spanish outputs, along with sliders for each metric. Responses were saved to disk per-evaluator for later aggregation.

## Analysis

For each metric, we compute mean scores (and variability) for each system aggregated over speakers and sentences. We also report:

- **Per-speaker breakdowns**, since speaker similarity is speaker-dependent.
- **Preference rates** (fraction of items where fine-tuned was preferred over zero-shot).

If multiple evaluators are collected, we aggregate across evaluators and optionally run paired significance tests (e.g., Wilcoxon signed-rank) comparing fine-tuned vs zero-shot on matched items.



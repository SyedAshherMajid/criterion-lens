# CriterionLens

**Evidence-grounded, criterion-level clinical-trial eligibility screening with a fine-tuned open-weight Transformer, plus hand-built KV-cached decoding, tiled online-softmax attention and greedy speculative decoding.**

> ⚠️ **Educational research screening prototype — not a clinical enrollment decision.**

| | |
|---|---|
| Live app | _TBD (Hugging Face Space link)_ |
| Paper (IEEE) | _TBD (`report/CriterionLens_IEEE.pdf`)_ |
| Fine-tuned adapters | _TBD (HF Hub links)_ |
| Demo video | _TBD_ |
| Split manifest | [`data/splits/split_manifest.json`](data/splits/split_manifest.json) |

## What it does
Given a patient note and a trial's inclusion/exclusion criteria, CriterionLens judges **each criterion** (`included` / `not included` / `excluded` / `not excluded` / `not enough information` / `not applicable`), cites the **patient-note sentence IDs** that support the decision, shows the exact evidence text and a grounded explanation, and aggregates the per-criterion results into a *preliminary* patient–trial outcome using an explicit rule.

- **Data:** `ncbi/TrialGPT-Criterion-Annotations` only. A fixed patient-grouped split (37/8/8 patients, `default_rng(42)`). GPT-4 columns are never used.
- **Model:** `Qwen2.5-1.5B-Instruct` + LoRA (target); `Qwen2.5-0.5B-Instruct` (draft).
- **Inference mechanisms (implemented by hand, verified against references):** manual KV-cache decoding, tiled online-softmax attention, greedy speculative decoding.
- **App:** a Gradio browser app with screening, live inference/attention benchmarking and report export.

## Results (held-out test split, 8 patients)
_TBD — generated from `results/eval/` by `scripts/make_figures.py`._

## Quick start
_TBD — see `docs/EXECUTION_PLAN.md` Phase 10/12._

## Project plan
- [`docs/PLAN.md`](docs/PLAN.md): full design (source of truth)
- [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md): step-by-step build order
- [`docs/RUBRIC_CHECKLIST.md`](docs/RUBRIC_CHECKLIST.md): rubric → evidence map
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decision log
- [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md): paper outline, demo script, AI-use statement

## License
Code: MIT. Dataset and model weights are under their own licences (see `docs/MODEL_SELECTION.md`).

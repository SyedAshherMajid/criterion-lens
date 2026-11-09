# CriterionLens

**Evidence-grounded, criterion-level clinical-trial eligibility screening with a fine-tuned open-weight Transformer, plus hand-built KV-cached decoding, tiled online-softmax attention and greedy speculative decoding.**

> ⚠️ **Educational research screening prototype — not a clinical enrollment decision.**

| | |
|---|---|
| Official app | Locally hosted Gradio app (see [Windows quick start](#windows-quick-start)) |
| Optional hosted app | Hugging Face Space files are ready in `space/`; creation requires HF PRO for `cpu-basic` |
| Paper (IEEE) | _TBD (`report/CriterionLens_IEEE.pdf`)_ |
| Fine-tuned adapters | [Qwen2.5 1.5B LoRA](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-lora) · [Qwen2.5 0.5B LoRA](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-lora) |
| Merged 0.5B model | [Qwen2.5 0.5B merged](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-merged) |
| Demo video | _TBD_ |
| Split manifest | [`data/splits/split_manifest.json`](data/splits/split_manifest.json) |

## What it does
Given a patient note and a trial's inclusion/exclusion criteria, CriterionLens judges **each criterion** (`included` / `not included` / `excluded` / `not excluded` / `not enough information` / `not applicable`), cites the **patient-note sentence IDs** that support the decision, shows the exact evidence text and a grounded explanation, and aggregates the per-criterion results into a *preliminary* patient–trial outcome using an explicit rule.

- **Data:** `ncbi/TrialGPT-Criterion-Annotations` only. A fixed patient-grouped split (37/8/8 patients, `default_rng(42)`). GPT-4 columns are never used.
- **Model:** `Qwen2.5-1.5B-Instruct` + LoRA (target); `Qwen2.5-0.5B-Instruct` (draft).
- **Inference mechanisms (implemented by hand, verified against references):** manual KV-cache decoding, tiled online-softmax attention, greedy speculative decoding.
- **App:** a Gradio browser app with screening, live inference/attention benchmarking and report export.
- **Deployment:** locally hosted on Windows; the assignment allows a browser-accessible local application. A free Hugging Face `cpu-basic` Space could not be created because HF requires PRO (2026); deployment files remain ready if PRO becomes available.

## Results (held-out test split, 8 patients)
_TBD — generated from `results/eval/` by `scripts/make_figures.py`._

## Windows quick start

1. Clone the repository and create/activate the project environment as described in [`docs/LOCAL_ENVIRONMENT.md`](docs/LOCAL_ENVIRONMENT.md).
2. Install the project and app dependencies in `.venv` if not already installed:

   ```powershell
   .venv\Scripts\python.exe -m pip install -e ".[app]"
   ```

3. Double-click `run_app.bat`, or run `.\run_app.bat` from PowerShell. It starts the app at `http://127.0.0.1:7860` and opens the browser. To create a temporary public Gradio link for a defense, run `.\run_app.bat --share` only if the evaluator needs remote access.
4. Choose a synthetic example or enter an appropriately de-identified note and trial criteria. The selected model is downloaded from Hugging Face Hub on first use and cached locally; allow time and disk space for the initial download.

The app automatically uses CUDA when available and otherwise runs on CPU. The GTX 1060 can run the smaller 0.5B model; the larger 1.5B model may need CPU or lower-memory settings depending on available VRAM. CPU inference is supported but slower. Use `CL_DEVICE=cpu` to force CPU, or `CL_DEVICE=cuda` to require CUDA; `CL_DTYPE` can select `float32`, `float16`, `bfloat16`, or `auto`. The 1.5B and 0.5B models can be selected in the app.

### Troubleshooting

- **Missing `.venv\Scripts\python.exe`:** create the environment and install the app dependencies using the project setup instructions above.
- **Model download/authentication error:** confirm internet access and that the required Hugging Face model is accessible; restart the app after resolving credentials or network access.
- **CUDA unavailable or out of memory:** set `CL_DEVICE=cpu` and restart; expect slower inference. Select the 0.5B model to reduce memory use.
- **Port 7860 is busy:** close the other app using that port before restarting.
- **Remote access:** `--share` creates a temporary public link. Use it only when needed for the evaluator; do not submit real or identifiable patient information.

The application is a research prototype, not a medical device or clinical eligibility decision. Human review of source records is required.

## Project plan
- [`docs/PLAN.md`](docs/PLAN.md): full design (source of truth)
- [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md): step-by-step build order
- [`docs/RUBRIC_CHECKLIST.md`](docs/RUBRIC_CHECKLIST.md): rubric → evidence map
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decision log
- [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md): paper outline, demo script, AI-use statement

## License
Code: MIT. Dataset and model weights are under their own licences (see `docs/MODEL_SELECTION.md`).

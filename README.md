# CriterionLens

**Evidence-grounded, criterion-level clinical-trial eligibility screening with a fine-tuned open-weight Transformer, plus hand-built KV-cached decoding, tiled online-softmax attention and greedy speculative decoding.**

> ⚠️ **Educational research screening prototype — not a clinical enrollment decision.**

| | |
|---|---|
| Official app | Locally hosted Gradio app (see [Windows quick start](#windows-quick-start)) |
| Optional hosted app | Hugging Face Space files are ready in `space/`; creation requires HF PRO for `cpu-basic` |
| Paper (IEEE) | [`report/CriterionLens_IEEE.pdf`](report/CriterionLens_IEEE.pdf) |
| Fine-tuned adapters | [Qwen2.5 1.5B LoRA](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-lora) · [Qwen2.5 0.5B LoRA](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-lora) |
| Merged models | [Qwen2.5 1.5B merged](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-merged) · [Qwen2.5 0.5B merged](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-merged) |
| Demo video | [YouTube (5:35, captioned)](https://youtu.be/Z3mVr0ccfKE) |
| Split manifest | [`data/splits/split_manifest.json`](data/splits/split_manifest.json) |

## What it does
Given a patient note and a trial's inclusion/exclusion criteria, CriterionLens judges **each criterion** (`included` / `not included` / `excluded` / `not excluded` / `not enough information` / `not applicable`), cites the **patient-note sentence IDs** that support the decision, shows the exact evidence text and a grounded explanation, and aggregates the per-criterion results into a *preliminary* patient–trial outcome using an explicit rule.

- **Data:** `ncbi/TrialGPT-Criterion-Annotations` only. A fixed patient-grouped split (37/8/8 patients, `default_rng(42)`). GPT-4 columns are never used.
- **Model:** `Qwen2.5-1.5B-Instruct` + LoRA (target); `Qwen2.5-0.5B-Instruct` (draft).
- **Inference mechanisms (implemented by hand, verified against references):** manual KV-cache decoding, tiled online-softmax attention, greedy speculative decoding.
- **App:** a Gradio browser app with screening, live inference/attention benchmarking and report export.
- **Deployment:** locally hosted on Windows; the assignment allows a browser-accessible local application. A free Hugging Face `cpu-basic` Space could not be created because HF requires PRO (2026); deployment files remain ready if PRO becomes available.

## Results (held-out test split, 8 patients, 206 criteria)
Evaluated **once** at tag `v1.0-frozen` (see [`results/TEST_ACCESS_LOG.md`](results/TEST_ACCESS_LOG.md)). Greedy decoding. 95% bootstrap CIs (1,000 resamples) are in `report/tables/main_results.tex`.

| Model | Accuracy [95% CI] | Macro-F1 (all 6 labels) | Evidence micro-F1 | Evidence macro-F1* | Abstention acc. |
|---|---|---|---|---|---|
| Base Qwen2.5-0.5B | 0.325 | 0.093 | 0.020 | 0.007 | 1.000 |
| Base Qwen2.5-1.5B | 0.204 | 0.158 | 0.242 | 0.498 | 0.000 |
| Fine-tuned 0.5B (LoRA) | 0.694 | 0.410 | 0.473 | 0.526 | 0.890 |
| **Fine-tuned 1.5B (LoRA, headline)** | **0.762 [0.704, 0.816]** | **0.476 [0.420, 0.529]** | **0.542** | **0.584** | 0.897 |

\*Averaged over criteria with non-empty expert evidence. Macro-F1 averages all six fixed labels, so labels that never occur in the 8 test patients count as 0. Averaged over only the labels present, the fine-tuned 1.5B scores 0.571. JSON parse failures for both fine-tuned models: 0%. Validation results, per-class scores and confusion matrices: `results/eval/`, `results/figures/`. Error analysis (12 distinct cases): [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md).

Inference-mechanism benchmarks (Kaggle T4): see [Benchmarks](#benchmarks-kaggle-tesla-t4).

## Benchmarks (Kaggle Tesla T4)
> **Addendum (after submission):** the full benchmark grid finished later. KV cache reaches **24.8×** at 2048 prompt tokens. In fp16, speculative decoding diverged from target greedy output on 4 of 20 prompts, a numerical effect independent of draft and k; it is exact in fp32. See [`docs/BENCHMARK_FULL_RUN.md`](docs/BENCHMARK_FULL_RUN.md) and `results/bench_full/`.

Raw logs: [`results/bench/`](results/bench/) (JSON + CSV per benchmark). Kaggle kernel `criterion-lens-benchmarks-quick` (`kaggle/jobs/benchmarks_quick/`), 1 warm-up + 3 repeats (speculative: 6 synthetic prompts, 2 repeats). Figures: `report/figures/`.

| Mechanism | Result |
|---|---|
| **KV cache** (fine-tuned 1.5B, fp16, 128 new tokens) | 1.5× / 4.9× / **9.8×** faster than full-prefix recomputation at 128 / 512 / 1024 prompt tokens (2.76 → 29.1 tokens/s at 1024). 0.5B: 1.1× / 2.5× / 6.2×. Cache grows 28 KiB/token. Identical greedy tokens in all precision checks. |
| **Tiled online-softmax attention** | 80/80 configurations within tolerance (max abs. error 1.2e-6 fp32, 9.7e-4 fp16). Slower than the naive reference (pure PyTorch, not fused), but much less memory: 93 MB vs 878 MB at 2048 tokens; at 8192 tokens the reference runs out of memory and tiled finishes in 345 MB. |
| **Greedy speculative decoding** (1.5B target, 0.5B draft) | Exact: token equality 1.0, 0 mismatches. Acceptance 0.80→0.46 (base draft) and 0.83→0.48 (fine-tuned draft) for k = 1→8. Target forwards drop to 0.23–0.57 per token, but wall-clock speedup is **< 1** (0.60–0.94×): on a T4 at batch 1 the draft passes cost more than they save. Reported as measured. |

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

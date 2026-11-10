# CriterionLens Handoff

The official application is the locally hosted Gradio app. Start it on Windows
from the repository root with `run_app.bat`; it opens
`http://127.0.0.1:7860`. See the README for setup, first-use model downloads,
CPU/GPU behavior, and troubleshooting.

The Hugging Face Space was not created: in 2026, Hugging Face requires PRO for
free `cpu-basic` Gradio Spaces. The assignment permits a browser-accessible
locally hosted app. `space/` and `scripts/deploy_space.py` remain available if
PRO becomes available, but Space deployment is not required and should not be
retried as a prerequisite.

## Checkpoint status — 2026-11-10 05:44:40 +05:00

- Latest checkpoint commit: [`d101a5a`](https://github.com/SyedAshherMajid/criterion-lens/commit/d101a5a).
- No Kaggle kernel is currently running. The train kernel is `COMPLETE`; the
  prior evaluation/benchmark kernel is `ERROR` after all four validation
  evaluations completed and the benchmark step failed because the 1.5B merged
  model was not yet on the Hub. The validation outputs are already committed;
  do not rerun or download them again.
- The public 1.5B merged-model repository is still uploading. The last check
  saw 69% progress, and the Hub API listed only `.gitattributes`; verify that
  `config.json` and model weights are listed and publicly accessible before
  launching benchmarks.
- The private Kaggle code dataset is ready, but re-sync the latest committed
  source before the next job.

## Current links

- GitHub: [CriterionLens repository](https://github.com/SyedAshherMajid/criterion-lens).
- Hugging Face models:
  - [Qwen2.5 0.5B LoRA adapter](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-lora)
  - [Qwen2.5 1.5B LoRA adapter](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-lora)
  - [Qwen2.5 0.5B merged model](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-merged)
  - [Qwen2.5 1.5B merged model](https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-merged) — upload in progress; not ready for benchmarking until weights/config are visible.
- Kaggle:
  - [Private code dataset](https://www.kaggle.com/datasets/syedashhermajid/criterion-lens-code) — ready.
  - [Training kernel](https://www.kaggle.com/code/syedashhermajid/criterion-lens-train) — `COMPLETE`.
  - [Prior evaluation/benchmark kernel](https://www.kaggle.com/code/syedashhermajid/criterion-lens-eval-bench) — `ERROR` after the four completed validation runs; do not rerun validation.
  - [Old smoke-test kernel](https://www.kaggle.com/code/syedashhermajid/criterion-lens-smoketest) — ignore.
  - Benchmarks-only kernel slug to push: `syedashhermajid/criterion-lens-benchmarks`; it has not been pushed. Source: [`kaggle/jobs/benchmarks/`](../kaggle/jobs/benchmarks/).
  - Final-test kernel slug to push only after `v1.0-frozen`: `syedashhermajid/criterion-lens-eval-test-final`; source metadata: [`kernel-metadata.json`](../kaggle/jobs/eval_test_final/kernel-metadata.json).

## Ordered next steps

1. **Benchmarks-only Kaggle job:** verify the 1.5B merged model is public with
   its weights/config, sync the latest source, push
   `syedashhermajid/criterion-lens-benchmarks` with T4, then download only
   benchmark outputs (KV, attention, and speculative CSV/JSON; do not rerun
   validation).
2. **Freeze:** freeze the selected prompt/decoding and adapters; commit/tag
   `v1.0-frozen`, then sync the source dataset from that exact tag.
3. **Final test evaluation:** push
   `syedashhermajid/criterion-lens-eval-test-final` once; download the four
   `test_*` result folders and `results/TEST_ACCESS_LOG.md`.
4. **Error analysis + figures:** run `scripts/error_analysis.py` on the final
   test outputs, then regenerate figures and tables.
5. **Report:** fill generated final numbers and compile `report/main.tex`
   locally with MiKTeX to `report/CriterionLens_IEEE.pdf`.
6. **README + HANDOFF.md:** replace pending results/PDF placeholders, add
   verified links, and leave only video recording, submission, and live
   defense as human tasks.

## Defense day

- [ ] Start the app with `run_app.bat` and confirm it opens at
  `http://127.0.0.1:7860`.
- [ ] Test one unseen **synthetic or appropriately de-identified** patient note
  with trial criteria; verify evidence, missing information, the preliminary
  outcome, and report export.
- [ ] Use `run_app.bat --share` only if the evaluator needs remote access. The
  link is temporary; do not submit real or identifiable patient information.

## Human-only remaining tasks

- Record the demo video.
- Submit the required materials.
- Attend and complete the live defense.

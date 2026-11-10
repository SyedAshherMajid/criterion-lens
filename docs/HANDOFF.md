# CriterionLens Handoff

Everything automated is done. Only the human tasks at the bottom remain.

## Application
The official application is the **locally hosted Gradio app** (the assignment allows "a browser-accessible locally hosted or deployed application").
Start it on Windows from the repository root with `run_app.bat`. It opens `http://127.0.0.1:7860`. See the README for setup, first-use model downloads, CPU/GPU behaviour and troubleshooting.
No Hugging Face Space was created, because in 2026 Hugging Face requires PRO for `cpu-basic` Gradio Spaces (D-025). `space/` and `scripts/deploy_space.py` remain ready if PRO is ever available.

## Links to submit
- GitHub: https://github.com/SyedAshherMajid/criterion-lens (**make public before submitting**, see below)
- Report PDF: `report/CriterionLens_IEEE.pdf` (in the repo)
- Weights (public on the Hugging Face Hub):
  - 1.5B LoRA adapter (headline): https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-lora
  - 1.5B merged: https://huggingface.co/Ashher123/criterion-lens-qwen2.5-1.5b-merged
  - 0.5B LoRA adapter: https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-lora
  - 0.5B merged: https://huggingface.co/Ashher123/criterion-lens-qwen2.5-0.5b-merged
- Split manifest: `data/splits/split_manifest.json` (attach the file too)
- Benchmark logs: `results/bench/` (Kaggle job `criterion-lens-benchmarks-quick`)
- Test access log: `results/TEST_ACCESS_LOG.md`
- Demo video: https://youtu.be/Z3mVr0ccfKE

## Final state (10 Nov 2026)
- Freeze tag `v1.0-frozen` = commit `b1b089d`. The one-time test evaluation ran on Kaggle (2× T4) from that tag: four rows in `results/TEST_ACCESS_LOG.md`.
- Headline (test, fine-tuned 1.5B): accuracy 0.762 [0.704, 0.816], fixed-label macro-F1 0.476 [0.420, 0.529]. Base 1.5B: accuracy 0.204.
- Error analysis: `docs/ERROR_ANALYSIS.md` (12 curated cases + automated candidates).
- Kaggle kernels (private, account `syedashhermajid`): `criterion-lens-train`, `criterion-lens-eval-bench` (validation), `criterion-lens-benchmarks`, `criterion-lens-eval-test-final`.

## Human-only remaining tasks
1. ~~Record and upload the demo video~~ (done: https://youtu.be/Z3mVr0ccfKE).
2. ~~Make the repo public~~ (done): `gh repo edit SyedAshherMajid/criterion-lens --visibility public --accept-visibility-change-consequences`, then open it in a logged-out browser to check.
3. **Submit on Google Classroom:** the report PDF, GitHub URL, startup instructions (README), weight links, the split manifest JSON, benchmark logs and the video link.
4. **After grading:** regenerate the Hugging Face and Kaggle tokens. They were visible in a screenshot.

## Defense day
- [ ] Start the app with `run_app.bat` and confirm it opens at `http://127.0.0.1:7860`. Run one example first so the model is downloaded and warm.
- [ ] Test one unseen **synthetic or de-identified** patient note with trial criteria. Show the evidence, missing information, the preliminary outcome and the report export.
- [ ] Show the Inference Lab (no-cache vs KV cache vs speculative) and the Attention Lab live.
- [ ] Use `run_app.bat --share` only if the evaluator needs remote access. The link is temporary.
- [ ] Be ready to explain: the KV-cache position handling (`docs/PLAN.md` §7), online softmax (§8), speculative accept/rollback (§9), and why absence-as-negation is the main error (`docs/ERROR_ANALYSIS.md`).

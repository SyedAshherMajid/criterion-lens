# CriterionLens — Execution Plan

Work top to bottom. Each step has an **owner** ([COPILOT] = coding agent, [YOU] = the student, who has to do it by hand), **files**, and an **acceptance check**. Tick `- [x]` and commit after each step.
Technical details for every step are in `docs/PLAN.md` (section numbers given as §).

**Deadline: 10 Nov 2026 + 2 h grace.** This plan was written on 9 Nov 2026, so time is extremely tight. Follow the priority tiers:
- **P0** = needed for marks in every rubric row. Do these first, end to end.
- **P1** = adds strong evidence (ablations, integration demos). Do these only after all P0 steps work.
- **P2** = stretch. Skip if behind schedule.

**Work runs in two parallel lanes. Read `docs/PARALLEL_WORKFLOW.md` first.** The Kaggle GPU lane (training, 1.5B evaluation, T4 benchmarks) is launched and collected **from the laptop by [COPILOT] using the Kaggle CLI** (token already in `.env`, verified). The laptop lane (GTX 1060 + CPU) does coding, tests, 0.5B runs, attention benchmarks, the app and the report at the same time. The Kaggle steps below that say **[YOU]** are now done by **[COPILOT]** through the CLI. The only exception: if Kaggle assigns an unsupported GPU, the student switches the accelerator to T4 x2 once on kaggle.com. Weights are pushed to the HF Hub from the laptop, so **no Kaggle Secrets are needed**.

## Suggested timeline (≈ 22 working hours with both lanes)
| Hours | Kaggle lane (GPU T4 x2) | Laptop lane (GTX 1060 + CPU) |
|---|---|---|
| 0–2 | — | Phase 0–2: package, data, split, audit; Phase 3 code; `sync_code.py` |
| 2–3 | **push `train`**: 1.5B on GPU0 ∥ 0.5B on GPU1 | Phase 6–8 code with tiny-model tests |
| 3–6 | train → **push `eval_val`** | Phase 5 metrics + pipeline; base-0.5B val smoke eval on the GTX 1060; Phase 7 attention tests + local bench |
| 6–9 | **push `bench`** (KV, speculative, attention on T4) | push adapters/merged models to HF; Phase 10 app |
| 9–12 | (idle / P1 `sweep`) | app finished; deploy the Space; Phase 9.3 figures |
| 12–13 | **freeze → push `eval_test_final` (once)** | error-analysis tooling |
| 13–19 | — | Phase 9.2 error analysis; Phase 11 report (local MiKTeX) |
| 19–22 | — | Phase 12 video **[YOU]**, README, submission |

---

## Phase 0 — Repository & environment (P0)

- [x] **0.1 [COPILOT]** The repo `criterion-lens` already exists with `docs/` and `.github/copilot-instructions.md`. Create the layout from PLAN §12 (empty `__init__.py` files, folders with `.gitkeep`), `.gitignore` (Python, `data/raw/`, `models/`, `outputs/`, `*.safetensors`, `*.bin`, `.env`, `wandb/`, `report/*.aux|log|out`), `.gitattributes` (`* text=auto`), `LICENSE` (MIT, author "Syed Ashher Majid").
- [x] **0.2 [COPILOT]** `pyproject.toml` (package `criterion_lens` in `src/`, Python `>=3.10`, pytest config with marker `slow`, ruff). Pin `requirements.txt` to the versions **already installed** (see `docs/LOCAL_ENVIRONMENT.md`): `transformers==4.57.6`, `peft==0.19.1`, `accelerate==1.14.0`, `datasets==5.0.0`, `huggingface_hub==0.36.2`, `numpy==2.2.6`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.9`, `pyyaml`, `tqdm`, `jinja2`, `pytest`; `torch>=2.4` left unpinned (local = 2.6.0+cu124, Kaggle uses its preinstalled one). `requirements-app.txt`: `-r requirements.txt` + `gradio==5.50.0`, `pysbd`, `requests`, `psutil`. On Kaggle, install only `-r requirements.txt` (never reinstall torch).
- [x] **0.3 [DONE by Claude]** The local env is **already built**: `criterion-lens/.venv` (Python 3.10.11) reuses the ML stack from `D:\Nastp_Project\venv` via a `.pth` file, and gradio/matplotlib/pytest/pysbd were added. **Do not reinstall anything.** Read `docs/LOCAL_ENVIRONMENT.md` for the rules. [COPILOT] only runs `.venv\Scripts\python.exe -m pip install -e . --no-deps` after creating `pyproject.toml`.
- [x] **0.4 [YOU]** `.env` already holds `HF_TOKEN`, `HF_USERNAME=Ashher123`, `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`. **The HF token is currently READ-only (verified via whoami), so replace it with a WRITE token**, which is needed to upload adapters and the Space. Kaggle Secrets are not needed (see PARALLEL_WORKFLOW.md). [COPILOT] writes `src/criterion_lens/utils/env.py::load_env()` (a tiny `.env` parser with no extra dependency; it never logs the value). All scripts call it, and `huggingface_hub` picks up `HF_TOKEN` from the environment. **Never commit or print the token.**
- [x] **0.5 [COPILOT]** `README.md` skeleton (keep the existing one and fill sections as phases complete).
**Accept:** `pytest -q` runs (0 tests OK), `python -c "import criterion_lens"` works, first commit pushed.

## Phase 1 — Research notes (P0, ~1 h, can be done any time before Phase 11)
- [x] **1.1 [COPILOT]** Create `docs/LITERATURE.md`: for each source in PLAN §11, record a verified citation (title, authors, venue, year, URL/DOI), a 2–3 line summary, its **limitation**, and **which design decision it informs**. Minimum 8 verified entries, target 12+. Mark any entry you could not verify and do not cite it.
- [x] **1.2 [COPILOT]** Create `docs/MODEL_SELECTION.md` from the PLAN §3.1 table. Fill each number from the model's `config.json` / model card. Write a script `scripts/inspect_models.py` that prints layers, heads, kv heads, head_dim, vocab, context and parameter counts from the configs (no weight download needed: `AutoConfig`) and writes `results/model_specs.json`. Cite it in the report.
**Accept:** ≥8 verified references, a specs JSON produced by the script.

## Phase 2 — Data, split, audit (P0)
- [x] **2.1 [COPILOT]** `src/criterion_lens/data/load.py::load_annotations(revision=None) -> pd.DataFrame`: load with `datasets.load_dataset("ncbi/TrialGPT-Criterion-Annotations", revision=...)`, **drop GPT-4 columns + `training` immediately** (PLAN §2.1), parse `expert_sentences` with `json.loads` into `list[int]`, normalise label strings (strip/lower), drop null/empty `criterion_text` (log count). Cache to `data/raw/annotations.parquet`.
- [x] **2.2 [COPILOT]** `data/sentences.py`: `parse_numbered_note(note) -> list[tuple[int,str]]` (regex PLAN §4.1, validates contiguous IDs) and `split_sentences(text) -> list[tuple[int,str]]` for unseen notes (use numbering if present; else pysbd; fallback regex). Tests in `tests/test_sentences.py` (dataset-style note, free text, edge cases: decimals like "3.5 mg", abbreviations "Dr.", empty).
- [x] **2.3 [COPILOT]** `data/split.py::make_split(df)` with the **exact** recipe in PLAN §2.2. `scripts/make_split.py` writes `data/splits/split_manifest.json` + `.csv` (with dataset revision SHA, counts, label counts per split). Commit the manifest.
- [x] **2.4 [COPILOT]** `tests/test_split.py` (disjoint, 37/8/8, every annotation once, deterministic, manifest equals recomputation) and `tests/test_no_leakage.py` (loaded frame contains no `gpt4_*`/`explanation_correctness`/`training` columns; for a sample of rows, the built prompt + target contains none of the GPT-4 explanation text).
- [x] **2.5 [COPILOT]** `scripts/audit_data.py` produces `results/data_audit.json` + `results/figures/label_dist.png`, `note_length_tokens.png` (everything listed in PLAN §2.3; token lengths use the Qwen2.5 tokenizer, which only needs a tokenizer download). Write a short summary in `docs/DECISIONS.md` (label/type inconsistencies, empty criteria, out-of-range IDs, decision taken).
**Accept:** manifest shows 37/8/8 patients; tests pass; audit JSON exists; the max prompt token length is known (used for `max_seq_len`).

## Phase 3 — Prompting & training code (P0)
- [x] **3.1 [COPILOT]** `data/prompts.py`: `SYSTEM_PROMPT`, `ALLOWED_LABELS`, `build_messages(sentences, criterion_type, criterion_text)`, `format_target(evidence_ids, label)` exactly as in PLAN §4.1. `tests/test_prompts.py` (inclusion/exclusion label lists correct, target is valid JSON with sorted unique IDs, no GPT-4 text).
- [x] **3.2 [COPILOT]** `data/tokenize.py::encode_example(tokenizer, example, max_len)` → `input_ids`, `attention_mask`, `labels` (prompt masked with -100), plus truncation handling + log (PLAN §4.4). Test: decoding the unmasked labels gives exactly the target JSON + end token.
- [x] **3.3 [COPILOT]** `inference/parse.py` (`parse_output(text, criterion_type, n_sentences) -> ParsedOutput`), `inference/schema.py` (PLAN §6.1). Tests on malformed outputs.
- [x] **3.4 [COPILOT]** `training/sft.py` + `scripts/train.py --config configs/train_qwen25_1p5b.yaml`: load model (fp16 or fp32 via config), PEFT LoRA (PLAN §4.3), HF `Trainer` (eval on val loss each epoch, save each epoch, `logging_steps=5`, `report_to="none"`), grad clipping 1.0, seed 42. Saves `log_history.json`, `train_config.json`, the trainable-params count, adapters → `outputs/<run_name>/`. A `--max_steps 5 --model <tiny> --cpu` smoke mode.
- [x] **3.5 [COPILOT]** `configs/train_qwen25_0p5b.yaml` (same recipe, draft model).
- [x] **3.6 [COPILOT]** Added the Kaggle training notebook and kernel metadata, private allowlisted code-dataset sync, CLI credential wrapper/status checker, and laptop-side Hub uploader with a provenance card and metric placeholders. The notebook verifies T4 x2, preserves Kaggle's Torch, regenerates the fixed split, trains both models concurrently, and packages adapters/logs. Per `docs/PARALLEL_WORKFLOW.md`, HF credentials stay on the laptop and all Hub uploads happen there.
- [ ] **3.7 [COPILOT]** Sync the private code dataset and push `kaggle/jobs/train` with `--accelerator NvidiaTeslaT4`; monitor without blocking laptop work, then download adapters/logs to `outputs/` and `results/train/`. If the job remains queued for 45 minutes, start the documented local 0.5B fallback. Never put the HF token on Kaggle.
**Accept:** adapters for both models on the HF Hub; training logs in `results/train/`.

## Phase 4 — (P1) Hyper-parameter sweep on val
- [ ] **4.1 [COPILOT]** `configs/sweep/*.yaml` for r∈{8,16} × lr∈{1e-4,2e-4} (+ oversampling on/off if time); `results/train/runs.csv` aggregator.
- [ ] **4.2 [YOU]** Run on Kaggle only if Phase 3 finished early. The choice is made **on val macro-F1 only**.

## Phase 5 — Evaluation pipeline (P0)
- [x] **5.1 [COPILOT]** `eval/metrics.py` (PLAN §5.1, incl. bootstrap CI) and `eval/evidence_metrics.py` (PLAN §5.2, all conventions). Unit tests with hand-computed toy cases, **including the empty-gold and empty-pred combinations**.
- [x] **5.2 [COPILOT]** `inference/predict.py` (full pipeline PLAN §6.2 with fallback scoring), `inference/explain.py`, `inference/aggregate.py` (PLAN §6.4) + tests (`test_aggregate.py` covers each verdict branch and the "absent evidence ≠ proof" rule).
- [x] **5.3 [COPILOT]** Added `scripts/evaluate.py` with fixed-split selection, JSONL predictions (all `CriterionResult` fields + gold), metrics JSON, confusion figures, and a sealed test path that requires `--final`, HEAD tagged `v1.0-frozen`, and no prior test access. Greedy no-cache is the current default; `greedy_kv` is wired to the Phase 6 decoder. Decoding settings are recorded in the metrics.
- [ ] **5.4 [COPILOT]** `notebooks/02_kaggle_eval_bench.ipynb`: on **val**: base 1.5B, fine-tuned 1.5B (each epoch checkpoint → pick best by val macro-F1), base 0.5B, fine-tuned 0.5B, SmolLM2-1.7B zero-shot (P1), base 1.5B 2-shot (P1). Then run the benchmarks from Phases 6–8 (cells call `scripts/bench_*.py`). Finally zip `results/` for download.
- [ ] **5.5 [YOU]** Run `02_kaggle_eval_bench.ipynb` on a T4. Bring `results/` back into the repo and commit.
**Accept:** val metrics for base vs fine-tuned exist, the best checkpoint is chosen and recorded in DECISIONS.md, and it is merged + pushed (`...-1.5b-merged`) for the app.

## Phase 6 — KV-cached decoding (P0) — PLAN §7
- [x] **6.1 [COPILOT]** `decoding/cache_utils.py` (`cache_seq_len`, `cache_bytes`, `crop_cache`, compatible with the installed transformers Cache API) and `decoding/timing.py` (`Timer` context manager with CUDA sync, `measure_peak_memory`, warm-up/repeat helper returning mean/std/median).
- [x] **6.2 [COPILOT]** `decoding/kv_cache.py`: `greedy_no_cache(model, input_ids, max_new, eos_id=None) -> GenResult` and `greedy_kv_cache(...) -> GenResult` (explicit prefill + decode loop, explicit `position_ids`, `cache_position`, `attention_mask`; records per-step logits margin, cache length and bytes; `GenResult` holds tokens, prefill_s, decode_s, per-step stats). Rich docstrings explaining positions, masking, growth and numerics (needed for the defense).
- [x] **6.3 [COPILOT]** `tests/test_kv_cache.py` (tiny Qwen2, CPU fp32): identical tokens for 3 prompt lengths × 3 output lengths; cache length == processed tokens after every step; bytes == formula; **negative test**: wrong positions (always 0) → output differs.
- [ ] **6.4 [COPILOT]** `scripts/bench_kv.py --config configs/bench.yaml` (PLAN §7.5 grid, warm-up 2, repeats 5, fp16 + fp32 equality runs, mismatch investigation dump) → `results/bench/kv_cache.json|csv`; `--device cpu --model 0.5B` mode for local runs.
- [ ] **6.5 [P2, COPILOT]** `decoding/mini_qwen2.py` + `SimpleKVCache` (pre-allocated): a hand-written forward pass using the HF weights; test that logits match HF (`atol 1e-4` fp32) on the tiny model and on real 0.5B (slow test).
- [ ] **6.6 [P2, COPILOT]** prefix-cache reuse for many criteria per patient (PLAN §7.6) + equality test.
**Accept:** tests pass on CPU; the bench script runs on CPU with the 0.5B model in < 5 min (small grid).

## Phase 7 — Tiled attention (P0) — PLAN §8
- [x] **7.1 [COPILOT]** `attention/reference.py::sdpa_reference` and `attention/tiled.py::tiled_attention` exactly as PLAN §8.1–8.2 (GQA, query offset, padding mask, fully-masked rows, fp32 accumulation, block sizes not dividing N). Docstrings derive the online-softmax rescaling.
- [x] **7.2 [COPILOT]** `tests/test_attention.py`: the grid from PLAN §8.3 vs an fp64 reference; also checks the tiled code never calls `scaled_dot_product_attention` (monkeypatch it to raise during the test).
- [ ] **7.3 [COPILOT]** `scripts/bench_attention.py` → `results/bench/attention_accuracy.csv`, `attention_perf.csv` (+ CPU mode). Catch OOM and record it as "OOM", never crash.
- [ ] **7.4 [P1, COPILOT]** `attention/hf_integration.py`: register the tiled attention with transformers (`AttentionInterface`) or monkeypatch; `scripts/bench_attention.py --model-integration` checks the same greedy tokens as `eager` on 5 val prompts.
**Accept:** all attention tests pass on CPU; the accuracy CSV has max/mean error per case.

## Phase 8 — Speculative decoding (P0) — PLAN §9
- [x] **8.1 [COPILOT]** `decoding/speculative.py::speculative_greedy(target, draft, input_ids, max_new, k, eos_id) -> SpecResult` implementing the invariant algorithm in PLAN §9.2 with **our own** draft loop, a single target verify forward, accept/reject/correct, and cache crop for both models. `SpecResult`: tokens, iterations, proposed, accepted, accepted-per-iteration list, target_forwards, draft_forwards, timings, peak memory.
- [x] **8.2 [COPILOT]** `tests/test_speculative.py` (tiny models, CPU fp32): the 4 unit tests in PLAN §9.3; `tests/test_tokenizers_compatible.py` (slow; real tokenizers).
- [ ] **8.3 [COPILOT]** `scripts/bench_speculative.py` (PLAN §9.3 grid) → `results/bench/speculative.json|csv` (+ optional HF `assistant_model` reference row clearly labelled).
**Accept:** unit tests pass; a CPU run with 0.5B as target and 0.5B as draft gives 100% acceptance and equal tokens (sanity check).

## Phase 9 — Final held-out evaluation (P0, ONCE)
- [ ] **9.1 [YOU+COPILOT]** Freeze everything (prompt, checkpoint, fallback, decoding). Commit and tag `v1.0-frozen`. On Kaggle run `evaluate.py --split test --final` for: base 1.5B, fine-tuned 1.5B (headline), base 0.5B, fine-tuned 0.5B. Also run speculative/KV benchmarks on test prompts if desired. Commit `results/eval/test_*` and `TEST_ACCESS_LOG.md`.
- [ ] **9.2 [COPILOT]** `scripts/error_analysis.py` → candidate errors by category (PLAN §5.4); write `docs/ERROR_ANALYSIS.md` with **≥10 distinct analysed errors** (≥8 required), including unsupported-decision, missing-information and contradictory cases. The student must read and correct these diagnoses.
- [ ] **9.3 [COPILOT]** `scripts/make_figures.py` → all report figures into `results/figures/` and `report/figures/` (training curves, val F1 per epoch, confusion matrices, per-class F1 bars base vs FT, evidence P/R/F1, KV latency vs length, tokens/s, memory vs length, cache bytes vs tokens, attention error table/heatmap, attention time & memory vs N, speculative acceptance vs k, speedup vs k, latency breakdown). Also LaTeX tables (`report/tables/*.tex`) generated from the JSON, so no numbers are typed by hand.

## Phase 10 — Application (P0) — PLAN §10
- [ ] **10.1 [COPILOT]** `modeling/load_model.py` (load merged or base+adapter; device/dtype from env; lazy singleton cache; draft loader).
- [ ] **10.2 [COPILOT]** `trials/criteria_parser.py` (+ tests on 3 real-style criteria texts) and `trials/ctgov.py` (API v2 fetch, timeout 10 s, errors → typed exceptions).
- [ ] **10.3 [COPILOT]** `report/export.py` + Jinja template → HTML / JSON / Markdown with provenance and disclaimer (PLAN §10.1).
- [ ] **10.4 [COPILOT]** `app/` Gradio Blocks with the 5 tabs from PLAN §10.1, the validation/error handling from §10.2, a disclaimer banner on top of every tab, and `app/examples/` with 3 **synthetic** patient notes + criteria written for the demo (not dataset test patients). `python -m app` starts it.
- [ ] **10.5 [COPILOT]** `tests/test_app_smoke.py`: import the app, run the screening function with the tiny model on an example, assert the report exports; and invalid inputs return friendly errors.
- [ ] **10.6 [COPILOT]** `space/` deployment files + `scripts/deploy_space.py` (uses `huggingface_hub.HfApi().upload_folder` to `Ashher123/criterion-lens` Space, SDK gradio, hardware cpu-basic).
- [ ] **10.7 [YOU]** Run locally (`.venv\Scripts\python.exe -m app`, which uses the GTX 1060 for the 0.5B model and the CPU for 1.5B), screen 2 examples, export reports, try invalid inputs. Then deploy the Space: `.venv\Scripts\python.exe scripts\deploy_space.py`, which uploads `space/` to huggingface.co/spaces/Ashher123/criterion-lens using the `.env` token. HF builds it and gives a public URL in ~5 min. Check it from an incognito browser. Take screenshots (each tab, results, report, error message) → `report/figures/screens/`.
**Accept:** the Space URL works with no code execution; the exported report contains provenance + disclaimer.

## Phase 11 — IEEE report (P0) — see `docs/REPORT_OUTLINE.md`
- [ ] **11.1 [COPILOT]** `report/main.tex` using `\documentclass[conference]{IEEEtran}`, the sections in `sections/`, `references.bib` from `docs/LITERATURE.md` (verified entries only), and figures/tables from Phase 9.3. Every figure/table is `\ref`'d **and interpreted** in the text. Architecture diagrams in TikZ or a PNG drawn with draw.io (system pipeline, fine-tuning I/O, KV cache flow, tiled attention, speculative loop).
- [ ] **11.2 [COPILOT]** Compile **locally**: MiKTeX is installed (see `docs/LOCAL_ENVIRONMENT.md`): `cd report; pdflatex main; bibtex main; pdflatex main; pdflatex main`. Fix every error and overfull box, then copy the output to `report/CriterionLens_IEEE.pdf`. **[YOU]** read every paragraph, since you must defend it.
- [ ] **11.3 [COPILOT]** AI-use appendix (template in REPORT_OUTLINE.md), filled honestly.

## Phase 12 — Video, README, submission (P0)
- [ ] **12.1 [YOU]** Record a 5–7 min demo following `docs/REPORT_OUTLINE.md` → "Demo video script" (OBS or Xbox Game Bar `Win+Alt+R`). Upload unlisted to YouTube or Google Drive (link visible to anyone with the link).
- [ ] **12.2 [COPILOT]** Final README: badges/links (Space, video, weights, paper PDF), headline results table (generated), quick start (Space / local / reproduce training), the exact reproduction commands per phase, the hardware used, the split manifest link, the benchmark log links, limitations, disclaimer, AI-use note, citation.
- [ ] **12.3 [YOU]** Make the GitHub repo **public** (`gh repo edit --visibility public --accept-visibility-change-consequences`); open it logged out to check. Check that the HF model repos and Space are public.
- [ ] **12.4 [YOU]** Submit on Google Classroom: report PDF, GitHub URL, Space URL, weights links, video link, the split manifest (attach the JSON too), benchmark logs (link to `results/bench/`).

## Defense preparation (H3) — [YOU], 1 h before the demo
Be ready to: explain the split code line by line; explain why the prompt masks loss; derive KV bytes/token for Qwen2.5-1.5B; explain why the decode step needs `position_ids=cur_len`; derive the online-softmax rescale `α = exp(m_old − m_new)`; walk through one speculative iteration with a rejection; change `k` or the block size live; run the app on an unseen note; change the aggregation rule live (it is in one function: `inference/aggregate.py`). Practise running `pytest -q` and `python scripts/bench_attention.py --device cpu --quick` live.

## Definition of Done (check before submitting)
- [ ] Every row of `docs/RUBRIC_CHECKLIST.md` points to an existing file/figure/URL.
- [ ] `pytest -q` is green (fast tests) on a clean clone.
- [ ] No test-set use before `v1.0-frozen` (see `TEST_ACCESS_LOG.md`).
- [ ] All numbers in the README/report come from `results/` files.
- [ ] Space, weights, video and repo are all publicly accessible.

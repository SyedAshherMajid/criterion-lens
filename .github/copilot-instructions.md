# Copilot Agent Instructions — CriterionLens

You are the **execution agent** for this repository. The design is already decided.
Your job is to implement it exactly, step by step, following `docs/EXECUTION_PLAN.md`.

## Read these first (in this order)
1. `docs/EXECUTION_PLAN.md` — the ordered task list. Work through it phase by phase. Tick boxes (`- [x]`) as you finish steps and commit.
2. `docs/PLAN.md` — design, interfaces, prompt format, metrics definitions, algorithms. **This is the source of truth for every technical detail.**
3. `docs/RUBRIC_CHECKLIST.md` — what the grader will look for. Every rubric row must end up with evidence.
4. `docs/DECISIONS.md` — design decision log. Append to it whenever you hit a difficulty or change something.
5. `docs/REPORT_OUTLINE.md` — IEEE paper outline, demo script, AI-use appendix.

## Hard rules (never break these)
1. **Test-set seal.** Never use test-split patients for training, prompt design, hyper-parameter choice, threshold choice or "quick checks". All tuning uses the validation split. Test evaluation happens only in Phase 9 through `scripts/evaluate.py --split test --final`, and every test run is appended to `results/TEST_ACCESS_LOG.md`.
2. **No leakage of GPT-4 fields.** Columns `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, `explanation_correctness` must never appear in any prompt, training target or feature. The dataset's own `training` column is ignored; the split comes only from `scripts/make_split.py`. A unit test enforces this.
3. **Split recipe is fixed:** `ids = np.array(sorted(df["patient_id"].unique())); rng = np.random.default_rng(42); rng.shuffle(ids)`; first 37 → train, next 8 → val, last 8 → test. Do not change it.
4. **Never collapse labels.** Keep all labels: inclusion = {included, not included, not enough information, not applicable}; exclusion = {excluded, not excluded, not enough information, not applicable}.
5. **No paid APIs, no proprietary models.** Only open-weight models from the Hugging Face Hub.
6. **Do not fake results.** Never write numbers into README/report that were not produced by a script in this repo. If a number is pending a GPU run, write `TBD (run Phase X)`.
7. **Do not just call library switches** for the graded mechanisms: KV cache (Task 3), tiled attention (Task 4) and speculative decoding (Task 5) must be implemented by hand as described in `docs/PLAN.md`. Library versions (`use_cache=True` generate, `assistant_model=`, `F.scaled_dot_product_attention`) are allowed **only as reference baselines** for comparison.
8. **Every citation must be real.** Before adding a reference to `report/references.bib`, confirm its title/authors/venue/year. Never invent a citation.
9. Keep the medical disclaimer everywhere the model output is shown: *"Educational research screening prototype — not a clinical enrollment decision."*

## Coding conventions
- Python 3.11 (Kaggle and local). Package code lives in `src/criterion_lens/`; runnable entry points live in `scripts/`; the app lives in `app/`.
- Every script takes `--config configs/<file>.yaml` and/or CLI flags, sets seeds, and writes outputs (JSON/CSV) under `results/`. Never print-only.
- Type hints and docstrings on public functions. Docstrings explain *why*, since the student must defend the code.
- Tests in `tests/` with `pytest`. Tests that need a GPU or a large download are marked `@pytest.mark.slow`. Fast tests use a **tiny randomly initialised Qwen2 model built from a config** (no download), see `tests/conftest.py` in PLAN §10.
- Use `pathlib`, `logging` (not bare prints in library code), and `json` with `indent=2`.
- Commit after every completed step with a message like `feat(data): patient-grouped split + manifest (Phase 2.3)`. End commit messages with the attribution line the user asks for, if any.
- Never commit: model weights, `data/raw/`, `.env`, HF tokens. Large files go to the Hugging Face Hub.

## Where you cannot act
Some steps must be done by the human (marked **[YOU]** in the plan): running Kaggle GPU notebooks, creating the HF token, recording the video, compiling LaTeX on Overleaf. For those steps, prepare everything (notebook, commands, instructions) and stop with a clear message saying what the human needs to run and which files to bring back.

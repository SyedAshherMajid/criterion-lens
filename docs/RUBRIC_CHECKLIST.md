# Rubric → Evidence Checklist (135 marks)

Fill the **Evidence** column with real paths/URLs as work completes. A row is done only when the evidence exists **and** is described in the report.

| Code | Marks | What earns full credit | Evidence we will produce | Status |
|---|---|---|---|---|
| A1 Literature | 5 | ≥5 credible sources, alternatives + limitations, linked to problem | `docs/LITERATURE.md`, report §II Related Work (12+ refs, each with a limitation and the design decision it informs) | ☐ |
| A2 Model choice | 5 | architecture, params, tokenizer, context, licence, compute; justified | `results/model_specs.json` (from `scripts/inspect_models.py`), `docs/MODEL_SELECTION.md`, report Table "Candidate models", §III-C compute/memory | ☐ |
| A3 Research-informed decisions | 5 | alternatives, rejected options, difficulties, evidence | `docs/DECISIONS.md` (ADR log), val ablations (label-first vs evidence-first, r/lr, 0.5B vs 1.5B, SmolLM2 zero-shot), report "Design decisions" table | ☐ |
| B1 Dataset & split | 5 | only TrialGPT data; 37/8/8 seed 42; IDs; manifest; leakage checks | `data/splits/split_manifest.json/.csv`, `scripts/make_split.py`, `tests/test_split.py` | ☐ |
| B2 I/O processing | 5 | patient text, type, criterion, labels, sentence IDs; GPT-4 excluded | `data/load.py`, `data/prompts.py`, `data/tokenize.py`, `tests/test_no_leakage.py`, `results/data_audit.json`, truncation log | ☐ |
| B3 Fine-tuning | 5 | executable FT, adapters, config, curves/logs, validated checkpoint | `scripts/train.py`, `configs/train_*.yaml`, `results/train/*/log_history.json`, training-curve figure, HF adapter links, best-epoch selection in DECISIONS | ☐ |
| B4 Before/after | 5 | identical val/test examples before/after; decoding documented | `results/eval/{val,test}_{base,ft}_*/metrics.json`, report comparison table + discussion | ☐ |
| C1 Label correctness | 5 | inclusion/exclusion label sets, NEI, N/A; held-out accuracy + macro-F1 | `results/eval/test_ft_1p5b/metrics.json` (with bootstrap CI) | ☐ |
| C2 Detailed metrics | 5 | per-class P/R/F1, confusion matrices, inclusion vs exclusion | `confusion_{overall,inclusion,exclusion}.png`, per-class table | ☐ |
| C3 Evidence localisation | 5 | original sentence IDs; micro/macro P/R/F1; empty-set handling | `eval/evidence_metrics.py` + tests, metrics.json evidence block, report paragraph on conventions | ☐ |
| C4 Explanations & failures | 5 | every prediction linked to evidence; ≥8 distinct errors incl. unsupported/missing-info | `predictions.jsonl` (evidence text + explanation), `docs/ERROR_ANALYSIS.md` (≥10 cases), report error table | ☐ |
| D1 Cache-aware decoder | 5 | explicit forward passes with cache reuse/update, masking, positions | `decoding/kv_cache.py` (+ P2 `mini_qwen2.py` + `SimpleKVCache`) | ☐ |
| D2 Correctness tests | 5 | greedy equality ≥3 input/output lengths; mismatch investigation | `tests/test_kv_cache.py`, `results/bench/kv_cache.json` equality fields, fp16 vs fp32 mismatch analysis | ☐ |
| D3 Benchmark quality | 5 | prefill, decode, tok/s, peak memory; warm-up; synchronized repeats | `scripts/bench_kv.py`, `results/bench/kv_cache.csv`, figures | ☐ |
| E1 Reference attention | 5 | correct scaling, masking, stable softmax | `attention/reference.py` + tests | ☐ |
| E2 Tiled online-softmax | 5 | blocked attention, running max/normaliser, no fused backend | `attention/tiled.py`, test that monkeypatches SDPA to raise | ☐ |
| E3 Validation | 5 | multiple shapes + causal; max/mean error; honest runtime/memory | `results/bench/attention_accuracy.csv`, `attention_perf.csv`, report discussion of why slower than fused | ☐ |
| F1 Draft/target design | 5 | compatible tokenizer pair; memory + generation settings justified | `tests/test_tokenizers_compatible.py`, report §Speculative (memory arithmetic, k choice) | ☐ |
| F2 Working algorithm | 5 | drafting, verify, accepted prefix, rejection correction, cache sync | `decoding/speculative.py` + `tests/test_speculative.py` | ☐ |
| F3 Correctness & efficiency | 5 | token equality; acceptance; latency; throughput; speedup/slowdown | `results/bench/speculative.csv`, acceptance-vs-k and speedup figures | ☐ |
| G1 Independent app | 5 | no IDE/notebook; documented startup; model loading; validation; errors | Assignment-permitted locally hosted Gradio app; `run_app.bat`, `python -m app`, README startup/troubleshooting, screenshots of error handling. HF Space is optional and blocked by HF's 2026 PRO requirement. | ☐ |
| G2 Screening workflow | 5 | unseen note + criteria; each decision, evidence, missing info, overall outcome | Screen tab; video segment | ☐ |
| G3 Inference visibility | 5 | live inference modes + comparison (not static charts) | Inference Lab + Attention Lab tabs (live runs) | ☐ |
| G4 Usability & export | 5 | readable UI; downloadable report with provenance; recovery from invalid input | HTML/JSON/MD export; invalid-input demo in video | ☐ |
| H1 IEEE paper | 5 | full methods, metrics, graphs/tables, discussion, limitations, citations | `report/CriterionLens_IEEE.pdf` | ☐ |
| H2 Reproducible source + demo | 5 | repo, setup, model link, split, logs, 5–7 min video | README, requirements-lock, HF links, video link | ☐ |
| H3 Defense | 5 | explain + modify components live | EXECUTION_PLAN "Defense preparation" | ☐ |

**Grading traps to avoid:** results copied from TrialGPT (never); metrics not on the fixed test split; claiming cache/attention without validation tests; a notebook-only app; a missing demo.

# CriterionLens — Master Design Plan

> Evidence-grounded clinical-trial criterion matching with a fine-tuned open-weight decoder,
> hand-built KV-cached decoding, tiled online-softmax attention, greedy speculative decoding,
> and a browser application.
>
> GenAI Assignment #2. **Deadline: 10 Nov 2026** (2-hour grace period). Individual work.

This document is the **source of truth** for every design detail. `EXECUTION_PLAN.md` says *in what order* to build things; this file says *what* to build and *how*.

---

## 0. Requirements extracted from the assignment (nothing omitted)

| # | Requirement | Where handled |
|---|---|---|
| R1 | Use only `ncbi/TrialGPT-Criterion-Annotations` for supervised training/comparison | §2 |
| R2 | Never use GPT-4 predictions/explanations as input or in evaluation | §2.3, test `test_no_leakage.py` |
| R3 | Split: sort patient IDs, shuffle with `np.random.default_rng(42)`, 37 train / 8 val / 8 test, grouped by patient | §2.2 |
| R4 | Publish split manifest (IDs + counts); include leakage checks | §2.2 |
| R5 | Test set untouched until final evaluation; no tuning on test | §2.4, `TEST_ACCESS_LOG.md` |
| R6 | Patient–trial verdict = explicit aggregation rule, not ground truth | §6.4 |
| T1a | ≥5 credible sources: eligibility reasoning, Transformers, supervised adaptation, efficient inference | §11, report |
| T1b | ≥2 open-weight decoder candidates; justify with params, license, memory, architecture, hardware fit | §3 |
| T1c | Fine-tune: input = patient note + criterion type + criterion text → structured label + sentence IDs | §4 |
| T1d | Report preprocessing, tokenization, output schema, objective, config, hyper-parameters, val performance, held-out results | §4, report |
| T1e | Same model **before vs after** fine-tuning on identical examples & decoding settings | §5.3 |
| T1f | Document truncation, especially if evidence could be cut | §4.4 |
| T2a | Inclusion + exclusion; keep NEI and N/A labels | §2.1, §6 |
| T2b | Output per prediction: criterion, category, cited IDs, evidence text, explanation | §6.1 |
| T2c | Accuracy, macro-F1, per-class P/R/F1, confusion matrices, overall + inclusion vs exclusion | §5.1 |
| T2d | Evidence P/R/F1 (micro + macro); state how empty-gold records are treated | §5.2 |
| T2e | Representative errors, contradictory cases, missing-information failures (≥8 distinct errors) | §5.4 |
| T2f | Never treat absent evidence as proof of eligibility/ineligibility | §6.3 |
| T3 | Manual KV-cache decoding vs no-cache reference; token agreement; prefill/decode time, tok/s, peak memory over several input & output lengths; explain positions, masking, cache growth, numerics | §7 |
| T4 | Reference SDPA + tiled online-softmax attention; causal tests; max/mean error; speed/memory; explain why slower than fused | §8 |
| T5 | Draft model with compatible tokenizer; greedy speculative decoding (draft, verify, accept, reject-correct, cache sync); equality vs greedy; acceptance, latency, tok/s, memory | §9 |
| T6 | Browser app: patient record in, criteria select/enter, run, per-criterion results + exact evidence, export report, disclaimer, decoding benchmark view, no code execution needed by evaluator, documented startup | §10 |
| S1 | IEEE LaTeX paper with all listed sections, curves, tables, runtimes, error analysis, screenshots; every figure/table referenced and interpreted | `REPORT_OUTLINE.md` |
| S2 | Submit: PDF, GitHub URL, reproduction instructions, weights/adapter link, split manifest, benchmark logs, 5–7 min video link, AI-use appendix | `EXECUTION_PLAN.md` Phase 12 |
| S3 | Explain compute resources and memory limits | §3.3 |
| S4 | Acknowledge AI assistants; test reused code; be able to defend/modify any component | `REPORT_OUTLINE.md` §AI-use, `DEFENSE_PREP` in EXECUTION_PLAN |

---

## 1. System overview

```
                 ┌─────────────────────────── offline (Kaggle T4 GPU) ────────────────────────────┐
HF dataset ──► data audit ──► patient-grouped split (seed 42) ──► prompt builder ──► LoRA SFT (Qwen2.5-1.5B-Instruct)
                                         │                                     └──► LoRA SFT (Qwen2.5-0.5B-Instruct = draft)
                                         └──► evaluate (base vs fine-tuned, val → test once) ──► metrics, error analysis
                 └─────────────────────────────────────────────────────────────────────────────────┘
                                                      │ adapters / merged weights pushed to HF Hub
                                                      ▼
Browser (Gradio app, HF Space CPU or local) ──► sentence splitter ──► per-criterion prompt ──►
     decoding engine [no-cache | manual KV-cache | speculative(draft→verify)] ──► JSON parse + validate
     ──► fallback constrained label scoring ──► evidence text + grounded explanation ──► aggregation rule
     ──► results table, highlighted note, HTML/JSON/Markdown report export
     + Inference Lab (live decoding comparison) + Attention Lab (live reference vs tiled attention)
```

**Project name:** CriterionLens. **Repo:** `criterion-lens`. **Package:** `criterion_lens`.

---

## 2. Data

### 2.1 Dataset facts (verify in Phase 2 audit — numbers below are expected, not guaranteed)
- HF id `ncbi/TrialGPT-Criterion-Annotations`, single split `train`, ~1,015–1,020 rows, Parquet, license shown as public-domain (read the repo `LICENSE` and quote it in the report).
- Columns: `annotation_id` (int), `patient_id` (str, 53 unique, e.g. `sigir-20141`), `note` (str), `trial_id` (NCT id, ~103 unique), `trial_title`, `criterion_type` (`inclusion` / `exclusion`), `criterion_text` (nullable!), `gpt4_explanation`, `explanation_correctness`, `gpt4_sentences`, `expert_sentences` (string like `"[0, 1, 2]"` or `"[]"`), `gpt4_eligibility`, `expert_eligibility`, `training` (bool).
- `note` is already sentence-numbered inline: `"0. A 58-year-old ... 1. The pain started ..."`.
- Label values: inclusion → `included`, `not included`, `not enough information`, `not applicable`; exclusion → `excluded`, `not excluded`, `not enough information`, `not applicable`.

**Columns we use:** `annotation_id`, `patient_id`, `note`, `trial_id`, `trial_title` (metadata only, not in prompt), `criterion_type`, `criterion_text`, `expert_eligibility`, `expert_sentences`.
**Columns we drop at load time:** `gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, `explanation_correctness`, `training`. The loader deletes them immediately so nothing downstream can touch them. (Exception: a separate, clearly-labelled script may compute GPT-4's agreement with experts on the *test split* purely as a published reference point in the report's Related Work discussion — optional, never used for training/tuning.)

### 2.2 Split (exact recipe — do not change)
```python
ids = np.array(sorted(df["patient_id"].unique()))   # lexicographic string sort
rng = np.random.default_rng(42)
rng.shuffle(ids)
train_ids, val_ids, test_ids = ids[:37], ids[37:45], ids[45:53]
```
- Assert `len(ids) == 53`. Assert the three sets are disjoint and union = all IDs.
- Write `data/splits/split_manifest.json`:
  `{"seed": 42, "method": "...recipe text...", "dataset_revision": "<HF commit sha>", "train": {"patient_ids": [...], "n_patients": 37, "n_annotations": N, "n_inclusion": .., "n_exclusion": .., "label_counts": {...}}, "val": {...}, "test": {...}}` plus `split_manifest.csv` (`patient_id,split,n_annotations`).
- Record the dataset commit SHA (`huggingface_hub.HfApi().dataset_info(...).sha`) and load with `revision=<sha>` for reproducibility.
- Leakage checks (`tests/test_split.py`): disjoint patient sets; every annotation of a patient in one split; determinism (run twice → identical); `note` texts do not cross splits; manifest matches recomputation.

### 2.3 Audit (Phase 2 — results go in `results/data_audit.json` and the report)
- Row count; null/empty `criterion_text` (drop them, report count); label distribution overall / by type / by split; labels inconsistent with criterion type (e.g. an inclusion row labelled `excluded`) → count, list, and decide in `DECISIONS.md` (default: keep in eval as-is, exclude from training); `expert_sentences` IDs out of range for the note → count; empty-evidence rate per label; note length in sentences, words and **tokens** (with the chosen tokenizer); criterion length in tokens; number of criteria per patient/trial.

### 2.4 Test-set seal
`scripts/evaluate.py` refuses `--split test` unless `--final` is also given, and appends `{timestamp, git commit, model, adapter, decoding config}` to `results/TEST_ACCESS_LOG.md`. All hyper-parameter, prompt and fallback choices are made on val first.

---

## 3. Model selection

### 3.1 Candidates (report table; verify every number from the model card / `config.json` and cite it)
| Model | Params | Arch | Context | Tokenizer / vocab | License | fp16 weights | Draft available (same tokenizer) | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 1.54B (1.31B non-emb) | decoder-only, RoPE, SwiGLU, RMSNorm, GQA 12Q/2KV heads, 28 layers, head_dim 128, tied embeddings | 32,768 | byte-level BPE, ~151.6k tokens | Apache-2.0 | ~3.1 GB | **Qwen2.5-0.5B-Instruct** (0.49B, 24 layers, 14Q/2KV, head_dim 64) | **Selected target** |
| SmolLM2-1.7B-Instruct | 1.7B | Llama-style, MHA 32 heads, 24 layers | 8,192 | 49,152 | Apache-2.0 | ~3.4 GB | SmolLM2-360M / 135M | Runner-up (evaluated zero-shot on val) |
| Llama-3.2-1B-Instruct | 1.24B | Llama, GQA, 16 layers | 128k | 128,256 | Llama 3.2 Community (gated) | ~2.5 GB | none smaller in family (3B would be target) | Rejected: gated licence, no smaller draft for 1B |
| Phi-3.5-mini-instruct | 3.8B | dense, MHA | 128k | 32,064 | MIT | ~7.6 GB | none same-tokenizer small | Rejected: too large for free CPU demo & T4 full-precision |
| Gemma-2-2B-it | 2.6B | GQA, local/global attention, soft-capping | 8,192 | 256k | Gemma terms (gated) | ~5.2 GB | none | Rejected: licence + huge vocab cost, soft-capping complicates custom attention |
| Qwen3-1.7B | 1.7B | Qwen3 (QK-norm) | 32k | same family as 0.6B | Apache-2.0 | ~3.4 GB | Qwen3-0.6B | Considered; rejected for the thinking-mode chat template and to keep transformers API stable — mention as future work |

**Justification for Qwen2.5-1.5B-Instruct + Qwen2.5-0.5B-Instruct:**
- Permissive Apache-2.0 licence, ungated download (no friction for graders).
- Identical tokenizer across the family → valid draft/target pair for speculative decoding (verify: same `tokenizer.json` hash and same vocab; note `lm_head` rows may be padded beyond tokenizer size — slice logits to `len(tokenizer)`).
- Small enough for: LoRA training in fp16/fp32 on a free Kaggle T4 (16 GB); CPU inference in a free HF Space (16 GB RAM); the KV-cache and speculative experiments on one GPU.
- GQA (2 KV heads) makes KV cache tiny → good teaching example of cache-growth arithmetic.
- 32k context ≫ longest note+criterion (~1k tokens expected) → **no truncation expected** (must be verified in audit).

### 3.2 Expected memory (to be confirmed by measurement)
- KV cache per token = `2 × layers × kv_heads × head_dim × bytes`.
  - 1.5B fp16: 2×28×2×128×2 B = **28,672 B ≈ 28 KB/token** → 2,048 tokens ≈ 56 MB.
  - 0.5B fp16: 2×24×2×64×2 B = **12,288 B ≈ 12 KB/token**.
- Weights: 1.5B fp16 ≈ 3.1 GB, fp32 ≈ 6.2 GB; 0.5B fp16 ≈ 1.0 GB.
- LoRA training (r=16, all linear layers, grad-checkpointing, seq ≤1,536, micro-batch 4): expected 7–10 GB on T4.

### 3.3 Compute resources (state in report)
- **Local laptop:** NVIDIA GTX 1060 3 GB (Pascal, sm_61). The existing torch 2.6.0+cu124 **does** support it (verified). It runs 0.5B fp16, attention benchmarks, tiny-model tests and app inference with the 0.5B model. The 1.5B fp16 (3.1 GB) does not fit, so it runs on CPU locally. This is a second hardware point for the benchmarks. Environment details: `docs/LOCAL_ENVIRONMENT.md`.
- **Kaggle (free):** use **GPU T4 ×2** (Turing sm_75, 16 GB each, fp16, no bf16); ~30 GPU-h/week, 12 h per session. Do **not** use the P100 option (Pascal again). All training, evaluation and GPU benchmarks run here.
- **Hugging Face Hub (free):** model/adapter hosting; **HF Space CPU-basic (free, 2 vCPU, 16 GB RAM)** for the deployed app.
- No paid resources anywhere.

---

## 4. Fine-tuning (Task 1)

### 4.1 Prompt / output schema
Use the model's chat template (`tokenizer.apply_chat_template`). The **patient note comes first** so the note prefix can be KV-cached and reused across criteria (stretch goal §7.6).

System message (constant, `src/criterion_lens/data/prompts.py::SYSTEM_PROMPT`):
```
You are a clinical-trial screening assistant. Judge ONE eligibility criterion against a patient note.
Use only facts stated in the note. Cite the IDs of the note sentences that support your judgment.
If the note does not contain the needed information, answer "not enough information".
If the criterion cannot apply to this patient (e.g., wrong sex, pregnancy criterion for a male), answer "not applicable".
Respond with one JSON object only.
```
User message:
```
Patient note (sentences are numbered):
0. A 58-year-old African-American woman presents ...
1. The pain started while ...
...

Criterion type: inclusion
Criterion: Patients of chronic stable angina with abnormal Exercise Myocardial Perfusion ...
Allowed labels: "included" | "not included" | "not enough information" | "not applicable"

Return: {"evidence": [<sentence ids>], "label": "<one allowed label>"}
```
Assistant target (exact formatting, compact, sorted unique IDs):
```
{"evidence": [0, 1, 2], "label": "not enough information"}
```
- **Evidence before label** (a light "cite, then decide" ordering). Ablation on val: label-first vs evidence-first (P1).
- Sentence parsing: split `note` with regex `(?:^|\s)(\d+)\.\s` into `[(id, text)]`; IDs must be 0..n-1 contiguous; assert and log failures. Re-render each sentence on its own line `"{id}. {text}"`.
- **Explanation is NOT a training target.** The only explanations in the dataset are GPT-4's, and using them as targets would leak GPT-4 outputs into the model. Explanations are produced at inference by a grounded template (§6.2). This is a key decision for `DECISIONS.md`.

### 4.2 Training objective
Causal-LM cross-entropy on **assistant-response tokens only** (prompt tokens get label `-100`), i.e. supervised fine-tuning = maximum likelihood of the expert JSON given the prompt. Implement tokenization manually (build prompt ids with `apply_chat_template(..., add_generation_prompt=True)`, append target ids + `<|im_end|>`), use HF `Trainer` + `DataCollatorForSeq2Seq(pad_to_multiple_of=8)`. (Avoids TRL API churn; TRL `SFTTrainer` is an acceptable alternative if completion-only loss is verified by a unit test.)

### 4.3 Method & hyper-parameters (`configs/train_qwen25_1p5b.yaml`)
- **LoRA** (PEFT): r=16, alpha=32, dropout=0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, bias none. Report trainable params (~18M, ~1.2%).
- Base weights fp16 on T4 with fp32 LoRA params + AMP fp16. **If loss becomes NaN/inf** (Qwen2 fp16 overflow is a known risk): load base in fp32 + gradient checkpointing (fits in 16 GB) and log this in `DECISIONS.md`. QLoRA 4-bit is the fallback if memory fails.
- lr 2e-4, cosine, warmup 3%, weight-decay 0, epochs 3, micro-batch 4, grad-accum 4 (effective 16), max_seq_len 1,536 (adjust from audit), grad-checkpointing on, seed 42, `group_by_length=True`.
- Save adapter every epoch and log training/validation loss. The completed Kaggle run did not record generation-based macro-F1 per epoch; do not infer a best epoch from loss or rerun the four recovered validation evaluations. Select the headline model only from the completed final-adapter comparison on the fixed validation split (D-006, D-027).
- Small sweep on val only (P1, if time): r ∈ {8,16}, lr ∈ {1e-4, 2e-4}; class-balanced oversampling of rare labels on/off. Record every run in `results/train/runs.csv`.
- Logs: `trainer_state.json` `log_history` → `results/train/<run>/log_history.json`; plot the recorded train loss vs step and validation loss vs epoch. Compare final-adapter validation macro-F1 across models separately; do not fabricate epoch-level F1.
- Same recipe for the **draft** `Qwen2.5-0.5B-Instruct` (`configs/train_qwen25_0p5b.yaml`) — gives (a) a fine-tuned draft with higher acceptance and (b) a second model-size comparison point.

### 4.4 Truncation policy
- Compute token lengths of all prompts+targets with the real tokenizer; report max / p95 / mean per split.
- `max_seq_len` set ≥ max observed length → expected **zero truncation**. If any example still exceeds it: drop **whole sentences from the end of the note** that are not gold evidence (training only), log `{annotation_id, removed_sentence_ids, removed_gold_evidence: bool}` to `results/truncation_log.json`, and never silently cut the target. At inference (app) a note exceeding the budget triggers a visible warning listing which sentence IDs were not seen.

### 4.5 Artifacts
- Adapters: HF Hub `Ashher123/criterion-lens-qwen2.5-1.5b-lora` and `...-0.5b-lora` (README model card with training config, split, metrics).
- Merged fp16 weights (for app speed, optional): `Ashher123/criterion-lens-qwen2.5-1.5b-merged`.
- `scripts/download_weights.py` → `snapshot_download` into `models/`. README documents it.

---

## 5. Evaluation (Tasks 1 & 2)

### 5.1 Classification metrics (`src/criterion_lens/eval/metrics.py`)
- Overall (6-label space), inclusion-only (4 labels), exclusion-only (4 labels).
- Accuracy, macro-F1 (over labels present in gold ∪ pred, also report fixed 4-label macro), per-class precision/recall/F1/support (`sklearn.metrics.classification_report`, `zero_division=0`), confusion matrices (counts + row-normalised PNGs).
- Supplementary unified view: map to {criterion met (`included`/`excluded`), not met (`not included`/`not excluded`), NEI, N/A} — clearly labelled supplementary.
- Parse-failure rate, fallback-usage rate, invalid-label rate.
- Bootstrap 95% CI (1,000 resamples over criteria, seed 42) for accuracy and macro-F1 on test — small test set, so CIs matter.
- Baselines for context: majority class (from train), and base model zero-shot.

### 5.2 Evidence metrics (`eval/evidence_metrics.py`) — conventions stated explicitly in report
For each record: gold set G, predicted set P (invalid IDs removed and counted separately).
- **Micro P/R/F1**: sum TP=|G∩P|, FP=|P\G|, FN=|G\P| over all records. Records with G=∅ and P=∅ contribute nothing; G=∅, P≠∅ contributes FPs.
- **Macro P/R/F1 (primary macro)**: average per-record scores over records with **G≠∅**. If P=∅ there: P=0, R=0, F1=0.
- **Empty-gold handling**: for records with G=∅ report *abstention accuracy* = fraction with P=∅, and mean #false citations.
- **Secondary macro (all records)**: convention G=∅∧P=∅ → P=R=F1=1; G=∅∧P≠∅ → 0. Reported for transparency.
- Report overall and by criterion type and by gold label (evidence for NEI is usually empty).
- Hallucinated-ID rate (IDs not in note).

### 5.3 Before/after comparison (B4)
Same prompts, same examples (full val; full test at final), same decoding: greedy, `max_new_tokens=64`, no sampling, same parser and fallback. Rows: base 1.5B zero-shot, base 1.5B 2-shot (P1, shots from **train**), fine-tuned 1.5B, base 0.5B, fine-tuned 0.5B, SmolLM2-1.7B zero-shot (val only, model-choice evidence). Discuss differences: format compliance, NEI over-use, polarity confusion.

### 5.4 Error analysis (C4) — `scripts/error_analysis.py` → `results/eval/error_cases.json` + `docs/ERROR_ANALYSIS.md`
Select **≥ 10 distinct** test (or val, labelled) errors covering these categories, each with note excerpt, criterion, gold vs pred, gold vs cited evidence, and a 2–3 sentence diagnosis:
1. Inclusion/exclusion polarity confusion (`included` vs `not excluded`)
2. Predicts NEI when evidence exists (missed evidence)
3. Asserts a decision with **no** cited evidence (unsupported decision)
4. Hallucinated / irrelevant sentence IDs
5. Negation errors ("denies chest pain")
6. Numeric / threshold reasoning (age, lab values, durations)
7. Temporal reasoning (history vs current)
8. `not applicable` confusion (sex/pregnancy-specific)
9. Contradictory case: label contradicts the model's own cited evidence
10. Gold-annotation disagreement/ambiguity (case where expert label is debatable)
Also: confusion between NEI and N/A; a missing-information failure where the model guessed instead of abstaining.

---

## 6. Inference pipeline (Task 2) — `src/criterion_lens/inference/`

### 6.1 Output record (`schema.py`)
```python
@dataclass
class CriterionResult:
    criterion_id: str            # e.g. "inc-3"
    criterion_type: Literal["inclusion","exclusion"]
    criterion_text: str
    label: str                   # one of the allowed labels for the type
    evidence_ids: list[int]      # valid, sorted, unique
    evidence_text: list[str]     # exact sentence strings for those IDs
    explanation: str             # grounded template (§6.2)
    missing_information: bool    # label == NEI or evidence_ids empty
    unsupported_decision: bool   # decisive label (met/not met) with no evidence
    invalid_ids_dropped: list[int]
    raw_output: str
    parse_ok: bool
    invalid_label: bool          # an explicit label was present but outside the allowed schema
    fallback_used: bool
    decoding: dict               # mode, max_new_tokens, k (spec), timing stats
```

### 6.2 Steps (`predict.py::predict_criterion`)
1. Build prompt (§4.1). Notes from users are sentence-split by `sentences.py::split_sentences` (use the numbering if already present; else `pysbd` or a regex splitter; IDs from 0).
2. Generate with the selected decoding engine (`no_cache` | `kv_cache` | `speculative`), greedy, `max_new_tokens=64`, stop at `<|im_end|>`.
3. Parse: `json.loads` on the first `{...}`; regex fallback for `"label"` and `"evidence"`.
4. Validate: label ∈ allowed set for criterion type; IDs ∈ note range (drop + record invalid).
5. **Fallback** if parse/validation fails: constrained label scoring — for each allowed label `L`, compute sum log-prob of the continuation `{"evidence": [<parsed or []>], "label": "L"}` given the prompt (one batched forward); pick argmax; set `fallback_used=True`.
6. Evidence text = exact sentences for IDs.
7. Explanation (grounded template; never invents facts):
   - decisive + evidence: `"Judged '{label}' for this {type} criterion based on sentence(s) {ids}: “{s1}” … The decision relies only on these statements."`
   - NEI: `"The note does not state the information needed to judge this criterion{; closest related sentence(s): ids}. Absence of information is not treated as proof of eligibility or ineligibility."`
   - N/A: `"The criterion does not apply to this patient given sentence(s) {ids}."`
   - unsupported (decisive, no evidence): adds `"⚠ No supporting sentence was cited; treat this decision as unverified."`
   - Optional (P2) toggle "LLM rationale": base model (adapter disabled via `with model.disable_adapter():`) writes a 1–2 sentence rationale restricted to the cited sentences; clearly labelled as generated.

### 6.3 Absent evidence rule
Absent evidence → `missing_information=True`. A decisive label with no evidence is flagged `unsupported_decision` and the aggregator treats it as **needs review**, never as a pass/fail.

### 6.4 Patient–trial aggregation (explicit rule; not ground truth — say so in UI and report)
- Ignore `not applicable` criteria.
- **Likely ineligible**: any inclusion `not included` or any exclusion `excluded` with cited evidence.
- **Potentially eligible (preliminary)**: all remaining inclusion criteria `included`, no exclusion `excluded`, no NEI, no unsupported decisions.
- **Needs manual review**: otherwise (any NEI or unsupported decision, with no evidence-backed failure).
- Output also counts: met / not met / NEI / N/A / unsupported, and the list of missing-information criteria.

---

## 7. KV-cached autoregressive generation (Task 3) — `src/criterion_lens/decoding/kv_cache.py`

### 7.1 Reference path (no cache reuse)
```
ids = prompt_ids
for t in range(max_new):
    logits = model(input_ids=ids, attention_mask=ones, use_cache=False).logits[:, -1]
    nxt = argmax(logits); ids = cat(ids, nxt); stop on EOS (unless forced length)
```
Cost per step O(L) full forward → total O(L·T) token-forwards, quadratic attention per step.

### 7.2 Cache-aware path (explicit, hand-managed)
```
cache = DynamicCache()
# PREFILL
out = model(input_ids=prompt, attention_mask=ones(1,L), position_ids=arange(L),
            cache_position=arange(L), past_key_values=cache, use_cache=True)
nxt = argmax(out.logits[:, -1]); cur_len = L
# DECODE
for t in range(1, max_new):
    out = model(input_ids=nxt[:,None], attention_mask=ones(1,cur_len+1),
                position_ids=[[cur_len]], cache_position=[cur_len],
                past_key_values=cache, use_cache=True)
    cur_len += 1; nxt = argmax(out.logits[:, -1])
```
- We call the model's `forward` ourselves, own the loop, the positions, the mask and the cache object; we never call `generate()`.
- Instrumentation: after every step record per-layer `cache.layers[i].keys.shape` (or `key_cache[i]` depending on transformers version — write a tiny accessor `cache_seq_len(cache)` and `cache_bytes(cache)` that handles both) → verify `seq_len == cur_len` and bytes match §3.2 formula. Plot cache bytes vs tokens.
- **Stretch P2 (strongly recommended if time): `SimpleKVCache`** — our own cache class with **pre-allocated** tensors `[layers, 2, B, kv_heads, max_len, head_dim]` and a write pointer, plugged into a **hand-written Qwen2 decoder forward** (`decoding/mini_qwen2.py`) that loads the HF weights (embeddings, RMSNorm, RoPE, GQA attention calling *our* attention functions, SwiGLU MLP, lm_head). Verify logits vs HF within tolerance. This fully demonstrates cache-aware attention, and lets the tiled attention (Task 4) run inside the real model.

### 7.3 Explanations required in the report
- **Positions**: RoPE rotates Q/K by absolute position; the new token must use position `cur_len`, not 0. Keys in cache are stored *already rotated*, so they are never re-rotated. Demonstrate the bug: feeding position 0 for every decode token → tokens diverge (add as a negative test).
- **Masking**: prefill uses a causal (lower-triangular) mask; a single decode query may attend to all cached keys, so no causal mask is needed beyond padding; with left-padded batches the 2D attention mask must keep the pad columns masked. Multi-token verification (spec decoding) needs a causal mask *within* the new block plus full visibility of the cache.
- **Cache growth**: linear in tokens; formula §3.2; measured curve. DynamicCache grows by `torch.cat` (reallocation each step) vs pre-allocated static cache.
- **Numerical differences**: cached and uncached paths use different matmul shapes (1×L vs L×L), different kernel selection/reduction order → fp16 logits differ by ~1e-3; argmax can flip when the top-2 logit margin is tiny. Report: max |Δlogit| per step, first divergence step, the top-2 margin at that step; re-run in fp32 to show agreement.

### 7.4 Correctness tests (D2)
- Greedy tokens identical between paths for **≥ 3 input lengths × ≥ 3 output lengths**: input ∈ {~128, ~512, ~1024, ~2048 tokens} (real notes, concatenated/padded with real notes for the long ones), output ∈ {32, 128, 256} (EOS ignored to force length). Run in fp16 and fp32. Any mismatch → logged with step index, margin and explanation.
- Unit tests with tiny random Qwen2 (CPU, fp32): exact token equality, cache length == sequence length, wrong-position negative test.

### 7.5 Benchmark (D3) — `scripts/bench_kv.py` → `results/bench/kv_cache.json` + `.csv`
Same model (fine-tuned 1.5B merged, fp16, T4) and same inputs for both paths. For every (input_len, output_len): 2 warm-up runs, 5 timed repeats, `torch.cuda.synchronize()` around timers (`time.perf_counter`), `torch.cuda.reset_peak_memory_stats()` before each run and `max_memory_allocated()` after. Metrics: prefill time (first forward), decode time (remaining steps), total latency, output tokens/s (decode tokens / decode time), time-per-output-token, peak memory; mean ± std, median. Also run a CPU version (0.5B) to show the effect on CPU. Record hardware, torch/transformers versions, dtype in the JSON.

### 7.6 Stretch (P2): prefix KV reuse in the app
Note first in the prompt → compute the note prefix cache once per patient, `copy.deepcopy`/crop it for each criterion → big speedup when screening many criteria for one patient on CPU. Verify identical outputs vs. no prefix reuse.

---

## 8. Tiled attention with online softmax (Task 4) — `src/criterion_lens/attention/`

### 8.1 Reference (`reference.py::sdpa_reference(q, k, v, causal, attn_mask=None, scale=None)`)
Shapes `[B, H, Nq, d]`, `[B, Hkv, Nk, d]` (GQA: repeat KV heads `H/Hkv` times). `scores = (q @ kᵀ) * (scale or d**-0.5)`; causal mask with **query offset** `Nk - Nq` (so decode-style Nq<Nk works): key j visible to query i iff `j ≤ i + (Nk − Nq)`; masked = `-inf`; stable softmax: subtract row max (fully masked rows → output 0, no NaN); `softmax @ v`. Computes in fp32 internally.

### 8.2 Tiled online-softmax (`tiled.py::tiled_attention(q, k, v, causal, block_q=64, block_k=64)`)
For each query block Qᵢ (size Br): init `m = -inf`, `l = 0`, `acc = 0` (fp32). For each key block Kⱼ, Vⱼ (size Bc), **skipping blocks entirely above the causal diagonal**:
```
S = Qᵢ Kⱼᵀ · scale ; apply causal/padding mask inside diagonal blocks
m_new = max(m, rowmax(S))
P = exp(S − m_new)                       # masked → 0
α = exp(m − m_new)                       # rescale old stats (handle -inf − -inf → use where)
l = α·l + rowsum(P)
acc = α·acc + P Vⱼ
m = m_new
```
Output `Oᵢ = acc / l` (rows with l=0 → 0). Never materialises the N×N matrix: memory O(Br·Bc + N·d). No call to `F.scaled_dot_product_attention` or any fused backend inside. Handles Nq, Nk not divisible by block sizes, GQA, fp16/fp32 inputs (accumulate in fp32), query offset for decode.

### 8.3 Validation (E3) — `tests/test_attention.py` + `scripts/bench_attention.py`
- Grid: (B,H,Nq,Nk,d) ∈ {(1,1,8,8,4), (2,4,64,64,32), (1,2,1,77,64) decode, (3,8,129,129,64) non-divisible, (1,12,512,512,128)}, causal ∈ {True, False}, dtype ∈ {fp32, fp16}, block sizes ∈ {16, 32, 64, 128}; plus a padding-mask case and a fully-masked-row case.
- Ground truth: `sdpa_reference` in **fp64**. Report max abs error and mean abs error per case into `results/bench/attention_accuracy.csv`. Tolerances: fp32 atol 1e-5, fp16 atol 2e-3 (justify).
- Also compare against `torch.nn.functional.scaled_dot_product_attention` (math / efficient / flash backends via `torch.nn.attention.sdpa_kernel`, whichever are available on T4) as **performance baselines only**.
- Performance: N ∈ {256, 512, 1k, 2k, 4k, 8k}, d=64/128, H=12: time (warm-up + synchronized repeats) and peak memory for reference vs tiled vs SDPA backends, on T4 and CPU. Expectation (explain honestly): reference OOMs/explodes in memory at large N; tiled uses far less memory but is **slower** because each block iteration launches separate small kernels from Python (launch overhead, no SRAM reuse, no warp-level fusion, HBM round-trips); fused FlashAttention keeps tiles in on-chip SRAM in one kernel.
- **Integration demo (P1)**: register the tiled function with transformers' attention interface (`transformers.AttentionInterface.register("criterion_tiled", fn)` — check the installed version supports it; fallback = monkeypatch the Qwen2 attention forward, or use the mini-Qwen2 from §7.2) and run the fine-tuned model with `attn_implementation="criterion_tiled"`: same greedy tokens as `"eager"` on 5 val prompts.

---

## 9. Greedy speculative decoding (Task 5) — `src/criterion_lens/decoding/speculative.py`

### 9.1 Pair
Target = fine-tuned Qwen2.5-1.5B (merged), draft = Qwen2.5-0.5B (two variants: base and fine-tuned). Check tokenizer compatibility programmatically (`test_tokenizers_compatible`: identical vocab dict & special tokens; `lm_head` out-dim may differ → slice logits to `len(tokenizer)` before argmax). Memory: 3.1 GB + 1.0 GB fp16 + two KV caches (28 KB + 12 KB per token).

### 9.2 Algorithm (invariant-based; implement exactly)
**Invariant:** `seq` = prompt + committed tokens. The target cache holds KV for `seq[:-1]`; the newest token `seq[-1]` is "pending" (not yet in the target cache). The draft cache holds KV for some prefix `seq[:d_len]`.
```
PREFILL: target forward on the full prompt → t0 = argmax(last logits); seq = prompt + [t0]
         (target cache now holds seq[:-1] = prompt → invariant holds; t0 is pending)
         draft forward on the full prompt → draft cache holds prompt; d_len = len(prompt)
loop until max_new or EOS:
  # 1. DRAFT k tokens greedily
  feed draft all tokens seq[d_len:] (catch-up, usually 1–2 tokens) → d1 = argmax
  for i in 2..k: feed d_{i-1} → d_i = argmax
  # 2. VERIFY: ONE target forward on [seq[-1], d1, ..., dk]  (k+1 tokens, positions cur..cur+k,
  #    causal within block, sees full cache) → target argmax p1..p_{k+1}
  # 3. ACCEPT longest prefix n where d_i == p_i (i = 1..n)
  # 4. CORRECT: append d1..dn then p_{n+1} (the target's own token: replacement on reject, bonus if n == k)
  # 5. SYNC CACHES: target cache.crop(len(seq_new) - 1); draft cache.crop(min(d_len_now, len(seq_new) - 1))
  #    and set d_len accordingly (the draft never keeps KV for rejected tokens)
  # 6. truncate to max_new exactly; stop if EOS is among appended tokens (cut after EOS)
```
Worked example (k=4): seq ends with pending token `x`; draft proposes d1..d4; target verifies `[x,d1,d2,d3,d4]` → p1..p5. If d1=p1, d2=p2, d3≠p3 → n=2; append d1,d2,p3; target cache had prompt..x,d1..d4 → crop to len(seq_new)−1 (keeps …x,d1,d2; p3 pending). Draft cache had …x,d1,d2,d3 → crop to keep …x,d1,d2. If n=k=4: append d1..d4,p5; draft cache lacks d4 → it is fed on the next catch-up. Write this convention in the docstring and test it.

Because the output = target argmax at every position (accepted tokens equal target argmax by construction; the correction is target argmax), output == ordinary greedy target decoding **in exact arithmetic**.

### 9.3 Tests & experiments (F3) — `scripts/bench_speculative.py`
- Unit (tiny models, CPU fp32): (a) draft == target ⇒ acceptance 100% and identical output; (b) random draft ⇒ low acceptance and still identical output; (c) cache lengths satisfy the invariant after every iteration; (d) EOS and max_new edge cases.
- Real (T4 fp16 and fp32): 20 val prompts (+ test prompts at final), k ∈ {1, 2, 3, 4, 6, 8}, draft ∈ {base 0.5B, fine-tuned 0.5B}; compare with our cached greedy (§7.2) on the same prompts. Metrics: token equality rate (sequence-level and token-level; investigate any fp16 mismatch exactly as in §7.3 — batched verify vs single-step numerics), acceptance rate = accepted/proposed, mean accepted per iteration, target forwards per output token, end-to-end latency, tokens/s, speedup = greedy_latency / spec_latency, peak memory. Warm-up + synchronized repeats.
- Optional reference: HF `generate(assistant_model=draft, do_sample=False)` timings and outputs — for comparison only.
- Explain slowdowns: short JSON outputs (~20 tokens) give little room to amortise; the draft costs ~1/3 of the target per token (0.5B vs 1.5B is not a large ratio); `crop` & Python overhead; on CPU the target verify cost is closer to linear in k.
- Bonus analysis: acceptance per position in the JSON (structural tokens like `{"evidence": [` are near-100% accepted).

---

## 10. Application (Task 6) — `app/`

**Framework:** Gradio Blocks (runs locally and as a free HF Space; no code execution needed by the evaluator).

### 10.1 Tabs
1. **Screen a patient**
   - Patient note: textbox + `.txt` upload; "Load example" (synthetic examples written by us + train/val patients; *never* test patients in examples).
   - Criteria input, three ways: (a) paste free text with "Inclusion Criteria:" / "Exclusion Criteria:" headings (parser `trials/criteria_parser.py`, bullets/numbers split into items); (b) enter an NCT ID → fetch eligibility text from the ClinicalTrials.gov API v2 (`https://clinicaltrials.gov/api/v2/studies/{nct}`, field `protocolSection.eligibilityModule.eligibilityCriteria`) with timeout and clear error/offline fallback — allowed as "public trial metadata to support the application"; (c) editable table (`gr.Dataframe` with columns type/text) to fix the parse.
   - Settings: model (fine-tuned 1.5B / base 1.5B / fine-tuned 0.5B), decoding mode (KV-cache greedy [default] / no-cache reference / speculative with k), max criteria.
   - Run → progress bar per criterion; results table (criterion, type, label badge, evidence IDs, evidence text, explanation, flags); numbered note view with cited sentences highlighted per selected criterion; overall preliminary outcome card with the rule text (§6.4); list of missing-information criteria.
   - Export: **HTML report** (self-contained, printable) + **JSON** (machine-readable) + Markdown; includes provenance: app version + git commit, model id + adapter revision, decoding settings, timestamp (UTC), SHA-256 of the input note & criteria, timings, the disclaimer. Optionally save to `reports/` on disk (persistent locally).
2. **Inference Lab (live)** — on the current (or example) prompt run: no-cache vs KV-cache vs speculative (choose k, draft) live; table of prefill, decode, total latency, tokens/s, peak memory (CUDA) / RSS delta (CPU), token-equality check (✓/✗ with first-divergence index), acceptance stats; live bar chart from *this* run; session run history. Also a "cache growth" plot from the instrumented run.
3. **Attention Lab (live)** — inputs N, d, heads, block size, causal, dtype → runs reference vs tiled vs torch SDPA on random tensors; shows max/mean error, time, memory; explains which is fused.
4. **Offline results** — tables/plots loaded from `results/*.json` (GPU benchmarks, test metrics, training curves), labelled "measured offline on Kaggle T4, date, commit".
5. **About** — model card, split manifest summary, metrics, limitations, disclaimer, how decisions are aggregated.

### 10.2 Validation & error handling (G1, G4)
Empty note; note too short (<1 sentence); note over token budget (warn + list unseen sentence IDs); no criteria parsed; >N criteria (cap, warn); invalid NCT ID format (`^NCT\d{8}$`); network failure / 404 from ClinicalTrials.gov; model load failure (clear message + fallback to smaller model); CUDA OOM → retry on CPU / smaller model; malformed model output (fallback scoring, flagged); non-UTF-8 upload. Use `gr.Warning`/`gr.Error`, never a stack trace. A "Try invalid input" demo button in the video.

### 10.3 Startup / deployment
- Official deployment: locally hosted Gradio app, allowed by the assignment. On Windows, `run_app.bat` launches `.venv\Scripts\python.exe -m app`, opens `http://127.0.0.1:7860`, and accepts optional `--share` for temporary remote access. Models download from HF Hub on first use and are cached by Hugging Face; document CPU/GPU behavior and troubleshooting.
- Optional deployment: retain `space/` and `scripts/deploy_space.py` as ready to deploy if Hugging Face PRO is available. Do not retry deployment or block project completion on it: HF's 2026 requirement for free `cpu-basic` Gradio Spaces requires PRO. Do not claim a live Space.
- Optional `Dockerfile`.

---

## 11. Literature (≥ 5 required; target 12–15). Verify every entry before citing.
Eligibility reasoning: Jin et al., "Matching patients to clinical trials with large language models" (TrialGPT), *Nature Communications* 2024; Koopman & Zuccon, "A test collection for matching patients to clinical trials", SIGIR 2016 (source of the patient notes); Wornow et al., zero-shot clinical trial patient matching with LLMs (NEJM AI) — verify; a criteria2query / pre-LLM matching paper — optional.
Transformers: Vaswani et al. 2017; Su et al., RoFormer (RoPE); Ainslie et al. 2023, GQA; Qwen2.5 Technical Report (arXiv 2412.15115).
Supervised adaptation: Hu et al., LoRA (ICLR 2022); Dettmers et al., QLoRA (NeurIPS 2023); optionally Ouyang et al. 2022 (instruction tuning) .
Efficient inference: Pope et al., "Efficiently Scaling Transformer Inference" (MLSys 2023); Kwon et al., PagedAttention/vLLM (SOSP 2023); Milakov & Gimelshein 2018, online softmax; Rabe & Staats 2021, "Self-attention does not need O(n²) memory"; Dao et al. 2022 FlashAttention; Dao 2023 FlashAttention-2; Leviathan et al., ICML 2023, speculative decoding; Chen et al. 2023, speculative sampling.
Each must be *compared* (limitations) and *connected* to a design decision in the Related Work section.

---

## 12. Repository layout (final)
```
criterion-lens/
├── README.md                     # overview, results table, quick start, links (video, Space, weights, paper)
├── LICENSE                       # MIT (code); dataset/model licences referenced
├── .gitignore  .gitattributes
├── pyproject.toml                # package metadata, pytest config, ruff
├── requirements.txt              # core (torch, transformers, peft, datasets, accelerate, sklearn, pandas, numpy, matplotlib, pyyaml, huggingface_hub)
├── requirements-app.txt          # + gradio, pysbd, requests, psutil
├── configs/   data.yaml  train_qwen25_1p5b.yaml  train_qwen25_0p5b.yaml  eval.yaml  bench.yaml
├── src/criterion_lens/
│   ├── data/        load.py split.py audit.py sentences.py prompts.py tokenize.py
│   ├── modeling/    load_model.py (base/adapter/merged, dtype/device logic)
│   ├── training/    sft.py (dataset build, Trainer setup, callbacks)
│   ├── inference/   schema.py parse.py predict.py explain.py aggregate.py fallback.py
│   ├── decoding/    kv_cache.py speculative.py mini_qwen2.py(P2) cache_utils.py timing.py
│   ├── attention/   reference.py tiled.py hf_integration.py
│   ├── eval/        metrics.py evidence_metrics.py bootstrap.py error_analysis.py
│   ├── trials/      ctgov.py criteria_parser.py
│   └── report/      export.py templates/report.html.j2
├── scripts/  make_split.py audit_data.py train.py evaluate.py error_analysis.py bench_kv.py
│             bench_attention.py bench_speculative.py make_figures.py push_to_hub.py download_weights.py
├── notebooks/ 01_kaggle_train.ipynb 02_kaggle_eval_bench.ipynb (thin: clone repo, pip install, call scripts)
├── app/      __main__.py ui.py screening.py inference_lab.py attention_lab.py examples/ assets/
├── space/    README.md app.py requirements.txt
├── tests/    conftest.py test_split.py test_no_leakage.py test_sentences.py test_prompts.py test_parse.py
│             test_metrics.py test_evidence_metrics.py test_aggregate.py test_attention.py
│             test_kv_cache.py test_speculative.py test_tokenizers_compatible.py(slow) test_app_smoke.py
├── data/splits/  split_manifest.json split_manifest.csv      (data/raw/ is git-ignored)
├── results/  data_audit.json  train/  eval/  bench/  figures/  TEST_ACCESS_LOG.md
├── report/   main.tex  sections/*.tex  references.bib  figures/   (IEEEtran; compile on Overleaf)
└── docs/     PLAN.md EXECUTION_PLAN.md DECISIONS.md RUBRIC_CHECKLIST.md REPORT_OUTLINE.md ERROR_ANALYSIS.md
```

### Tiny test model (`tests/conftest.py`)
```python
from transformers import Qwen2Config, Qwen2ForCausalLM
cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
torch.manual_seed(0); tiny = Qwen2ForCausalLM(cfg).eval()   # fp32, CPU
```
Second tiny model with another seed (and/or 1 layer) = "draft" for speculative tests.

---

## 13. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Deadline in ~1 day | Strict priority tiers (P0/P1/P2) in EXECUTION_PLAN; GPU jobs launched early, CPU work in parallel |
| Kaggle session dies | Save adapter each epoch to `/kaggle/working`, push to HF Hub at end; scripts resumable |
| fp16 NaNs (Qwen2 on T4) | fp32 base fallback; gradient clipping 1.0 |
| Model outputs invalid JSON | Constrained fallback scoring; parse-failure rate reported |
| Class imbalance (rare N/A, `excluded`) | Report per-class; optional oversampling ablation on val |
| CPU Space too slow | 0.5B option; prefix-cache reuse (P2); cap criteria; progress bar |
| Pascal GPU unsupported locally | Use CPU locally, T4 on Kaggle |
| transformers Cache API differences | `cache_utils.py` accessors handle both `layers[i].keys` and `key_cache[i]`; pin the version used |
| fp16 mismatch in KV/spec equality | Investigate + fp32 rerun, documented (this earns marks, not loses them) |

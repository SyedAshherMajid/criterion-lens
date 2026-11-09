# Design Decision Log (ADR-style)

Each entry: **Context → Options → Decision → Evidence → Consequences.** The execution agent appends new entries whenever something is changed, fails, or is measured. These entries feed rubric item A3 and the report's "Design decisions and failures" table.

---

### D-001 Target model: Qwen2.5-1.5B-Instruct
- **Options:** Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Llama-3.2-1B-Instruct, Phi-3.5-mini, Gemma-2-2B-it, Qwen3-1.7B.
- **Decision:** Qwen2.5-1.5B-Instruct (Apache-2.0, ungated, 32k context, GQA 2 KV heads, same-tokenizer 0.5B draft, fits T4 training and free-CPU serving).
- **Evidence:** inspected config/parameter facts in `results/model_specs.json` and rationale in `docs/MODEL_SELECTION.md`; zero-shot val comparison vs SmolLM2 (`results/eval/val_smollm2_zs/`) — TBD.
- **Consequences:** a small model means modest absolute accuracy, so fine-tuning matters. The draft/target size ratio (~1:3) limits the speculative speed-up.

### D-002 Fine-tuning method: LoRA (r=16, all linear layers), not full FT / QLoRA
- **Options:** full fine-tuning (too much memory for 1.5B on T4 with Adam: ~25 GB), QLoRA 4-bit (slower, and quantisation noise is unnecessary at 1.5B), LoRA in fp16/fp32.
- **Decision:** LoRA; QLoRA only as a fallback if memory fails.
- **Evidence:** peak training memory from Kaggle logs — TBD.

### D-003 Output schema: `{"evidence": [...], "label": "..."}` (evidence first); no free-text explanation target
- **Why:** the only explanations in the dataset are GPT-4 generated, and training on them would leak GPT-4 outputs (forbidden). Putting evidence first is a light cite-then-decide ordering, and the short output speeds up decoding.
- **Alternatives:** label first (ablation P1); including the GPT-4 explanation as a target (rejected: leakage).
- **Explanations** come from a grounded template over the cited sentences (traceable, cannot hallucinate facts).

### D-004 Patient note before criterion in the prompt
- **Why:** it enables reuse of the note prefix KV cache across the criteria of one patient (app speed on CPU), and the model reads the evidence before the question.

### D-005 Completion-only loss with a manual tokenization + HF Trainer
- **Why:** full control and testability of label masking; avoids TRL API version churn.

### D-006 Checkpoint selection by val macro-F1 (generation-based), not val loss
- **Why:** the target metric is label macro-F1. Val loss is dominated by the JSON formatting tokens.

### D-007 Fallback = constrained label scoring
- **Why:** it guarantees a valid label for every criterion (reliability, Task 2). The fallback rate is reported, so the fallback cannot hide model weakness.

### D-008 Hardware: Kaggle T4 for training + headline benchmarks; local GTX 1060 for 0.5B/attention; CPU in the HF Space
- **Why:** the local GTX 1060 3 GB works with the installed torch 2.6.0+cu124 (sm_61 verified), but it cannot hold 1.5B fp16 and has no fast fp16 tensor cores. Kaggle T4 = Turing, fp16 tensor cores, 16 GB. Prefer T4 over the Kaggle P100, because newer Kaggle torch builds may drop Pascal.
- **Reuse:** the local env reuses the already-installed HF stack (`docs/LOCAL_ENVIRONMENT.md`). The requirements are pinned to those versions so Kaggle matches.

### D-009 Evidence metric conventions
- Micro over all records; macro over records with non-empty gold; empty-gold records reported separately as abstention accuracy; a secondary "empty = perfect" macro is reported for transparency.

### D-010 App: Gradio on a Hugging Face Space (CPU basic)
- **Options:** Streamlit, FastAPI + React, Gradio. **Decision:** Gradio. It is the fastest to build, has native Space hosting, and its `gr.File` download and progress bars cover the required features.

### D-011 Candidate inspection and inaccessible configs
- **Context:** The model-spec run could not read the Llama 3.2 and Gemma 2 configs (`OSError`); the Qwen3-1.7B repo's safetensors metadata reports 2,031,739,904 parameters.
- **Options:** Fill inaccessible candidate rows from estimates in the plan, omit the candidates, or preserve their unavailable status; treat the Qwen3 repo label as the count or report the retrieved tensor total.
- **Decision:** Record Llama and Gemma as uninspected with no parameter estimates. Report the retrieved Qwen3 tensor total and retain its repo name as the model identifier.
- **Evidence:** `results/model_specs.json`, `docs/MODEL_SELECTION.md`.
- **Consequences:** Candidate comparisons remain reproducible and do not imply that gated configs were inspected; Qwen3's total count differs from the rounded repo name.

### D-012 Data audit and sequence length
- **Context:** At dataset revision `1cfcafde94a1560a33b4addc1638664fe28fc059`, 1,015 source rows yielded 1,014 annotations after dropping one empty criterion. The audit found zero label/type inconsistencies, zero out-of-range evidence IDs, and zero numbered-note parse failures.
- **Options:** Filter or relabel inconsistent rows, or preserve expert annotations; retain the planned 1,536-token sequence length, or size it from observed prompt-plus-target lengths.
- **Decision:** Keep the expert labels and evidence unchanged because no inconsistencies or invalid IDs were found; retain the loader's explicit empty-criterion drop. Use `max_seq_len=640` for training: the longest chat-formatted prompt plus target and end token is 634 tokens, so 640 is the smallest multiple of eight that covers the observed data.
- **Evidence:** `results/data_audit.json` reports 1,014 measurable prompt/target lengths, overall mean 338.54, p95 481.35, and maximum 634; the audit also reports empty-evidence rates by label.
- **Consequences:** Phase 3.2 verified all 602 training examples fit within 640 tokens with a maximum of 634 and zero truncations. The shared tokenization helper preserves the target and gold evidence and records any removed sentence IDs.

### D-013 Model-output parsing
- **Context:** Generated responses may contain extra prose, malformed JSON, labels incompatible with the criterion type, or evidence IDs outside the note.
- **Options:** Reject any non-JSON output or attempt a constrained regex recovery; silently accept bad fields or preserve a failed-parse state for constrained label fallback.
- **Decision:** Parse the first JSON object, then try regex recovery when its structure is unusable. Normalize and validate the label, sort/deduplicate in-range evidence, record out-of-range IDs, and set `parse_ok=False` whenever validation fails.
- **Evidence:** `tests/test_parse.py` covers fenced JSON, malformed-JSON recovery, type-incompatible labels, malformed evidence, invalid IDs, and parser arguments.
- **Consequences:** A recovered label/evidence pair can be used without hiding malformed output; failed validation remains explicit for the later fallback path.

### D-014 Training smoke model and tokenizer
- **Context:** The first CPU smoke attempt used a temporary tiny Qwen2 model without tokenizer files, so model override resolution also tried to load a tokenizer from that model directory and failed.
- **Options:** Require every tiny test model to package the project's tokenizer, or keep the tokenizer ID explicit and independently configurable.
- **Decision:** Pin `tokenizer_name_or_path` to Qwen2.5-1.5B-Instruct in the training config, independently of `--model`. The smoke model must retain compatible vocabulary and LoRA module names.
- **Evidence:** The selected tokenizer's maximum token ID is 151,664, which fits the Qwen model's 151,936-entry embedding; the five-step CPU Trainer smoke run evaluated validation loss and saved an adapter.
- **Consequences:** The smoke path tests the real dataset, prompt/tokenization path, Trainer, and LoRA save flow with a tiny compatible model; it does not represent a fine-tuned project model or reportable model metrics.

### D-015 Kaggle source sync and Hub credential boundary
- **Context:** The Kaggle training lane needs the private repository source, while the initial Phase 3.6 checklist described cloning the repository and placing `HF_TOKEN` in Kaggle Secrets. `docs/PARALLEL_WORKFLOW.md` instead specifies a private allowlisted source dataset and laptop-only Hub uploads.
- **Options:** Clone the private repository and expose the Hub token to Kaggle, or stage only required source/data files in the private Kaggle dataset and upload trained artifacts from the laptop.
- **Decision:** Use the allowlisted private Kaggle code dataset; do not stage `.env` or send `HF_TOKEN` to Kaggle. Keep Hub publishing on the laptop after training outputs are downloaded.
- **Evidence:** `scripts/kaggle/sync_code.py` stages only `src/`, `scripts/`, `configs/`, `data/splits/`, and required project files; notebook tests assert that `HF_TOKEN` and `kaggle_secrets` are absent.
- **Consequences:** Kaggle runs need no Hub secret. Model uploads still require a write-enabled laptop token, and outputs must be collected before publishing.

### D-016 Kaggle owner slug
- **Context:** The account appeared to be `syedashher` because a stale system-wide `KAGGLE_API_TOKEN` took precedence over the credentials in `.env`.
- **Options:** Continue using the account authenticated by the inherited process token, or make the explicitly configured `.env` Kaggle credentials authoritative for Kaggle CLI calls.
- **Decision:** Use `syedashhermajid`, as configured in `.env`; `scripts/kaggle/kaggle_env.py` overrides process environment values for `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME` only.
- **Evidence:** The earlier `syedashher` result came from a stale system-wide token, not `.env`. The switched account must be verified via `kaggle_env.py -- kernels list --mine` before syncing or launching.
- **Consequences:** Use only `syedashhermajid/criterion-lens-code` and `syedashhermajid/criterion-lens-train` for this workflow. Ignore the old `syedashhermajid/criterion-lens-smoketest` kernel.

### D-017 Kaggle dataset metadata filename
- **Context:** The installed Kaggle CLI help requested `datasets-metadata.json`, but the first dataset create rejected that filename and explicitly required `dataset-metadata.json`.
- **Options:** Follow the CLI help text or the server's validation error.
- **Decision:** Use `dataset-metadata.json`, as required by the dataset API.
- **Evidence:** The create request failed locally at metadata validation before the dataset was uploaded.
- **Consequences:** The staging test and workflow docs now use the server-accepted filename.

### D-018 Kaggle training kernel received no GPU
- **Context:** Training kernel version 1 was pushed with `--accelerator NvidiaTeslaT4`, began execution, then failed in the first preflight cell.
- **Options:** Continue to training without a GPU, weaken the GPU guard, or stop before installing/training and retry when a GPU runtime is available.
- **Decision:** Preserve the fail-fast GPU check. The downloaded Kaggle log reports `torch.cuda.is_available()` is false; no training or split regeneration ran.
- **Evidence:** `results/kaggle/train/criterion-lens-train.log` ends with `RuntimeError: A Kaggle GPU runtime is required`.
- **Consequences:** No adapter or training metrics exist from this run. Retry the kernel after other laptop work; the documented local fallback is tied specifically to a job remaining `QUEUED` for 45 minutes, which this failed job did not do.

### D-019 Kaggle GPU preflight failed on retry
- **Context:** After the first no-CUDA failure, the source dataset was versioned to the latest commit and training kernel version 2 was pushed again with `NvidiaTeslaT4`.
- **Options:** Remove the GPU preflight and risk CPU training, keep retrying immediately, or preserve the guard and continue the independent laptop lane.
- **Decision:** Preserve the preflight; the second run failed with the same `torch.cuda.is_available() == False` error before data setup or training.
- **Evidence:** The downloaded `results/kaggle/train/criterion-lens-train.log` from version 2 has the same first-cell traceback as version 1.
- **Consequences:** The retry confirms a Kaggle runtime issue rather than a source regression. No adapter or split evaluation was produced; continue local work and retry only if GPU availability changes.

### D-020 Do not report synthetic KV smoke timings as project benchmarks
- **Context:** The benchmark CLI and JSON/CSV outputs can be exercised with a tiny random Qwen2 model, but the fine-tuned 1.5B merged model is not available and Kaggle's T4 runtime failed both GPU preflights.
- **Options:** Save tiny CPU smoke timings as `results/bench/kv_cache.json` and present them as project results, or keep the smoke run isolated and wait for the intended model/hardware.
- **Decision:** Keep the smoke outputs in a temporary directory, clearly mark benchmark inputs as deterministic synthetic token IDs, and do not claim these timings as model results. Run the full precision/grid benchmark when the target model and hardware are available.
- **Evidence:** The tiny CPU CLI smoke produced one fp32 parity check with no mismatch; the project test suite (102 passed) and Ruff passed. No fine-tuned model artifacts exist.
- **Consequences:** Phase 6.4 remains unchecked until the planned benchmark is run; no benchmark metrics are invented.

### D-021 Record unavailable SDPA backends instead of treating them as failures
- **Context:** A quick CUDA benchmark on the GTX 1060 reports Flash Attention as unavailable while math and efficient SDPA execute.
- **Options:** Let an unsupported forced backend abort the entire benchmark, omit it without explanation, or preserve an `UNAVAILABLE` row with the backend error.
- **Decision:** Record unsupported backends as `UNAVAILABLE`; reserve `OOM` for memory failures and propagate other runtime errors.
- **Evidence:** The quick run wrote 8 accuracy and 10 performance rows; Flash Attention returned `No available kernel`, while math and efficient paths completed. The temporary smoke outputs were removed.
- **Consequences:** Backend availability is hardware-specific; the local GTX result is not a T4 performance result. Phase 7.3 remains unchecked until the full benchmark outputs are run and reviewed.

### D-022 Kaggle credentials and training fallback
- **Context:** A machine-level `KAGGLE_API_TOKEN` can silently select a different Kaggle account than the project `.env`; the previous 45-minute queued-only fallback also does not address immediate no-GPU errors under the current deadline.
- **Options:** Trust inherited process credentials and wait for queued kernels, or prioritize `.env` Kaggle credentials and start laptop fallback on a no-GPU error or 20-minute queue.
- **Decision:** Override process values from `.env` for `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME` only. Launch training as `syedashhermajid/criterion-lens-train`; start local 0.5B training on GTX 1060 immediately on a no-GPU `ERROR` or after 20 minutes QUEUED.
- **Evidence:** The Kaggle owner previously observed was authenticated by the stale system token, while the verified project account and existing private code dataset are `syedashhermajid`.
- **Consequences:** Other environment variables retain their existing process-over-.env precedence. Never print or log the token; leave queued Kaggle training active while local fallback runs.

---
<!-- Agent: append D-015+ below: data-audit findings, NaN/OOM incidents, fp16 mismatch investigations, sweep results, speculative slowdowns, etc. -->

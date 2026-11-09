# Design Decision Log (ADR-style)

Each entry: **Context → Options → Decision → Evidence → Consequences.** The execution agent appends new entries whenever something is changed, fails, or is measured. These entries feed rubric item A3 and the report's "Design decisions and failures" table.

---

### D-001 Target model: Qwen2.5-1.5B-Instruct
- **Options:** Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Llama-3.2-1B-Instruct, Phi-3.5-mini, Gemma-2-2B-it, Qwen3-1.7B.
- **Decision:** Qwen2.5-1.5B-Instruct (Apache-2.0, ungated, 32k context, GQA 2 KV heads, same-tokenizer 0.5B draft, fits T4 training and free-CPU serving).
- **Evidence:** `results/model_specs.json`; zero-shot val comparison vs SmolLM2 (`results/eval/val_smollm2_zs/`) — TBD.
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

### D-008 Hardware: Kaggle T4 (not P100), CPU locally and in the HF Space
- **Why:** the local GTX 1060 3 GB (Pascal) cannot hold 1.5B fp16, and new PyTorch CUDA builds may drop Pascal. Kaggle P100 is also Pascal. T4 = Turing, fp16 tensor cores, 16 GB.

### D-009 Evidence metric conventions
- Micro over all records; macro over records with non-empty gold; empty-gold records reported separately as abstention accuracy; a secondary "empty = perfect" macro is reported for transparency.

### D-010 App: Gradio on a Hugging Face Space (CPU basic)
- **Options:** Streamlit, FastAPI + React, Gradio. **Decision:** Gradio. It is the fastest to build, has native Space hosting, and its `gr.File` download and progress bars cover the required features.

---
<!-- Agent: append D-011+ below: data-audit findings, NaN/OOM incidents, fp16 mismatch investigations, sweep results, speculative slowdowns, etc. -->

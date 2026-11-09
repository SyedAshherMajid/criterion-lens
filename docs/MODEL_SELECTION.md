# Model Selection

## Inspection method

`scripts/inspect_models.py` loads only each model's `config.json` through
`AutoConfig` and reads parameter totals and license metadata from the Hugging Face
Hub API. It does not download model weights. `results/model_specs.json` records
the inspected fields and explicitly marks models whose configs could not be read.
Parameter totals below are the Hub safetensors metadata totals; fp16 storage is
estimated as two bytes per parameter and excludes runtime memory.

## Candidate comparison

| Candidate | Parameters | Layers; Q/KV heads; head dim | Context; config vocab | License | Assessment |
|---|---:|---|---|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1,543,714,304 (about 3.09 GB fp16) | 28; 12/2; 128 | 32,768; 151,936 | Apache-2.0 | **Selected target.** GQA keeps the KV cache small; fp16 weights fit a T4, while local app inference uses the 0.5B option. |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 494,032,768 (about 0.99 GB fp16) | 24; 14/2; 64 | 32,768; 151,936 | Apache-2.0 | Selected draft and smaller-model comparison. Its family and config vocabulary match the target; the dedicated tokenizer compatibility test remains required before speculative decoding. |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1,711,379,456 (about 3.42 GB fp16) | 24; 32/32; 64 | 8,192; 49,152 | Apache-2.0 | Runner-up; planned zero-shot validation comparison. Its MHA cache is larger per token than Qwen2.5's GQA cache. |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Not inspected | Not inspected | Not inspected | Gated; license acceptance required | Excluded: gated access and no smaller same-tokenizer draft were specified in the plan. The inspection run could not read its config; no parameter estimate is asserted. |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3,821,079,552 (about 7.64 GB fp16) | 32; 32/32; 96 | 131,072; 32,064 | MIT | Rejected: larger CPU memory/latency burden and poor fit for T4 full-precision training. |
| [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it) | Not inspected | Not inspected | Not inspected | Gated terms | Excluded: gated access and architecture-specific attention behavior complicate the planned custom-attention comparison. The inspection run could not read its config; no parameter estimate is asserted. |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 2,031,739,904 (about 4.06 GB fp16) | 28; 16/8; 128 | 40,960; 151,936 | Apache-2.0 | Considered for future work; not selected to avoid the thinking-mode chat-template and Transformers-API changes noted in the plan. |

## Selection rationale

Qwen2.5-1.5B-Instruct is the target because it is openly downloadable under
Apache-2.0, fits a free T4 training run, and has an inspected 0.5B family member
with matching configuration vocabulary for a draft/target pair. Its two KV heads
make the cache-growth calculation concrete. The 0.5B checkpoint also fits the
local GTX 1060 and is the fallback headline model if the larger training job is
not available in time.

Config vocabulary equality is not a substitute for tokenizer identity. Before
speculative decoding, `tests/test_tokenizers_compatible.py` must verify the
tokenizer vocabularies and special-token mappings directly. The parameter counts
and architecture facts in this file are from `results/model_specs.json`; the
gated candidates remain explicitly uninspected rather than filled with estimates.

# Addendum: full benchmark grid (added 11 Nov 2026, after submission)

The submitted report (`report/CriterionLens_IEEE.pdf`) uses the **reduced grid** from Kaggle job
`criterion-lens-benchmarks-quick` (`results/bench/`). The **full grid** job `criterion-lens-benchmarks`
(`kaggle/jobs/benchmarks/`, configs `configs/bench.yaml`, `configs/speculative_bench.yaml`) finished after
submission. Its raw outputs are in [`results/bench_full/`](../results/bench_full/) with the Kaggle log
`kaggle_benchmarks_full.log`. Same hardware (Tesla T4, PyTorch 2.11.0+cu130, Transformers 4.57.6),
2 warm-up + 5 timed repeats. The submitted PDF is unchanged; this note records what the larger run adds.

## KV cache (fine-tuned 1.5B, float16)
| Prompt tokens | 32 new | 128 new | 256 new | KV tokens/s | No-cache tokens/s (256 new) |
|---|---|---|---|---|---|
| 128 | 1.30× | 1.48× | 1.94× | ≈28 | 14.2 |
| 512 | 4.13× | 4.80× | 5.30× | ≈28 | 5.1 |
| 1024 | 7.76× | 9.32× | 10.5× | ≈27 | 2.5 |
| 2048 | 14.9× | 22.5× | **24.8×** | ≈28 | 1.0 |

The KV-cache decoder stays at about 28 tokens/s regardless of context. Full-prefix recomputation degrades
roughly linearly with context. 24/24 precision checks per model: identical greedy tokens.
This agrees with the reduced grid (9.8× at 1024/128 there vs 9.3× here).

## Tiled attention
The same 80/80 accuracy configurations are within tolerance. The results match the reduced run.

## Speculative decoding (20 synthetic prompts, 16 and 64 new tokens, k ∈ {1,2,3,4,6,8})
- Acceptance: 0.83 → 0.49 (64 tokens) and 0.65 → 0.31 (16 tokens) as k grows 1 → 8. The fine-tuned draft
  is accepted marginally more often than the base draft.
- Wall-clock speedup is below 1 everywhere: 0.94× at best (k=1, 64 tokens) and 0.45× at worst (k=8, 16 tokens).
  This matches the reduced grid.
- **Exactness:** 26 of 240 (prompt, length, k) cells per draft contain a divergence from target-only greedy
  decoding. Mean token equality is 0.927 and sequence equality 0.892. The reduced grid (prompts 0–5) had none.

### What the mismatches are
- They occur on only 4 of the 20 prompts (#10, 14, 17, 18).
- For each prompt the first divergence is at the **same step for every k (including k=1) and for both
  drafts**, and it is deterministic across all 5 repeats.
- The draft and the accept/reject rule therefore cannot be the cause, because both change with k and with
  the draft. The difference comes from the **target model's float16 numerics**. Verification scores k+1
  tokens in one forward pass, while the reference decodes one token per pass. Different kernel shapes give
  slightly different fp16 rounding, and on these random-token prompts the top two logits are nearly tied
  (e.g. the continuation `16, 17, 18` vs `17, 18, 19`), so the argmax flips.
- The float32 unit test `tests/test_speculative.py` passes: random and identical drafts reproduce target
  greedy output exactly and resynchronise caches. This confirms the algorithm. The fp16 benchmark exposes
  numerical, not algorithmic, non-determinism.

**Correction to the submitted report:** its sentence "speculative output matched target greedy decoding exactly
(... zero mismatches)" is true for the reduced grid it reports. On the full grid, exact equality holds in fp32
and for 89% of fp16 sequences. Running the target in float32 (or bfloat16-accumulated verification) would be the
fix, at a speed cost.

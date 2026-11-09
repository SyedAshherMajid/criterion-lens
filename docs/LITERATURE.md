# Verified Literature Notes

The bibliographic fields below were checked against the linked publisher, proceedings,
PubMed, Crossref, or arXiv record. Each note records a limitation and the project
decision it informs; the sources are not claims of clinical validity for this prototype.

## Clinical-trial matching and data

### Jin et al. (2024) — TrialGPT
- **Citation:** Qiao Jin, Zifeng Wang, Charalampos S. Floudas, Fangyuan Chen, Changlin Gong,
  Dara Bracken-Clarke, Elisabetta Xue, Yifan Yang, Jinyu Sun, and Zhiyong Lu.
  “Matching patients to clinical trials with large language models.”
  *Nature Communications*, 15, article 9074, 2024.
  [DOI: 10.1038/s41467-024-53081-z](https://doi.org/10.1038/s41467-024-53081-z).
- **Summary:** Introduces TrialGPT for matching patients to clinical trials and structures
  the task around criterion-level decisions and explanations.
- **Limitation:** It relies on large language models and its evaluation setup does not by
  itself establish that a small locally fine-tuned model is reliable for enrollment.
- **Design decision informed:** Keep CriterionLens criterion-level, return traceable
  evidence, and present the result as research screening rather than a clinical decision.

### Koopman and Zuccon (2016) — SIGIR clinical-trial collection
- **Citation:** Bevan Koopman and Guido Zuccon. “A Test Collection for Matching Patients to
  Clinical Trials.” In *Proceedings of the 39th International ACM SIGIR Conference on
  Research and Development in Information Retrieval*, pp. 669–672, 2016.
  [DOI: 10.1145/2911451.2914672](https://doi.org/10.1145/2911451.2914672).
- **Summary:** Defines a test collection for patient-to-trial matching that supports
  reproducible retrieval and matching experiments.
- **Limitation:** Its patient population and annotation collection are small and
  benchmark-specific; results should not be generalized to clinical deployment.
- **Design decision informed:** Use the associated TrialGPT annotations as the single
  supervised dataset and split by patient to prevent note-level leakage.

## Transformers and adaptation

### Vaswani et al. (2017) — Transformer
- **Citation:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need.”
  *Advances in Neural Information Processing Systems* 30, 2017.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- **Summary:** Presents the attention-based Transformer architecture, replacing recurrence
  with self-attention and enabling parallel sequence processing during training.
- **Limitation:** Full self-attention has quadratic sequence-length compute and materialized
  attention can consume quadratic memory.
- **Design decision informed:** Use a decoder-only Transformer for generation and measure
  explicit cache and attention implementations rather than assume they are efficient.

### Hu et al. (2022) — LoRA
- **Citation:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li,
  Shean Wang, Lu Wang, and Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.”
  In *International Conference on Learning Representations*, 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- **Summary:** Freezes pretrained weights and learns low-rank updates, substantially
  reducing trainable parameters and memory compared with updating all weights.
- **Limitation:** The quality and suitable rank are task-dependent; parameter efficiency
  does not guarantee better outcomes on this dataset.
- **Design decision informed:** Fine-tune the 1.5B and 0.5B models with LoRA and report
  trainable parameter counts and validation-based checkpoint selection.

### Dettmers et al. (2023) — QLoRA
- **Citation:** Tim Dettmers et al. “QLoRA: Efficient Finetuning of Quantized LLMs.”
  *arXiv:2305.14314*, 2023; extended NeurIPS submission.
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- **Summary:** Combines 4-bit quantized frozen base weights with trainable low-rank
  adapters to reduce fine-tuning memory requirements.
- **Limitation:** Quantization adds implementation and numerical considerations and is
  unnecessary if the target model and LoRA fit the available T4 memory.
- **Design decision informed:** Keep standard LoRA as the primary recipe and reserve
  QLoRA as a documented fallback if training memory is insufficient.

### Su et al. (2021) — Rotary Position Embedding
- **Citation:** Jianlin Su et al. “RoFormer: Enhanced Transformer with Rotary Position
  Embedding.” *arXiv:2104.09864*, 2021.
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- **Summary:** Introduces RoPE, encoding absolute positions by rotations while making
  relative position dependencies explicit in self-attention.
- **Limitation:** The paper's experiments do not establish that one positional encoding
  is best for every architecture, context length, or downstream task.
- **Design decision informed:** Use absolute decode positions in the KV-cache path and
  include a negative test showing why resetting positions to zero is incorrect.

### Ainslie et al. (2023) — Grouped-query attention
- **Citation:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy,
  Federico Lebrón, and Sumit Sanghai. “GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints.” *arXiv preprint arXiv:2305.13245*, 2023; accepted
  at EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- **Summary:** Generalizes multi-query attention by sharing key/value heads in groups,
  reducing autoregressive KV-cache cost while retaining quality close to multi-head
  attention in the reported experiments.
- **Limitation:** Its uptraining experiments and evaluated model scales differ from this
  project's fixed pretrained Qwen checkpoints.
- **Design decision informed:** Include the model's actual KV-head count in cache-byte
  calculations and compare cache growth against the GQA-aware formula.

### Qwen Team (2024; revised 2025) — Qwen2.5
- **Citation:** An Yang et al. (Qwen Team). “Qwen2.5 Technical Report.”
  *arXiv:2412.15115*, first submitted December 2024, revised January 2025.
  [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
- **Summary:** Describes the Qwen2.5 model series, including instruction-tuned open-weight
  models and their training and evaluation approach.
- **Limitation:** The report's broad benchmark results do not establish clinical
  eligibility accuracy or validate the smaller checkpoints on this dataset.
- **Design decision informed:** Select the 1.5B instruction model as target and its
  0.5B same-family checkpoint as a tokenizer-compatible draft; verify checkpoint
  specifications directly from model configs.

## Efficient inference

### Milakov and Gimelshein (2018) — Online softmax
- **Citation:** Maxim Milakov and Natalia Gimelshein. “Online normalizer calculation for
  softmax.” *arXiv:1805.02867*, 2018.
  [arXiv:1805.02867](https://arxiv.org/abs/1805.02867).
- **Summary:** Describes an online method for computing a numerically stable softmax
  normalizer with fewer memory accesses, including a running maximum and normalizer.
- **Limitation:** The paper focuses on softmax and does not specify the complete
  query-key-value attention implementation or its causal and padding masks.
- **Design decision informed:** Derive the running-max rescale used by the tiled
  attention implementation and test fully masked and non-divisible tiles.

### Dao et al. (2022) — FlashAttention
- **Citation:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré.
  “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.”
  In *Advances in Neural Information Processing Systems* 35, 2022.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- **Summary:** Shows how tiling and IO-aware scheduling can compute exact attention while
  reducing high-bandwidth-memory traffic and avoiding a full attention matrix.
- **Limitation:** Its fused GPU kernels are hardware- and implementation-optimized;
  a clear pedagogical PyTorch implementation is not expected to match their speed.
- **Design decision informed:** Implement and validate an unfused tiled online-softmax
  reference, and explicitly treat fused SDPA only as a performance baseline.

### Kwon et al. (2023) — PagedAttention
- **Citation:** Woosuk Kwon et al. “Efficient Memory Management for Large Language Model
  Serving with PagedAttention.” In *Proceedings of the 29th Symposium on Operating
  Systems Principles (SOSP)*, 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- **Summary:** Applies paging to KV-cache management to reduce fragmentation and allow
  flexible cache sharing in high-throughput LLM serving.
- **Limitation:** The main contribution is a serving and memory-management system; it
  does not provide a drop-in implementation for this project's single-request decoder.
- **Design decision informed:** Explain cache growth and memory accounting, while keeping
  the required single-sequence decoder explicit rather than claiming PagedAttention.

### Leviathan et al. (2023) — Speculative decoding
- **Citation:** Yaniv Leviathan, Matan Kalman, and Yossi Matias. “Fast Inference from
  Transformers via Speculative Decoding.” In *Proceedings of the 40th International
  Conference on Machine Learning*, PMLR 202, pp. 19274–19286, 2023.
  [PMLR paper](https://proceedings.mlr.press/v202/leviathan23a.html).
- **Summary:** Uses a faster draft model to propose tokens that a larger target verifies
  in parallel, preserving the target model's output distribution while potentially
  reducing serial decoding work.
- **Limitation:** Speedup depends on draft quality, acceptance rate, and hardware; the
  paper's sampling guarantee is not identical to a greedy-only implementation.
- **Design decision informed:** Implement the project's greedy draft/verify/accept/reject
  algorithm as a separately tested path and report equality, acceptance, and latency
  rather than assume a speedup.

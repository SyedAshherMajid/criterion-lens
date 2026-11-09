# IEEE Report Outline, Demo Script, AI-Use Appendix

## A. Paper (IEEEtran conference, 2-column; target 8–10 pages + appendix)
Title: **CriterionLens: Evidence-Grounded Criterion-Level Clinical Trial Matching with a Fine-Tuned Small Decoder and Hand-Built Efficient Inference**
Author: Syed Ashher Majid, FAST-NUCES Islamabad, i230007@isb.nu.edu.pk (confirm the affiliation with the student).

| § | Section | Content | Figures / Tables (all `\ref`'d and interpreted) |
|---|---|---|---|
| — | Abstract + Index Terms | problem, method, headline test macro-F1/accuracy (base→FT), evidence F1, KV speed-up, spec acceptance/speed-up, app | — |
| I | Introduction & problem statement | trial screening burden; criterion-level formulation; contributions list (5 bullets: FT model, evidence pipeline, KV cache, tiled attention, speculative decoding + app) | Fig. 1 system overview |
| II | Related Work | eligibility matching (TrialGPT, SIGIR 2016 collection, LLM zero-shot matching); Transformers/RoPE/GQA/Qwen2.5; LoRA/QLoRA; KV cache & serving (Pope, vLLM); online softmax / memory-efficient / FlashAttention; speculative decoding (Leviathan, Chen). Each with limitation + link to our choice | Table I: technique ↔ source ↔ our use |
| III | Materials & Methods | A. dataset + audit + leakage policy; B. split (recipe, manifest); C. model selection + compute/memory; D. prompt/output schema, tokenization, truncation; E. fine-tuning objective + config; F. inference pipeline (parse, validate, fallback, explanation, aggregation rule); G. KV-cached decoding (algorithm box, positions/masking/growth); H. tiled online-softmax attention (algorithm box + derivation); I. greedy speculative decoding (algorithm box + invariant); J. application workflow | Table II data stats/split; Table III candidate models; Fig. 2 prompt/IO; Table IV hyper-parameters; Alg. 1 KV decode; Alg. 2 tiled attn; Alg. 3 spec decode; Fig. 3 app workflow |
| IV | Experimental Setup & Metrics | hardware/software versions; decoding settings; metric definitions incl. evidence empty-set conventions; timing methodology (warm-up, sync, repeats); test-seal protocol | Table V settings |
| V | Results & Discussion | A. training curves; B. before/after (val+test); C. per-class + confusion, inclusion vs exclusion; D. evidence; E. error analysis (≥8 cases table); F. KV cache correctness + benchmark; G. attention accuracy + perf (why slower than fused); H. speculative (equality, acceptance vs k, speed-up/slowdown explained); I. app screenshots | Fig. 4 curves; Table VI main results w/ CI; Fig. 5 confusion matrices; Table VII per-class; Table VIII evidence; Table IX error cases; Fig. 6 KV latency/throughput/memory; Fig. 7 cache bytes; Table X attention errors; Fig. 8 attention time/mem; Fig. 9 acceptance & speed-up vs k; Fig. 10 app screenshots |
| VI | Design decisions & failures | from DECISIONS.md: alternatives, rejected options, incidents (NaN, OOM, fp16 mismatches) | Table XI |
| VII | Limitations | 53 patients / small test set (wide CIs); SIGIR notes are short synthetic-style cases; single annotator labels; small model; template explanations; CPU latency; not a clinical tool | — |
| VIII | Conclusion & future work | Qwen3, constrained decoding, larger drafts/EAGLE, Triton kernel | — |
| — | References | verified BibTeX only | — |
| App. A | AI-use statement | see C | — |
| App. B | Reproducibility | commands, links, hashes | — |

Writing rules: no screenshots of tables, since tables are LaTeX generated from the JSON. Interpret every number ("FT improves NEI recall from x to y because…"). Report the model and decoding settings next to every comparison.

## B. Demo video script (target 6:00, max 7:00)
| Time | Show |
|---|---|
| 0:00–0:30 | Title, problem, disclaimer, architecture figure |
| 0:30–1:00 | Repo tour: split manifest, tests passing (`pytest -q`), results folder |
| 1:00–3:00 | Space app → paste an **unseen** synthetic note → fetch criteria by NCT ID (or paste) → run → walk through 3 criteria (decisive with evidence highlight, NEI/missing info, N/A) → overall preliminary outcome and its rule → export the HTML report and open it (provenance) |
| 3:00–3:30 | Invalid input: empty note, bad NCT ID, then recovery |
| 3:30–4:45 | Inference Lab: no-cache vs KV-cache vs speculative (k=4) live, token equality ✓, acceptance, tokens/s; Attention Lab live run with error ≈ 1e-6 |
| 4:45–5:45 | Key results: before/after table, confusion matrix, KV benchmark plot, speculative speed-up/slowdown explanation |
| 5:45–6:15 | Limitations + where the weights/paper are |

## C. AI-use appendix (template — make it truthful)
"AI assistants were used during this project: Claude (Anthropic) for requirements analysis and producing the design and execution plan; GitHub Copilot (agent mode) for implementing code from that plan, writing tests and drafting documentation. All generated code was reviewed, executed and tested by the author (see `tests/`); all experimental numbers were produced by scripts in the repository; all references were verified manually. The author can explain and modify every component. Libraries used: PyTorch, Hugging Face Transformers/PEFT/Datasets, scikit-learn, Gradio, (others)."

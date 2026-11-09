# Progress Log

`time (local) | phase.step | result | next`

- 2026-11-09 | planning | Claude: plan, env (.venv reusing Nastp libs), MiKTeX, Kaggle CLI, tokens verified, Kaggle T4 smoke test | Copilot starts at Phase 0.1
- 2026-11-09 22:05 +05:00 | 0.1 | Added package/layout scaffold, `.gitattributes`, and MIT license; verified package import and `.env` ignore. Pytest found no tests yet (exit 5) | Phase 0.2
- 2026-11-09 22:09 +05:00 | 0.2 | Added Python packaging, pinned core/app requirements, pytest/ruff config, and import smoke test; editable install, fast tests (1 passed), ruff, and import passed | Phase 0.3
- 2026-11-09 22:12 +05:00 | 0.4 | Implemented and tested the secret-safe `.env` loader; verified the local file parses without displaying values and remains git-ignored; credentials left untouched | Phase 0.5
- 2026-11-09 22:13 +05:00 | 0.5 | Retained the existing README skeleton for later result-backed completion; fast tests (4 passed) and Ruff passed | Phase 1.1
- 2026-11-09 22:20 +05:00 | 1.1 | Added 12 verified sources with summaries, limitations, and design implications; fast tests (4 passed), Ruff, and diff checks passed | Phase 1.2
- 2026-11-09 22:30 +05:00 | 1.2 | Added config-only model inspection and selection notes; specs JSON has 5 accessible configs and explicitly records 2 unavailable gated configs; tests (5 passed) and Ruff passed | Phase 2.1
- 2026-11-09 22:44 +05:00 | 2.1 | Implemented and tested the revision-pinned annotation loader; 1,015 source rows yielded 1,014 after dropping one empty criterion, 53 patients, and no leakage columns; cached locally (git-ignored) at revision `1cfcafde94a1560a33b4addc1638664fe28fc059` | Phase 2.2
- 2026-11-09 22:49 +05:00 | 2.2 | Added numbered-note parsing with contiguous-ID validation and pysbd/free-text splitting; decimals, abbreviations, malformed numbering, and empty input covered; fast tests (12 passed) and Ruff passed | Phase 2.3
- 2026-11-09 22:52 +05:00 | 2.3 | Implemented the fixed patient-level split and wrote JSON/CSV manifests at dataset revision `1cfcafde94a1560a33b4addc1638664fe28fc059`; counts are train 37/602, val 8/206, test 8/206; fast tests (12 passed) and Ruff passed | Phase 2.4
- 2026-11-09 23:05 +05:00 | 3.1 | Implemented the prompt builder ahead of Phase 2.4 because its leakage test depends on it; inclusion/exclusion schemas and target JSON validated; fast tests (16 passed) and Ruff passed | Phase 2.4
- 2026-11-09 23:12 +05:00 | 2.4 | Added deterministic split/disjointness/manifest tests and an end-to-end loader-to-prompt leakage test with a GPT-4 sentinel; fast tests (22 passed) and Ruff passed | Phase 2.5
- 2026-11-09 | 2.5 | Added the reproducible data audit, label/note-length figures, and exact Qwen2.5 chat prompt-plus-target length summaries; 1,014 annotations audited, max sequence length 634; documented zero label/type and evidence-range issues, dropped one empty criterion; fast tests (25 passed) and Ruff passed | Phase 3.2
- 2026-11-09 | 3.2 | Added completion-only tokenization with protected evidence/targets and structured truncation records; decoded labels match the exact target plus `<|im_end|>`; all 602 training rows fit within 640 tokens (max 634, zero truncations); fast tests (29 passed) and Ruff passed | Phase 3.3

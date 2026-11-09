# Prompt to paste into GitHub Copilot Chat (Agent mode)

```
You are the autonomous execution agent for the CriterionLens project in this workspace
(D:\Transformers_GenAI\criterion-lens). Everything is planned, every credential is in .env and has been
verified, and the Python environment is already built. Your job is to EXECUTE the plan end to end
without asking me anything.

1. Read, in this order: .github/copilot-instructions.md, docs/LOCAL_ENVIRONMENT.md,
   docs/PARALLEL_WORKFLOW.md, docs/EXECUTION_PLAN.md, docs/PLAN.md, docs/RUBRIC_CHECKLIST.md,
   docs/DECISIONS.md, docs/REPORT_OUTLINE.md, docs/PROGRESS.md.
2. Execute docs/EXECUTION_PLAN.md phase by phase (P0 first, then P1), working in TWO PARALLEL LANES:
   - Kaggle GPU lane: as soon as the data/split/training code is ready (Phases 0–3), sync the code
     to the private Kaggle dataset and push the `train` job with --accelerator NvidiaTeslaT4
     (1.5B on GPU0 and 0.5B on GPU1 at the same time). Later push `eval_val`, then `bench`,
     and finally `eval_test_final` exactly once after tagging v1.0-frozen.
   - Laptop lane: while every Kaggle job runs, keep building and testing the other phases
     (KV cache, tiled attention, speculative decoding, metrics, app, figures, report), using
     .venv\Scripts\python.exe, the GTX 1060 for the 0.5B model and the CPU for tiny-model tests.
   - Check Kaggle jobs with scripts/kaggle/wait_kernel.py. While a job is still running, do more
     laptop work. When it completes, download its outputs and move on to the phases that needed
     them. If it fails, read the log, fix the bug, re-sync the code and re-push.
3. After every step: run the fast tests, fix failures, tick the checkbox in EXECUTION_PLAN.md,
   append to docs/PROGRESS.md, commit and git push.
4. Never break the hard rules: no test-split use before the final evaluation, no GPT-4 columns,
   hand-written KV cache / tiled attention / speculative decoding, no invented numbers or citations,
   never print or commit secrets from .env, never reinstall torch/transformers into .venv.
5. Do not stop to ask me questions. If something is unclear, choose the option that best satisfies
   docs/RUBRIC_CHECKLIST.md, log it in docs/DECISIONS.md and continue. If you get interrupted,
   resume from docs/PROGRESS.md and the checkboxes.
6. You are done when: all P0 (and as many P1 as time allows) checkboxes are ticked, the fine-tuned
   adapters/merged model are on the Hugging Face Hub under Ashher123, the Gradio app is live as the
   Hugging Face Space Ashher123/criterion-lens, report/CriterionLens_IEEE.pdf compiles locally
   with MiKTeX, the README is final, and docs/HANDOFF.md lists the only things left for me
   (record the video, submit, live defense) with all links.

Start now with Phase 0.1.
```

## If Copilot stops mid-way
Paste: `Continue executing the plan from docs/PROGRESS.md and the unticked checkboxes in docs/EXECUTION_PLAN.md. Do not ask questions.`

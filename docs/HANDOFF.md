# CriterionLens Handoff

The official application is the locally hosted Gradio app. Start it on Windows
from the repository root with `run_app.bat`; it opens
`http://127.0.0.1:7860`. See the README for setup, first-use model downloads,
CPU/GPU behavior, and troubleshooting.

The Hugging Face Space was not created: in 2026, Hugging Face requires PRO for
free `cpu-basic` Gradio Spaces. The assignment permits a browser-accessible
locally hosted app. `space/` and `scripts/deploy_space.py` remain available if
PRO becomes available, but Space deployment is not required and should not be
retried as a prerequisite.

## Defense day

- [ ] Start the app with `run_app.bat` and confirm it opens at
  `http://127.0.0.1:7860`.
- [ ] Test one unseen **synthetic or appropriately de-identified** patient note
  with trial criteria; verify evidence, missing information, the preliminary
  outcome, and report export.
- [ ] Use `run_app.bat --share` only if the evaluator needs remote access. The
  link is temporary; do not submit real or identifiable patient information.

## Human-only remaining tasks

- Record the demo video.
- Submit the required materials.
- Attend and complete the live defense.

# Parallel Workflow: Kaggle GPU lane + Laptop lane

GPU work (training, 1.5B evaluation, headline benchmarks) runs on **Kaggle T4 GPUs**. At the same time, the laptop (GTX 1060 3 GB + CPU) does all coding, tests, 0.5B experiments, attention benchmarks, the app and the report. Kaggle jobs are **launched and collected from the laptop with the Kaggle CLI**, so the agent can drive both lanes without anyone clicking in the Kaggle website.

## Verified tooling (9 Nov 2026)
- Kaggle CLI **2.2.5**, installed as an isolated uv tool (it needs Python ≥ 3.11, so it is not in `.venv`): `C:\Users\ASUS\.local\bin\kaggle.exe`.
- Auth: `KAGGLE_API_TOKEN` (new `KGAT_` style) + `KAGGLE_USERNAME=syedashhermajid` in `.env`. Verified with `kaggle kernels list --mine`.
- The CLI reads the token from the **environment**. Load `.env` first. Helper (PowerShell):
  ```powershell
  Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object { $k,$v = $_ -split '=',2; Set-Item "env:$k" $v }
  kaggle kernels list --mine
  ```
  [COPILOT] also writes `scripts/kaggle/kaggle_env.py`, which loads `.env` and runs a kaggle subcommand with the token in its env. It **never echoes the token**.

## How a Kaggle job is run from the laptop
1. **Ship code as a private Kaggle dataset** (the GitHub repo is private, so Kaggle cannot `git clone` it):
   `scripts/kaggle/sync_code.py` copies `src/ scripts/ configs/ data/splits/ requirements.txt pyproject.toml` into `build/kaggle_code/`, writes `dataset-metadata.json` (`id: syedashhermajid/criterion-lens-code`, private), then runs `kaggle datasets create -p build/kaggle_code --dir-mode zip` the first time and `kaggle datasets version -p ... -m "<git sha>" --dir-mode zip` afterwards. Wait until `kaggle datasets status syedashhermajid/criterion-lens-code` reports ready.
2. **One folder per job** under `kaggle/jobs/<job>/`, containing `<job>.ipynb` (or `.py`) + `kernel-metadata.json`:
   ```json
   {"id": "syedashhermajid/criterion-lens-<job>", "title": "criterion-lens-<job>",
    "code_file": "<job>.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": true, "enable_gpu": true, "enable_internet": true,
    "dataset_sources": ["syedashhermajid/criterion-lens-code"],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}
   ```
   Check `kaggle kernels push --help` for an accelerator option. If one exists, request **T4 x2** (e.g. `NvidiaTeslaT4`). Every job's first cell prints `torch.cuda.get_device_name()`, the device count and `torch.cuda.get_arch_list()`, and **aborts** if the GPU is unsupported. If the CLI cannot select T4 and gives a P100 that fails, ask the student to open the kernel on kaggle.com once, set *Accelerator → GPU T4 x2*, and *Save & Run All*.
3. The job's first cells: `pip install -q -r /kaggle/input/criterion-lens-code/requirements.txt` (never torch), `pip install -e . --no-deps` from a copy in `/kaggle/working/code`, then run the scripts. Write all outputs to `/kaggle/working/out/`.
4. Launch: `kaggle kernels push -p kaggle/jobs/<job>`. Monitor: `kaggle kernels status syedashhermajid/criterion-lens-<job>` (poll every 5–10 min, not more often). Collect: `kaggle kernels output syedashhermajid/criterion-lens-<job> -p results/kaggle/<job>/`. Then copy the logs/metrics into `results/` and commit them. Adapters/weights go to `outputs/` (git-ignored).
5. **Pushing weights to the HF Hub happens from the laptop**, using the `.env` `HF_TOKEN` (`scripts/push_to_hub.py`). The HF token therefore never has to be stored on Kaggle (no Kaggle Secrets needed). Kaggle only downloads public Qwen models and the public dataset, which needs no token.

## Using both T4s in one Kaggle session
Kaggle "GPU T4 x2" gives two 16 GB GPUs. The training job runs both trainings **concurrently**:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/train_qwen25_1p5b.yaml > out/log_1p5b.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/train_qwen25_0p5b.yaml > out/log_0p5b.txt 2>&1 &
wait
```
Likewise, the eval/bench job runs the val evaluations on GPU 0 and the benchmarks on GPU 1. Kaggle may also allow more than one GPU session at a time: if a second `push` gets queued, just let it queue.

## Jobs (Kaggle lane)
| Job | Contents | Needs | Output → repo |
|---|---|---|---|
| `train` | LoRA 1.5B (GPU0) ∥ LoRA 0.5B (GPU1), per-epoch adapters, log_history | Phase 2–3 code | `outputs/train/*` (adapters), `results/train/*/log_history.json` |
| `eval_val` | base vs fine-tuned per-epoch on **val**, 1.5B (GPU0) ∥ 0.5B + SmolLM2 zero-shot (GPU1) | train done, Phase 5 code | `results/eval/val_*` |
| `bench` | KV-cache grid, speculative grid (k sweep, both drafts), attention perf on T4, tiled-attention model integration | Phases 6–8 code + merged models | `results/bench/*` |
| `eval_test_final` | **once**, after tag `v1.0-frozen`: test eval for base/FT 1.5B and 0.5B | everything frozen | `results/eval/test_*`, `TEST_ACCESS_LOG.md` |
| `sweep` (P1) | r/lr sweep on val | spare GPU time | `results/train/runs.csv` |

## Laptop lane (same time)
| While Kaggle runs… | Laptop does |
|---|---|
| `train` (~40–60 min) | Phase 6–8 code + tiny-model tests (CPU); Phase 5 metrics code; zero-shot **base 0.5B** val eval on the GTX 1060 (pipeline smoke test with real data) |
| `eval_val` | attention accuracy tests + CPU/GTX 1060 attention benchmarks; KV-cache & speculative runs with 0.5B on the GTX 1060 (second hardware point); app skeleton |
| `bench` | app (Phase 10), figure scripts, report sections that don't need numbers |
| `eval_test_final` | error-analysis tooling, README, report Methods |

**Rule:** the laptop never runs the 1.5B in fp16 on the GPU (it does not fit). Running the 1.5B in CPU fp32 is acceptable for app demos only.

## Kaggle quota awareness
About 30 GPU-hours/week, 12 h max per session. Estimated use: train ~1 h, eval_val ~1 h, bench ~1.5 h, test ~0.5 h. That leaves plenty of room for one retry of each job.

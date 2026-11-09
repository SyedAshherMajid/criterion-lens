# Local Environment (already set up — do NOT reinstall)

Audited on 9 Nov 2026 on the student's laptop (Windows 11, GTX 1060 3 GB). The goal is to **reuse libraries that are already installed** rather than download them again.

## Project virtualenv: `criterion-lens/.venv` (ready)
- Base interpreter: `C:\Users\ASUS\AppData\Local\Programs\Python\Python310\python.exe` (Python **3.10.11**).
- **Shared packages:** `.venv\Lib\site-packages\_shared_nastp_ml.pth` adds
  `D:\Nastp_Project\venv\Lib\site-packages` to the path. The heavy ML stack is therefore reused from that existing environment, with no download and no change to that project.
- Installed locally in `.venv` (they were missing): `gradio 5.50.0`, `matplotlib 3.10.9`, `pytest 9.2.0`, `pysbd`, `ruff` and their dependencies. Conflicting duplicates (`huggingface_hub 1.x`, `hf-xet`, `fsspec`) were removed so that the shared versions are used.

| Package | Version | Source |
|---|---|---|
| torch | 2.6.0+cu124 (**CUDA works on the GTX 1060**, arch list includes sm_61) | shared (Nastp) |
| transformers | 4.57.6 (has `AttentionInterface`, `DynamicCache.layers[i].keys/values` API) | shared |
| peft | 0.19.1 | shared |
| accelerate | 1.14.0 | shared |
| datasets | 5.0.0 | shared |
| huggingface_hub | 0.36.2 | shared |
| tokenizers / safetensors / sentencepiece | 0.22.2 / 0.8.0 / 0.2.1 | shared |
| numpy / pandas / scikit-learn | 2.2.6 / 2.3.3 / 1.7.2 | shared |
| pyyaml / jinja2 / psutil / requests / nltk | yes | shared |
| gradio | 5.50.0 | local |
| matplotlib / pytest / pysbd / ruff | 3.10.9 / 9.2.0 / yes / yes | local |

**Activate:** `D:\Transformers_GenAI\criterion-lens\.venv\Scripts\Activate.ps1` (PowerShell), or select `.venv\Scripts\python.exe` as the VS Code interpreter.

### Rules for the agent
1. **Never** run `pip install torch/transformers/peft/datasets/accelerate/huggingface_hub` into `.venv`. They come from the shared path, and reinstalling would shadow them or create conflicts.
2. To add a new *small* package: `uv pip install --python .venv\Scripts\python.exe <pkg>`. Afterwards run `.venv\Scripts\python.exe -m pip check` and the import smoke test below. If the new package pulls in `huggingface_hub>=1.0` or a different `fsspec`, uninstall those duplicates from `.venv` again.
3. Smoke test:
   ```powershell
   .venv\Scripts\python.exe -c "import torch,transformers,peft,datasets,gradio; print(torch.cuda.is_available(), transformers.__version__)"
   ```
4. The project depends on `D:\Nastp_Project\venv` staying in place. If it is ever deleted, rebuild with `uv pip install -r requirements.txt` (pinned to the same versions).

## GPU facts (changes vs. the first plan)
- The GTX 1060 3 GB **is usable** with this torch build (cu124 still ships sm_61 kernels).
- It fits: Qwen2.5-0.5B in fp16 (~1 GB) + KV cache; the attention benchmarks; the tiny-model tests; app inference with the **0.5B** model.
- It does **not** fit: Qwen2.5-1.5B fp16 (3.1 GB of weights > 3 GB VRAM). Run the 1.5B on CPU locally (fp32, ~6.2 GB RAM) or on Kaggle T4. Training is always done on **Kaggle T4**.
- Pascal has no fast fp16 tensor cores. Local fp16 timings are therefore not representative, so the headline benchmarks come from the T4 and local numbers are reported as a second hardware point.

## Other tools found
| Tool | Path / status |
|---|---|
| Kaggle CLI 2.2.5 | `C:\Users\ASUS\.local\bin\kaggle.exe`: an isolated uv tool on Python 3.14 (it needs ≥ 3.11, so it is **not** in `.venv`). Auth from `.env` `KAGGLE_API_TOKEN` (verified, account `syedashher`). Usage: `docs/PARALLEL_WORKFLOW.md` |
| Accounts | GitHub `SyedAshherMajid` · Hugging Face **`Ashher123`** (all HF repo ids use this) · Kaggle `syedashher` |
| uv | `C:\Users\ASUS\.local\bin\uv.exe` (global wheel cache `C:\Users\ASUS\AppData\Local\uv\cache`) |
| git 2.51, gh 2.102 (logged in as SyedAshherMajid) | ready. **Fixed:** this repo's local git config uses `gh auth git-credential`, so plain `git push` works here (verified). |
| **MiKTeX 25.12 (LaTeX)** | installed by Claude: `C:\Users\ASUS\AppData\Local\Programs\MiKTeX\miktex\bin\x64\` (`pdflatex`, `bibtex`). `IEEEtran.cls` present; on-the-fly package install is enabled. Test compile with IEEEtran + amsmath + algorithm + algpseudocode + tikz + booktabs + hyperref succeeded. **No Overleaf needed.** Open a new terminal so the PATH update applies. |
| HF cache | `C:\Users\ASUS\.cache\huggingface\hub`, shared by all projects. Qwen2.5 models and the dataset are not cached yet: they download **once** and are reused afterwards. |
| Other Python envs (not used) | `D:\Anaconda3` (3.13, no torch), `D:\i230007_GenAi_A1\.venv` (3.10, torch 2.7.1+cu118, no transformers), `D:\Youtube_videos_Generation\.venv-*` (old transformers 4.46), `D:\Assesment\.venv*` (torch only). The Nastp env was chosen because it alone has the whole HF fine-tuning stack. |

## Compile the report locally
```powershell
cd D:\Transformers_GenAI\criterion-lens\report
pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
```
(or `latexmk -pdf main` if Perl is available).

## Hugging Face token
- Put it in `criterion-lens/.env` as `HF_TOKEN=hf_...` (the file exists and is **git-ignored**; `.env.example` shows the format). Local scripts read it via `criterion_lens.utils.env.load_env()`.
- Kaggle **cannot read your local `.env`**. In the Kaggle notebook, open *Add-ons → Secrets → Add secret*, name it `HF_TOKEN`, paste the same token, and tick it for the notebook. The notebook's first cell reads the Kaggle secret and falls back to the environment variable.

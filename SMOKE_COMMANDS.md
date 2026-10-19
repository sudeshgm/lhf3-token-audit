# Smoke commands (GPU machine)

Only the GPU-required steps. Datasets are NOT downloaded here: the 20-sample smoke subsets
are committed in `data/smoke/*.parquet` (built from the lmms-lab parquet shards on the CPU box).
The repo is private, so the GPU box needs GitHub access as sudeshgm (`gh auth login` once, or a PAT
at the git password prompt). Needs ~15 GB free disk for the model (fp16 safetensors).

```bash
# 1) GPU check (versions printed after install, step 3)
nvidia-smi

# 2) get the repo + install (pinned)
git clone https://github.com/sudeshgm/lhf3-token-audit && cd lhf3-token-audit
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python -c "import sys,torch,transformers;print('python',sys.version.split()[0],'| torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'| transformers',transformers.__version__,'| gpu',torch.cuda.get_device_name(0))"

# 3) model weights (~14 GB, safetensors only)
huggingface-cli download llava-hf/llava-1.5-7b-hf --exclude "*.bin"

# 4) smoke: 20 samples per benchmark (POPE: 20 per split) x {full-576, random k=58 seed0, keep_last k=58}
#    prints visual tokens seen by the LLM per sample, scores, s/sample, peak VRAM; ends with a summary table
python -m pytest -q tests && bash scripts/smoke.sh 2>&1 | tee smoke_output.txt
```

Paste back: the version line, the final summary table, and any error. Per-run JSONs land in
`results/smoke/<bench>/<arm>_k<k>_seed<s|NA>.json`.

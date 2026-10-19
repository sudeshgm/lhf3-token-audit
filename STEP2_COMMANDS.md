# Step 2 commands (Azure T4, same env as the smoke)

The TextVQA gate is built into `scripts/sweep.py` and runs first in the same process, so the model loads
only once. It runs TextVQA full-576 on the first 200 val samples and exits with code 2 before any sweep config
if acc < 0.53. Those 200 samples are reused by the full-576 TextVQA run. The sweep can be resumed: just rerun the
same nohup line. Finished configs are skipped and a half-done config continues from its `.partial.jsonl`.
(`bash scripts/gate_textvqa.sh` runs the gate alone if you ever want it separately.)

```bash
# activate the env used for the smoke first (e.g. conda activate azureml_py38)
export USE_TF=0 TRANSFORMERS_NO_TF=1
export HF_HOME=/home/azureuser/cloudfiles/code/Users/miu3kor/models/hf_cache
export MODEL=/home/azureuser/cloudfiles/code/Users/miu3kor/models/llava-1.5-7b-hf
cd /home/azureuser/cloudfiles/code/Users/miu3kor/lhf3-token-audit
git pull
mkdir -p logs "$HF_HOME"
nohup python scripts/sweep.py > logs/step2_sweep.log 2>&1 &
echo $! > logs/step2_sweep.pid
sleep 60; grep -E "GATE|loaded|Error|Traceback" logs/step2_sweep.log
tail -f logs/step2_sweep.log        # Ctrl-C stops tailing only; the sweep keeps running
```

Check later / after it finishes:
```bash
grep -E "GATE|DONE|SKIP|ALL DONE|Error|Traceback" logs/step2_sweep.log | cut -c1-200
ps -p $(cat logs/step2_sweep.pid) >/dev/null && echo running || echo stopped
python scripts/summarize_step2.py results/step2
```
Paste back: the `GATE` lines (incl. the 3 `GATE PROMPT` lines), then the full `summarize_step2.py` output when it is done.
If the gate FAILs (exit 2), paste the gate lines plus `head -c 3000 results/step2/gate/textvqa_full576_first200.json`.

Disk: about 1.25 GB of dataset parquet goes into `$HF_HOME/hub` (GQA testdev-balanced 68 MB, POPE 255 MB,
TextVQA val 920 MB, with no `datasets` Arrow copy), plus about 150 MB of result JSONs in `results/step2`.
Runtime estimate on a T4, from the smoke s/sample: full-576 about 3 h, the four k=58/144 main configs about 5.5 h,
the seed-1/2 runs on the 1k subsets about 1.3 h, so about 10 h in total.

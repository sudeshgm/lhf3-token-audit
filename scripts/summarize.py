"""Print a compact table of every result JSON under a directory."""
import glob
import json
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "results"
print(f"{'bench':8} {'arm':12} {'k':>4} {'seed':>4} {'n':>4} {'vis_tok':>8} {'score':>40} {'s/sample':>8} {'peakGiB':>7}")
for f in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
    d = json.load(open(f))
    m = d["metrics"]
    if d["bench"] == "pope":
        sc = " ".join(f"{c[:3]}:F1={m[c]['f1']:.3f}/acc={m[c]['acc']:.3f}" for c in ("random", "popular", "adversarial") if c in m)
    else:
        sc = f"acc={m['acc']:.4f}"
    peak = d.get("peak_vram_gib")
    print(f"{d['bench']:8} {d['arm']:12} {d['k']:>4} {str(d['seed']):>4} {d['n']:>4} {str(d['visual_tokens_to_llm']):>8} "
          f"{sc:>40} {d['sec_per_sample_mean']:8.3f} {('%.2f' % peak) if peak else '-':>7}")

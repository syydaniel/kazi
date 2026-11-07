"""Score the English with/without-Kazi test and write the film's bench data.

Same rules as the Chinese test (skill/bench/score.py + offsets.py):
- kazi_check on each run's video against timeline.json (one shared word timeline)
- offset: within keyword -3 s .. +2 s, the frame with the biggest change (gray 480x270), minus the keyword time
- energy: per-0.05 s max of that change curve, normalised by the run's own 99th percentile (for the chart)
Usage (from the repo root): python bench/planets_en/score_en.py [--out bench/planets_en/results.json]
Reads bench/planets_en/videos/<run>.mp4 and bench/planets_en/timeline.json; needs ffmpeg, numpy and tools/kazi_check.py.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
ap.add_argument("--out", default="")
ap.add_argument("--timeline", default=str(HERE/"timeline.json"))
a = ap.parse_args()
TL = json.loads(Path(a.timeline).read_text("utf-8"))
KW = [(k["t"], k["word"]) for k in TL["keywords"]]
W, H, FPS = 480, 270, 60
out = {}
for name in ["opus_base", "opus_kazi", "sonnet_base", "sonnet_kazi"]:
    v = HERE/"videos"/f"{name}.mp4"
    if not v.exists(): print(f"{name:12s} no {v.name}"); continue
    j = HERE/"runs"/name/"check.json"
    subprocess.run([sys.executable, str(ROOT/"tools"/"kazi_check.py"), str(v), "--timeline", a.timeline, "--json", str(j)], stdout=subprocess.DEVNULL, check=False)
    c = json.loads(j.read_text("utf-8"))
    c = json.loads(json.dumps(c).replace(str(HERE) + "/", ""))  # keep the saved check free of local paths
    j.write_text(json.dumps(c, ensure_ascii=False, indent=1), "utf-8")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(v), "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.int16)
    d = np.r_[0, np.abs(np.diff(f, axis=0)).mean((1, 2))]
    kws = []
    for (t, w), det in zip(KW, c["detail"]):
        a0, a1 = max(1, int((t - 3)*FPS)), min(len(d), int((t + 2)*FPS))
        k = a0 + int(np.argmax(d[a0:a1]))
        kws.append({"w": w, "t": round(t, 2), "ok": bool(det["hit"]), "off": round(k/FPS - t, 2)})
    step = 3
    e = d[: len(d)//step*step].reshape(-1, step).max(1)
    e = np.clip(e/(np.percentile(e, 99) + 1e-9), 0, 1.5)
    out[name] = {"hit": c["hits"], "n": c["keywords"], "chance": round(100*c["chance"]), "keywords": kws,
                 "energy_dt": step/FPS, "energy": [round(float(x), 3) for x in e]}
    print(f"{name:12s} {c['hits']}/{c['keywords']}  chance {round(100*c['chance'])}%  " + "  ".join(f"{x['w']}{x['off']:+.2f}{'*' if x['ok'] else ''}" for x in kws))
if a.out:
    lines = [{"start": l["start"], "end": l["end"], "text": l["show"]} for l in TL["lines"]]
    about = ("Eight-planets test, English rerun (2026-11-07). Four videos in videos/, shared narration in narration.wav "
             "(Yiyang's cloned voice, English). All four runs got the same art kit and layout, so they differ only in motion and timing. "
             "energy: per energy_dt seconds, how much the picture changes (0..1.5, normalised per run); off: where the biggest change within "
             "keyword -3 s .. +2 s sits relative to the keyword (seconds, negative = early).")
    Path(a.out).write_text(json.dumps({**out, "_lines": lines, "_about": about}, ensure_ascii=False), "utf-8")
    print("wrote", a.out)

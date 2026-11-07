"""Snap keyword onsets out of silences.

The forced aligner works on an 80 ms grid and sometimes puts a word's start inside the pause in front
of it (a comma or colon pause: "home, Earth", "side: Uranus", "all: Neptune"). For every keyword, if the
20 ms window at the aligned start is silent (< -45 dBFS), move the start forward to the first window that
is clearly voiced (> -38 dBFS). The word entry in lines[].words/cap is moved with it.
Writes timeline.json in place and keeps the aligner's original as timeline_aligned.json.
"""
import json, shutil
from pathlib import Path

import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
tlp = ROOT / "timeline.json"; raw = ROOT / "timeline_aligned.json"
if not raw.exists(): shutil.copy(tlp, raw)
tl = json.loads(raw.read_text("utf-8"))
x, sr = sf.read(ROOT / "input" / "narration.wav")
win = int(0.02 * sr)


def db(t):
    i = int(t * sr); s = x[i:i + win]
    return 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9)


for k in tl["keywords"]:
    t = k["t"]
    if db(t) < -45:
        u = t
        while u < t + 0.35 and db(u) < -38: u += 0.005
        new = round(u, 3)
        print(f"{k['word']:14s} {t:.3f} -> {new:.3f}  (+{(new - t) * 1000:.0f} ms)")
        k["t"] = new
        line = next(l for l in tl["lines"] if l["i"] == k["line"])
        first = k["word"].split()[0].lower()
        for w in line["words"]:
            if abs(w["s"] - t) < 1e-6 and w["u"].lower().strip(".,:;!?").startswith(first): w["s"] = new
        for c in line.get("cap", []):
            if abs(c["s"] - t) < 1e-6: c["s"] = new
    else:
        print(f"{k['word']:14s} {t:.3f}    ok ({db(t):.0f} dB)")
tlp.write_text(json.dumps(tl, ensure_ascii=False, indent=1), "utf-8")

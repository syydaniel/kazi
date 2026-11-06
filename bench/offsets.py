"""每个关键词附近，画面变化最大的那一下离关键词差几秒（负数是画面抢在前面，正数是落在后面）。
kazi_check 只回答"卡没卡上"；这个脚本回答"差了多少"，方便看清楚没卡上的那几遍是早了还是晚了。
找法：关键词前 3 秒到后 2 秒里，逐帧变化量（灰度、480×270）最大的那一帧。
用法：python bench/offsets.py <测试根目录> [--topic planets] [--json out.json]
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--topic", default="planets"); ap.add_argument("--json", default="")
a = ap.parse_args()
tl = json.loads((HERE / a.topic / "timeline.json").read_text("utf-8"))
kws = []
for l in tl["lines"]:
    cs, i = l["cap"], 0
    while i < len(cs):
        if cs[i].get("k"):
            j = i
            while j < len(cs) and cs[j].get("k"): j += 1
            kws.append((cs[i]["s"], "".join(c["c"] for c in cs[i:j]))); i = j
        else: i += 1
W, H = 480, 270
out = {}
for name in ["opus_base", "opus_kazi", "sonnet_base", "sonnet_kazi"]:
    v = Path(a.root) / name / "out" / "final.mp4"
    if not v.exists(): continue
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(v), "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.int16)
    d = np.r_[0, np.abs(np.diff(f, axis=0)).mean((1, 2))]
    fps = 60
    rows = []
    for t, w in kws:
        a0, a1 = max(1, int((t - 3) * fps)), min(len(d), int((t + 2) * fps))
        k = a0 + int(np.argmax(d[a0:a1]))
        rows.append({"w": w, "t": round(t, 2), "peak": round(k / fps, 2), "off": round(k / fps - t, 2)})
    # 画面变化曲线（每 0.05 秒取最大值，按这一遍自己的 99 分位归一），片子里画"心电图"用
    step = 3
    e = d[: len(d) // step * step].reshape(-1, step).max(1)
    e = np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1.5)
    out[name] = {"rows": rows, "energy_dt": step / fps, "energy": [round(float(x), 3) for x in e]}
    print(f"{name:12s} " + "  ".join(f"{r['w']}{r['off']:+.2f}" for r in rows))
if a.json: Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")

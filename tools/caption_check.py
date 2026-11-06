"""字幕和配音对不对得上：拿真实音频量每条字幕出现、消失的时机，以及逐字变亮的时机。

为什么要量：字幕时间来自对齐表，可对齐表在句首常常不准（把句首静音算进第一个字），
配音模型每句前后的静音也长短不一，结果字幕会比声音早出现、字比声音先亮，看着就"对不齐"。

量什么（都和音频里实际开口、收声的时刻比，音量比全片响的部分低 32 dB 以下算没声音）：
1. 出现：每条字幕出现的时刻 - 它第一个字实际开口的时刻。建议 -0.15 到 0 秒（先半拍出现，正好读完淡入）。
2. 逐字变亮：每个停顿之后的第一个字，对齐表的开始 - 实际开口。建议在 ±0.08 秒以内。
3. 停留：字幕消失 - 最后一个字收声。太短来不及读完，太长会压到下一句的空白里。
4. 太短的字幕：显示不到 0.8 秒的条目。

用法：python tools/caption_check.py timeline.json narration.wav [--show -0.1] [--hold 0.7] [--maxc 16]
--show/--hold/--maxc 要和画面里字幕组件的设置一致（starter 里的 Caption 默认就是这几个值）。
"""
import argparse, json, re
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("timeline"); ap.add_argument("wav")
ap.add_argument("--show", type=float, default=-0.1, help="字幕比第一个字早多少秒出现（负数是提前）")
ap.add_argument("--hold", type=float, default=0.7, help="最后一个字念完后最多再停留多少秒")
ap.add_argument("--maxc", type=int, default=16)
a = ap.parse_args()

T = json.load(open(a.timeline, encoding="utf-8"))
y, sr = sf.read(a.wav, dtype="float32", always_2d=True); y = y.mean(1)
HOP = 0.01; h = int(HOP*sr); n = len(y)//h
r = np.sqrt(np.mean(y[:n*h].reshape(n, h)**2, 1) + 1e-12); db = 20*np.log10(r/np.percentile(r, 95))
loud = np.convolve(db > -32, np.ones(3), "same") >= 2

BREAK = set("，。？！；：、,.?!;:")
width = lambda cs: sum(0.55 if re.match(r"[A-Za-z0-9 .\-']", c["c"]) else 0 if c["c"] in BREAK else 1 for c in cs)
chunks = []
for l in T["lines"]:
    if not l.get("real"): continue
    phrases, cur = [], []
    for c in l["cap"]:
        cur.append(c)
        if c["c"] in BREAK: phrases.append(cur); cur = []
    if cur: phrases.append(cur)
    acc = []
    def flush():
        global acc
        if acc: chunks.append({"i": l["i"], "chars": [c for c in acc if c["c"] not in BREAK]})
        acc = []
    for p in phrases:
        if width(acc) + width(p) <= a.maxc: acc = acc + p
        else:
            flush()
            if width(p) <= a.maxc: acc = p
            else:
                for k in range(0, len(p), a.maxc): acc = p[k:k+a.maxc]; flush()
    flush()
for k, c in enumerate(chunks):
    c["show"] = c["chars"][0]["s"] + a.show; c["hide"] = c["chars"][-1]["e"] + a.hold
    # 和字幕组件一致：停顿很短时，上一条等最后一个字念完再换，新的一条最晚在开口前 0.02 秒出现
    if k > 0 and chunks[k-1]["chars"][-1]["e"] + 0.03 > c["show"]:
        c["show"] = min(chunks[k-1]["chars"][-1]["e"] + 0.03, c["chars"][0]["s"] - 0.02)
for c, d in zip(chunks, chunks[1:]):
    c["hide"] = min(c["hide"], d["show"])
    if d["show"] - c["hide"] < 0.25: c["hide"] = d["show"]   # 两条之间只空不到 0.25 秒：不闪空，直接接上（和画面里的 Caption 组件一致）


def onset_near(t, back=0.25, fwd=0.25):
    """t 附近实际开口的时刻：从 t-back 往后找第一段持续有声的位置。"""
    i0, i1 = max(0, int((t-back)/HOP)), min(n, int((t+fwd)/HOP))
    # 先退到静音里，再往后找开口
    k = i0
    while k < i1 and loud[k]: k += 1
    while k < i1 and not loud[k]: k += 1
    return k*HOP if k < i1 else None


def offset_near(t, back=0.25, fwd=0.3):
    i0, i1 = max(0, int((t-back)/HOP)), min(n, int((t+fwd)/HOP))
    k = i1 - 1
    while k > i0 and not loud[k]: k -= 1
    return (k+1)*HOP if k > i0 else None


show_d, hold_d, short, light = [], [], [], []
for k, c in enumerate(chunks):
    first = c["chars"][0]
    # 只看前面是停顿的字幕（句首、逗号后），这时开口时刻量得准
    on = onset_near(first["s"])
    if on is not None: show_d.append((round(c["show"] - on, 3), c["i"], "".join(x["c"] for x in c["chars"])[:8]))
    nxt = chunks[k+1]["chars"][0]["s"] if k + 1 < len(chunks) else c["chars"][-1]["e"] + 1
    off = offset_near(c["chars"][-1]["e"], fwd=max(0.02, min(0.3, nxt - 0.03 - c["chars"][-1]["e"])))
    if off is not None: hold_d.append((round(c["hide"] - off, 3), c["i"]))
    if c["hide"] - c["show"] < 0.8: short.append((c["i"], "".join(x["c"] for x in c["chars"]), round(c["hide"] - c["show"], 2)))
    on2 = onset_near(first["s"], back=0.15, fwd=0.15)
    if on2 is not None: light.append(first["s"] - on2)

sd = np.array([x[0] for x in show_d]); hd = np.array([x[0] for x in hold_d]); ld = np.array(light)
print(f"字幕 {len(chunks)} 条")
print(f"出现 - 开口：中位 {np.median(sd):+.2f}s，p10 {np.percentile(sd,10):+.2f}，p90 {np.percentile(sd,90):+.2f}；早于 0.25 秒的 {int((sd < -0.25).sum())} 条，晚于开口的 {int((sd > 0.02).sum())} 条")
print(f"逐字变亮（每条第一个字）对齐表 - 开口：中位 {np.median(ld):+.3f}s，|偏差|>0.08 秒的 {int((abs(ld) > 0.08).sum())} 条")
print(f"消失 - 收声：中位 {np.median(hd):+.2f}s，p10 {np.percentile(hd,10):+.2f}；收声前就消失的 {int((hd < 0).sum())} 条")
early = sorted([x for x in hold_d if x[0] < 0])
print("收声前就消失：", early[:20])
print(f"显示不到 0.8 秒的 {len(short)} 条：", short[:10])
bad = sorted(show_d, key=lambda x: x[0])[:5] + sorted(show_d, key=lambda x: -x[0])[:5]
print("出现时机最偏的：", bad)

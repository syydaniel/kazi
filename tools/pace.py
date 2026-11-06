"""停顿规整：让每处停顿听起来都是该停的长度。

为什么要单独做：配音模型每句开头结尾自带的静音长短不一（0 到 0.5 秒），句子里的逗号有时一口气念过去，有时又拖很久。
稿子里写"这句后面停 0.3 秒"，实际听到的空白可能是 0.2 秒，也可能是 1 秒，有的段落就显得太挤，有的又太松。

做三件事，都只动静音，不动念出来的声音：
1. 句首句尾：按音量找出真正开口和收声的位置，前面只留 0.04 秒、后面只留 0.08 秒静音，记下来交给 timeline.py。
   timeline.py 加 --speech-gap 后，稿子里的"| 0.5"就是"这句收声到下一句开口，听到的空白是 0.5 秒"。
2. 句内标点：逗号、顿号、冒号、分号、句号、问号处，量出实际的静音长度，短了就在静音最深处补一段，长了就从静音中间剪掉一段。
3. 字的时间跟着挪：补了多少、剪了多少，后面每个字的对齐时间就加减多少，不用重新对齐。

用法：python tools/pace.py vo/fast vo/paced [--report vo/paced_report.tsv]
之后：python tools/timeline.py vo/script.txt vo/paced <timeline.json> --wav <narration.wav> --speech-gap
"""
import argparse, hashlib, json, re, shutil
from pathlib import Path
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--report", default="")
ap.add_argument("--head", type=float, default=0.04); ap.add_argument("--tail", type=float, default=0.08)
a = ap.parse_args()
SRC, DST = Path(a.src), Path(a.dst)
for d in ("lines", "align"): (DST/d).mkdir(parents=True, exist_ok=True)
key = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-']*|[㐀-鿿]")
# 句内标点处的静音（秒，按最终语速）：最短，最长
RULE = {"，": (0.14, 0.32), ",": (0.14, 0.32), "、": (0.06, 0.18), "：": (0.16, 0.36), "；": (0.24, 0.42),
        "。": (0.30, 0.48), "？": (0.30, 0.48), "！": (0.30, 0.48)}
HOP = 0.01
THR_IN = -32.0  # 句内停顿：换气声听起来也是"停"，所以门槛放宽，比响的部分低 32 dB 就算
THR = -40.0  # 比全句响的部分（95 分位）低 40 dB 以下算静音；量静音、找开口收声都用同一个标准
FADE = 0.008


def frames_db(y, sr):
    h = int(HOP*sr); n = len(y)//h
    r = np.sqrt(np.mean(y[:n*h].reshape(n, h)**2, axis=1) + 1e-12)
    ref = np.percentile(r, 95) + 1e-9
    return 20*np.log10(r/ref)


def silent_run(db, t0, t1, thr=THR):
    """t0..t1 之间最长的一段静音：返回 (开始秒, 结束秒)；没有静音就返回音量最低的那一帧（长度 0）。"""
    i0, i1 = max(0, int(t0/HOP)), min(len(db), int(np.ceil(t1/HOP)))
    if i1 <= i0: return (t0, t0)
    seg = db[i0:i1] < thr
    best, cur, bs = (0, 0), 0, 0
    for k, s in enumerate(seg):
        if s:
            if cur == 0: bs = k
            cur += 1
            if cur > best[1]-best[0]: best = (bs, k+1)
        else: cur = 0
    if best[1] > best[0]: return ((i0+best[0])*HOP, (i0+best[1])*HOP)
    m = i0 + int(np.argmin(db[i0:i1])); return (m*HOP + HOP/2, m*HOP + HOP/2)


def fade_join(y, cut_s, cut_e, insert, sr):
    """把 y 的 [cut_s, cut_e) 换成 insert 长度的静音，接口处做很短的淡入淡出，避免咔哒声。"""
    a_, b_ = int(cut_s*sr), int(cut_e*sr); f = int(FADE*sr)
    left, right = y[:a_].copy(), y[b_:].copy()
    if len(left) > f: left[-f:] *= np.linspace(1, 0, f)
    if len(right) > f: right[:f] *= np.linspace(0, 1, f)
    return np.concatenate([left, np.zeros(int(round(insert*sr)), np.float32), right])


rows, n = [], 0
for meta in sorted((SRC/"lines").glob("L*.json")):
    d = json.loads(meta.read_text("utf-8")); k = key(d["say"])
    y, sr = sf.read(meta.with_suffix(".wav"), dtype="float32")
    if y.ndim > 1: y = y[:, 0]
    alp = SRC/"align"/f"{k}.json"
    al = json.loads(alp.read_text("utf-8")) if alp.exists() else []
    say = d["say"]
    db = frames_db(y, sr)
    # ---- 句内标点 ----
    # 对齐结果按顺序对上念法里的单位，找出每个标点前后的两个字
    us = [(m.group(0), m.start(), m.end()) for m in WORD.finditer(say)]
    times, j = [None]*len(us), 0
    for q, (u, _, _) in enumerate(us):
        for jj in range(j, min(j+4, len(al))):
            if al[jj]["t"].lower().strip() == u.lower(): times[q] = jj; j = jj+1; break
    edits = []  # (静音开始, 静音结束, 换成多长)
    for q in range(len(us)-1):
        between = say[us[q][2]:us[q+1][1]].strip()
        p = between[:1]
        if p not in RULE or times[q] is None or times[q+1] is None: continue
        lo, hi = RULE[p]
        e0, s1 = al[times[q]]["e"], al[times[q+1]]["s"]
        cs, ce = silent_run(db, min(e0, s1) - 0.10, max(e0, s1) + 0.10, thr=THR_IN)
        sil = ce - cs
        # 补和剪都落在这段静音里最安静的那一帧附近，换气声尽量不被切开
        i0, i1 = int(cs/HOP), max(int(cs/HOP)+1, int(np.ceil(ce/HOP)))
        quiet = (i0 + int(np.argmin(db[i0:i1])))*HOP + HOP/2 if i1 <= len(db) else (cs+ce)/2
        if sil < lo - 0.02:
            edits.append((quiet, quiet, lo - sil, sil, p))
        elif sil > hi + 0.02:
            cut = sil - hi; c0_ = min(max(cs, quiet - cut/2), ce - cut); edits.append((c0_, c0_ + cut, 0.0, sil, p))
    # 从后往前改，前面的位置不受影响
    shifts = []
    for cs, ce, ins, sil, p in sorted(edits, key=lambda x: -x[0]):
        y = fade_join(y, cs, ce, ins, sr)
        shifts.append((cs, ins - (ce - cs)))
        rows.append((meta.stem, p, round(sil, 3), round(sil + ins - (ce - cs), 3)))

    def moved(t):
        return t + sum(dl for at, dl in shifts if t > at + 1e-4)
    al2 = [{**x, "s": round(moved(x["s"]), 3), "e": round(moved(x["e"]), 3)} for x in al]
    # 对齐器有时把标点处的静音算进下一个字（字幕会提前出现、字提前变亮），按实际静音把字的边界吸过去
    db2 = frames_db(y, sr)
    for q in range(len(us)-1):
        if say[us[q][2]:us[q+1][1]].strip()[:1] not in RULE or times[q] is None or times[q+1] is None: continue
        A, B = al2[times[q]], al2[times[q+1]]
        cs, ce = silent_run(db2, min(A["e"], B["s"]) - 0.15, max(A["e"], B["s"]) + 0.15, thr=THR_IN)
        if ce - cs < 0.06: continue
        B["s"] = round(min(max(B["s"], ce - 0.02), B["e"] - 0.03), 3)
        A["e"] = round(max(min(A["e"], cs + 0.02), A["s"] + 0.03), 3)
    # ---- 句首句尾 ----
    db = frames_db(y, sr)
    loud = db > THR
    run = np.convolve(loud, np.ones(3), "same") >= 3  # 连续 3 帧都响才算开口，免得被一下咔哒声骗
    on = np.nonzero(run)[0]; off = np.nonzero(loud)[0]
    s_on = max(0, on[0]-1)*HOP if len(on) else 0.0
    s_off = (off[-1]+1)*HOP if len(off) else len(y)/sr
    c0 = max(0.0, s_on - a.head); c1 = min(len(y)/sr, s_off + a.tail)
    y = y[int(c0*sr):int(c1*sr)].copy()
    f = int(0.005*sr); y[:f] *= np.linspace(0, 1, f)
    f = int(min(0.05, c1 - s_off)*sr)
    if f > 0: y[-f:] *= np.linspace(1, 0, f)
    al2 = [{**x, "s": round(max(0, x["s"]-c0), 3), "e": round(max(0, x["e"]-c0), 3)} for x in al2]
    head = round(s_on - c0, 3); tail = round(c1 - s_off, 3)
    # 对齐器常把第一个字的开始算到 0（句首静音里），用实际开口的位置修正；最后一个字同理
    if al2:
        # 第一个字从真正开口算起（用句内停顿的门槛，换气声不算开口）；对齐器差得不多就不动
        dbt = frames_db(y, sr); run32 = np.nonzero(np.convolve(dbt > THR_IN, np.ones(3), "same") >= 3)[0]
        on32 = max(0, run32[0]-1)*HOP if len(run32) else head
        al2[0]["s"] = round(on32 if abs(al2[0]["s"] - on32) > 0.04 else max(al2[0]["s"], head), 3)
        al2[0]["s"] = round(min(al2[0]["s"], al2[0]["e"] - 0.03), 3)
        al2[-1]["e"] = round(min(al2[-1]["e"], len(y)/sr - tail), 3)
    sf.write(DST/"lines"/meta.with_suffix(".wav").name, y, sr, subtype="PCM_16")
    (DST/"lines"/meta.name).write_text(json.dumps({**d, "dur": round(len(y)/sr, 3), "head": head, "tail": tail, "paced": len(edits)},
                                                  ensure_ascii=False), "utf-8")
    if al: (DST/"align"/f"{k}.json").write_text(json.dumps(al2, ensure_ascii=False), "utf-8")
    n += 1
have = {m.name for m in (SRC/"lines").glob("L*.*")}
for m in (DST/"lines").glob("L*.*"):
    if m.name not in have: m.unlink()
if a.report:
    with open(a.report, "w", encoding="utf-8") as fo:
        fo.write("句\t标点\t原来静音\t改后静音\n")
        for r in rows: fo.write("\t".join(map(str, r)) + "\n")
fix_short = sum(1 for r in rows if r[3] > r[2]); fix_long = len(rows) - fix_short
print(f"{n} 句；句内停顿补长 {fix_short} 处，剪短 {fix_long} 处")

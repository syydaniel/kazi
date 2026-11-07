"""对时间：把逐句配音按停顿拼成一条，并用 Qwen3-ForcedAligner 给每个字标上起止时间。

中文稿按字对齐，英文稿（没有汉字）按词对齐。
用法：
  逐句配音：python timeline.py <稿子.txt> <配音目录> <输出 timeline.json> [--wav 拼好的配音.wav] [--lead 1.6]
  一整条现成录音：python timeline.py <稿子.txt> - <输出 timeline.json> --whole 录音.wav
    （录音要和稿子一字不差；建议 5 分钟以内，更长的先按段切开。）
输出里除了每句每字的时间，还有 keywords：稿子里 *关键词* 的起止时间，画面和检查都用它。
- 配好的句子用真实时长和逐字对齐；还没配的句子按每秒 5 个字估算（real=false），方便先做画面。
- 稿子里的 "@gap 0.6" 在上一句后面多停 0.6 秒；"@gap 22 demo" 再给这段空白起个名字，画面用来放别的声音。
- 对齐结果按念法内容缓存在 <配音目录>/align/，稿子改了只重算改过的句子。
- --pause-scale 0.8 把句间停顿整体缩短（配合 tempo.py 提速）。
- --speech-gap：稿子里的停顿按"这句收声到下一句开口、听到的空白"算，配合 pace.py 规整过的配音用（推荐）。
"""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("vodir"); ap.add_argument("out")
ap.add_argument("--wav", default=""); ap.add_argument("--lead", type=float, default=1.6)
ap.add_argument("--tail", type=float, default=3.0); ap.add_argument("--cps", type=float, default=5.0)
ap.add_argument("--wps", type=float, default=2.6, help="英文稿还没配音时，按每秒几个词估时长")
ap.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
ap.add_argument("--whole", default="", help="一整条现成录音，不用逐句配音")
ap.add_argument("--speech-gap", action="store_true", help="句后停顿按听到的空白算：减掉这句句尾和下一句句首自带的静音（pace.py 写在每句的 json 里）")
ap.add_argument("--pause-scale", type=float, default=1.0, help="句后停顿和不带名字的空当整体缩放；带名字的空当（演示、试听）不动")
a = ap.parse_args()
VO = Path(a.vodir if a.vodir != "-" else Path(a.out).parent/".kazi"); AL = VO/"align"; AL.mkdir(parents=True, exist_ok=True)
SR = 24000
key = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-']*|[㐀-鿿]")
HAN = re.compile(r"[㐀-鿿]")
lang = lambda s: "Chinese" if HAN.search(s) else "English"  # 稿子里没有汉字就按英文对齐（按词）


def parse(path):
    items, section, gaps = [], None, []
    for raw in Path(path).read_text("utf-8").splitlines():
        l = raw.strip()
        if not l or l.startswith("#"): continue
        if l.startswith("@gap"):
            parts = l.split(); gaps.append({"after": len(items), "dur": float(parts[1]), "name": parts[2] if len(parts) > 2 else None}); continue
        if l.startswith("["): section = l[1:-1]; continue
        pause = 0.4
        if "|" in l: l, p = l.rsplit("|", 1); pause = float(p); l = l.strip()
        pieces = [[x or z, y or z] for x, y, z in re.findall(r"\{([^}]*)=>([^}]*)\}|([^{}]+)", l)]
        pieces = [[x.replace("*", ""), y] for x, y in pieces]  # *词* 只标字幕关键词
        say = "".join(p[0] for p in pieces); show = "".join(p[1] for p in pieces).replace("*", "")
        items.append({"i": len(items)+1, "sec": section, "say": say, "show": show, "pause": pause, "pieces": pieces})
    return items, gaps


def units(s):
    """念法里能念出声的单位：一个汉字，或者一个连在一起的英文/数字词，带上它在字符串里的位置。"""
    return [(m.group(0), m.start(), m.end()) for m in WORD.finditer(s)]


aligner = None


def align(wav, say):
    global aligner
    c = AL/f"{key(say)}.json"
    if c.exists(): return json.loads(c.read_text("utf-8"))
    if aligner is None:
        from mlx_audio.stt import load
        aligner = load(a.aligner)
    r = aligner.generate(str(wav), text=say, language=lang(say))
    out = [{"t": x["text"], "s": round(float(x["start"]), 3), "e": round(float(x["end"]), 3)} for x in r.segments] if hasattr(r, "segments") else [{"t": x.text, "s": x.start_time, "e": x.end_time} for x in r]
    c.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    return out


def unit_times(say, dur, al):
    """每个念法单位的起止时间。对齐结果按顺序一一对上；对不上的单位按长度均分补齐。"""
    us = units(say)
    times = [None]*len(us)
    if al:
        j = 0
        for k, (u, _, _) in enumerate(us):
            # 对齐器对英文按词、对中文按字给结果；按顺序向前找同一个单位
            for jj in range(j, min(j+4, len(al))):
                if al[jj]["t"].lower().strip() == u.lower():
                    times[k] = (al[jj]["s"], al[jj]["e"]); j = jj+1; break
    # 补齐：没对上的，夹在前后已知时间中间均分
    known = [k for k, t in enumerate(times) if t]
    if not known:
        w = np.array([max(1, len(u)/2.5) if u.isascii() else 1 for u, _, _ in us], float); edges = np.concatenate([[0], np.cumsum(w)])/w.sum()*dur
        return [(u, s0, e0, float(edges[k]), float(edges[k+1])) for k, (u, s0, e0) in enumerate(us)]
    k = 0
    while k < len(us):
        if times[k]: k += 1; continue
        k2 = k
        while k2 < len(us) and not times[k2]: k2 += 1
        lo = times[k-1][1] if k > 0 else 0.0
        hi = times[k2][0] if k2 < len(us) else dur
        n = k2-k
        for q in range(n): times[k+q] = (lo+(hi-lo)*q/n, lo+(hi-lo)*(q+1)/n)
        k = k2
    return [(u, s0, e0, float(times[k][0]), float(times[k][1])) for k, (u, s0, e0) in enumerate(us)]


def snap_onsets(ut, y, sr):
    """对齐器偶尔把一个词的起点放进它前面的停顿里，或者前一个词的尾音里（2026-11-07 英文实测：Earth、Uranus、Neptune
    早了 0.11 到 0.20 秒）。起点处或起点后 40 毫秒内开始有一段安静（50 毫秒低于门限），就把起点挪到这段安静之后、
    声音连续响 30 毫秒的地方。只往后挪，最多 0.35 秒，不越过这个词的结尾。"""
    hop = sr // 100
    db = 20*np.log10(np.array([np.sqrt(np.mean(y[i:i+hop]**2)) for i in range(0, max(1, len(y)-hop), hop)]) + 1e-9)
    thr = min(max(db.max() - 34.5, -52.0), -44.0)
    out = []
    for u, s0, e0, s, e in ut:
        k = int(round(s*100)); lim = min(int((e-0.03)*100), int((s+0.35)*100), len(db)-3)
        j0 = next((j for j in range(k, min(k+5, lim)) if (db[j:j+5] < thr).all()), None)
        if j0 is not None:
            j = next((j for j in range(j0, lim) if (db[j:j+3] > thr).all()), None)
            if j is not None: s = max(s, j/100)
        out.append((u, s0, e0, s, e))
    return out


def build_cap(it, ut, t):
    """念法单位的时间 → 字幕每个字的时间；{念=>显} 的一段把时间均分给显示的字；* 是关键词开关。"""
    # 念法 → 字幕：每一段 {念=>显} 的时间跨度，均分给显示的字
    cap, pos, key_on = [], 0, False
    for say_p, show_p in it["pieces"]:
        s0, e0 = pos, pos+len(say_p); pos = e0
        inside = [x for x in ut if x[1] >= s0 and x[2] <= e0]
        # 字幕里的 * 是关键词开关，不占字
        flags, plain = [], []
        for ch in show_p:
            if ch == "*": key_on = not key_on; continue
            plain.append(ch); flags.append(1 if key_on else 0)
        if say_p == show_p.replace("*", ""):
            tm = {x[1]: (x[3], x[4]) for x in inside}
            last = (inside[0][3], inside[0][3]) if inside else ((cap[-1]["e"]-t) if cap else 0, (cap[-1]["e"]-t) if cap else 0)
            for k, ch in enumerate(plain):
                if s0+k in tm: last = tm[s0+k]
                cap.append({"c": ch, "s": round(t+last[0], 3), "e": round(t+last[1], 3), **({"k": 1} if flags[k] else {})})
        else:
            ps = inside[0][3] if inside else ((cap[-1]["e"]-t) if cap else 0); pe = inside[-1][4] if inside else ps
            n = max(1, len(plain))
            for k, ch in enumerate(plain):
                cap.append({"c": ch, "s": round(t+ps+(pe-ps)*k/n, 3), "e": round(t+ps+(pe-ps)*(k+1)/n, 3), **({"k": 1} if flags[k] else {})})
    return cap


items, gaps = parse(a.script)
for it in items: it["pause"] = round(it["pause"]*a.pause_scale, 3)
for g in gaps:
    if not g["name"]: g["dur"] = round(g["dur"]*a.pause_scale, 3)
gap_after = {}
for g in gaps: gap_after.setdefault(g["after"], []).append(g)
t = a.lead; lines, named, chunks = [], [], []
if a.whole:
    import subprocess, tempfile
    src = Path(a.whole)
    tmp = Path(tempfile.mkdtemp())/"whole.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
    y, _ = sf.read(tmp, dtype="float32"); total_dur = len(y)/SR
    if total_dur > 300: print("提醒：录音超过 5 分钟，对齐可能不准，建议按段切开")
    SEP = "" if lang("".join(it["say"] for it in items)) == "Chinese" else " "  # 英文句子之间要留空格
    full = SEP.join(it["say"] for it in items)
    al = align(tmp, full)
    ut_all = snap_onsets(unit_times(full, total_dur, al), y, SR)
    k0 = 0
    for it in items:
        n_u = len(units(it["say"])); seg = ut_all[k0:k0+n_u]; k0 += n_u
        k_it = items.index(it); base = len(SEP.join(x["say"] for x in items[:k_it])) + (len(SEP) if k_it else 0)
        ut = [(u, s0-base, e0-base, s1, e1) for u, s0, e0, s1, e1 in seg]
        cap = build_cap(it, ut, 0.0)
        words = [{"u": u, "s": round(s1, 3), "e": round(e1, 3)} for u, _, _, s1, e1 in ut]
        st = words[0]["s"] if words else 0.0; en = words[-1]["e"] if words else st
        lines.append({"i": it["i"], "sec": it["sec"], "start": st, "end": en, "pause": it["pause"], "say": it["say"],
                      "show": it["show"], "real": True, "words": words, "cap": cap})
    t = total_dur - a.tail; real_n = len(items); items_left = []
else:
    items_left = items
for g in ([] if a.whole else gap_after.get(0, [])):  # 第一句之前的空当（比如开场先放一段演示）
    if g["name"]: named.append({"name": g["name"], "start": round(t, 3), "end": round(t+g["dur"], 3)})
    t += g["dur"]
real_n = 0 if not a.whole else real_n
def edge(i, k):
    """pace.py 记下的句首（head）/句尾（tail）静音秒数；这句没规整过就当 0。"""
    m = VO/"lines"/f"L{i:03d}.json"
    if not m.exists(): return 0.0
    d = json.loads(m.read_text("utf-8"))
    return float(d.get(k, 0.0)) if d["say"] == next((x["say"] for x in items if x["i"] == i), None) else 0.0


for it in items_left:
    w = VO/"lines"/f"L{it['i']:03d}.wav"; mj = w.with_suffix(".json")
    real = w.exists() and mj.exists() and json.loads(mj.read_text("utf-8"))["say"] == it["say"]
    if real:
        y, sr = sf.read(w, dtype="float32")
        if y.ndim > 1: y = y[:, 0]
        assert sr == SR, sr
        dur = len(y)/sr; al = align(w, it["say"]); real_n += 1
    else:
        y = None; dur = max(0.8, len(units(it["say"]))/(a.cps if lang(it["say"]) == "Chinese" else a.wps)); al = None
    ut = snap_onsets(unit_times(it["say"], dur, al), y, SR)
    cap = build_cap(it, ut, t)
    words = [{"u": u, "s": round(t+s, 3), "e": round(t+e, 3)} for u, _, _, s, e in ut]
    lines.append({"i": it["i"], "sec": it["sec"], "start": round(t, 3), "end": round(t+dur, 3), "pause": it["pause"],
                  "say": it["say"], "show": it["show"], "real": real, "words": words, "cap": cap})
    if y is not None: chunks.append((t, y))
    gap = it["pause"]
    if a.speech_gap and real: gap = max(0.02, gap - edge(it["i"], "tail") - edge(it["i"]+1, "head"))
    t += dur+gap
    for g in gap_after.get(it["i"], []):
        if g["name"]: named.append({"name": g["name"], "start": round(t, 3), "end": round(t+g["dur"], 3)})
        t += g["dur"]
total = round(t+a.tail, 3)
keywords = []
for l in lines:
    run = []
    for c in l["cap"] + [{"c": "", "s": 0, "e": 0}]:
        if c.get("k"): run.append(c)
        elif run:
            keywords.append({"line": l["i"], "word": "".join(x["c"] for x in run), "t": run[0]["s"], "e": run[-1]["e"]}); run = []
secs = {}
for l in lines:
    s = secs.setdefault(l["sec"], {"start": l["start"], "end": l["end"], "first": l["i"], "last": l["i"]})
    s["end"] = l["end"]; s["last"] = l["i"]
Path(a.out).write_text(json.dumps({"fps": 60, "total": total, "lead": a.lead, "lines": lines, "sections": secs, "gaps": named, "keywords": keywords},
                                  ensure_ascii=False), "utf-8")
if a.wav and not a.whole:
    buf = np.zeros(int(total*SR)+SR, np.float32)
    for st, y in chunks: i0 = int(round(st*SR)); buf[i0:i0+len(y)] += y
    sf.write(a.wav, buf[:int(total*SR)], SR, subtype="PCM_16")
print(f"{len(lines)} 句，{real_n} 句是真实配音，总长 {total:.1f} 秒（{total/60:.2f} 分）")

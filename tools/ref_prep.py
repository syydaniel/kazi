"""把你自己的录音变成克隆用的参考段（ref.wav + 一字不差的 ref.txt），顺便切出"像不像你"的对照句子。

用法：python tools/ref_prep.py 朗读.wav [朗读2.wav ...] --script 朗读稿.txt --out ref_en [--lang en] [--rate 4.2]
      python tools/ref_prep.py --free 随便说.wav [...] --out ref_en          没有稿子的录音（随便聊的）
- 朗读稿：一句一行，# 开头的行和 [段落] 行跳过。读两遍、中途重读都没关系，每一句念在哪里会自己找出来。
- --free 的录音用识别结果当逐字稿，只留切下来再识别一遍还对得上的段落。
1. 量：采样率、峰值、削波、底噪、信噪比。
2. 清理：去直流、70 Hz 高通；底噪高于 -62 dBFS 才做轻度降噪（最多压 12 dB）。不压缩、不均衡，声音保持原样。
3. Whisper 逐词时间，找出每一句的位置，切点放在前后最安静的地方。
4. 候选参考段：同一条录音里连着念的整句，6.5 到 15 秒，中间没有重读和多余的词。
   打分：切下来再识别一遍要和文字对上（0.95 以上才进前排）、语速接近 --rate、8 到 13 秒最好、
   0.7 秒以上的长停顿少、没有削波、音高有起伏。
5. 写出 OUT/ref.wav（48 kHz 24 bit，-23 LUFS）和 ref.txt，备选 ref_2、ref_3；
   OUT/like/：参考段以外的每一句（1.5 秒以上），给 voice.py --like 和 --voiceprint 用；OUT/report.md。
"""
import argparse, difflib, json, math, re
from pathlib import Path
import numpy as np, soundfile as sf, librosa, pyloudnorm as pyln
from scipy.ndimage import uniform_filter
from scipy.signal import butter, sosfiltfilt
import mlx_whisper

ap = argparse.ArgumentParser()
ap.add_argument("read", nargs="*", help="照着 --script 念的录音")
ap.add_argument("--script", default="")
ap.add_argument("--free", nargs="*", default=[], help="没有稿子的录音")
ap.add_argument("--out", required=True)
ap.add_argument("--lang", default="en", choices=["en", "zh"])
ap.add_argument("--rate", type=float, default=0, help="目标语速，每秒音节数（中文是字数），不算停顿；默认英文 4.2、中文 4.5")
ap.add_argument("--asr", default="mlx-community/whisper-large-v3-turbo")
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--lufs", type=float, default=-23.0)
a = ap.parse_args()
assert a.read or a.free, "至少给一条录音"
assert not a.read or a.script, "照稿念的录音要给 --script"
EN = a.lang == "en"; RATE = a.rate or (4.2 if EN else 4.5)
OUT = Path(a.out); LIKE = OUT/"like"; LIKE.mkdir(parents=True, exist_ok=True)
for old in LIKE.glob("*.wav"): old.unlink()

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def spell(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else ONES[n % 10])
    if n < 1000: return ONES[n // 100] + "hundred" + ("and" + spell(n % 100) if n % 100 else "")
    return str(n)
def norm(s):
    s = s.lower()
    if EN: s = re.sub(r"\d+", lambda m: spell(int(m.group())), s.replace(",", ""))
    return re.sub(r"[^\w]", "", s)
def syllables(text):
    """英文按元音组估音节；中文一个字一个音节"""
    n = len(re.findall(r"[一-鿿]", text))
    for w in re.findall(r"[A-Za-z']+", text):
        w = w.lower().replace("'", ""); k = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1: k -= 1
        n += max(k, 1)
    return n


def frames_db(y, sr, win=.01):
    hop = int(win*sr); n = max(1, len(y)//hop)
    return 20*np.log10(np.sqrt(np.mean(y[:n*hop].reshape(n, hop)**2, axis=1)) + 1e-9)

def load(p):
    info = sf.info(p); y, sr = sf.read(p, dtype="float64", always_2d=True)
    if y.shape[1] > 1:   # 一边没声音或者两边不一样：取响的那一边
        rms = np.sqrt((y**2).mean(0)); c = np.corrcoef(y.T)[0, 1] if rms.min() > 1e-6 else 0
        y = y.mean(1) if c > .9 else y[:, int(rms.argmax())]
    else: y = y[:, 0]
    return y, sr, info

def measure(y, sr):
    db = frames_db(y, sr, .05); floor = float(np.percentile(db, 10)); speech = float(np.percentile(db, 95))
    try: lu = float(pyln.Meter(sr).integrated_loudness(y))
    except Exception: lu = float("nan")
    return {"floor": round(floor, 1), "speech": round(speech, 1), "snr": round(speech - floor, 1), "lufs": round(lu, 1),
            "peak": round(20*math.log10(np.abs(y).max() + 1e-12), 2), "clipped": int((np.abs(y) >= .999).sum())}

def clean(y, sr, m):
    y = sosfiltfilt(butter(4, 70, "highpass", fs=sr, output="sos"), y - y.mean())
    if m["floor"] <= -62: return y, False
    # 轻度降噪：噪声样本取最安静的那些帧，每个频点按维纳滤波压，最多压 12 dB，时间和频率上抹平，免得发颤
    hop = 512; db = frames_db(y, sr, hop/sr); q = np.argsort(db)[:max(20, int(3*sr/hop))]
    noise = np.concatenate([y[i*hop:(i+1)*hop] for i in sorted(q)])
    S = librosa.stft(y, n_fft=2048, hop_length=hop)
    prof = np.abs(librosa.stft(noise, n_fft=2048, hop_length=hop)).mean(1, keepdims=True)
    g = np.sqrt(np.clip(1 - (1.5*prof/np.maximum(np.abs(S), 1e-10))**2, 0, 1))
    g = uniform_filter(np.maximum(g, 10**(-12/20)), size=(5, 3))
    return librosa.istft(S*g, hop_length=hop, length=len(y)), True

def pauses(y, sr):
    """0.12 秒以上的静音（比最响的帧低 35 dB），和 voice.py 量语速的规则一样"""
    db = frames_db(y, sr); on = db > max(db.max()-35, -50); idx = np.where(on)[0]; out = []
    if len(idx) == 0: return out
    k = idx[0]
    while k <= idx[-1]:
        if not on[k]:
            j = k
            while j <= idx[-1] and not on[j]: j += 1
            if j - k >= 12: out.append(round((j-k)*.01, 2))
            k = j
        else: k += 1
    return out

def pitch_spread(y, sr):
    f0, _, _ = librosa.pyin(librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=24000),
                            fmin=60, fmax=420, sr=24000, frame_length=1024, hop_length=240)
    f = f0[np.isfinite(f0)]
    return round(float((12*np.log2(f/np.median(f))).std()), 2) if len(f) > 10 else 0.0

def level(y, sr):
    y = y*10**((a.lufs - pyln.Meter(sr).integrated_loudness(y))/20); pk = np.abs(y).max()
    return y*(10**(-1/20)/pk) if pk > 10**(-1/20) else y

def fade(y, sr):
    n = min(int(.01*sr), len(y)//2); r = np.linspace(0, 1, n); y = y.copy(); y[:n] *= r; y[-n:] *= r[::-1]; return y

def quietest(db, lo, hi, w=3):
    lo, hi = max(0, lo), min(len(db), hi)
    if hi - lo <= w: return lo
    return lo + int(np.convolve(db[lo:hi], np.ones(w)/w, "valid").argmin()) + w//2

def cut(db, t0, t1, prev_end, next_start, thresh):
    """词 [t0, t1] 前后最安静的地方下刀，不切进旁边的词；开口前留 80 毫秒，收声后留 150 毫秒"""
    f = lambda t: int(round(t*100))
    s = quietest(db, max(f(t0-.5), f(prev_end)), f(t0)+5)
    while s < f(t0) and db[s+1] < thresh: s += 1
    e = quietest(db, f(t1)-5, min(f(t1+.6), f(next_start)))
    while e > f(t1) and db[e-1] < thresh: e -= 1
    return max(s*.01 - .08, prev_end), min(e*.01 + .15, next_start)

def hear(y, sr):
    p = OUT/".tmp.wav"; sf.write(p, y, sr, subtype="PCM_24")
    t = mlx_whisper.transcribe(str(p), path_or_hf_repo=a.asr, language=a.lang, condition_on_previous_text=False)["text"].strip()
    p.unlink(); return t


SENTS = []
if a.script:
    for raw in Path(a.script).read_text("utf-8").splitlines():
        l = raw.strip()
        if l and not l.startswith(("#", "[")): SENTS.append({"j": len(SENTS), "text": l, "n": norm(l)})

files, reads = [], []
for fi, (p, free) in enumerate([(p, False) for p in a.read] + [(p, True) for p in a.free]):
    y, sr, info = load(p); m = measure(y, sr); yc, gated = clean(y, sr, m)
    res = mlx_whisper.transcribe(str(p), path_or_hf_repo=a.asr, language=a.lang, word_timestamps=True,
                                 condition_on_previous_text=False)
    words = [{"w": w["word"].strip(), "n": norm(w["word"]), "s": float(w["start"]), "e": float(w["end"])}
             for seg in res["segments"] for w in seg.get("words", [])]
    words = [w for w in words if w["n"]]; W = [w["n"] for w in words]; used = np.zeros(len(W), bool); found = []
    if free:   # 没有稿子：按识别结果的句末标点断句
        st = 0
        for k, w in enumerate(words):
            if re.search(r"[.?!。？！]$", w["w"]) or k == len(words) - 1:
                found.append((None, st, k + 1 - st, " ".join(x["w"] for x in words[st:k+1]) if EN else "".join(x["w"] for x in words[st:k+1])))
                st = k + 1
        used[:] = True
    else:      # 每一句在识别结果里找最像的一段词（长度在稿子的 0.7 到 1.3 倍之间），先认最有把握的
        hits = []
        for s in SENTS:
            for i in range(len(W)):
                best, txt = None, ""
                for L in range(1, len(W) - i + 1):
                    txt += W[i+L-1]
                    if len(txt) > 1.3*len(s["n"]): break
                    if len(txt) < .7*len(s["n"]): continue
                    sm = difflib.SequenceMatcher(None, s["n"], txt, autojunk=False)
                    if sm.real_quick_ratio() < .8: continue
                    r = sm.ratio()
                    if best is None or r > best[0]: best = (r, L)
                if best and best[0] >= .8: hits.append((best[0], s["j"], i, best[1]))
        for r, j, i, L in sorted(hits, key=lambda h: (-h[0], h[2])):
            if used[i:i+L].any(): continue
            used[i:i+L] = True; found.append((j, i, L, SENTS[j]["text"]))
    db = frames_db(yc, sr); thresh = max(db.max() - 35, m["floor"] + 8)
    for j, i, L, text in sorted(found, key=lambda x: x[1]):
        prev_end = words[i-1]["e"] if i > 0 else 0.0
        next_start = words[i+L]["s"] if i+L < len(words) else len(yc)/sr
        s, e = cut(db, words[i]["s"], words[i+L-1]["e"], prev_end, next_start, thresh)
        reads.append({"file": fi, "j": j, "w0": i, "w1": i+L, "s": round(s, 3), "e": round(e, 3), "text": text, "free": free})
    files.append({"path": p, "y": yc, "sr": sr, "info": info, "raw": m, "clean": measure(yc, sr), "gated": gated,
                  "used": used, "n_found": len(found), "free": free})

# 候选：同一条录音里连着念的整句（照稿念的要按稿子顺序），中间没有多余的词
cands = []
for fi, F in enumerate(files):
    rs = sorted([r for r in reads if r["file"] == fi], key=lambda r: r["s"]); sr = F["sr"]
    for x in range(len(rs)):
        for z in range(x, len(rs)):
            span = rs[x:z+1]
            if not F["free"] and any(span[q+1]["j"] != span[q]["j"] + 1 for q in range(len(span)-1)): break
            s0, e0 = span[0]["s"], span[-1]["e"]; dur = e0 - s0
            if dur > 15: break
            if dur < 6.5 or not F["used"][span[0]["w0"]:span[-1]["w1"]].all(): continue
            y = F["y"][int(s0*sr):int(e0*sr)]; text = " ".join(r["text"] for r in span) if EN else "".join(r["text"] for r in span)
            ps = pauses(y, sr); rate = syllables(text)/max(dur - sum(ps), .1)
            cands.append({"file": fi, "span": [r["j"] for r in span], "s": s0, "e": e0, "dur": round(dur, 2), "text": text,
                          "rate": round(rate, 2), "long_pauses": sum(1 for q in ps if q > .7),
                          "peak": round(20*math.log10(np.abs(y).max() + 1e-12), 2)})
pre = lambda c: -30*abs(math.log(c["rate"]/RATE)) - 1.5*max(0, 8 - c["dur"], c["dur"] - 13) - c["long_pauses"]
cands.sort(key=pre, reverse=True)
for c in cands[:16]:   # 识别和音高比较慢，只给初筛前 16 个做
    F = files[c["file"]]; y = F["y"][int(c["s"]*F["sr"]):int(c["e"]*F["sr"])]
    c["heard"] = hear(y, F["sr"]); c["asr"] = round(difflib.SequenceMatcher(None, norm(c["text"]), norm(c["heard"]), autojunk=False).ratio(), 3)
    c["st_std"] = pitch_spread(y, F["sr"])
    c["score"] = round(pre(c) + 20*c["asr"] + .5*min(c["st_std"], 4) - (5 if c["peak"] > -.1 else 0), 2)
# ref.txt 必须和声音一字不差：识别对不上的，排在所有对得上的后面
scored = sorted([c for c in cands if "score" in c], key=lambda c: (c["asr"] >= .95, c["score"]), reverse=True)
top = []
for c in scored:
    if all(c["file"] != t["file"] or c["e"] <= t["s"] or c["s"] >= t["e"] for t in top): top.append(c)
    if len(top) == a.top: break
assert top, "没有找到 6.5 到 15 秒、连着念对的一段：多录一点，或者念慢一点、句子之间别重来"
for n, c in enumerate(top, 1):
    F = files[c["file"]]; name = "ref" if n == 1 else f"ref_{n}"
    sf.write(OUT/f"{name}.wav", level(fade(F["y"][int(c["s"]*F["sr"]):int(c["e"]*F["sr"])], F["sr"]), F["sr"]), F["sr"], subtype="PCM_24")
    (OUT/f"{name}.txt").write_text(c["text"] + "\n", "utf-8")
    (OUT/f"{name}.json").write_text(json.dumps({**c, "source": F["path"]}, ensure_ascii=False, indent=1), "utf-8")

# like/：参考段以外的每一句，给 voice.py 量"像不像你"、算平均声纹
R = top[0]; n_like = 0
for k, r in enumerate(sorted(reads, key=lambda r: (r["file"], r["s"]))):
    if r["e"] - r["s"] < 1.5 or (r["file"] == R["file"] and r["s"] < R["e"] and r["e"] > R["s"]): continue
    F = files[r["file"]]; y = fade(F["y"][int(r["s"]*F["sr"]):int(r["e"]*F["sr"])], F["sr"])
    sf.write(LIKE/f"{Path(F['path']).stem}_{k:03d}.wav", level(y, F["sr"]), F["sr"], subtype="PCM_24"); n_like += 1

L = ["# 参考录音：清理和切段报告", ""]
for F in files:
    r, c, i = F["raw"], F["clean"], F["info"]
    L += [f"## {Path(F['path']).name}" + ("（没有稿子）" if F["free"] else ""), "",
          f"- 格式：{F['sr']} Hz，{i.subtype}，{i.channels} 声道，{len(F['y'])/F['sr']:.1f} 秒" + ("" if F["sr"] >= 44100 else "（**采样率偏低**）"),
          f"- 原始：底噪 {r['floor']} dBFS，说话 {r['speech']} dBFS，信噪比约 {r['snr']} dB，峰值 {r['peak']} dBFS，"
          f"削波 {r['clipped']} 个采样" + ("（**有削波，离话筒远一点或调小增益重录**）" if r["clipped"] > 10 else ""),
          f"- 清理：去直流、70 Hz 高通" + ("、轻度降噪（最多压 12 dB）" if F["gated"] else "，底噪够低，没有降噪") + f"；之后底噪 {c['floor']} dBFS",
          f"- 找到 {F['n_found']} 句" + ("" if F["free"] else f"（稿子 {len(SENTS)} 句）"), ""]
if SENTS:
    miss = [s["j"]+1 for s in SENTS if not any(r["j"] == s["j"] for r in reads)]
    if miss: L += [f"稿子里没找到的句子：第 {miss} 句（念错太多或者漏了）", ""]
L += ["## 参考段", ""]
for n, c in enumerate(top, 1):
    L.append(f"- {'ref' if n == 1 else f'ref_{n}'}.wav：{Path(files[c['file']]['path']).name} {c['s']:.1f} 到 {c['e']:.1f} 秒，{c['dur']} 秒，"
             f"语速 {c['rate']}（目标 {RATE}），识别吻合度 {c['asr']}，音高起伏 {c['st_std']} 半音，得分 {c['score']}")
    L.append(f"  - 文字：{c['text']}")
    if c["asr"] < .95: L.append(f"  - **识别出来是**：{c['heard']}（对不上，ref.txt 要按实际念的改）")
L += ["", f"对照句子：{n_like} 段，在 `like/`（不含 ref.wav 那一段）。"]
(OUT/"report.md").write_text("\n".join(L) + "\n", "utf-8")
print("\n".join(L)); print("REF_PREP DONE")

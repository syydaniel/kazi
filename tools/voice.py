"""逐句生成旁白：千问 Qwen3-TTS Base 克隆 ref.wav 的声音，每句生成 N 遍，用 Whisper 回听打分，留最好的一遍。

用法：python voice.py <稿子.txt> <输出目录> [--takes 4] [--only 1,2,5] [--lang en]
- --lang en：念英文稿。语音识别、语速打分都按英文来。参考录音最好也是你自己说的英文（tools/ref_prep.py 切），
  中文参考也能念，但像你的程度低一截，偶尔整句声音变样。
- --like 录音或文件夹：你本人用这门语言说话的录音（ref_prep.py 的 like/）。每遍量和它们的声纹相似度，挑最像你的。
- --voiceprint 录音或文件夹：用这些录音的平均声纹代替参考段自己的声纹，参考段只管念法和节奏。
- --rate 4.2：英文目标语速（每秒音节数，不算停顿），挑遍时偏好接近的。模型的语速跟着参考段走，差得远就换参考段。
- --ref-tempo 1.1：参考段先加快 1.1 倍（变速不变调）再拿去克隆，模型跟着念快一点，比生成完再整体变速自然。
- --save-voice 声音包：试听定下来以后，把这次的参考段（已加快）、逐字稿、声纹、对照声纹和参数存成一个文件夹。
- --voice 声音包：直接用定好的声音包，不再重新算声纹、不再试参考段；包里的参数（语言、语速、遍数等）当默认值，命令行写了的优先。
- --voiceprint-only：只借音色。不给参考段的逐字稿，模型只用声纹（要配 --voiceprint 或声音包），节奏和语调是模型自己的。
  念英文时中文腔最少，同时还是你的音色（卡字英文版定的就是这个）。给了逐字稿时，模型连参考段的念法、口音一起学。
- --temperature 0.9：采样温度，不写就用模型默认（0.9）。
- --chunk-words 60：同一段里相邻几句拼起来一口气念（每次最多这么多词），念完按 Whisper 逐词时间在句间最安静处切回一句一个文件。
  整段念的语调是连着的，比一句一句念自然；句间停顿交给 pace.py / timeline.py 按稿子定。
- --fit-rate：挑好的那遍整体变速到 --rate（变速不变调，最多 ±12%），句内超过 0.45 秒的停顿缩短。一口气念会比逐句念快一成左右。
- 稿子格式：一句一行；[段落名] 开新段；{念法=>字幕}；行尾 "| 0.4" 是句后停顿秒数。
- 可以断点续跑：念法和声音（参考段、声纹）都没变的句子直接跳过。所有遍都留在 takes/，方便换耳朵挑。
"""
import argparse, datetime, difflib, hashlib, json, math, re, shutil, subprocess, tempfile, time
from pathlib import Path
import librosa, mlx_whisper, numpy as np, soundfile as sf
import mlx.core as mx
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("out")
ap.add_argument("--ref", default=str(Path(__file__).with_name("ref.wav")))
ap.add_argument("--takes", type=int, default=4)
ap.add_argument("--only", default="")
ap.add_argument("--model", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
ap.add_argument("--asr", default="mlx-community/whisper-large-v3-turbo")
ap.add_argument("--hint", default="")
ap.add_argument("--lang", default="zh", choices=["zh", "en"])
ap.add_argument("--like", nargs="*", default=[])
ap.add_argument("--voiceprint", nargs="*", default=[])
ap.add_argument("--rate", type=float, default=0)
ap.add_argument("--ref-tempo", type=float, default=1.0)
ap.add_argument("--voice", default=""); ap.add_argument("--save-voice", default="")
ap.add_argument("--voiceprint-only", action="store_true")
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--chunk-words", type=int, default=0)
ap.add_argument("--fit-rate", action="store_true")
pre, _ = ap.parse_known_args()
if pre.voice:   # 声音包：ref.wav（已经加快过）、ref.txt、voiceprint.npy、like.npy、voice.json（只借音色的包没有 ref）
    V = Path(pre.voice); cfg = json.loads((V/"voice.json").read_text("utf-8")) if (V/"voice.json").exists() else {}
    ap.set_defaults(**({"ref": str(V/"ref.wav")} if (V/"ref.wav").exists() else {}),
                    **{k: cfg[k] for k in ("lang", "rate", "takes", "voiceprint_only", "temperature", "chunk_words", "fit_rate") if k in cfg},
                    **({"voiceprint": [str(V/"voiceprint.npy")]} if (V/"voiceprint.npy").exists() else {}),
                    **({"like": [str(V/"like.npy")]} if (V/"like.npy").exists() else {}))
a = ap.parse_args()
EN = a.lang == "en"
if not a.hint: a.hint = "The following is an English sentence." if EN else "以下是普通话的句子。"

OUT = Path(a.out); LINES = OUT/"lines"; TAKES = OUT/"takes"; WORK = OUT/".work"
for d in (LINES, TAKES, WORK): d.mkdir(parents=True, exist_ok=True)
REF = Path(a.ref)
# 只借音色时不给逐字稿：模型只用声纹，参考段的念法、口音不带过来
REF_TEXT = None if a.voiceprint_only else REF.with_suffix(".txt").read_text("utf-8").strip()
if a.ref_tempo != 1.0 and not a.voiceprint_only:   # 加快的参考段放在 .work 里，逐字稿不变
    fast = WORK/f"ref_x{a.ref_tempo:g}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(REF), "-af", f"atempo={a.ref_tempo}", "-c:a", "pcm_s24le", str(fast)], check=True)
    REF = fast
# 文件夹里的 .wav 都算；.npy 是存好的声纹（--save-voice 存的）
wavs = lambda xs: [q for x in xs for q in (sorted(Path(x).glob("*.wav")) if Path(x).is_dir() else [Path(x)])]
LIKE, VPS = wavs(a.like), wavs(a.voiceprint)
# 这句是用哪个声音念的（按文件内容算，项目挪了地方也认得）：换了参考段或声纹，缓存里的旧句子不能直接拿来用
digest = lambda p: hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]
if a.voiceprint_only:
    if not VPS: raise SystemExit("--voiceprint-only 要配 --voiceprint（录音或 .npy）或带 voiceprint.npy 的声音包")
    VOICE = {"mode": "voiceprint_only", "voiceprint": sorted(digest(p) for p in VPS)}
else:
    VOICE = {"ref": digest(REF) + hashlib.sha1(REF_TEXT.encode("utf-8")).hexdigest()[:12], "voiceprint": sorted(digest(p) for p in VPS)}
# 用声音包时沿用包里记的声音标记：存包那次用录音算的声纹和包里的 .npy 是同一个声音，缓存要认得
if pre.voice and cfg.get("voice_key") and REF == V/"ref.wav" and VPS == ([V/"voiceprint.npy"] if (V/"voiceprint.npy").exists() else []):
    VOICE = cfg["voice_key"]
# 温度、整段念、变速也会改出来的声音，记进声音标记（都用默认时标记和以前一样，旧缓存照样认）
VOICE.update({k: v for k, v in (("temperature", a.temperature), ("chunk_words", a.chunk_words),
                                 ("fit_rate", a.rate if a.fit_rate else 0)) if v})
if EN and REF_TEXT and re.search(r"[一-鿿]", REF_TEXT):
    print("注意：参考录音是中文的。用中文参考念英文，像你的程度低一截，偶尔整句声音变样；"
          "最好录一段英文参考，见 docs/02_配音.md 的「英文片」。", flush=True)


def parse(path):
    items, section = [], None
    for raw in Path(path).read_text("utf-8").splitlines():
        l = raw.strip()
        if not l or l.startswith("#") or l.startswith("@"): continue
        # 带【】的是等数据的占位句，不配音（timeline 里按字数估时长）
        if "【" in l and not l.startswith("["): items.append(None); continue
        if l.startswith("["): section = l[1:-1]; continue
        pause = 0.4
        if "|" in l: l, p = l.rsplit("|", 1); pause = float(p); l = l.strip()
        # *词* 只标字幕里的关键词，念法里去掉
        say = re.sub(r"\{([^}]*)=>[^}]*\}", r"\1", l).replace("*", "")
        show = re.sub(r"\{[^}]*=>([^}]*)\}", r"\1", l).replace("*", "")
        pieces = [[x or z, y or z] for x, y, z in re.findall(r"\{([^}]*)=>([^}]*)\}|([^{}]+)", l)]
        items.append({"i": len(items)+1, "section": section, "say": say, "show": show, "pause": pause, "pieces": pieces})
    return [x for x in items if x is not None]


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n):
    """英文稿用：识别结果里的阿拉伯数字念成英文单词，再和稿子比（稿子里写的是单词）。"""
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else ONES[n % 10])
    if n < 1000: return ONES[n // 100] + "hundred" + ("and" + spell(n % 100) if n % 100 else "")
    return str(n)


def norm(s):
    s = s.lower()
    if EN: s = re.sub(r"\d+", lambda m: spell(int(m.group())), s.replace(",", ""))
    return re.sub(r"[^\w]", "", s)


def asr(p):
    return mlx_whisper.transcribe(str(p), path_or_hf_repo=a.asr, language=a.lang, initial_prompt=a.hint,
                                  condition_on_previous_text=False)["text"].strip()


def trim(y, sr):
    hop = int(.01*sr)
    db = 20*np.log10(np.array([np.sqrt(np.mean(y[i:i+hop]**2)) for i in range(0, len(y)-hop, hop)])+1e-6)
    idx = np.where(db > max(db.max()-45, -55))[0]
    return y[max(0, idx[0]*hop-int(.04*sr)):min(len(y), (idx[-1]+1)*hop+int(.08*sr))]


def pitch_spread(y, sr):
    """音高起伏（半音的标准差）。太平的那一遍听起来像念经，打分时略微偏好起伏大一点的。"""
    f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=60, fmax=420, sr=sr, frame_length=1024, hop_length=240)
    f = f0[np.isfinite(f0)]
    return round(float((12*np.log2(f/np.median(f))).std()), 2) if len(f) > 10 else 0.0


def syllables(text):
    n = 0
    for w in re.findall(r"[A-Za-z']+", text):
        w = w.lower().replace("'", ""); k = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1: k -= 1
        n += max(k, 1)
    return n


def speech_secs(y, sr):
    """去掉 0.12 秒以上的停顿（比最响处低 35 dB）以后的说话时长，英文语速按它算"""
    hop = int(.01*sr); n = len(y)//hop
    db = 20*np.log10(np.sqrt((y[:n*hop].reshape(n, hop)**2).mean(1)) + 1e-9); on = db > max(db.max()-35, -50)
    idx = np.where(on)[0]; gap, k = 0, idx[0] if len(idx) else 0
    while len(idx) and k <= idx[-1]:
        j = k
        while j <= idx[-1] and not on[j]: j += 1
        if j - k >= 12: gap += j - k
        k = max(j, k + 1)
    return max(len(y)/sr - gap*.01, .1)


def load24(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True); y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=24000) if sr != 24000 else y


TARGET = VP = None


def setup(model):
    """--like：对照录音的平均声纹（千问自己的声纹编码器）；--voiceprint：换掉参考段的声纹。给的是 .npy 就直接读，不再算"""
    global EXTRACT, TARGET, VP
    EXTRACT = model.extract_speaker_embedding
    raw = lambda ps: np.stack([np.array(EXTRACT(mx.array(load24(p))), dtype=np.float32).reshape(-1) for p in ps])
    if LIKE:
        if LIKE[0].suffix == ".npy": TARGET = np.load(LIKE[0]).astype(np.float32).reshape(-1)
        else: e = raw(LIKE); e /= np.linalg.norm(e, axis=1, keepdims=True); TARGET = e.mean(0)
        TARGET = TARGET/np.linalg.norm(TARGET)
    if VPS:   # 平均方向，长度取单段录音声纹的典型长度（模型只见过单段录音的声纹）
        if VPS[0].suffix == ".npy": VP = np.load(VPS[0]).astype(np.float32).reshape(-1)
        else: v = raw(VPS); VP = v.mean(0)/np.linalg.norm(v.mean(0))*float(np.linalg.norm(v, axis=1).mean())
        vp = mx.array(VP.reshape(1, -1)); model.extract_speaker_embedding = lambda *x, **k: vp


def likeness(y, sr):
    y = librosa.resample(y, orig_sr=sr, target_sr=24000) if sr != 24000 else y
    e = np.array(EXTRACT(mx.array(y.astype(np.float32))), dtype=np.float32).reshape(-1)
    return round(float(e @ TARGET/np.linalg.norm(e)), 4)


items = parse(a.script)
only = {int(x) for x in a.only.split(",") if x}
# 按念法内容缓存：稿子里插句、删句以后，序号会变，但念过的句子不用重念
CACHE = OUT/"cache"; CACHE.mkdir(exist_ok=True)
key = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
for m in LINES.glob("L*.json"):
    d = json.loads(m.read_text("utf-8")); w = m.with_suffix(".wav"); c = CACHE/key(d["say"])
    if w.exists() and not c.with_suffix(".wav").exists():
        shutil.copy(w, c.with_suffix(".wav")); c.with_suffix(".json").write_text(json.dumps(d, ensure_ascii=False), "utf-8")
model = None; t0 = time.time()
SR = 24000


def cached_of(it):
    """以前没记声音的缓存照旧能用；记了声音的，要同一个声音才能用"""
    c = CACHE/key(it["say"]); j = c.with_suffix(".json")
    d = json.loads(j.read_text("utf-8")) if j.exists() else None
    return d if d and c.with_suffix(".wav").exists() and d.get("voice", VOICE) == VOICE else None


def from_cache(it, d):
    shutil.copy((CACHE/key(it["say"])).with_suffix(".wav"), LINES/f"L{it['i']:03d}.wav")
    (LINES/f"L{it['i']:03d}.json").write_text(json.dumps({**d, **it}, ensure_ascii=False), "utf-8")


def put(it, y, sr, info):
    dest = LINES/f"L{it['i']:03d}.wav"; meta = dest.with_suffix(".json"); c = CACHE/key(it["say"])
    sf.write(dest, y, sr, subtype="PCM_16")
    meta.write_text(json.dumps({**it, **info, "voice": VOICE}, ensure_ascii=False), "utf-8")
    shutil.copy(dest, c.with_suffix(".wav")); shutil.copy(meta, c.with_suffix(".json"))


def synth(text, budget, prefix):
    """生成一遍，返回 wav 路径"""
    global model
    if model is None: model = load_model(a.model); setup(model)
    for old in WORK.glob(Path(prefix).name + "*.wav"): old.unlink()
    samp = {"temperature": a.temperature} if a.temperature else {}
    if a.voiceprint_only:
        # 直接调 model.generate：generate_audio 见到没有逐字稿会自动转写参考段，又变回连念法一起学。
        # 参考音频只用来取声纹，而声纹已经换成 VP，所以给一段静音就行
        res = model.generate(text=text, ref_audio=mx.zeros(SR), ref_text=None, lang_code=a.lang, max_tokens=budget, **samp)
        y = np.concatenate([np.array(r.audio, dtype=np.float32).reshape(-1) for r in res])
        p = Path(prefix + ".wav"); sf.write(p, y, SR, subtype="PCM_16"); return p
    generate_audio(text=text, model=model, ref_audio=str(REF), ref_text=REF_TEXT, lang_code=a.lang,
                   file_prefix=prefix, verbose=False, max_tokens=budget, **samp)
    return next(WORK.glob(Path(prefix).name + "*.wav"))


def atempo(y, sr, f):
    if abs(f - 1) < .002: return y
    with tempfile.TemporaryDirectory() as d:
        s, t = Path(d)/"a.wav", Path(d)/"b.wav"; sf.write(s, y, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(s), "-af", f"atempo={f:.4f}", "-c:a", "pcm_f32le", str(t)], check=True)
        return sf.read(t, dtype="float32")[0]


def db10(y, sr):
    """每 10 毫秒的音量，相对这段里较响的地方（第 95 百分位）"""
    h = int(.01*sr); k = len(y)//h; r = np.sqrt((y[:k*h].reshape(k, h)**2).mean(1)) + 1e-9
    return 20*np.log10(r/np.percentile(r, 95))


def quiet_runs(y, sr, minf=6):
    s = db10(y, sr) < -32; out, k = [], 0
    while k < len(s):
        if s[k]:
            j = k
            while j < len(s) and s[j]: j += 1
            if j - k >= minf: out.append((k*.01, j*.01))
            k = j
        else: k += 1
    return out


def cap_pauses(y, sr, cap=.45):
    """句内超过 cap 秒的停顿从中间剪短到 cap，从不往里加静音（加静音就是"卡顿感"的来源）"""
    keep = np.ones(len(y), bool)
    for r0, r1 in quiet_runs(y, sr, minf=12):
        if r1 - r0 > cap and r0 > .05 and r1 < len(y)/sr - .05:
            mid = int((r0 + r1)/2*sr); n = int((r1 - r0 - cap)*sr); keep[mid - n//2: mid - n//2 + n] = False
    return y[keep]


def fit(y, sr, say):
    """整体变速到 --rate（音节/秒，不算停顿），最多 ±12%"""
    f = min(max(a.rate/(syllables(say)/speech_secs(y, sr)), .88), 1.12)
    return atempo(y, sr, f), round(f, 3)


def heard_words(p):
    r = mlx_whisper.transcribe(str(p), path_or_hf_repo=a.asr, language=a.lang, initial_prompt=a.hint,
                               word_timestamps=True, condition_on_previous_text=False)
    ws = [{"w": x["word"].strip(), "s": float(x["start"]), "e": float(x["end"])} for s in r["segments"] for x in s.get("words", [])]
    return r["text"].strip(), [x for x in ws if norm(x["w"])]


def stumbles(y, sr, ws, h2t, punct):
    """听起来像卡住的停顿（秒）：两个词之间没有标点却断开 0.1 秒以上，按超出 0.05 秒的部分 ×3；词中间 0.12 秒以上的静音 ×6"""
    out = 0.0
    for r0, r1 in quiet_runs(y, sr):
        if r0 < .05 or r1 > len(y)/sr - .05: continue
        b = [k for k in range(len(ws) - 1) if r0 < ws[k + 1]["s"] + .08 and r1 > ws[k]["e"] - .08]
        if b:
            if not (b[0] in h2t and punct[h2t[b[0]]]) and r1 - r0 > .1: out += 3*(r1 - r0 - .05)
        elif ws and ws[0]["s"] < r0 and r1 < ws[-1]["e"] and r1 - r0 >= .12: out += 6*(r1 - r0)
    return round(out, 2)


def chunked(items):
    """同一段里相邻的句子拼成一组，每组最多 --chunk-words 个词。一段分成几组时尽量分得一样长，
    免得最后剩一句单独念（单句一口气念得慢，拉不回目标语速）。一句超过上限就自己一组"""
    secs, out = [], []
    for it in items:
        if secs and secs[-1][-1]["section"] == it["section"]: secs[-1].append(it)
        else: secs.append([it])
    for sec in secs:
        W = [len(x["say"].split()) for x in sec]; goal = sum(W)/math.ceil(sum(W)/a.chunk_words); cur, cw = [], 0
        for it, w in zip(sec, W):
            if cur and (cw + w > a.chunk_words or cw + w/2 > goal): out.append(cur); cur, cw = [], 0
            cur.append(it); cw += w
        out.append(cur)
    return out


def run_chunk(g):
    """一组句子一口气念 N 遍，挑一遍，变速，按逐词时间在句间最安静处切开"""
    says = [it["say"] if re.search(r"[.?!,;:]['\")]*$", it["say"]) else it["say"] + "." for it in g]
    text = " ".join(says); toks, owner = [], []
    for k, s in enumerate(says): toks += s.split(); owner += [k]*len(s.split())
    punct = [bool(re.search(r"[,.;:?!]['\")]*$", t)) for t in toks]
    budget = int(len(toks)/2.6*12*1.8) + 40; tag = f"C{g[0]['i']:03d}-{g[-1]['i']:03d}"; cands = []
    for t in range(a.takes):
        p = synth(text, budget, str(WORK/f"{tag}_t{t}"))
        y, sr = sf.read(p, dtype="float32", always_2d=True); y = trim(y[:, 0], sr)
        sf.write(TAKES/f"{tag}_t{t}.wav", y, sr, subtype="PCM_16")
        heard, ws = heard_words(TAKES/f"{tag}_t{t}.wav")
        ratio = difflib.SequenceMatcher(None, norm(text), norm(heard), autojunk=False).ratio()
        A = [norm(x) for x in toks]; B = [norm(x["w"]) for x in ws]; h2t = {}
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
            if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1): h2t.update({j1 + k: i1 + k for k in range(i2 - i1)})
        rate = syllables(text)/speech_secs(y, sr); chop = stumbles(y, sr, ws, h2t, punct)
        cand = {"y": y, "sr": sr, "take": t, "ratio": round(ratio, 3), "heard": heard, "ws": ws, "h2t": h2t,
                "chunk_rate": round(rate, 2), "stumbles": chop,
                "penalty": chop + (abs(math.log(rate/a.rate))/math.log(1.05) if a.rate else 0)}
        if LIKE: cand["like"] = likeness(y, sr)
        cands.append(cand)
    # 先看念没念对：只在吻合度最高的几遍里挑（和最高的差 0.01 以内；整段长，错一个词也有 0.97，所以不能只卡 0.95），
    # 再看卡顿少、语速接近 --rate（差 5% 约等于 1 秒卡顿）
    top = max(c["ratio"] for c in cands)
    best = min((c for c in cands if c["ratio"] >= top - .01), key=lambda c: c["penalty"])
    y, sr, ws = best["y"], best["sr"], best["ws"]; f = 1.0
    if a.fit_rate and a.rate:
        y, f = fit(y, sr, text); ws = [{**x, "s": x["s"]/f, "e": x["e"]/f} for x in ws]
    t2h = {v: k for k, v in best["h2t"].items()}; d = db10(y, sr)
    d = np.convolve(d, np.ones(3)/3, mode="same")   # 平滑一下，别切在塞音闭合的那一下
    cuts = []
    for k in range(len(g) - 1):
        last = max(j for j, o in enumerate(owner) if o == k); first = last + 1
        L = next((t2h[j] for j in range(last, -1, -1) if j in t2h), None)
        R = next((t2h[j] for j in range(first, len(toks)) if j in t2h), None)
        lo = ws[L]["e"] if L is not None else (cuts[-1] if cuts else 0.0)
        hi = ws[R]["s"] if R is not None else len(y)/sr
        if hi - lo < .06: mid = (lo + hi)/2; lo, hi = mid - .08, mid + .08
        f0, f1 = max(int(lo*100), int(cuts[-1]*100) + 1 if cuts else 0), min(int(hi*100) + 1, len(d))
        cuts.append(((f0 + int(np.argmin(d[f0:f1]))) if f1 > f0 else f0)*.01 + .005)
    edges = [0] + [int(c*sr) for c in cuts] + [len(y)]
    for k, it in enumerate(g):
        piece = cap_pauses(trim(y[edges[k]:edges[k + 1]], sr), sr)
        mine = [x["w"] for x in ws if edges[k]/sr <= (x["s"] + x["e"])/2 < edges[k + 1]/sr]
        heard = " ".join(mine); dur = len(piece)/sr
        info = {"take": best["take"], "chunk": [g[0]["i"], g[-1]["i"]], "chunk_ratio": best["ratio"], "stumbles": best["stumbles"],
                "ratio": round(difflib.SequenceMatcher(None, norm(it["say"]), norm(heard), autojunk=False).ratio(), 3),
                "heard": heard, "dur": round(dur, 3), "cps": round(len(it["say"].split())/max(dur, .1), 2),
                "syl_rate": round(syllables(it["say"])/speech_secs(piece, sr), 2), "chunk_rate": best["chunk_rate"], "stretch": f,
                "st_std": pitch_spread(piece, sr), **({"like": best["like"]} if "like" in best else {})}
        put(it, piece, sr, info)
        if info["ratio"] < .98: print(f"  注意：第 {it['i']} 句听着像「{heard}」，稿子是「{it['say']}」，听一下", flush=True)
    print(tag, g[0]["section"], f"{len(g)} 句", "ratio", best["ratio"], "stumbles", best["stumbles"], "rate", best["chunk_rate"], "x", f,
          "|", best["heard"][:60], f"[{round(time.time()-t0)}s]", flush=True)
    if best["ratio"] < .95: print(f"  注意：{tag} 这组 {a.takes} 遍都没念对（最好 {best['ratio']}），听一下或加 --takes 再跑", flush=True)


if a.chunk_words:
    for g in chunked(items):
        if only and not any(it["i"] in only for it in g): continue
        ds = [cached_of(it) for it in g]
        if all(ds):
            for it, d in zip(g, ds): from_cache(it, d)
            continue
        run_chunk(g)
for it in ([] if a.chunk_words else items):
    if only and it["i"] not in only: continue
    cached = cached_of(it)
    if cached: from_cache(it, cached); continue
    # 生成长度上限：按预计秒数 × 12 帧/秒 × 1.8 倍余量（中文约 4 字/秒，英文约 2.6 词/秒）
    secs = len(it["say"].split())/2.6 if EN else len(it["say"])/4.0
    best, budget = None, int(secs*12*1.8)+40
    for t in range(a.takes):
        p = synth(it["say"], budget, str(WORK/f"L{it['i']:03d}_t{t}"))
        y, sr = sf.read(p, dtype="float32", always_2d=True); y = trim(y[:, 0], sr)
        sf.write(TAKES/f"L{it['i']:03d}_t{t}.wav", y, sr, subtype="PCM_16")
        heard = asr(p)
        ratio = difflib.SequenceMatcher(None, norm(it["say"]), norm(heard), autojunk=False).ratio()
        s = pitch_spread(y, sr); dur = len(y)/sr
        # 语速：中文按字/秒，英文按词/秒
        cps = (len(it["say"].split()) if EN else len(norm(it["say"])))/max(dur, .1)
        lo, hi = (2.0, 3.9) if EN else (2.5, 8.5)
        # 念对字是第一位的：吻合度差 0.05 就压过起伏的差别（做片时遇到过，0.94 的那遍起伏大，被挑中了，结果"不装"念得像"不中"）
        # 语速异常的扣分；吻合度差不多时，才看起伏
        score = s + (-6 if ratio < .9 else 0) + (-4 if cps < lo or cps > hi else 0) + 25*ratio
        cand = {"y": y, "sr": sr, "take": t, "ratio": round(ratio, 3), "st_std": s,
                "heard": heard, "dur": round(dur, 3), "cps": round(cps, 2)}
        if EN: cand["syl_rate"] = round(syllables(it["say"])/speech_secs(y, sr), 2)
        # 像不像你：相似度差 0.003 约等于 1 分（同一句几遍之间一般差 0.001 到 0.005）
        if LIKE: cand["like"] = likeness(y, sr); score += 300*(cand["like"] - .98)
        if EN and a.rate: score -= 40*abs(math.log(cand["syl_rate"]/a.rate))
        # 英文先看念没念对（吻合度 0.95 以上的排前面），再比像不像、语速和起伏
        cand["score"] = (ratio >= .95 if EN else True, round(score, 3))
        if best is None or cand["score"] > best["score"]: best = cand
    keep = ("take", "ratio", "st_std", "heard", "dur", "cps", "syl_rate", "like")
    info = {k: best[k] for k in keep if k in best}; y = best["y"]
    if EN and a.fit_rate and a.rate:
        y, info["stretch"] = fit(y, best["sr"], it["say"]); y = cap_pauses(y, best["sr"])
        info.update(dur=round(len(y)/best["sr"], 3), syl_rate=round(syllables(it["say"])/speech_secs(y, best["sr"]), 2))
    put(it, y, best["sr"], info)
    print(it["i"], it["section"], info["dur"], "s", info.get("syl_rate", best["cps"]), "syl/s" if EN else "cps",
          "ratio", best["ratio"], *(["like", best["like"]] if LIKE else []), "|", best["heard"][:60],
          f"[{round(time.time()-t0)}s]", flush=True)
if a.save_voice:   # 定下来的声音存成包，以后 --voice 直接用
    if model is None and (VPS or LIKE): model = load_model(a.model); setup(model)
    V = Path(a.save_voice); V.mkdir(parents=True, exist_ok=True)
    if not a.voiceprint_only:   # 只借音色的包用不到参考段
        shutil.copy(REF, V/"ref.wav"); (V/"ref.txt").write_text(REF_TEXT + "\n", "utf-8")
    if VP is not None: np.save(V/"voiceprint.npy", VP.reshape(1, -1).astype(np.float32))
    if TARGET is not None: np.save(V/"like.npy", TARGET.astype(np.float32))
    (V/"voice.json").write_text(json.dumps({"lang": a.lang, "rate": a.rate, "takes": a.takes, "voiceprint_only": a.voiceprint_only,
        "temperature": a.temperature, "chunk_words": a.chunk_words, "fit_rate": a.fit_rate,
        **({} if a.voiceprint_only else {"ref_from": a.ref, "ref_tempo_applied": a.ref_tempo}),
        "voiceprint_from": [str(p) for p in VPS], "like_from": [str(p) for p in LIKE], "model": a.model, "voice_key": VOICE,
        "saved": datetime.date.today().isoformat()}, ensure_ascii=False, indent=1), "utf-8")
    print("声音包已存：", V, "（以后 --voice", V, "直接用）", flush=True)
print("VOICE LINES DONE", flush=True)

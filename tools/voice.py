"""逐句生成旁白：千问 Qwen3-TTS Base 克隆 ref.wav 的声音，每句生成 N 遍，用 Whisper 回听打分，留最好的一遍。

用法：python voice.py <稿子.txt> <输出目录> [--takes 4] [--only 1,2,5]
- 稿子格式：一句一行；[段落名] 开新段；{念法=>字幕}；行尾 "| 0.4" 是句后停顿秒数。
- 可以断点续跑：念法没变的句子直接跳过。所有遍都留在 takes/，方便换耳朵挑。
"""
import argparse, difflib, hashlib, json, re, shutil, time
from pathlib import Path
import librosa, mlx_whisper, numpy as np, soundfile as sf
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

ap = argparse.ArgumentParser()
ap.add_argument("script"); ap.add_argument("out")
ap.add_argument("--ref", default=str(Path(__file__).with_name("ref.wav")))
ap.add_argument("--takes", type=int, default=4)
ap.add_argument("--only", default="")
ap.add_argument("--model", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
ap.add_argument("--asr", default="mlx-community/whisper-large-v3-turbo")
ap.add_argument("--hint", default="以下是普通话的句子。")
a = ap.parse_args()

OUT = Path(a.out); LINES = OUT/"lines"; TAKES = OUT/"takes"; WORK = OUT/".work"
for d in (LINES, TAKES, WORK): d.mkdir(parents=True, exist_ok=True)
REF = Path(a.ref); REF_TEXT = REF.with_suffix(".txt").read_text("utf-8").strip()


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


norm = lambda s: re.sub(r"[^\w]", "", s.lower())


def asr(p):
    return mlx_whisper.transcribe(str(p), path_or_hf_repo=a.asr, language="zh", initial_prompt=a.hint,
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
for it in items:
    if only and it["i"] not in only: continue
    dest = LINES/f"L{it['i']:03d}.wav"; meta = LINES/f"L{it['i']:03d}.json"; c = CACHE/key(it["say"])
    if c.with_suffix(".wav").exists():
        shutil.copy(c.with_suffix(".wav"), dest)
        meta.write_text(json.dumps({**json.loads(c.with_suffix(".json").read_text("utf-8")), **it}, ensure_ascii=False), "utf-8"); continue
    if model is None: model = load_model(a.model)
    best, budget = None, int(len(it["say"])/4.0*12*1.8)+40
    for t in range(a.takes):
        pre = str(WORK/f"L{it['i']:03d}_t{t}")
        for old in WORK.glob(f"L{it['i']:03d}_t{t}*.wav"): old.unlink()
        generate_audio(text=it["say"], model=model, ref_audio=str(REF), ref_text=REF_TEXT, lang_code="zh",
                       file_prefix=pre, verbose=False, max_tokens=budget)
        p = next(WORK.glob(f"L{it['i']:03d}_t{t}*.wav"))
        y, sr = sf.read(p, dtype="float32", always_2d=True); y = trim(y[:, 0], sr)
        sf.write(TAKES/f"L{it['i']:03d}_t{t}.wav", y, sr, subtype="PCM_16")
        heard = asr(p)
        ratio = difflib.SequenceMatcher(None, norm(it["say"]), norm(heard), autojunk=False).ratio()
        s = pitch_spread(y, sr); dur = len(y)/sr; cps = len(norm(it["say"]))/max(dur, .1)
        # 念对字是第一位的：吻合度差 0.05 就压过起伏的差别（做片时遇到过，0.94 的那遍起伏大，被挑中了，结果"不装"念得像"不中"）
        # 语速异常的扣分；吻合度差不多时，才看起伏
        score = s + (-6 if ratio < .9 else 0) + (-4 if cps < 2.5 or cps > 8.5 else 0) + 25*ratio
        cand = {"score": score, "y": y, "sr": sr, "take": t, "ratio": round(ratio, 3), "st_std": s,
                "heard": heard, "dur": round(dur, 3), "cps": round(cps, 2)}
        if best is None or score > best["score"]: best = cand
    sf.write(dest, best["y"], best["sr"], subtype="PCM_16")
    meta.write_text(json.dumps({**it, **{k: best[k] for k in ("take", "ratio", "st_std", "heard", "dur", "cps")}},
                               ensure_ascii=False), "utf-8")
    shutil.copy(dest, c.with_suffix(".wav")); shutil.copy(meta, c.with_suffix(".json"))
    print(it["i"], it["section"], best["dur"], "s", best["cps"], "cps ratio", best["ratio"], "|", best["heard"][:60],
          f"[{round(time.time()-t0)}s]", flush=True)
print("VOICE LINES DONE", flush=True)

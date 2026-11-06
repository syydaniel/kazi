"""整体提速：把逐句配音用 ffmpeg 的 atempo（WSOLA，专门给语音用的变速算法）加快一点，音高和音色不变。

为什么不在生成时调语速：千问克隆模型没有可靠的语速参数，重新生成还要重新挑遍数；
变速只动时间轴，念对的句子还是那一遍。
实测 12 句：atempo 1.07 倍后语音识别吻合度 0.943，和原声 0.943 一样；Rubber Band R3 是 0.934，R2 是 0.939。
对齐结果直接按比例缩放（变速的时间映射是均匀的），不用重新对齐。

用法：python vo/tempo.py vo/main vo/fast --tempo 1.07
之后：python vo/timeline.py vo/script.txt vo/fast <timeline.json> --pause-scale 0.8
"""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--tempo", type=float, default=1.07)
a = ap.parse_args()
SRC, DST = Path(a.src), Path(a.dst)
for d in ("lines", "align", "cache"): (DST/d).mkdir(parents=True, exist_ok=True)
key = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
tag = f"at{a.tempo:.3f}"

n_new = 0
for meta in sorted((SRC/"lines").glob("L*.json")):
    d = json.loads(meta.read_text("utf-8")); k = key(d["say"])
    wav = meta.with_suffix(".wav"); c = DST/"cache"/f"{k}_{tag}.wav"
    if not c.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", f"atempo={a.tempo}", "-c:a", "pcm_s16le", str(c)], check=True)
        n_new += 1
    shutil.copy(c, DST/"lines"/wav.name)
    d2 = {**d, "tempo": a.tempo}
    if "dur" in d2: d2["dur"] = round(d2["dur"]/a.tempo, 3)
    (DST/"lines"/meta.name).write_text(json.dumps(d2, ensure_ascii=False), "utf-8")
    al = SRC/"align"/f"{k}.json"
    if al.exists():
        segs = json.loads(al.read_text("utf-8"))
        (DST/"align"/f"{k}.json").write_text(json.dumps([{**s, "s": round(s["s"]/a.tempo, 3), "e": round(s["e"]/a.tempo, 3)} for s in segs],
                                                         ensure_ascii=False), "utf-8")
# 已经不在稿子里的旧句子（序号比现在多出来的）删掉，免得 timeline 误用
have = {m.name for m in (SRC/"lines").glob("L*.*")}
for m in (DST/"lines").glob("L*.*"):
    if m.name not in have: m.unlink()
print(f"变速 {a.tempo}×，新处理 {n_new} 句")

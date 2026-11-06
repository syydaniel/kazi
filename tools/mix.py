"""混音：旁白、配乐、音效三轨，按逐字时间让位，统一响度，可以直接封进视频。

用法：
  python mix.py --timeline timeline.json --voice vo/narration.wav --out audio/mix.wav \
      [--music audio/music.wav] [--cues out/cues.json --sfx-dir audio/sfx] [--stems audio/stems] \
      [--clips out/clips.json --clip-root film/public] [--video out/video.mp4 --final out/final.mp4]

让位（ducking）怎么做
  - 从 timeline.json 取每句话第一个字到最后一个字，前面提前 0.2 秒、后面延后 0.15 秒，算"在说话"。
  - 两句之间空当不到 1.2 秒，就当一直在说话，音乐不浮上来（来回起伏会很吵）。
  - 说话时音乐比旁白低 15 LU；空当里低 6 LU。压下去快（约 80 毫秒），浮上来慢（约 0.5 秒）。
  - 说话时再把音乐的 1 到 4 kHz 压低 3 dB，给人声的清晰度让出位置。
  - 音效在说话时也低 3 dB。
音效
  - cues.json：[{"t": 秒, "kind": "pop", "gain": -10, "pan": 0.2}]，t 是声音峰值要落的时刻。
  - --sfx-dir 下按 kind 建子文件夹放 wav；同一种有多个文件时轮流用，避免听出重复。
  - 没有对应文件夹时，用内置合成音效兜底（soft_in、pop、tick、whoosh、paper），可以先用着，正式片子换成好的音效库。
现成的音频片段（--clips）
  - clips.json：[{"t": 秒, "src": "audio/xx.wav", "from": 0, "dur": 18.7, "gain": 0}]，src 相对 --clip-root。
    用来放片子里要原样播放的声音：测试视频的配音、A/B 试听。
  - 默认把片段的响度对齐到旁白，gain 在这之上再加减（dB）；首尾各 20 毫秒淡入淡出。
  - 放片段的那段时间也算"在说话"，音乐和音效一样让位。
母带
  - 积分响度 -14 LUFS，真峰值不超过 -1 dBTP（4 倍过采样估算），48 kHz / 24 bit 立体声。
    wav 本身压到 -1.8：封成 AAC 以后真峰值会抬高约 0.5 dB（实测 -1.2 变 -0.7），留出这个余量。
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

import numpy as np, pyloudnorm as pyln, soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

ap = argparse.ArgumentParser()
ap.add_argument("--timeline", required=True)
ap.add_argument("--voice", required=True)
ap.add_argument("--music", default="")
ap.add_argument("--cues", default="")
ap.add_argument("--sfx-dir", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--stems", default="")
ap.add_argument("--lufs", type=float, default=-14.0)
ap.add_argument("--tp", type=float, default=-1.0)
ap.add_argument("--music-under", type=float, default=15.0, help="说话时音乐比旁白低多少 LU")
ap.add_argument("--music-gap", type=float, default=6.0, help="空当里音乐比旁白低多少 LU")
ap.add_argument("--clips", default="")
ap.add_argument("--clip-root", default=".")
ap.add_argument("--video", default="")
ap.add_argument("--final", default="")
a = ap.parse_args()
SR = 48000


def load(path, ch=2):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        g = np.gcd(sr, SR); y = resample_poly(y, SR // g, sr // g, axis=0).astype(np.float32)
    if y.shape[1] == 1 and ch == 2: y = np.repeat(y, 2, axis=1)
    return y[:, :ch]


def fit(y, n):
    return y[:n] if len(y) >= n else np.pad(y, ((0, n - len(y)), (0, 0)))


tl = json.loads(Path(a.timeline).read_text("utf-8"))
voice = load(a.voice)
N = max(len(voice), int(round(tl["total"] * SR)))
voice = fit(voice, N)
meter = pyln.Meter(SR)

# 说话区间（现成的音频片段也算）
clips = json.loads(Path(a.clips).read_text("utf-8")) if a.clips else []
spans = []
for l in tl["lines"]:
    if l["words"]: spans.append([l["words"][0]["s"] - 0.2, l["words"][-1]["e"] + 0.15])
for c in clips:
    spans.append([c["t"] - 0.2, c["t"] + c.get("dur", 0) + 0.15])
spans.sort(); merged = []
for s, e in spans:
    if merged and s - merged[-1][1] < 1.2: merged[-1][1] = max(merged[-1][1], e)
    else: merged.append([s, e])
RATE = 100
mask = np.zeros(int(N / SR * RATE) + 2)
for s, e in merged: mask[max(0, int(s * RATE)):int(e * RATE) + 1] = 1


def smooth_db(target, down=0.08, up=0.5):
    """压下去快、浮上来慢的一阶平滑（单位 dB，100 Hz）。"""
    out = np.empty_like(target); cur = target[0]
    ad, au = np.exp(-1 / (down * RATE)), np.exp(-1 / (up * RATE))
    for i, x in enumerate(target):
        k = ad if x < cur else au
        cur = k * cur + (1 - k) * x; out[i] = cur
    return out


def to_audio_rate(ctrl):
    return np.interp(np.arange(N) / SR, np.arange(len(ctrl)) / RATE, ctrl).astype(np.float32)


voice_l = meter.integrated_loudness(voice) if np.abs(voice).max() > 0 else -23.0

music = np.zeros((N, 2), np.float32)
if a.music:
    m = fit(load(a.music), N)
    ml = meter.integrated_loudness(m)
    base = (voice_l - a.music_gap) - ml
    target = np.where(mask > 0, base - (a.music_under - a.music_gap), base)
    g = 10 ** (to_audio_rate(smooth_db(target)) / 20)
    # 说话时把 1 到 4 kHz 再压 3 dB
    band = sosfiltfilt(butter(2, [1000, 4000], "bandpass", fs=SR, output="sos"), m, axis=0).astype(np.float32)
    dip = -to_audio_rate(smooth_db(-mask * (1 - 10 ** (-3 / 20))))  # 说话时约 0.29，其余 0
    music = (m - band * dip[:, None]) * g[:, None]


def synth(kind, seed):
    """内置兜底音效：都是滤波噪声和简单包络，不带音高旋律。"""
    rng = np.random.default_rng(seed)
    def env(n, att, dec):
        t = np.arange(n) / SR
        return np.minimum(1, t / max(att, 1e-4)) * np.exp(-np.maximum(0, t - att) / dec)
    def bp(x, lo, hi):
        return sosfiltfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), x)
    if kind == "tick":
        n = int(0.08 * SR); y = bp(rng.standard_normal(n), 1800, 7000) * env(n, 0.001, 0.012)
    elif kind == "pop":
        n = int(0.25 * SR); t = np.arange(n) / SR
        body = np.sin(2 * np.pi * (180 * t + 220 * (1 - np.exp(-t / 0.02)) * 0.02)) * env(n, 0.002, 0.05)
        y = body * 0.8 + bp(rng.standard_normal(n), 800, 4000) * env(n, 0.001, 0.008) * 0.5
    elif kind == "whoosh":
        n = int(0.6 * SR); t = np.arange(n) / SR; x = rng.standard_normal(n)
        lo = sosfiltfilt(butter(2, 500, "lowpass", fs=SR, output="sos"), x); hi = bp(x, 1500, 6000)
        sweep = t / t[-1]
        y = (lo * (1 - sweep) + hi * sweep) * np.sin(np.pi * sweep) ** 2
    elif kind == "paper":
        n = int(0.35 * SR); x = bp(rng.standard_normal(n), 2000, 9000)
        grain = (rng.random(n) < 0.004).astype(float); grain = np.convolve(grain, np.hanning(int(0.01 * SR)), "same")
        y = x * (0.3 + grain) * np.hanning(n)
    else:  # soft_in 及其他
        n = int(0.5 * SR); t = np.arange(n) / SR
        y = bp(rng.standard_normal(n), 300, 3000) * np.sin(np.pi * np.minimum(1, t / 0.5)) ** 2
    y = y / (np.abs(y).max() + 1e-9)
    return np.stack([y, y], 1).astype(np.float32)


def sample_for(kind, i):
    d = Path(a.sfx_dir) / kind if a.sfx_dir else None
    files = sorted(d.glob("*.wav")) if d and d.is_dir() else []
    if files:
        return load(str(files[i % len(files)]))
    return synth(kind, int(hashlib.md5(f"{kind}{i}".encode()).hexdigest()[:8], 16))


sfx = np.zeros((N, 2), np.float32)
if a.cues:
    cues = json.loads(Path(a.cues).read_text("utf-8")); count = {}
    for c in cues:
        k = c["kind"]; count[k] = count.get(k, 0) + 1
        y = sample_for(k, count[k] - 1)
        fade = min(len(y) // 4, int(0.005 * SR)); y = y.copy(); y[:fade] *= np.linspace(0, 1, fade)[:, None]; y[-fade:] *= np.linspace(1, 0, fade)[:, None]
        peak = int(np.argmax(np.abs(y).max(1)))  # 峰值对齐到 t
        pan = float(c.get("pan", 0)); th = (pan + 1) * np.pi / 4
        y = y * np.array([np.cos(th), np.sin(th)]) * np.sqrt(2) * 10 ** (c.get("gain", -12) / 20)
        i0 = int(round(c["t"] * SR)) - peak
        j0, j1 = max(0, i0), min(N, i0 + len(y))
        if j1 > j0: sfx[j0:j1] += y[j0 - i0:j1 - i0]
    sfx *= 10 ** (to_audio_rate(smooth_db(mask * -3.0)) / 20)[:, None]

clip = np.zeros((N, 2), np.float32)
for c in clips:
    y = load(str(Path(a.clip_root) / c["src"]))
    i0 = int(round(c.get("from", 0) * SR)); y = y[i0:]
    if c.get("dur"): y = y[:int(round(c["dur"] * SR))]
    if not len(y): continue
    lv = meter.integrated_loudness(y) if len(y) > SR * 0.5 else voice_l
    y = y * 10 ** ((voice_l - lv + c.get("gain", 0)) / 20)
    fade = min(len(y) // 4, int(0.02 * SR)); y = y.copy(); y[:fade] *= np.linspace(0, 1, fade)[:, None]; y[-fade:] *= np.linspace(1, 0, fade)[:, None]
    j0 = int(round(c["t"] * SR)); j1 = min(N, j0 + len(y))
    if j1 > j0 >= 0: clip[j0:j1] += y[:j1 - j0]
    print(f"片段 {c['src']} @ {c['t']:.2f}s，{len(y) / SR:.1f} 秒，响度对齐旁白 {voice_l - lv + c.get('gain', 0):+.1f} dB")

mix = voice + music + sfx + clip
gain = 10 ** ((a.lufs - meter.integrated_loudness(mix)) / 20)


def limit(x, ceiling_db):
    """真峰值限制：4 倍过采样估峰值，提前 3 毫秒压，60 毫秒恢复。"""
    c = 10 ** (ceiling_db / 20)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[:len(x)] if len(up) >= 4 * len(x) else np.abs(x).max(1)
    need = np.minimum(1, c / np.maximum(pk, 1e-9))
    look = int(0.003 * SR)
    # 保持窗：每个点取前后 look 范围内的最小增益
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * look + 1)
    out = np.empty_like(need); cur = 1.0; rel = np.exp(-1 / (0.06 * SR))
    for i, v in enumerate(need):
        cur = v if v < cur else rel * cur + (1 - rel) * v
        out[i] = cur
    return x * out[:, None]


HEAD = 0.8  # 给 AAC 编码留的真峰值余量（dB）
master = limit(mix * gain, a.tp - HEAD)
for _ in range(2):  # 限制会让响度掉一点，补回来再限一次
    d = a.lufs - meter.integrated_loudness(master)
    if abs(d) < 0.2: break
    gain *= 10 ** (d / 20); master = limit(mix * gain, a.tp - HEAD)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, master, SR, subtype="PCM_24")
if a.stems:
    st = Path(a.stems); st.mkdir(parents=True, exist_ok=True)
    for name, y in (("voice", voice), ("music", music), ("sfx", sfx), ("clips", clip)):
        sf.write(st / f"{name}.wav", y * gain, SR, subtype="PCM_24")
print(f"混音 {N / SR:.1f} 秒，响度 {meter.integrated_loudness(master):.1f} LUFS，样本峰值 {20 * np.log10(np.abs(master).max() + 1e-9):.1f} dBFS → {out}")
if a.video and a.final:
    # 有苹果的 AAC 编码器（aac_at）就用它：ffmpeg 自带的 aac 在 320k 下会冒出 1 dB 多的真峰值尖刺
    encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    aenc = ["-c:a", "aac_at", "-aac_at_mode", "cbr"] if " aac_at " in encs else ["-c:a", "aac"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-i", str(out), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    *aenc, "-b:a", "320k", "-shortest", a.final], check=True)
    print("已封进视频 →", a.final)

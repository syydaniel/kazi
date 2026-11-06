"""卡字检查：量一条成片的画面有没有卡在旁白的关键词上，顺便查黑帧、闪帧、长时间不动、响度和音轨。

用法：
  python kazi_check.py out/final.mp4 --timeline timeline.json [--json out/check.json]
  python kazi_check.py out/final.mp4 --keywords keywords.json      # [{"word": "变大", "t": 3.42}, ...]

卡字率
  每个关键词第一个字开始念的时刻记作 t0。看两段时间里画面动了多少：
    关键词这一下：[t0 - 0.35, t0 + 0.15] 秒
    它前面一秒：  [t0 - 1.35, t0 - 0.45] 秒
  "动了多少"是每一帧和上一帧相比、变化超过 6 个灰阶的像素占比（画面缩到 192×108 灰度再算）。
  关键词这一下的平均动量，至少是前面一秒的 1.5 倍、并且多出画面的 0.1% 以上，就算卡上。
  窗口前宽后窄：画面略早于声音不容易被察觉，晚了很容易被察觉（ITU-R BT.1359：声音超前约 45 ms 可察觉，
  画面超前约 125 ms 才可察觉）。一个 15 到 24 帧的动作要在 t0 前 3 帧落定，起点就在 t0 前 0.3 秒左右。
  画面一直在动的片子拿不到分：要的是"在这个词上，画面明显多动了一下"。
随机基线
  把同样的判断放到旁白里的每一个时刻上，有多大比例也会判成"卡上"。卡字率减去基线，报告里叫"高出随机"。
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--timeline", default="")
ap.add_argument("--keywords", default="")
ap.add_argument("--early", type=float, default=0.35, help="关键词前多少秒内开始动算卡上")
ap.add_argument("--late", type=float, default=0.15, help="关键词后多少秒内开始动算卡上")
ap.add_argument("--offset", type=float, default=0.0, help="视频第 0 秒对应时间表里的第几秒（只渲染了一段时用）")
ap.add_argument("--json", default="")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
W, H = 192, 108


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def keywords_from_timeline(tl):
    """kazi 的 timeline.json：cap 里 k=1 的连续字是一个关键词。也接受顶层的 keywords 列表。"""
    if tl.get("keywords"):
        return [{"word": k["word"], "t": k["t"], "line": k.get("line")} for k in tl["keywords"]]
    out = []
    for l in tl["lines"]:
        run = []
        for c in l["cap"] + [{"c": "", "s": 0}]:
            if c.get("k"):
                run.append(c)
            elif run:
                out.append({"word": "".join(x["c"] for x in run), "t": run[0]["s"], "line": l["i"]}); run = []
    return out


info = probe(a.video)
vs = next(s for s in info["streams"] if s["codec_type"] == "video")
num, den = map(int, vs.get("avg_frame_rate", vs["r_frame_rate"]).split("/"))
fps = num / den if den else 30.0
dur = float(info["format"].get("duration", vs.get("duration", 0)))
auds = [s for s in info["streams"] if s["codec_type"] == "audio"]

# 逐帧读灰度小图，边读边算，不把整条片子放进内存
cmd = ["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
N = W * H
act, mad, lum, flash = [], [], [], []
prev = prev2 = None
while True:
    buf = proc.stdout.read(N)
    if len(buf) < N: break
    f = np.frombuffer(buf, np.uint8).astype(np.int16).reshape(H, W)
    lum.append(float(f.mean()))
    if prev is None:
        act.append(0.0); mad.append(0.0)
    else:
        d = np.abs(f - prev)
        act.append(float((d > 6).mean())); mad.append(float(d.mean()))
        # 一闪而过：这一帧和前后都差很多，但前后两帧几乎一样
        if prev2 is not None and mad[-2] > 10 and float(np.abs(f - prev2).mean()) < 2.5:
            flash.append(len(act) - 2)
    prev2, prev = prev, f
proc.wait()
act = np.array(act); mad = np.array(mad); lum = np.array(lum); n = len(act)
if n < 2: sys.exit("读不到画面")

# 动量对比
sm = np.convolve(act, np.ones(3) / 3, mode="same")


def mean_in(t1, t2):
    i1, i2 = max(0, int(round(t1 * fps))), min(n, int(round(t2 * fps)) + 1)
    return float(sm[i1:i2].mean()) if i2 > i1 else 0.0


def judge(t0):
    near = mean_in(t0 - a.early, t0 + a.late); before = mean_in(t0 - a.early - 1.0, t0 - a.early - 0.1)
    return near >= 1.5 * before + 0.001, near, before


kws = []
if a.timeline:
    tl = json.loads(Path(a.timeline).read_text("utf-8")); kws = keywords_from_timeline(tl)
elif a.keywords:
    kws = json.loads(Path(a.keywords).read_text("utf-8"))
if a.offset:
    kws = [{**k, "t": round(k["t"] - a.offset, 3)} for k in kws]
kws = [k for k in kws if 0 <= k["t"] <= dur]

report = {"video": str(a.video), "fps": round(fps, 3), "frames": n, "duration": round(dur, 3),
          "size": f"{vs.get('width')}x{vs.get('height')}", "audio_streams": len(auds)}
problems = []

if kws:
    rows = []
    for k in kws:
        ok, near, before = judge(k["t"])
        rows.append({**k, "hit": bool(ok), "near": round(near * 100, 2), "before": round(before * 100, 2)})
    hits = sum(r["hit"] for r in rows)
    lo = max(min(k["t"] for k in kws) - 1.0, a.early + 1.0); hi_ = min(max(k["t"] for k in kws) + 1.0, n / fps - a.late)
    grid = np.arange(lo, hi_, 1 / fps)
    chance = float(np.mean([judge(t)[0] for t in grid])) if len(grid) else 0.0
    rate = hits / len(rows)
    report.update({"keywords": len(rows), "hits": hits, "kazi_rate": round(rate, 3), "chance": round(chance, 3),
                   "lift": round(rate - chance, 3), "window": [-a.early, a.late], "detail": rows})

# 其他检查
black = np.where(lum < 8)[0]
black = [int(i) for i in black if 0.5 * fps < i < n - 0.5 * fps]
if black: problems.append(f"黑帧 {len(black)} 帧，第一处在 {black[0] / fps:.2f} 秒")
if flash: problems.append(f"一闪而过的帧 {len(flash)} 处：" + "、".join(f"{i / fps:.2f}s" for i in flash[:8]))
cuts = [i for i in range(1, n) if act[i] > 0.5]
static, run_, longest, at_ = act < 0.0005, 0, 0, 0
for i, s in enumerate(static):
    run_ = run_ + 1 if s else 0
    if run_ > longest: longest, at_ = run_, i - run_ + 1
if longest / fps > 6: problems.append(f"画面有 {longest / fps:.1f} 秒完全不动（从 {at_ / fps:.1f} 秒开始）")
report.update({"black_frames": len(black), "flash_frames": [round(i / fps, 3) for i in flash], "hard_cuts": len(cuts),
               "longest_static": round(longest / fps, 2)})

if auds:
    r = subprocess.run(["ffmpeg", "-nostats", "-i", a.video, "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    m_i = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr); m_p = re.findall(r"Peak:\s+(-?[\d.inf]+) dBFS", r.stderr)
    if m_i: report["lufs"] = float(m_i[-1])
    if m_p:
        try: report["true_peak"] = float(m_p[-1])
        except ValueError: pass
    if "lufs" in report and not (-16.5 <= report["lufs"] <= -12.5): problems.append(f"响度 {report['lufs']} LUFS，网络视频一般在 -14 左右")
    if report.get("true_peak", -99) > -1.0: problems.append(f"真峰值 {report['true_peak']} dBTP，超过 -1")
else:
    problems.append("没有音轨")
if a.timeline:
    total = tl.get("total")
    if total and not a.offset and abs(total - dur) > 0.5: problems.append(f"片长 {dur:.2f} 秒，和时间表 {total:.2f} 秒不一致")
report["problems"] = problems

if a.json: Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
if not a.quiet:
    print(f"{a.video}  {report['size']}  {fps:.2f}fps  {dur:.2f}s  音轨 {len(auds)} 条")
    if kws:
        print(f"卡字率 {report['hits']}/{report['keywords']} = {report['kazi_rate']:.0%}，随机基线 {report['chance']:.0%}，高出随机 {report['lift']:+.0%}")
        for r_ in report["detail"]:
            mark = "卡上" if r_["hit"] else "没卡上"
            print(f"  {mark:3s} {r_['t']:7.2f}s {r_['word']:6s} 这一下动了 {r_['near']:.2f}%，前一秒 {r_['before']:.2f}%")
    if "lufs" in report: print(f"响度 {report['lufs']} LUFS，真峰值 {report.get('true_peak')} dBTP")
    print("问题：" + ("；".join(problems) if problems else "没有"))

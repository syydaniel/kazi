"""音效打点筛选：动画里每个动作都打了点，全放出来会很吵（这条片子导出了 861 个，平均每分钟 58 个）。
按"轻重缓急"留下最要紧的：

1. 优先级：重击、叮（关键词、结论）> 嗖（镜头大移动）> 弹出 > 点击、纸、笔、键盘、电机、提示音 > 滴答、轻亮。
2. 从优先级高的往下挑，一个点要同时满足这些才留下：
   - 离已经留下的任何一个点不少于 GAP 秒（默认 0.3）；
   - 同一种声音离上一个同类不少于它自己的最小间隔（滴答 0.6 秒、嗖 0.8 秒……见 SPACING）；
   - 前后 1.5 秒里最多 3 个；
   - 前后 30 秒里不超过 PER_MIN 个（默认 22，也就是每分钟 22 个左右）；同一种声音不超过 CAP 里它的数。
3. 最后按时间排好写出来，打印每种留了多少、每分钟多少。

经验值：轻的音效每分钟 4 到 10 个，中等的 1 到 3 个，重的全片 8 到 12 个；加起来不该超过每分钟二三十个，再多就像一直在敲东西。
用法：python tools/thin_cues.py out/cues.json out/cues_thin.json [--per-min 22] [--gap 0.3]
"""
import argparse, json
from pathlib import Path

PRI = {"thud": 5, "chime": 5, "whoosh": 4, "swoosh": 4, "pop": 3,
       "click": 2, "paper": 2, "pen": 2, "type": 2, "servo": 2, "beep": 2, "tick": 1, "soft_in": 1}
# 每种声音每分钟最多几个（前后 30 秒里数）：重的少、轻的多
CAP = {"thud": 2, "chime": 2, "whoosh": 3, "swoosh": 3, "pop": 8, "click": 3, "paper": 2,
       "pen": 1, "type": 1, "servo": 1, "beep": 1, "tick": 3, "soft_in": 1}
SPACING = {"thud": 0.5, "chime": 0.8, "whoosh": 0.8, "swoosh": 0.5, "pop": 0.25, "click": 0.3, "paper": 0.5,
           "pen": 0.8, "type": 0.4, "servo": 0.8, "beep": 0.5, "tick": 0.6, "soft_in": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("cues"); ap.add_argument("out")
ap.add_argument("--per-min", type=int, default=22)
ap.add_argument("--gap", type=float, default=0.3)
a = ap.parse_args()

cues = json.loads(Path(a.cues).read_text("utf-8"))
order = sorted(cues, key=lambda c: (-PRI.get(c["kind"], 1), -c.get("gain", -12), c["t"]))
kept = []
for c in order:
    t, k = c["t"], c["kind"]
    if any(abs(t - x["t"]) < a.gap for x in kept): continue
    if any(x["kind"] == k and abs(t - x["t"]) < SPACING.get(k, 0.4) for x in kept): continue
    if sum(abs(t - x["t"]) <= 1.5 for x in kept) >= 3: continue
    if sum(abs(t - x["t"]) <= 30 for x in kept) >= a.per_min: continue
    if sum(x["kind"] == k and abs(t - x["t"]) <= 30 for x in kept) >= CAP.get(k, 2): continue
    kept.append(c)
kept.sort(key=lambda c: c["t"])
Path(a.out).write_text(json.dumps(kept, ensure_ascii=False, indent=1), "utf-8")

span = max(c["t"] for c in cues) / 60 if cues else 1
print(f"{len(cues)} 个 → 留下 {len(kept)} 个（平均每分钟 {len(kept) / span:.1f} 个）")
by = {}
for c in kept: by[c["kind"]] = by.get(c["kind"], 0) + 1
print("按种类：" + "，".join(f"{k} {by[k]}/{sum(x['kind'] == k for x in cues)}" for k in sorted(by, key=lambda k: -PRI.get(k, 1))))
mins = [0] * (int(span) + 1)
for c in kept: mins[int(c["t"] // 60)] += 1
print("每分钟：" + " ".join(str(m) for m in mins))

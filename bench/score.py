"""给对照测试打分：对每个目录的 out/final.mp4 跑 kazi_check，汇总成一张表。

用法：python bench/score.py <测试根目录> [--json bench/results_mine.json]
关键词的时刻用 bench/<题目>/timeline.json（同一段旁白逐字对齐的结果），四组用的是同一张表。
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--json", default="")
ap.add_argument("--topic", default="planets", choices=["planets", "anc"])
a = ap.parse_args()
R = Path(a.root).resolve()
out = {}
for name in ["opus_base", "opus_kazi", "sonnet_base", "sonnet_kazi"]:
    v = R/name/"out"/"final.mp4"
    if not v.exists():
        print(f"{name}: 没有 out/final.mp4"); continue
    j = R/name/"check.json"
    subprocess.run([sys.executable, str(HERE.parent/"tools"/"kazi_check.py"), str(v), "--timeline", str(HERE/a.topic/"timeline.json"), "--json", str(j)],
                   stdout=subprocess.DEVNULL, check=False)
    r = json.loads(j.read_text("utf-8"))
    out[name] = r
    print(f"{name:12s} 卡字率 {r.get('hits')}/{r.get('keywords')}  随机基线 {round(100 * r.get('chance', 0))}%  高出随机 {round(100 * r.get('lift', 0)):+d}%")
if a.json:
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")

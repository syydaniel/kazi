"""稿子检查：把写稿那一步的规矩变成能跑的检查，不靠记性。

查的东西：破折号、"不是……而是"句式、禁用词、一口气太长的短语、字幕切段后每行字数和关键词个数、估算时长。
用法: python vo/lint.py vo/script.txt [--cps 5.0] [--max-line 16]
"""
import argparse
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("--cps", type=float, default=5.0, help="估算用的语速（字/秒）")
ap.add_argument("--max-line", type=int, default=16, help="字幕每行最多几个字")
ap.add_argument("--max-breath", type=int, default=22, help="两个标点之间最多几个字")
a = ap.parse_args()

UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-']*|[㐀-鿿]")
BREAK = "，。？！；：、,.?!;:"
BANNED = ["最强", "颠覆", "一键", "零门槛", "总结一下", "赋能", "干货满满", "保姆级"]
END_ONLY = ["最后", "总之"]  # 只允许出现在结尾段

problems, total, sec_t, cur = [], 0.0, {}, None
lines = Path(a.script).read_text("utf-8").splitlines()
sections = [l.strip()[1:-1] for l in lines if l.strip().startswith("[")]
last_sec = sections[-1] if sections else None
for n, raw in enumerate(lines, 1):
    l = raw.strip()
    if not l or l.startswith("#"):
        continue
    if l.startswith("@gap"):
        d = float(l.split()[1]); total += d; sec_t[cur] = sec_t.get(cur, 0) + d
        continue
    if l.startswith("["):
        cur = l[1:-1]
        continue
    pause = 0.4
    if "|" in l:
        l, p = l.rsplit("|", 1); pause = float(p); l = l.strip()
    say = re.sub(r"\{([^}]*)=>[^}]*\}", r"\1", l).replace("*", "")
    show = re.sub(r"\{[^}]*=>([^}]*)\}", r"\1", l)
    dur = len(UNIT.findall(say)) / a.cps + pause
    total += dur; sec_t[cur] = sec_t.get(cur, 0) + dur
    where = f"第 {n} 行 [{cur}]"
    if "【" in l:
        problems.append(f"{where} 占位句，还没定稿")
    if "—" in l or "——" in l:
        problems.append(f"{where} 有破折号")
    if re.search(r"不是[^，。？！]{0,14}[，,]?(而是|是)", say):
        problems.append(f"{where} 有\"不是……是……\"句式：{say[:30]}")
    for w in BANNED:
        if w in say:
            problems.append(f"{where} 禁用词「{w}」")
    for w in END_ONLY:
        if w in say and cur != last_sec:
            problems.append(f"{where} 「{w}」只该出现在结尾段")
    for ph in re.split(f"[{BREAK}]", say):
        if len(UNIT.findall(ph)) > a.max_breath:
            problems.append(f"{where} 一口气太长（{len(UNIT.findall(ph))} 字）：{ph}")
    # 模拟字幕切段：按标点切短语，再贪心拼成不超过 max-line 的小段
    plain, keys, on = [], [], False
    for ch in show:
        if ch == "*":
            on = not on; continue
        plain.append(ch); keys.append(on)
    phrases, buf = [], []
    for ch, k in zip(plain, keys):
        buf.append((ch, k))
        if ch in BREAK:
            phrases.append(buf); buf = []
    if buf:
        phrases.append(buf)
    width = lambda cs: sum(0.55 if re.match(r"[A-Za-z0-9 .\-']", c) else 0 if c in BREAK else 1 for c, _ in cs)
    chunks, acc = [], []
    for ph in phrases:
        if width(acc) + width(ph) <= a.max_line:
            acc += ph
        else:
            if acc: chunks.append(acc)
            acc = ph
    if acc: chunks.append(acc)
    for c in chunks:
        txt = "".join(ch for ch, _ in c)
        if width(c) > a.max_line:
            problems.append(f"{where} 字幕一行 {width(c):.0f} 字超过 {a.max_line}：{txt}")
        nk = len(re.findall(r"(?:^|(?<=False))True", "".join(str(k) for _, k in c)))
        runs = sum(1 for i, (_, k) in enumerate(c) if k and (i == 0 or not c[i - 1][1]))
        if runs > 1:
            problems.append(f"{where} 一条字幕里有 {runs} 个关键词：{txt}")

print("\n".join(problems) if problems else "没有发现问题")
print(f"\n估算总长 {total / 60:.2f} 分钟（语速按每秒 {a.cps} 字，含停顿和空当）")
for k, v in sec_t.items():
    print(f"  {k or '(开头)'}: {v:.0f} 秒")

"""对照测试的准备脚本：建四个互相隔离的工作目录，每个目录一份完全一样的旁白和逐字稿，再把要发给 AI 的提示词写好。

我们实际是这样测的：在 Claude Code 里开四个子任务（subagent），模型分别是 Opus 和 Sonnet，
每个模型一遍不装卡字、一遍装上卡字。四个任务的提示词除了"已安装的 skill"那两行，一字不差。

用法：
  python bench/run.py <测试根目录> [--python /path/to/python]
然后对每个目录：在 Claude Code 里选好模型（opus 或 sonnet），把目录里 PROMPT.txt 的全文发给它（或者交给一个子任务），
中途不要回复它，等它做出 out/final.mp4。四个都做完以后：
  python bench/score.py <测试根目录>
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--topic", default="planets", choices=["planets", "anc"], help="planets：太阳系八颗行星（第二版，默认）；anc：降噪耳机（第一版）")
ap.add_argument("--python", default=sys.executable, help="提示词里告诉 AI 用哪个 Python（要装好 numpy、scipy、soundfile、librosa、pillow，Mac 上再加 mlx-audio、mlx-whisper）")
a = ap.parse_args()

R = Path(a.root).resolve(); R.mkdir(parents=True, exist_ok=True)
TOPICS = {"planets": '"太阳系的八颗行星"', "anc": '"降噪耳机为什么能降噪"'}
base = (HERE/"prompt.txt").read_text("utf-8").replace("{TOPIC}", TOPICS[a.topic])
extra = (HERE/"prompt_kazi_extra.txt").read_text("utf-8")

for name in ["opus_base", "opus_kazi", "sonnet_base", "sonnet_kazi"]:
    d = R/name
    if d.exists():
        print(f"{d} 已经存在，跳过（要重测就先删掉它）"); continue
    (d/"input").mkdir(parents=True)
    for f in ["narration.wav", "script.txt"]:
        shutil.copy(HERE/a.topic/f, d/"input"/f)
    if name.endswith("_kazi"):
        shutil.copytree(SKILL, d/".claude/skills/kazi",
                        ignore=shutil.ignore_patterns("bench", ".git", "node_modules", "__pycache__", "*.pyc", "out"))
    subprocess.run(["git", "init", "-q"], cwd=d)
    prompt = base.replace("{WORKDIR}", str(d)).replace("{PYTHON}", a.python) + (extra if name.endswith("_kazi") else "")
    (d/"PROMPT.txt").write_text(prompt, "utf-8")
    print(f"{name}: {d}  模型 {name.split('_')[0]}，{'装' if name.endswith('_kazi') else '不装'}卡字")
print("\n下一步：把每个目录里的 PROMPT.txt 发给对应模型；PROMPT.txt 本身不算任务的一部分，AI 看到也没关系。")

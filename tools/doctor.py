"""查环境：卡字要用到的程序和 Python 包在不在，缺了怎么装。只检查，不安装。

用法：python tools/doctor.py
"""
import importlib.util, platform, shutil, subprocess, sys

ok, miss = [], []


def check(name, good, fix):
    (ok if good else miss).append((name, fix))


mac_arm = sys.platform == "darwin" and platform.machine() == "arm64"
check("Python 3.10 以上", sys.version_info >= (3, 10), "装一个新的 Python，比如 https://www.python.org 或 uv")
for b, fix in [("ffmpeg", "Mac: brew install ffmpeg；Windows: winget install ffmpeg"), ("ffprobe", "和 ffmpeg 一起装"),
               ("node", "装 Node.js 20 以上：https://nodejs.org"), ("npx", "和 Node.js 一起装")]:
    check(b, shutil.which(b) is not None, fix)
if shutil.which("node"):
    v = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
    check(f"Node.js 版本 {v}", v.startswith("v") and int(v[1:].split(".")[0]) >= 20, "升级到 Node.js 20 以上")
pkgs = {"numpy": "numpy", "soundfile": "soundfile", "scipy": "scipy", "librosa": "librosa", "pyloudnorm": "pyloudnorm"}
if mac_arm:
    pkgs.update({"mlx_audio": "mlx-audio", "mlx_whisper": "mlx-whisper"})
for mod, pip in pkgs.items():
    check(f"Python 包 {pip}", importlib.util.find_spec(mod) is not None, f"pip install {pip}")

opt = []
for b, what, fix in [("dovi_tool", "出杜比视界", "brew install dovi_tool，或在 github.com/quietvoid/dovi_tool 下载"),
                     ("MP4Box", "封装杜比视界 MP4", "brew install gpac")]:
    if shutil.which(b) is None:
        opt.append((b, what, fix))
if shutil.which("ffmpeg"):
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx265" not in enc:
        opt.append(("ffmpeg 的 libx265", "HDR 编码", "换一个带 x265 的 ffmpeg（brew install ffmpeg 自带），或者用 ENC=vt"))

print(f"系统：{platform.system()} {platform.machine()}，Python {platform.python_version()}")
if not mac_arm:
    print("注意：配音（voice.py）和对齐（timeline.py）目前只在苹果芯片的 Mac 上实测过，用的是 MLX。别的平台还没实测，可以这样绕过：")
    print("  1. 配音：按 Qwen3-TTS 官方仓库（github.com/QwenLM/Qwen3-TTS）的说明装 PyTorch 版，用 NVIDIA 显卡跑；或者干脆自己录旁白。")
    print("  2. 对齐：模型是 Hugging Face 上的 Qwen/Qwen3-ForcedAligner-0.6B，照它模型卡上的用法，把 timeline.py 里 from mlx_audio.stt import load 那几行换掉，输出格式不变。")
    print("  3. 画面、检查、混音（Remotion、kazi_check.py、mix.py）不依赖 MLX，各平台都能用。")
for n, _ in ok: print("  有  ", n)
for n, f in miss: print("  缺  ", n, "→", f)
for n, what, f in opt: print("  可选", n, f"（{what}才要）→", f)
print("\n第一次配音和对齐会自动下载模型：Qwen3-TTS 1.7B、Qwen3-ForcedAligner 0.6B、Whisper large-v3-turbo，合计几个 GB。")
print("没有问题" if not miss else f"\n缺 {len(miss)} 项，装好再开工。")

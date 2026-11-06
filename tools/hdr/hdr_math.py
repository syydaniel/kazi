"""HDR 的换算（hdr_lut.py 和 hdr_mask.py 共用）。

画面是 Remotion 出的 sRGB SDR。换算顺序：
  sRGB 码值 → 线性光 → 显示亮度（纸色 #F4EFE5 定到 PAPER_NITS 尼特）→ 可选提亮 GAIN 倍并软剪切到 PEAK
  → BT.709 原色换成 BT.2020 原色 → 1000 尼特显示器的 HLG 反向 OOTF（系统伽马 1.2，按亮度算）→ HLG OETF。
B 站杜比视界 8.4 的基础层是 HLG，所以最后输出 HLG 信号。杜比视界 L1 元数据仍然按 PQ 计，pq_of_nits 给它用。
"""
import os
import numpy as np

PAPER_NITS = 150.0          # 纸色的亮度：整屏都是纸时，比 BT.2408 的图形白 203 低一些，看着像纸而不是灯箱
PEAK = 1000.0               # 母版峰值
GAIN = 3.5                  # 遮罩满值时提亮的倍数（约 1.8 档）：金色、青色发光线约 100 尼特 → 约 360 到 400 尼特；暗场里的发光字也在这一档，不刺眼
KNEE = 600.0                # 软剪切从这里开始，往 PEAK 平滑地收


def srgb_to_linear(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


Y709 = np.array([0.2126, 0.7152, 0.0722])
Y2020 = np.array([0.2627, 0.6780, 0.0593])
M709_2020 = np.array([[0.6274, 0.3293, 0.0433],
                      [0.0691, 0.9195, 0.0114],
                      [0.0164, 0.0880, 0.8956]])

# 纸色：画面里最常见的大面积浅色，定到 PAPER_NITS。换了配色就设环境变量 HDR_PAPER，比如 HDR_PAPER=#FFFFFF
_hex = os.environ.get('HDR_PAPER', '#F4EFE5').lstrip('#')
PAPER = srgb_to_linear(np.array([int(_hex[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0)
WHITE_NITS = PAPER_NITS / float(PAPER @ Y709)      # SDR 纯白对应多少尼特（约 173）


def softclip(x):
    """KNEE 以下原样，以上平滑地逼近 PEAK。"""
    x = np.asarray(x, np.float64)
    k, r = KNEE, PEAK - KNEE
    return np.where(x <= k, x, k + r * (1 - np.exp(-(x - k) / r)))


def display_nits(rgb_code, gain=1.0):
    """sRGB 码值（0..1，最后一维是 RGB）→ BT.709 原色下每个通道的显示亮度（尼特）。"""
    lin = srgb_to_linear(rgb_code) * WHITE_NITS * gain
    mx = lin.max(-1, keepdims=True)
    scale = np.where(mx > KNEE, softclip(mx) / np.maximum(mx, 1e-9), 1.0)   # 按最大通道收，色相不变
    return lin * scale


def nits_to_hlg(nits709):
    """BT.709 原色的显示亮度（尼特）→ BT.2020 原色的 HLG 信号（0..1），按 1000 尼特显示器。"""
    fd = np.clip(nits709 @ M709_2020.T, 0, None) / PEAK
    yd = np.maximum(fd @ Y2020, 1e-12)
    gamma = 1.2
    es = fd * yd[..., None] ** ((1 - gamma) / gamma)                 # 反向 OOTF
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    es = np.clip(es, 0, 1)
    return np.where(es <= 1 / 12, np.sqrt(3 * es), a * np.log(np.maximum(12 * es - b, 1e-12)) + c)


def pq_of_nits(n):
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    y = np.clip(np.asarray(n, np.float64) / 10000.0, 0, 1) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2

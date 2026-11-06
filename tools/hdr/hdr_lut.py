"""生成两张 65³ 的 3D LUT：sRGB 码值 → BT.2020 HLG 信号。
  base.cube   普通映射（纸 150 尼特）
  boost.cube  同样的映射，但亮度先乘 GAIN 再软剪切（发光线用）
编码时两条一起算，再按增益遮罩逐像素混合（ffmpeg maskedmerge）。
用法：python tools/hdr/hdr_lut.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np
from hdr_math import display_nits, nits_to_hlg, WHITE_NITS, PAPER_NITS, GAIN

N = 65
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
g = np.linspace(0, 1, N)
# .cube 的顺序：R 变化最快，然后 G，最后 B
b, gg, r = np.meshgrid(g, g, g, indexing='ij')
rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
for name, gain in (('base', 1.0), ('boost', GAIN)):
    hlg = nits_to_hlg(display_nits(rgb, gain))
    with open(out / f'{name}.cube', 'w') as f:
        f.write(f'TITLE "kazi sRGB to HLG {name}"\nLUT_3D_SIZE {N}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        f.write('\n'.join(f'{x:.6f} {y:.6f} {z:.6f}' for x, y, z in hlg))
        f.write('\n')
print(f'SDR 白 = {WHITE_NITS:.0f} 尼特，纸 = {PAPER_NITS:.0f} 尼特，提亮 {GAIN} 倍；写到 {out}')

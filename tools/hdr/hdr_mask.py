"""从 SDR 成片算出 HDR 增益遮罩，顺便量出杜比视界要的每个镜头的亮度（L1）。

为什么从成片算、不在代码里另画一遍：讲解动画里"发光"的东西通常只有一两种颜色，而且只在暗底上出现。
所以规则可以写得很窄，不会误伤纸面上的其他颜色：
  1. 只看暗的帧（整帧平均码值低于 0.35 才开始算，越暗权重越高，亮暗之间的转场里平滑过渡）；
  2. 每个像素看它是不是"暗底色和某种发光色按某个比例混出来的"：混的比例就是遮罩值（光晕的半透明边缘自然变小），
     离这条混色线远的像素（白字、灰线、别的颜色）不算；
  3. --exclude 给的几块区域固定不提亮（字幕条、角标这类，字幕里的关键词也常是发光色）；
  4. 一帧里提亮的总面积超过 2% 时按比例压低，大块的发光色不会整片发亮。
遮罩满值 = 提亮 GAIN 倍（见 hdr_math.py），编码时由 ffmpeg maskedmerge 在两张 LUT 的结果之间混合。

用法：python tools/hdr/hdr_mask.py BASE.mp4 OUT_DIR [--bg '#0E1A1D'] [--glow '#7FE6D8,#F2C36B'] [--exclude '0,896,1920,1080']
  --bg       暗场的底色
  --glow     要提亮的发光色，逗号隔开
  --exclude  不提亮的矩形，按 1920×1080 的坐标写 x0,y0,x1,y1，多块用分号隔开；默认只排除底部字幕条
输出：OUT_DIR/mask.mkv（1920×1080 灰度，FFV1 无损）、OUT_DIR/frames.json（每帧 L1）、OUT_DIR/dv_generator.json（杜比视界 8.4）
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from hdr_math import srgb_to_linear, WHITE_NITS, GAIN, PEAK, softclip, pq_of_nits

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('out')
ap.add_argument('--bg', default='#0E1A1D')
ap.add_argument('--glow', default='#7FE6D8,#F2C36B')
ap.add_argument('--exclude', default='0,896,1920,1080')
args = ap.parse_args()
hexrgb = lambda h: np.array([int(h.strip().lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
src, out = args.src, Path(args.out)
out.mkdir(parents=True, exist_ok=True)
W, H = 1920, 1080
BG = hexrgb(args.bg)
GLOWS = [hexrgb(c) for c in args.glow.split(',') if c.strip()]
RESID = 0.16          # 离混色线多远以内算这种发光色（按发光色和底色之差归一化）
AREA = 0.02           # 一帧提亮面积的上限
excl = np.zeros((H, W), bool)
for box in filter(None, (b.strip() for b in args.exclude.split(';'))):
    x0, y0, x1, y1 = (int(float(v)) for v in box.split(','))
    excl[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
lut_nits = (srgb_to_linear(np.arange(256) / 255.0) * WHITE_NITS).astype(np.float32)
lut_pq = pq_of_nits(lut_nits).astype(np.float32)
FLOOR = 0.08          # 混色比例低于这个当作底色（暗场纸纹的噪点、极淡的底色），不提亮


def build_key_lut():
    """键值只取决于像素的 RGB，先把 256³ 种颜色都算好（16 MB），每帧查表即可。"""
    c = np.arange(256, dtype=np.float32) / 255
    key = np.zeros((256, 256, 256), np.float32)
    for r0 in range(0, 256, 32):
        rgb = np.stack(np.meshgrid(c[r0:r0 + 32], c, c, indexing='ij'), -1) - BG
        k = np.zeros(rgb.shape[:3], np.float32)
        for g in GLOWS:
            v = g - BG
            vv = float(v @ v)
            a = (rgb @ v) / vv
            res = np.sqrt(np.maximum((rgb * rgb).sum(-1) - a * a * vv, 0) / vv)
            a = np.clip((a - FLOOR) / (1 - FLOOR), 0, 1)
            k = np.maximum(k, a * np.clip(1 - res / RESID, 0, 1))
        key[r0:r0 + 32] = k
    return np.round(key * 255).astype(np.uint8).reshape(-1)


KEY = build_key_lut()

probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                                   'stream=nb_read_packets,r_frame_rate', '-of', 'json', src], capture_output=True, text=True, check=True).stdout)
NF = int(probe['streams'][0]['nb_read_packets'])
dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'scale={W}:{H}:flags=area:in_color_matrix=bt709:in_range=tv',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', '60', '-i', '-',
                        '-c:v', 'ffv1', '-level', '3', '-slices', '16', str(out / 'mask.mkv')], stdin=subprocess.PIPE)

frames, prev, diffs = [], None, []
zero = np.zeros((H, W), np.uint8)
n = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    small = img[::4, ::4]
    mean = float(small.mean()) / 255
    w = float(np.clip((0.35 - mean) / 0.15, 0, 1))
    m = None
    if w > 0:
        idx = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2]
        key = KEY[idx].astype(np.float32) / 255
        key[excl] = 0
        frac = float(key.mean())
        if frac > AREA:
            key *= AREA / frac
        m = key * w
    enc.stdin.write((zero if m is None else np.round(m * 255).astype(np.uint8)).tobytes())
    # L1：按 maxRGB 的显示亮度（PQ），在 480×270 上量（L1 只要最小、平均、最大三个数，够用）。
    # 没提亮的帧直接查表（256 种码值 → PQ），提亮的帧才逐像素算。
    # 最大值在整幅图上量（细小的亮点在缩小的图上会被平均掉）。
    smax = small.max(-1); fmax = img.max(-1)
    if m is None:
        pq = lut_pq[smax]; mx = lut_nits[smax]; pqmax = float(lut_pq[fmax.max()])
    else:
        mx = softclip(lut_nits[smax] * (1 + (GAIN - 1) * m[::4, ::4]))
        pq = pq_of_nits(mx)
        nz = m > 0   # 只在被提亮的那一小块（不超过 2%）上逐像素算，其余查表
        bmax = float(lut_nits[fmax[~nz].max()]) if (~nz).any() else 0.0
        hmax = float(softclip(lut_nits[fmax[nz]] * (1 + (GAIN - 1) * m[nz])).max()) if nz.any() else 0.0
        pqmax = float(pq_of_nits(max(bmax, hmax)))
    frames.append([int(round(float(pq.min()) * 4095)), int(round(float(pq.mean()) * 4095)), int(round(pqmax * 4095)), round(float(mx.mean()), 2)])
    g = small.mean(-1)
    diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()) / 255)
    prev = g
    n += 1
    if n % 1200 == 0:
        print(f'{n}/{NF} 帧', flush=True)
dec.wait(); enc.stdin.close(); enc.wait()
print('处理了', n, '帧（ffprobe 数到', NF, '帧）')
json.dump(frames, open(out / 'frames.json', 'w'))

# 镜头切分：相邻两帧平均变化超过 6% 记作一刀（讲解动画的换场多是硬切，镜头内部是连续运动），最短 0.5 秒
cuts = [0]
for i, d in enumerate(diffs):
    if d > 0.06 and i - cuts[-1] >= 30:
        cuts.append(i)
cuts.append(n)
f = np.array(frames, float)
shots = [{'start': a, 'duration': b - a, 'metadata_blocks': [{'Level1': {
    'min_pq': int(f[a:b, 0].min()), 'max_pq': int(f[a:b, 2].max()), 'avg_pq': int(round(f[a:b, 1].mean()))}}]}
         for a, b in zip(cuts[:-1], cuts[1:])]
maxfall = int(np.ceil(f[:, 3].max()))
cfg = {'cm_version': 'V40', 'profile': '8.4', 'length': n,
       'level6': {'max_display_mastering_luminance': int(PEAK), 'min_display_mastering_luminance': 1,
                  'max_content_light_level': int(PEAK), 'max_frame_average_light_level': maxfall},
       'shots': shots}
json.dump(cfg, open(out / 'dv_generator.json', 'w'), indent=1)
print(f'{len(shots)} 个镜头；MaxFALL {maxfall} 尼特；全片帧平均亮度的中位数 {np.median(f[:, 3]):.0f} 尼特；'
      f'L1 max 最高 {int(f[:, 2].max())}/4095')

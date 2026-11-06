// 缓动和小工具。所有函数都是时间的纯函数，任意一帧都能单独算出来。
export const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const inv = (a: number, b: number, x: number) => clamp((x - a) / (b - a));

// 三次贝塞尔缓动（与 CSS cubic-bezier 相同），牛顿迭代求解
export const bezier = (x1: number, y1: number, x2: number, y2: number) => {
  const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
  const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
  const sx = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sy = (t: number) => ((ay * t + by) * t + cy) * t;
  const dx = (t: number) => (3 * ax * t + 2 * bx) * t + cx;
  return (x: number) => {
    x = clamp(x);
    let t = x;
    for (let i = 0; i < 8; i++) {
      const e = sx(t) - x;
      const d = dx(t);
      if (Math.abs(e) < 1e-6 || Math.abs(d) < 1e-6) break;
      t -= e / d;
    }
    return sy(clamp(t));
  };
};

// 曲线族（数值以动效规范为起点，调研回来后统一校准）
export const E = {
  linear: (x: number) => clamp(x),
  // 标准：进出都柔和，用于位置移动、镜头
  standard: bezier(0.2, 0, 0, 1),
  // 进场：快出慢停
  enter: bezier(0.05, 0.7, 0.1, 1),
  // 退场：慢起快走
  exit: bezier(0.3, 0, 0.8, 0.15),
  // 两端都平滑，淡入淡出用
  smooth: (x: number) => {
    x = clamp(x);
    return x * x * x * (x * (x * 6 - 15) + 10);
  },
  inOut: bezier(0.65, 0, 0.35, 1),
};

// 有阻尼的弹簧（解析解），用于"落下、弹一下、停住"。zeta 越小越弹。
// 默认阻尼比 0.75，过冲约 3%；位置可以轻微过冲，透明度和颜色不要用弹簧。
export const spring = (t: number, freq = 2.2, zeta = 0.75) => {
  if (t <= 0) return 0;
  const w = 2 * Math.PI * freq;
  const wd = w * Math.sqrt(1 - zeta * zeta);
  return 1 - Math.exp(-zeta * w * t) * (Math.cos(wd * t) + ((zeta * w) / wd) * Math.sin(wd * t));
};

// 在时刻 at 开始、持续 dur 秒的进度
export const ramp = (t: number, at: number, dur: number, ease = E.standard) => ease(inv(at, at + dur, t));

// 进场 + 停留 + 退场的可见度
export const life = (t: number, a: number, b: number, fin = 0.35, fout = 0.3) =>
  Math.min(E.smooth(inv(a, a + fin, t)), 1 - E.smooth(inv(b - fout, b, t)));

// 可复现的伪随机
export const hash = (n: number) => {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
};
export const noise1 = (x: number, seed = 0) => {
  const i = Math.floor(x), f = x - i;
  const u = f * f * (3 - 2 * f);
  return lerp(hash(i + seed * 101), hash(i + 1 + seed * 101), u) * 2 - 1;
};

// 卡字最常用的一个：从 t0 开始动，在 t1 这一刻"第一次到位"（之后轻微过冲再停稳）。
// 把 t1 设成 hit('关键词')，动作就正好落在关键词上。阻尼比 0.75 时过冲约 3%。
export const springTo = (t: number, t0: number, t1: number, zeta = 0.75) => {
  const dur = Math.max(1e-3, t1 - t0);
  const freq = (Math.PI - Math.acos(zeta)) / (2 * Math.PI * Math.sqrt(1 - zeta * zeta) * dur);
  return spring(t - t0, freq, zeta);
};

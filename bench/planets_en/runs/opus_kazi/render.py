"""Eight planets explainer: every frame is a pure function of time; every narration-related
moment is looked up in timeline.json (kazi step 3), never estimated from word counts.

Usage: python film/render.py [--start S] [--end E] [--out path.mp4] [--still T --png path]
Renders 1920x1080 @ 60 fps video (no audio) by piping raw frames into ffmpeg.
"""
import argparse, json, math, os, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "input" / "art"
FPS, W, H = 60, 1920, 1080
TL = json.loads((ROOT / "timeline.json").read_text("utf-8"))
AUDIO_DUR = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(ROOT / "input" / "narration.wav")], capture_output=True, text=True).stdout)
NFRAMES = int(math.ceil(AUDIO_DUR * FPS - 1e-6))

# ---------------------------------------------------------------- time lookups (from the table)
LEAD, APPEAR = 3 / FPS, 12 / FPS
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")


def at(word, n=0, edge="s", line=None):
    """Start (or end) of the n-th occurrence of `word` (one or more words), optionally inside line i."""
    target = [norm(x) for x in word.split()]
    k = 0
    for l in TL["lines"]:
        if line is not None and l["i"] != line:
            continue
        units = [norm(w["u"]) for w in l["words"]]
        for i in range(len(units) - len(target) + 1):
            if units[i:i + len(target)] == target:
                if k == n:
                    return l["words"][i]["s"] if edge == "s" else l["words"][i + len(target) - 1]["e"]
                k += 1
    raise KeyError(f"word not found: {word!r} #{n} line={line}")


def kw(word):
    for k in TL["keywords"]:
        if norm(k["word"]).startswith(norm(word)) or norm(word) in norm(k["word"]):
            return k["t"]
    raise KeyError(f"keyword not found: {word}")


def hit(word):          # the frame an action lands: 3 frames before the keyword starts
    return kw(word) - LEAD


LINES = {l["i"]: l for l in TL["lines"]}
ls = lambda i: LINES[i]["start"]
le = lambda i: LINES[i]["end"]

# ---------------------------------------------------------------- easing
clamp = lambda x, a=0.0, b=1.0: min(b, max(a, x))
lerp = lambda a, b, t: a + (b - a) * t
inv = lambda a, b, x: clamp((x - a) / (b - a)) if b != a else float(x >= b)


def bezier(x1, y1, x2, y2):
    cx = 3 * x1; bx = 3 * (x2 - x1) - cx; ax = 1 - cx - bx
    cy = 3 * y1; by = 3 * (y2 - y1) - cy; ay = 1 - cy - by
    def f(x):
        x = clamp(x); t = x
        for _ in range(10):
            e = ((ax * t + bx) * t + cx) * t - x
            d = (3 * ax * t + 2 * bx) * t + cx
            if abs(e) < 1e-7 or abs(d) < 1e-7: break
            t -= e / d
        t = clamp(t)
        return ((ay * t + by) * t + cy) * t
    return f


E_STD = bezier(0.2, 0, 0, 1)          # position / camera
E_INOUT = bezier(0.65, 0, 0.35, 1)
E_EXIT = bezier(0.3, 0, 0.8, 0.15)   # slow start, fast leave
smooth = lambda x: (lambda u: u * u * u * (u * (u * 6 - 15) + 10))(clamp(x))


def spring(t, freq, zeta=0.75):
    if t <= 0: return 0.0
    w = 2 * math.pi * freq; wd = w * math.sqrt(1 - zeta * zeta)
    return 1 - math.exp(-zeta * w * t) * (math.cos(wd * t) + (zeta * w / wd) * math.sin(wd * t))


def spring_to(t, t0, t1, zeta=0.75):
    """0 -> 1, first reaching 1 exactly at t1 (then ~3 % overshoot and settle)."""
    dur = max(1e-3, t1 - t0)
    freq = (math.pi - math.acos(zeta)) / (2 * math.pi * math.sqrt(1 - zeta * zeta) * dur)
    return spring(t - t0, freq, zeta)


# ---------------------------------------------------------------- sprites
class Sprite:
    LEVELS = [1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125, 0.0884, 0.0625]

    def __init__(self, path, rest_scale=1.0):
        src = Image.open(path).convert("RGBA")
        self.size = src.size
        self.pyr = []
        for f in self.LEVELS:
            im = src if f == 1.0 else src.resize((max(1, round(src.width * f)), max(1, round(src.height * f))), Image.LANCZOS)
            self.pyr.append((f, im.convert("RGBa")))
        self.rest = rest_scale


def draw(canvas, spr, cx, cy, scale, rot=0.0, alpha=1.0, anchor=None):
    """Composite sprite so that its anchor point (source px; default centre) lands at (cx, cy)."""
    if alpha <= 0.002 or scale <= 1e-4: return
    sw, sh = spr.size
    ax, ay = anchor if anchor else (sw / 2, sh / 2)
    lvl, im = spr.pyr[0]
    for f, pim in spr.pyr:
        if f >= scale * 0.999: lvl, im = f, pim
        else: break
    r = scale / lvl                      # residual scale applied by the affine transform
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    # corners of the sprite on screen
    pts = []
    for px, py in ((0, 0), (sw, 0), (0, sh), (sw, sh)):
        dx, dy = (px - ax) * scale, (py - ay) * scale
        pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    x0 = max(0, int(math.floor(min(p[0] for p in pts))) - 1); x1 = min(W, int(math.ceil(max(p[0] for p in pts))) + 1)
    y0 = max(0, int(math.floor(min(p[1] for p in pts))) - 1); y1 = min(H, int(math.ceil(max(p[1] for p in pts))) + 1)
    if x1 <= x0 or y1 <= y0: return
    # screen (x, y) -> level-image (u, v)
    au, av = ax * lvl, ay * lvl
    a_ = c / r; b_ = s / r; d_ = -s / r; e_ = c / r
    # use pixel centres: sample at (x+0.5, y+0.5) -> u+0.5
    ox, oy = x0 + 0.5 - cx, y0 + 0.5 - cy
    cu = au + a_ * ox + b_ * oy - 0.5
    cv = av + d_ * ox + e_ * oy - 0.5
    tile = im.transform((x1 - x0, y1 - y0), Image.AFFINE, (a_, b_, cu, d_, e_, cv), resample=Image.BICUBIC)
    tile = tile.convert("RGBA")
    if alpha < 0.999:
        A = tile.getchannel("A").point(lambda v: int(v * alpha + 0.5))
        tile.putalpha(A)
    canvas.alpha_composite(tile, dest=(x0, y0))


# ---------------------------------------------------------------- assets & layout
NAMES = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
BG = None; SPR = {}
DISC = 720.0                      # planet disc diameter in planets/*.png (alpha > 128)
PLANET_H = 600.0                  # on-screen disc height
SATURN_W_SRC = 1451.0; SATURN_W = 1100.0
PX, PY = 700.0, 540.0             # planet rest centre (left of centre)
SAT_X = 650.0
LABEL_GAP = 95.0
LINEUP_C = (960.0, 595.0); LABEL8_C = (960.0, 372.0)
SUN_SCALE = 0.92; SUN_X = -215.0  # sun centre: mostly cut off at the left edge
LINEUP_MERC = (64.0, 209.0, 77.0)  # Mercury centre x, y and disc size inside lineup.png


def load():
    global BG
    BG = Image.open(ART / "background.png").convert("RGBA")
    for n in NAMES:
        SPR[n] = Sprite(ART / "planets" / f"{n}.png")
        SPR["L_" + n] = Sprite(ART / "labels" / f"{n}.png")
    SPR["lineup"] = Sprite(ART / "lineup.png")
    SPR["L_8"] = Sprite(ART / "labels" / "eight_planets.png")
    SPR["sun"] = Sprite(ART / "sun.png")
    # text bounding boxes of labels (so the text, not the padding, is placed)
    for k in list(SPR):
        if k.startswith("L_"):
            a = np.array(Image.open(ART / "labels" / (("eight_planets" if k == "L_8" else k[2:]) + ".png")))[..., 3]
            ys = np.where((a > 40).any(1))[0]; xs = np.where((a > 40).any(0))[0]
            SPR[k].bbox = (xs[0], ys[0], xs[-1] + 1, ys[-1] + 1)


def planet_scale(n):
    return SATURN_W / SATURN_W_SRC if n == "saturn" else PLANET_H / DISC


def planet_rest_x(n):
    return SAT_X if n == "saturn" else PX


def planet_half_w(n):
    return SATURN_W / 2 if n == "saturn" else PLANET_H / 2


def draw_label(canvas, key, left, cy, alpha=1.0, scale=1.0):
    spr = SPR[key]; bx0, by0, bx1, by1 = spr.bbox
    draw(canvas, spr, left, cy, scale, alpha=alpha, anchor=(bx0, (by0 + by1) / 2))


# ---------------------------------------------------------------- the beat sheet (all from timeline.json)
SWAP = 30 / FPS        # the pan that itself lands on a keyword (Venus)
ACC = 22 / FPS         # a keyword accent: name label lands, planet does its verb
TRAVEL = 0.70          # a pan in the pause before a line (42 frames, cross-screen)
SPAN = 1900.0          # distance between neighbouring planets during a pan (camera travels outward)
# per planet: (pan lands this long after its line starts, pan duration). A pan must settle before the
# 1-second window that precedes the name (so the name gets its own accent), and should start late
# enough that the previous name label stays readable.
ARRIVE = {"earth": (0.08, 0.70), "mars": (0.08, 0.55), "jupiter": (0.50, 0.70), "saturn": (0.06, 0.70),
          "uranus": (0.57, 0.70), "neptune": (0.70, 0.70)}

B = None; S = None


def beats():
    b = {}
    # line 1: "Our solar system has *eight planets*. Let's meet them one by one."
    b["eight"] = hit("eight")
    b["zoom0"] = at("one", 0, line=1) - 0.10                  # push in toward Mercury on "one by one"
    b["zoom1"] = ls(2) + 0.30                                  # settled as line 2 begins
    for n in NAMES:
        b[n] = hit(n)                                          # every name: label lands, planet does its verb
    return b


def schedule():
    """Arrival pan window [a0, a1] for each planet; the previous planet leaves on the same pan.
    Default: the camera pans to the planet in the pause before its own line and settles as the line
    starts, so the planet is alone on screen for its whole line; its name then gets a separate accent.
    Venus is the exception: its name comes 0.5 s into the line, so the pan itself lands on the name."""
    s = {}
    s["mercury"] = dict(a0=B["zoom1"], a1=B["zoom1"])
    s["venus"] = dict(a0=B["venus"] - SWAP, a1=B["venus"])
    for n, i in (("earth", 4), ("mars", 5), ("jupiter", 6), ("saturn", 7), ("uranus", 8), ("neptune", 9)):
        off, dur = ARRIVE[n]
        a1 = ls(i) + off
        s[n] = dict(a0=a1 - dur, a1=a1)
    return s


def pan(n, t):
    """Progress of the camera pan that brings planet n in (0 = n is one SPAN to the right, 1 = n at rest)."""
    if n is None: return 0.0
    a0, a1 = S[n]["a0"], S[n]["a1"]
    if n == "mercury": return 1.0
    if n == "venus": return spring_to(t, a0, a1)        # lands on the name with a small settle
    return E_INOUT(inv(a0, a1, t))                       # camera move: soft at both ends


def nxt(n):
    j = NAMES.index(n)
    return NAMES[j + 1] if j + 1 < len(NAMES) else None


def camera_x(n, t):
    """Horizontal offset of planet n's 'station' on screen."""
    return SPAN * (1 - pan(n, t)) - SPAN * pan(nxt(n), t)


def pose(n, t):
    k = planet_scale(n); y = PY; rot = 0.0; alpha = 1.0
    K = B[n]
    g = spring_to(t, K - ACC, K)                      # keyword accent progress (first reaches 1 on hit)
    gs = smooth(inv(K - ACC, K, t))                   # same window, no overshoot (for opacity)
    step = 0.0
    if n == "mercury":
        step = 90 * (1 - g)                           # "closest to the Sun": steps toward the Sun on its name
    elif n in ("earth", "mars"):
        k *= lerp(0.84, 1.0, g)                       # arrives a little small, swells on its name
    elif n == "jupiter":
        k *= lerp(0.30, 1.0, g)                       # "the biggest": grows from Mars-like size on its name
    elif n == "saturn":
        k *= lerp(0.86, 1.0, g); rot = lerp(-12.0, 0.0, g)   # rings tip level as it is named
    elif n in ("uranus", "neptune"):
        alpha = lerp(0.28, 1.0, gs); k *= lerp(0.86, 1.0, g)  # "hard to see by eye": faint until named
        if n == "uranus": rot = lerp(75.0, 0.0, g)   # "spins on its side": a roll that settles on the name
    a1 = S[n]["a1"]; lv = S[nxt(n)]["a0"] if nxt(n) else 1e9
    drift = -10.0 * inv(a1, min(lv, a1 + 8.0), t)     # slow drift while resting (below the motion threshold)
    cam = camera_x(n, t)
    return planet_rest_x(n) + cam + step + drift, y, k, rot, alpha, cam + drift


# ---------------------------------------------------------------- frame
def frame(t):
    cv = BG.copy()
    zu = inv(B["zoom0"], B["zoom1"], t); zt = E_INOUT(zu)
    merc_zoom = None

    # ---------------- scene 1: lineup + "8 Planets", then push in to Mercury
    if t < B["zoom1"] + 1e-9:
        lx, ly = LINEUP_C
        a_in = smooth(inv(0.0, 1.1, t))                       # slow fade-in before the voice starts
        sp = spring_to(t, B["eight"] - ACC, B["eight"])       # the row spreads out on "eight planets"
        s0 = lerp(0.93, 1.0, sp) * (1.0 + 0.012 * inv(B["eight"], B["zoom0"], t))
        mx, my, md = LINEUP_MERC
        P0 = (lx + (mx - 900) * s0, ly + (my - 210) * s0)     # the lineup's Mercury on screen
        z = ((PLANET_H / md) / s0) ** zt
        tx = lerp(P0[0], PX + 90, zt); ty = lerp(P0[1], PY, zt)   # move and zoom together
        draw(cv, SPR["lineup"], tx + (lx - P0[0]) * z, ty + (ly - P0[1]) * z, s0 * z,
             alpha=a_in * (1 - smooth(inv(0.30, 0.62, zt))))
        la = smooth(inv(B["eight"] - ACC, B["eight"] - ACC + 0.2, t)) * (1 - smooth(inv(0.0, 0.35, zt)))
        ldy = lerp(-110, 0, sp)
        draw(cv, SPR["L_8"], tx + (LABEL8_C[0] - P0[0]) * z, ty + (LABEL8_C[1] + ldy - P0[1]) * z,
             z * lerp(0.96, 1.0, sp), alpha=la)
        if zt > 0.2:                                          # hi-res Mercury takes over mid-zoom
            merc_zoom = (tx, ty, (md * s0 * z) / DISC, smooth(inv(0.22, 0.5, zt)))

    # ---------------- the Sun: comes in with the push toward Mercury, leaves on the pan to Venus
    sun_in = E_STD(inv(B["zoom0"] + 0.25, B["zoom1"] + 0.1, t))
    if sun_in > 0 and pan("venus", t) < 0.999:
        lv = S["venus"]["a0"]
        sx = lerp(SUN_X - 820, SUN_X, sun_in) - 4.0 * inv(B["zoom1"], lv, t) - SPAN * pan("venus", t)
        draw(cv, SPR["sun"], sx, PY, SUN_SCALE)

    # ---------------- planets and their name labels
    for n in NAMES:
        if n == "mercury" and merc_zoom is not None and t < B["zoom1"]:
            x, y, k, a = merc_zoom
            draw(cv, SPR[n], x, y, k, alpha=a)
            continue
        if t < S[n]["a0"] - 1e-9: continue
        if nxt(n) and pan(nxt(n), t) >= 0.999 and t > S[nxt(n)]["a1"]: continue
        x, y, k, rot, alpha, cam = pose(n, t)
        draw(cv, SPR[n], x, y, k, rot=rot, alpha=alpha)
        K = B[n]
        if t >= K - ACC:
            lp = spring_to(t, K - ACC, K)
            lab_a = smooth(inv(K - ACC, K - ACC + 0.18, t))
            left = planet_rest_x(n) + planet_half_w(n) + LABEL_GAP + 230 * (1 - lp) + cam
            if n == "mercury": left += 90 * (1 - spring_to(t, K - ACC, K))
            draw_label(cv, "L_" + n, left, PY, alpha=lab_a)

    return cv.convert("RGB")



def init():
    global B, S
    load(); B = beats(); S = schedule()


def render_one(i):
    return frame(i / FPS).tobytes()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "film" / "video.mp4"))
    ap.add_argument("--still", type=float, default=None); ap.add_argument("--png", default="")
    ap.add_argument("--beats", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    init()
    if a.beats:
        print(json.dumps({k: round(v, 3) for k, v in B.items()}, indent=1)); print(json.dumps(S, indent=1)); sys.exit()
    if a.still is not None:
        frame(a.still).save(a.png); sys.exit()
    f0 = int(round(a.start * FPS)); f1 = NFRAMES if a.end is None else min(NFRAMES, int(round(a.end * FPS)))
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", a.out], stdin=subprocess.PIPE)
    with Pool(a.jobs, initializer=init) as pool:
        for k, buf in enumerate(pool.imap(render_one, range(f0, f1), chunksize=4)):
            ff.stdin.write(buf)
            if k % 300 == 0: print(f"frame {f0 + k}/{f1}", flush=True)
    ff.stdin.close(); ff.wait()
    print("frames", f1 - f0, "->", a.out)

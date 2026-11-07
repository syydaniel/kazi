"""Render the eight-planets explainer: PIL compositing per frame -> ffmpeg (H.264, 60 fps).

Usage:
  python work/render.py stills t1 t2 ...   # write work/stills/t_XX.XX.png
  python work/render.py video              # write out/final.mp4
"""
import math
import os
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "input", "art")
W, H, FPS = 1920, 1080, 60
AUDIO = os.path.join(ROOT, "input", "narration.wav")
AUDIO_SECONDS = 942960 / 24000.0  # 39.29 s
NFRAMES = math.ceil(AUDIO_SECONDS * FPS)

# ---------------------------------------------------------------- timing
# Line spans from silence detection, keyword onsets from word-level alignment.
LINES = [  # (name, speech_start, speech_end, keyword_onset)
    ("eight_planets", 1.64, 5.49, 2.92),
    ("mercury", 6.29, 10.25, 8.08),
    ("venus", 10.93, 15.64, 11.40),
    ("earth", 16.35, 18.07, 17.78),
    ("mars", 18.53, 20.56, 19.98),
    ("jupiter", 21.03, 26.18, 23.20),
    ("saturn", 26.59, 28.50, 28.02),
    ("uranus", 28.92, 33.53, 32.88),
    ("neptune", 33.86, 36.86, 36.38),
]

PAN_DUR = 0.90     # camera pan from one planet to the next
PAN_DIST = 1700.0  # px; wider than any planet+label group, so groups never overlap
LABEL_LEAD = 0.05  # label starts this long before the spoken keyword
LABEL_DUR = 0.50
ZOOM_START, ZOOM_DUR = 5.50, 1.00   # lineup -> Mercury zoom
FINAL_EXIT_START, FINAL_EXIT_DUR = 37.75, 1.10


def transition_start(i):
    """When planet line i (1..8) starts leaving for line i+1."""
    _, _, e_cur, _ = LINES[i]
    _, s_next, _, k_next = LINES[i + 1]
    lo = e_cur + 0.05                 # never cut the outgoing line short
    hi = k_next - 0.20 - PAN_DUR      # land before the next name is spoken
    return max(lo, min(s_next - 0.25, hi))


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(u):
    return 1 - (1 - u) ** 3


def ease_in_cubic(u):
    return u ** 3


def ease_in_out_cubic(u):
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out_quad(u):
    return 1 - (1 - u) ** 2


def smoothstep(u):
    return u * u * (3 - 2 * u)


def ease_in_out_sine(u):
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_out_back(u, c=1.4):
    return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2


def lerp(a, b, u):
    return a + (b - a) * u


# ---------------------------------------------------------------- assets
class Asset:
    """Premultiplied, pre-scaled image plus an anchor point (in its own pixels)."""

    def __init__(self, path, scale, anchor):
        im = Image.open(path).convert("RGBA").convert("RGBa")
        if abs(scale - 1) > 1e-6:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        self.im = im
        self.ax, self.ay = anchor[0] * scale, anchor[1] * scale
        self.k = scale


def alpha_bbox(path, thr=8):
    im = Image.open(path).convert("RGBA")
    return im.getchannel("A").point(lambda v: 255 if v > thr else 0).getbbox()


PLANETS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
PLANET_H = 600.0
SATURN_W = 1100.0
PLANET_CY = 540.0
LABEL_GAP = 80.0
LABEL_AY = 74.0  # middle of the cap-height band (labels all start text at y=20, caps to ~128)

A = {}


def load_assets():
    if A:
        return
    for p in PLANETS:
        path = os.path.join(ART, "planets", p + ".png")
        x0, y0, x1, y1 = alpha_bbox(path)
        vw, vh = x1 - x0, y1 - y0
        s = SATURN_W / vw if p == "saturn" else PLANET_H / vh
        A["planet_" + p] = Asset(path, s, ((x0 + x1) / 2, (y0 + y1) / 2))
        A["planet_" + p].half_w = vw * s / 2
        lpath = os.path.join(ART, "labels", p + ".png")
        lx0, _, _, _ = alpha_bbox(lpath)
        A["label_" + p] = Asset(lpath, 1.0, (lx0, LABEL_AY))
    lpath = os.path.join(ART, "labels", "eight_planets.png")
    lx0, ly0, lx1, _ = alpha_bbox(lpath)
    A["label_eight"] = Asset(lpath, 1.1, ((lx0 + lx1) / 2, LABEL_AY))
    # lineup: anchor at Mercury's centre inside the lineup (so we can zoom into it)
    A["lineup"] = Asset(os.path.join(ART, "lineup.png"), 1.0, (64.0, 209.0))
    A["sun"] = Asset(os.path.join(ART, "sun.png"), 1.1, (720.0, 720.0))
    A["bg"] = Image.open(os.path.join(ART, "background.png")).convert("RGBA")


PLANET_CX = {p: 660.0 for p in PLANETS}
PLANET_CX["saturn"] = 650.0

# lineup placement: visible centre (899, 209.5) of lineup.png -> screen (960, 605)
LINEUP_VIS_C = (899.0, 209.5)
LINEUP_SCREEN_C = (960.0, 605.0)
MERC_IN_LINEUP = (64.0, 209.0)
MERC_LINEUP_D = 77.0
EIGHT_LABEL_POS = (960.0, 395.0)
SUN_CX, SUN_CY = -330.0, 540.0


# ---------------------------------------------------------------- drawing
def draw(canvas, asset, x, y, s=1.0, alpha=1.0):
    """Composite asset with its anchor at screen (x, y), scaled by s, at opacity alpha."""
    if alpha <= 0.003 or s <= 1e-4:
        return
    im = asset.im
    left = x - asset.ax * s
    top = y - asset.ay * s
    right = left + im.width * s
    bottom = top + im.height * s
    L, T = max(0, math.floor(left)), max(0, math.floor(top))
    R, B = min(W, math.ceil(right)), min(H, math.ceil(bottom))
    if R <= L or B <= T:
        return
    inv = 1.0 / s
    data = (inv, 0.0, (L - x) * inv + asset.ax, 0.0, inv, (T - y) * inv + asset.ay)
    layer = im.transform((R - L, B - T), Image.AFFINE, data, resample=Image.BICUBIC).convert("RGBA")
    if alpha < 0.999:
        a = layer.getchannel("A").point(lambda v: int(v * alpha + 0.5))
        layer.putalpha(a)
    canvas.alpha_composite(layer, (L, T))


# ---------------------------------------------------------------- scenes
def lineup_scene(canvas, t):
    _, s0, e0, k0 = LINES[0]
    # intro: fade + settle
    u_in = clamp((t - 0.20) / 1.10)
    a_in = ease_out_quad(u_in)
    s_in = lerp(0.94, 1.0, ease_out_cubic(u_in))
    h = 1.0 + 0.03 * smoothstep(clamp((t - 0.20) / (ZOOM_START - 0.20)))
    S0 = s_in * h
    # Mercury's screen position before zoom (lineup scaled about its visible centre)
    pm0 = (LINEUP_SCREEN_C[0] + (MERC_IN_LINEUP[0] - LINEUP_VIS_C[0]) * S0,
           LINEUP_SCREEN_C[1] + (MERC_IN_LINEUP[1] - LINEUP_VIS_C[1]) * S0)
    target = (PLANET_CX["mercury"], PLANET_CY)
    S_end = PLANET_H / MERC_LINEUP_D
    uz = clamp((t - ZOOM_START) / ZOOM_DUR)
    if uz > 0:
        ez = ease_in_out_cubic(uz)
        S = math.exp(lerp(math.log(S0), math.log(S_end), ez))
        pm = (lerp(pm0[0], target[0], ez), lerp(pm0[1], target[1], ez))
    else:
        S, pm = S0, pm0
    lineup_alpha = a_in * (1 - smoothstep(clamp((uz - 0.35) / 0.40)))
    if uz < 1:
        draw(canvas, A["lineup"], pm[0], pm[1], S, lineup_alpha)

    # "8 Planets" label pops in on the keyword, lifts away at the zoom
    lu = clamp((t - (k0 - LABEL_LEAD)) / LABEL_DUR)
    if lu > 0 and uz < 1:
        la = ease_out_quad(clamp(lu / 0.8)) * (1 - smoothstep(clamp(uz / 0.35)))
        ly = EIGHT_LABEL_POS[1] - 40 * (1 - ease_out_cubic(lu)) - 140 * ease_in_cubic(clamp(uz / 0.5))
        ls = lerp(0.88, 1.0, ease_out_back(lu)) * (1 + 0.02 * smoothstep(clamp((t - k0) / (ZOOM_START - k0))))
        draw(canvas, A["label_eight"], EIGHT_LABEL_POS[0], ly, ls, la)

    # big Mercury grows out of the lineup's Mercury (cross-fade at matching size)
    if uz > 0:
        rel = S / S_end
        ma = smoothstep(clamp((uz - 0.30) / 0.45))
        return pm, rel, ma
    return None


def pan_in(t, t0):
    """(x offset, alpha) for a group arriving with the pan that starts at t0."""
    e = ease_in_out_sine(clamp((t - t0) / PAN_DUR))
    return PAN_DIST * (1 - e), smoothstep(clamp(e / 0.35))


def pan_out(t, t0):
    e = ease_in_out_sine(clamp((t - t0) / PAN_DUR))
    return -PAN_DIST * e, 1 - smoothstep(clamp((e - 0.65) / 0.35))


def planet_group(canvas, idx, t, enter_t, exit_t, zoom_state=None, final=False):
    """Draw planet LINES[idx] with its label. enter_t=None means it arrives via the zoom."""
    name, s, e, k = LINES[idx]
    asset = A["planet_" + name]
    cx, cy = PLANET_CX[name], PLANET_CY

    if enter_t is not None:
        if t < enter_t:
            return
        dx_in, a_in = pan_in(t, enter_t)
        settle = enter_t + PAN_DUR
    else:
        dx_in, a_in = 0.0, 1.0
        settle = ZOOM_START + ZOOM_DUR

    # slow push-in while held
    hold = 1.0 + 0.04 * smoothstep(clamp((t - settle) / max(0.5, exit_t - settle)))

    if final:
        v = clamp((t - exit_t) / FINAL_EXIT_DUR)
        dx_out = -260 * ease_in_out_cubic(v)
        a_out = 1 - smoothstep(v)
        s_out = lerp(1.0, 0.9, ease_in_out_cubic(v))
    else:
        dx_out, a_out = pan_out(t, exit_t)
        s_out = 1.0
    if a_out <= 0:
        return

    sc = hold * s_out
    px, py, pa = cx + dx_in + dx_out, cy, a_in * a_out

    # planet (during the zoom it is driven by the lineup's Mercury)
    if enter_t is None and t < ZOOM_START + ZOOM_DUR:
        if zoom_state is not None:
            pm, rel, ma = zoom_state
            draw(canvas, asset, pm[0], pm[1], rel, ma)
    else:
        draw(canvas, asset, px, py, sc, pa)

    # label: pops in on the spoken name, to the right of the planet
    lu = clamp((t - (k - LABEL_LEAD)) / LABEL_DUR)
    if lu > 0:
        la = ease_out_quad(clamp(lu / 0.8)) * pa
        lx = px + (asset.half_w + LABEL_GAP) * sc + 50 * (1 - ease_out_cubic(lu))
        ls = lerp(0.9, 1.0, ease_out_back(lu)) * (1 + 0.02 * (hold - 1) / 0.04)
        draw(canvas, A["label_" + name], lx, py, ls, la)


def sun_layer(canvas, t, exit_t):
    u = clamp((t - (ZOOM_START + 0.45)) / 0.90)
    if u <= 0:
        return
    dx = -480 * (1 - ease_out_cubic(u))
    a = ease_out_quad(clamp(u / 0.7))
    dxo, ao = pan_out(t, exit_t)
    draw(canvas, A["sun"], SUN_CX + dx + dxo, SUN_CY, 1.0, a * ao)


def pan_starts():
    return {i: transition_start(i) for i in range(1, 8)}


def render(t):
    load_assets()
    canvas = A["bg"].copy()
    exits = pan_starts()
    zs = None
    if t < ZOOM_START + ZOOM_DUR + 0.01:
        zs = lineup_scene(canvas, t)
    if ZOOM_START < t < exits[1] + PAN_DUR + 0.02:
        sun_layer(canvas, t, exits[1])
    for i in range(1, 9):
        enter_t = None if i == 1 else exits[i - 1]
        if i < 8:
            exit_t, dur, final = exits[i], PAN_DUR, False
        else:
            exit_t, dur, final = FINAL_EXIT_START, FINAL_EXIT_DUR, True
        if enter_t is not None and t < enter_t:
            continue
        if t > exit_t + dur + 0.02:
            continue
        if i == 1 and t < ZOOM_START:
            continue
        planet_group(canvas, i, t, enter_t, exit_t, zoom_state=zs if i == 1 else None, final=final)
    return canvas.convert("RGB")


# Motion blur (180-degree shutter) only where things move fast: zoom and pans.
BLUR_SUB = 6


def fast_motion(t):
    if ZOOM_START - 0.02 < t < ZOOM_START + ZOOM_DUR + 0.02:
        return True
    return any(t0 - 0.02 < t < t0 + PAN_DUR + 0.02 for t0 in pan_starts().values())


def render_frame(n):
    import numpy as np
    t = n / FPS
    if not fast_motion(t):
        return render(t)
    acc = None
    for j in range(BLUR_SUB):
        ts = t + ((j + 0.5) / BLUR_SUB - 0.5) * (0.5 / FPS)
        a = np.asarray(render(ts), dtype=np.float32)
        acc = a if acc is None else acc + a
    return Image.fromarray(np.clip(acc / BLUR_SUB + 0.5, 0, 255).astype(np.uint8))


def frame_bytes(n):
    return render_frame(n).tobytes()


def main():
    mode = sys.argv[1]
    if mode == "stills":
        os.makedirs(os.path.join(ROOT, "work", "stills"), exist_ok=True)
        for ts in sys.argv[2:]:
            t = float(ts)
            render_frame(round(t * FPS)).save(os.path.join(ROOT, "work", "stills", f"t_{t:05.2f}.png"))
        return
    if mode == "timeline":
        for i in range(1, 8):
            print(LINES[i][0], "-> next: pan", round(transition_start(i), 3), "to", round(transition_start(i) + PAN_DUR, 3))
        return
    from multiprocessing import Pool

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    out = os.path.join(ROOT, "out", "final.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", AUDIO,
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-r", str(FPS), "-frames:v", str(NFRAMES),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8, initializer=load_assets) as pool:
        for n, buf in enumerate(pool.imap(frame_bytes, range(NFRAMES), chunksize=4)):
            proc.stdin.write(buf)
            if n % 300 == 0:
                print(f"frame {n}/{NFRAMES}", flush=True)
    proc.stdin.close()
    rc = proc.wait()
    print("ffmpeg exit", rc, "frames", NFRAMES)


if __name__ == "__main__":
    main()

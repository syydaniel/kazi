"""Shared art kit for the English with/without-Kazi test (2026-11-07).

Yiyang: the four test videos should look the same (same background, same planets), so the only visible
difference is whether the motion lands on the keyword. Every run gets these exact images and a fixed layout;
each run still chooses its own motion and timing.

Writes art/: background.png (1920x1080), sun.png, planets/<name>.png (transparent, lit from the upper left),
lineup.png (all eight in a row), labels/<name>.png (the planet's English name, white, transparent).
Usage: python make_kit.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import zoom, gaussian_filter

OUT = Path(sys.argv[1]); (OUT/"planets").mkdir(parents=True, exist_ok=True); (OUT/"labels").mkdir(exist_ok=True)
rng = np.random.default_rng(7)


def noise(h, w, octaves=5, base=4, seed=0, persist=0.55):
    r = np.random.default_rng(seed); acc = np.zeros((h, w)); amp = 1.0; tot = 0
    for o in range(octaves):
        n = base * 2**o
        g = r.random((n + 3, n + 3))
        z = zoom(g, ((h + 3*h/n) / (n + 3), (w + 3*w/n) / (n + 3)), order=3)[:h, :w]
        acc += amp * z; tot += amp; amp *= persist
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)


def lerp(a, b, t): return a + (b - a) * t


def ramp(stops, x):
    """stops: [(pos, (r,g,b))], x in 0..1 array -> rgb array"""
    x = np.clip(x, 0, 1); out = np.zeros(x.shape + (3,))
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = (x >= p0) & (x <= p1); t = ((x - p0) / max(p1 - p0, 1e-9))[..., None]
        out[m] = (np.array(c0) + (np.array(c1) - np.array(c0)) * t)[m]
    return out / 255.0


TW, TH = 1024, 512  # equirectangular texture size
SEED = {"mercury": 1, "venus": 3, "earth": 5, "mars": 7, "jupiter": 9, "saturn": 13, "uranus": 15, "neptune": 17}


def tex(name):
    lon = np.linspace(0, 1, TW)[None, :].repeat(TH, 0); lat = np.linspace(0, 1, TH)[:, None].repeat(TW, 1)
    n1 = noise(TH, TW, 6, 4, seed=SEED[name]); n2 = noise(TH, TW, 6, 8, seed=SEED[name] + 1)
    if name == "mercury":
        c = ramp([(0, (70, 66, 64)), (0.5, (140, 134, 128)), (1, (200, 194, 186))], 0.35 + 0.6*n1)
        for _ in range(140):  # craters
            cx, cy, r = rng.random()*TW, (0.15 + 0.7*rng.random())*TH, rng.random()**2*26 + 3
            yy, xx = np.ogrid[:TH, :TW]; d = np.sqrt(((xx - cx) * 0.5)**2 + (yy - cy)**2) / r
            c *= (1 - 0.18*np.clip(1 - d, 0, 1)**0.6 + 0.12*np.exp(-((d - 1)**2)/0.02))[..., None]
        return c
    if name == "venus":
        sw = noise(TH, TW, 5, 3, seed=11)
        v = lat + 0.08*np.sin(lon*np.pi*6 + sw*6) + 0.06*n2
        return ramp([(0, (196, 160, 104)), (0.35, (232, 206, 150)), (0.6, (214, 178, 118)), (1, (238, 214, 162))], (np.sin(v*np.pi*5)*0.5 + 0.5)*0.55 + 0.45*n1)
    if name == "earth":
        land = noise(TH, TW, 7, 3, seed=21) > 0.53
        ocean = ramp([(0, (18, 52, 120)), (1, (36, 96, 170))], n2)
        green = ramp([(0, (52, 98, 48)), (0.6, (120, 128, 70)), (1, (170, 150, 100))], n1)
        c = np.where(land[..., None], green, ocean)
        ice = np.abs(lat - 0.5) > 0.42; c[ice] = (0.93, 0.95, 0.97)
        cl = np.clip((noise(TH, TW, 6, 5, seed=23) - 0.52) * 3.2, 0, 1)[..., None]
        return lerp(c, np.ones(3)*0.97, cl*0.85)
    if name == "mars":
        c = ramp([(0, (120, 46, 24)), (0.5, (184, 84, 44)), (1, (222, 140, 92))], 0.25 + 0.75*n1)
        dark = np.clip((noise(TH, TW, 6, 4, seed=31) - 0.55)*3, 0, 1)[..., None]
        c = lerp(c, np.array([0.33, 0.15, 0.09]), dark*0.7)
        cap = np.abs(lat - 0.5) > 0.44; c[cap] = (0.95, 0.93, 0.9)
        return c
    if name in ("jupiter", "saturn"):
        warp = noise(TH, TW, 5, 6, seed=41 if name == "jupiter" else 51)
        v = lat + 0.018*np.sin(lon*np.pi*14 + warp*9) + 0.012*(warp - 0.5)
        if name == "jupiter":
            bands = (0.65*np.sin(v*np.pi*15) + 0.35*np.sin(v*np.pi*41 + 1.3))*0.5 + 0.5
            c = ramp([(0, (150, 96, 64)), (0.45, (206, 160, 118)), (0.75, (236, 214, 184)), (1, (246, 236, 220))], bands*0.8 + 0.2*n2)
            yy, xx = np.ogrid[:TH, :TW]; d = np.sqrt(((xx - 0.62*TW)/70)**2 + ((yy - 0.64*TH)/30)**2)
            spot = np.clip(1.2 - d, 0, 1)[..., None]
            return lerp(c, np.array([0.74, 0.33, 0.2]), spot*0.9)
        bands = np.sin(v*np.pi*11)*0.5 + 0.5
        return ramp([(0, (176, 144, 96)), (0.5, (218, 192, 140)), (1, (240, 224, 186))], bands*0.6 + 0.4*n2*0.6 + 0.2)
    if name == "uranus":
        return ramp([(0, (140, 206, 214)), (1, (182, 230, 234))], 0.6*(np.sin(lat*np.pi*7)*0.5 + 0.5)*0.4 + 0.6*n1*0.3 + 0.35)
    if name == "neptune":
        warp = noise(TH, TW, 5, 5, seed=61)
        v = lat + 0.02*np.sin(lon*np.pi*8 + warp*6)
        c = ramp([(0, (30, 60, 150)), (0.6, (52, 96, 200)), (1, (96, 146, 230))], (np.sin(v*np.pi*9)*0.5 + 0.5)*0.35 + 0.65*n1*0.7)
        yy, xx = np.ogrid[:TH, :TW]; d = np.sqrt(((xx - 0.4*TW)/46)**2 + ((yy - 0.6*TH)/22)**2)
        c = lerp(c, np.array([0.08, 0.15, 0.4]), np.clip(1.1 - d, 0, 1)[..., None]*0.8)
        streak = np.clip((noise(TH, TW, 4, 6, seed=63) - 0.7)*4, 0, 1)*(np.abs(np.sin(lat*np.pi*6)) > 0.9)
        return lerp(c, np.ones(3)*0.9, streak[..., None]*0.6)
    raise ValueError(name)


ATMOS = {"earth": (0.45, 0.7, 1.0), "venus": (1.0, 0.9, 0.7), "neptune": (0.45, 0.65, 1.0), "uranus": (0.7, 0.95, 1.0), "mars": (1.0, 0.6, 0.45)}
TILT = {"earth": 0.41, "mars": 0.44, "jupiter": 0.05, "saturn": 0.47, "uranus": 1.71, "neptune": 0.49, "venus": 0.05, "mercury": 0.0}


def sphere(name, D=720, ss=2, rot=0.15):
    S = D*ss; pad = int(S*0.08); N = S + 2*pad
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    x = (xx - N/2) / (S/2); y = (yy - N/2) / (S/2); r2 = x*x + y*y
    inside = r2 <= 1; z = np.sqrt(np.clip(1 - r2, 0, 1))
    a = TILT[name] * (0.35 if name != "uranus" else 1.0) if name != "uranus" else 1.25
    xr = x*np.cos(a) - y*np.sin(a); yr = x*np.sin(a) + y*np.cos(a)
    lat = np.arcsin(np.clip(yr, -1, 1)) / np.pi + 0.5
    lon = (np.arctan2(xr, z) / (2*np.pi) + 0.5 + rot) % 1.0
    t = tex(name)
    ti = np.clip((lat*(TH - 1)).astype(int), 0, TH - 1); tj = np.clip((lon*(TW - 1)).astype(int), 0, TW - 1)
    col = t[ti, tj]
    L = np.array([-0.55, -0.4, 0.73]); L /= np.linalg.norm(L)
    lam = np.clip(x*L[0] + y*L[1] + z*L[2], 0, 1)
    shade = 0.06 + 0.94*lam**0.85
    limb = 0.75 + 0.25*z
    rgb = col*(shade*limb)[..., None]
    alpha = np.clip((1 - np.sqrt(r2)) * S/2, 0, 1)  # anti-aliased edge (with supersampling)
    if name in ATMOS:
        rim = np.clip(1 - z, 0, 1)**3 * inside
        rgb = rgb + np.array(ATMOS[name])[None, None, :]*(rim*0.55*(0.25 + 0.75*lam))[..., None]
        d = np.sqrt(r2); halo = np.exp(-((d - 1)/0.035)**2) * (d > 1) * (0.25 + 0.75*np.clip(-(x*L[0] + y*L[1])/np.maximum(d, 1e-6), 0, 1))
        a_h = halo*0.55
        rgb = np.where(inside[..., None], rgb, np.array(ATMOS[name])[None, None, :])
        alpha = np.maximum(alpha*inside, a_h)
    else:
        alpha = alpha*inside
    img = np.dstack([np.clip(rgb, 0, 1), alpha])
    im = Image.fromarray((img*255).astype(np.uint8), "RGBA").resize((N//ss, N//ss), Image.LANCZOS)
    return im


def saturn(D=720):
    body = sphere("saturn", D)
    bw, bh = body.size; W = int(bw*2.3); H = int(bh*1.25)
    ss = 2; Ws, Hs = W*ss, H*ss
    yy, xx = np.mgrid[:Hs, :Ws].astype(np.float64)
    cx, cy = Ws/2, Hs/2; tilt = 0.32; ang = -0.32
    xr = (xx - cx)*np.cos(ang) + (yy - cy)*np.sin(ang); yr = -(xx - cx)*np.sin(ang) + (yy - cy)*np.cos(ang)
    R = D*ss/2
    rr = np.sqrt((xr/R)**2 + (yr/(R*tilt))**2)
    ring = ((rr > 1.22) & (rr < 2.15)).astype(float)
    ring *= np.where(rr < 1.45, 0.45, 1.0)  # faint inner ring
    band = 0.78 + 0.12*np.sin(rr*61) + 0.10*np.sin(rr*17.3)
    gap = 1 - 0.85*np.exp(-((rr - 1.78)/0.025)**2)
    a = ring*np.clip(band, 0, 1)*gap*0.92
    a *= np.clip((rr - 1.22)/0.05, 0, 1)*np.clip((2.15 - rr)/0.05, 0, 1)
    col = np.dstack([0.86 + 0*rr, 0.78 + 0*rr, 0.62 + 0*rr])*(0.75 + 0.25*band)[..., None]
    shadowed = (yr < 0) & (np.abs(xr) < R*1.02)  # far half of the ring goes behind the planet
    ring_img = np.dstack([np.clip(col, 0, 1), a])
    back = ring_img.copy(); back[..., 3] *= shadowed
    front = ring_img.copy(); front[..., 3] *= ~shadowed
    def to_im(arr): return Image.fromarray((arr*255).astype(np.uint8), "RGBA").resize((W, H), Image.LANCZOS)
    out = Image.new("RGBA", (W, H)); out.alpha_composite(to_im(back))
    out.alpha_composite(body, ((W - bw)//2, (H - bh)//2)); out.alpha_composite(to_im(front))
    return out


def sun(D=900):
    ss = 2; S = D*ss; N = int(S*1.6)
    yy, xx = np.mgrid[:N, :N].astype(np.float64); d = np.sqrt((xx - N/2)**2 + (yy - N/2)**2)/(S/2)
    gr = noise(N//4, N//4, 6, 6, seed=91); gr = zoom(gr, 4, order=1)[:N, :N]
    core = np.clip(1 - d, 0, 1)**0.35
    rgb = np.dstack([np.ones_like(d), 0.62 + 0.3*core, 0.18 + 0.5*core**2])*(0.88 + 0.12*gr[..., None])
    alpha = np.where(d <= 1, 1.0, np.exp(-(np.clip(d - 1, 0, None)/0.18)**1.2)*0.85)
    img = np.dstack([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)])
    return Image.fromarray((img*255).astype(np.uint8), "RGBA").resize((N//ss, N//ss), Image.LANCZOS)


def background():
    W, H = 1920, 1080
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    g = ramp([(0, (6, 9, 22)), (0.6, (11, 17, 40)), (1, (16, 24, 52))], (xx/W*0.5 + yy/H*0.5))
    neb = noise(H//4, W//4, 6, 3, seed=101); neb = zoom(neb, 4, order=1)[:H, :W]
    band = np.exp(-(((yy - H*0.35) - (xx - W/2)*0.32)/(H*0.22))**2)
    g = g + np.array([0.10, 0.08, 0.18])[None, None, :]*(neb**2*band)[..., None]
    v = 1 - 0.35*(((xx - W/2)/W)**2 + ((yy - H/2)/H)**2)*2.2
    g = g*v[..., None]
    im = Image.fromarray((np.clip(g, 0, 1)*255).astype(np.uint8), "RGB").convert("RGBA")
    dr = ImageDraw.Draw(im)
    for _ in range(900):
        x, y = rng.random()*W, rng.random()*H; b = rng.random()**3
        r = 0.6 + 1.4*b; c = int(120 + 135*b); tint = rng.choice([(1, 1, 1), (1, 0.9, 0.8), (0.85, 0.9, 1)])
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(int(c*tint[0]), int(c*tint[1]), int(c*tint[2]), int(120 + 135*b)))
    glow = im.filter(ImageFilter.GaussianBlur(1.2))
    return Image.alpha_composite(glow, im).convert("RGB")


NAMES = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
background().save(OUT/"background.png")
sun().save(OUT/"sun.png")
P = {}
for n in NAMES:
    P[n] = saturn() if n == "saturn" else sphere(n)
    P[n].save(OUT/"planets"/f"{n}.png")
    print(n, P[n].size)
# lineup: all eight in a row, sizes exaggerated for legibility (not to scale)
REL = {"mercury": 0.38, "venus": 0.56, "earth": 0.58, "mars": 0.45, "jupiter": 1.0, "saturn": 0.9, "uranus": 0.7, "neptune": 0.68}
LW, LH = 1800, 420; lu = Image.new("RGBA", (LW, LH)); x = 20
widths = {n: P[n].size[0]*REL[n] for n in NAMES}
k = (LW - 40 - 7*36)/sum(widths.values()); widths = {n: int(w*k) for n, w in widths.items()}; gap = (LW - 40 - sum(widths.values()))/7
for n in NAMES:
    w = widths[n]; im = P[n].resize((w, int(P[n].size[1]*w/P[n].size[0])), Image.LANCZOS)
    lu.alpha_composite(im, (int(x), (LH - im.size[1])//2)); x += w + gap
lu.save(OUT/"lineup.png")
# labels: one fixed type style for every run
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
def label(text, size=150, idx=10):
    f = ImageFont.truetype(FONT, size, index=idx)
    l, t, r, b = f.getbbox(text); pad = 20
    im = Image.new("RGBA", (r - l + 2*pad, b - t + 2*pad))
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=(246, 240, 228, 255))
    return im
for n in NAMES: label(n.capitalize()).save(OUT/"labels"/f"{n}.png")
label("8 Planets").save(OUT/"labels"/"eight_planets.png")
print("KIT DONE")

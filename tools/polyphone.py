"""多音字检查：从配好的音里，挑出"可能读错"的多音字，拼成一条试听音频给人听。它只负责缩小范围，最后要靠人耳判断。

为什么要单独查：配音那一步的回听，是把声音转回文字再和原稿比。可是识别模型会按上下文"猜对"字，
"往后数（shǔ）十句"念成了"数（shù）十句"，识别出来照样是"数十句"，比对发现不了。

怎么挑：
1. 按上下文给每个多音字标出该念的音（pypinyin 按词组判断；判断不对的，在 --expect 文件里改）。
2. 这个字的每一种读法，都换成一个读音唯一的同音字（shǔ → 鼠，shù → 树），连同前后各一个字，拼成几个候选写法。
3. 把这个字前后那一小段音频切出来，让识别模型（Whisper）给每个候选写法打分。
4. 得分最高的不是该念的那个音，而且差距够大，就列进试听音频。

要知道的局限（实测过）：
- 识别模型自带语言习惯，替换字本身常见不常见、通不通顺，会盖过读音的差别，所以列出来的大多是误报。
- 反过来也可能漏：我们拿读错的录音做过对照，用"原字"打分时原字总是赢，读错也照样听不出来。
所以不要把"没报出来"当成"没读错"。更稳的办法是写稿时就避开容易读错的多音字（替代表见 docs/02_配音.md），
配完后人把 flagged.wav 听一遍，觉得不对的句子再重配。

用法：
  python tools/polyphone.py timeline.json vo/narration.wav --out vo/polyphone [--expect vo/polyphone_expect.json] [--margin 1.0]
输出：
  <out>/report.tsv   每个多音字：第几句、上下文、该念、听起来像、分差
  <out>/flagged.wav  只含可能读错的地方，每处前后多留半秒，中间用提示音隔开
  <out>/flagged.txt  试听音频里每一处是什么
--expect 是一个 JSON：{"往后数十句": {"数": "shu3"}}，键是句子里的一段原文，值是这段里某个字该念的音（拼音加声调数字，轻声写 5）。

依赖：pypinyin、mlx-whisper（苹果芯片 Mac）。别的平台可以把 score() 换成 openai-whisper，算法一样。
"""
import argparse, json, re
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from pypinyin import pinyin, Style

ap = argparse.ArgumentParser()
ap.add_argument("timeline"); ap.add_argument("wav")
ap.add_argument("--out", default="polyphone")
ap.add_argument("--expect", default="")
ap.add_argument("--margin", type=float, default=1.0, help="得分最高的读法比该念的读法高出多少（对数概率）才报")
ap.add_argument("--asr", default="mlx-community/whisper-large-v3-turbo")
ap.add_argument("--lines", default="", help="只查这些句号，逗号分隔")
a = ap.parse_args()

HAN = re.compile(r"[㐀-鿿]")
# 轻声、变调由合成模型自己处理，这些字不查：一、不 的变调；助词和词尾的轻声
SKIP = set("一不了的着么吗呢啊吧个们子头里上下过来去得地和")
# 这几个字口语里常读错，又是全片高频词，单独加进来：得（děi）、地、和、着 在下面按词组补
FORCE = {"还得": "得", "得拉": "得", "得靠": "得"}

tl = json.loads(Path(a.timeline).read_text("utf-8"))
expect_over = json.loads(Path(a.expect).read_text("utf-8")) if a.expect else {}
only = {int(x) for x in a.lines.split(",") if x.strip()}

# 常见多音字和它们的常用读音（只比这些读音；字典里的生僻读音不比，否则误报太多）。
# 这张表也是写稿时的"多音字替代表"的来源，见 docs/02_配音.md。
POLY = {
    "长": "chang2 zhang3", "行": "xing2 hang2", "重": "zhong4 chong2", "还": "hai2 huan2", "得": "de2 dei3",
    "差": "cha4 cha1 chai1", "数": "shu4 shu3 shuo4", "只": "zhi3 zhi1", "量": "liang2 liang4", "调": "diao4 tiao2",
    "降": "jiang4 xiang2", "当": "dang1 dang4", "空": "kong1 kong4", "处": "chu4 chu3", "种": "zhong3 zhong4",
    "为": "wei4 wei2", "发": "fa1 fa4", "卡": "ka3 qia3", "率": "lv4 shuai4", "落": "luo4 la4 lao4", "没": "mei2 mo4",
    "背": "bei4 bei1", "弹": "tan2 dan4", "转": "zhuan3 zhuan4", "看": "kan4 kan1", "相": "xiang1 xiang4",
    "好": "hao3 hao4", "少": "shao3 shao4", "教": "jiao4 jiao1", "觉": "jue2 jiao4", "乐": "yue4 le4",
    "应": "ying1 ying4", "分": "fen1 fen4", "间": "jian1 jian4", "几": "ji3 ji1", "模": "mo2 mu2", "片": "pian4 pian1",
    "便": "bian4 pian2", "倒": "dao3 dao4", "都": "dou1 du1", "干": "gan1 gan4", "更": "geng4 geng1",
    "中": "zhong1 zhong4", "正": "zheng4 zheng1", "扁": "bian3 pian1", "拗": "ao4 niu4", "帧": "zhen1 zheng4",
    "屏": "ping2 bing3", "结": "jie2 jie1", "省": "sheng3 xing3", "似": "si4 shi4", "系": "xi4 ji4", "血": "xue4 xie3",
    "载": "zai3 zai4", "曾": "ceng2 zeng1", "朝": "chao2 zhao1", "称": "cheng1 chen4", "藏": "cang2 zang4",
    "薄": "bo2 bao2", "奇": "qi2 ji1", "供": "gong1 gong4", "给": "gei3 ji3", "露": "lu4 lou4", "强": "qiang2 qiang3",
    "切": "qie1 qie4", "传": "chuan2 zhuan4", "解": "jie3 xie4", "参": "can1 shen1", "冲": "chong1 chong4",
    "校": "xiao4 jiao4", "尽": "jin4 jin3", "劲": "jin4 jing4", "兴": "xing1 xing4", "将": "jiang1 jiang4",
    "难": "nan2 nan4", "答": "da2 da1", "提": "ti2 di1", "担": "dan1 dan4", "假": "jia3 jia4", "散": "san4 san3",
    "塞": "sai1 se4", "色": "se4 shai3", "识": "shi2 zhi4", "压": "ya1 ya4", "铺": "pu1 pu4", "观": "guan1 guan4",
    "旋": "xuan2 xuan4", "折": "zhe2 she2", "混": "hun4 hun2", "横": "heng2 heng4", "划": "hua2 hua4",
    "会": "hui4 kuai4", "创": "chuang4 chuang1", "仔": "zi3 zai3", "吐": "tu3 tu4", "扫": "sao3 sao4",
    "恶": "e4 wu4", "宿": "su4 xiu3", "冠": "guan1 guan4", "晕": "yun1 yun4", "蒙": "meng2 meng3",
    "号": "hao4 hao2", "要": "yao4 yao1", "华": "hua2 hua4", "喝": "he1 he4", "盛": "sheng4 cheng2",
}
POLY = {k: v.split() for k, v in POLY.items()}

def readings(ch):
    return list(POLY.get(ch, []))

# ---------- 每个读音挑一个读音唯一的常用字当替身 ----------
import mlx.core as mx
import mlx_whisper
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
from mlx_whisper.tokenizer import get_tokenizer

model = load_model(a.asr)
tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="zh", task="transcribe")
SUB = {}
for cp in range(0x4E00, 0x9FA6):
    c = chr(cp)
    rs = pinyin(c, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
    if len(rs) != 1 or not re.match(r"^[a-z]+[1-4]$", rs[0]): continue
    ids = tok.encode(c)
    key = (len(ids), ids[0])  # 记号越少越好（很多汉字在识别模型里是两个记号），同样多就挑编号小的（一般更常用）
    if rs[0] not in SUB or key < SUB[rs[0]][1]: SUB[rs[0]] = (c, key)

# ---------- 音频 ----------
wav, sr = sf.read(a.wav, dtype="float32", always_2d=True)
wav = wav.mean(1)
if sr != 16000:
    from math import gcd
    g = gcd(sr, 16000); wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)

PROMPT = list(tok.sot_sequence_including_notimestamps)

def score(seg, texts, prior=True):
    """每个候选写法在这段音频下的对数概率，减去它在一段静音下的对数概率（扣掉模型对替身字本身的偏好）。"""
    raw = _score(seg, texts)
    if not prior: return raw
    base = _score(np.zeros(int(0.6 * 16000), np.float32), texts)
    return [x - y for x, y in zip(raw, base)]

def _score(seg, texts):
    seg = np.pad(seg, (0, max(0, N_SAMPLES - len(seg))))[:N_SAMPLES]
    mel = log_mel_spectrogram(mx.array(seg), n_mels=model.dims.n_mels)
    feats = model.embed_audio(pad_or_trim(mel, N_FRAMES, axis=-2)[None])
    out = []
    for t in texts:
        ids = tok.encode(t) + [tok.eot]
        x = mx.array([PROMPT + ids])
        lp = model.logits(x, feats)[0]
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        n = len(PROMPT)
        out.append(float(sum(lp[n - 1 + j, ids[j]].item() for j in range(len(ids)))))
    return out

# ---------- 逐句找多音字 ----------
rows = []
for L in tl["lines"]:
    if not L.get("real") or (only and L["i"] not in only): continue
    units = L["words"]
    say = "".join(u["u"] for u in units)
    pos2u, ustart = [], []  # 句子里第几个字符属于第几个单元（英文单词是一个单元、好几个字符）
    for k, u in enumerate(units): ustart.append(len(pos2u)); pos2u += [k] * len(u["u"])
    # 该念的音：按连续的汉字段落用 pypinyin 判断（它会看词组）
    exp = [None] * len(units)
    k = 0
    while k < len(units):
        if not HAN.fullmatch(units[k]["u"]): k += 1; continue
        j = k
        while j < len(units) and HAN.fullmatch(units[j]["u"]): j += 1
        seg = "".join(u["u"] for u in units[k:j])
        py = pinyin(seg, style=Style.TONE3, neutral_tone_with_five=True)
        for m in range(j - k): exp[k + m] = py[m][0]
        k = j
    for frag, fix in expect_over.items():
        p = say.find(frag)
        while p >= 0:
            for ch, r in fix.items():
                q = frag.find(ch)
                if q >= 0: exp[pos2u[p + q]] = r
            p = say.find(frag, p + 1)
    for k, u in enumerate(units):
        c = u["u"]
        if not HAN.fullmatch(c): continue
        near = say[max(0, ustart[k] - 1):ustart[k] + 2]
        forced = any(w in near and c == ch for w, ch in FORCE.items())
        if c in SKIP and not forced: continue
        rs = readings(c)
        e = exp[k]
        if e and not e.endswith("5") and e not in rs: rs = [e] + rs
        if len(rs) < 2 or not e or e.endswith("5"): continue
        rs = [r for r in rs if r in SUB]
        if e not in rs or len(rs) < 2: continue
        lo, hi = max(0, k - 1), min(len(units) - 1, k + 1)
        while lo < k and not HAN.fullmatch(units[lo]["u"]): lo += 1
        while hi > k and not HAN.fullmatch(units[hi]["u"]): hi -= 1
        s0, s1 = units[lo]["s"] - 0.06, units[hi]["e"] + 0.06
        seg = wav[int(max(0, s0) * 16000):int(s1 * 16000)]
        pre = "".join(x["u"] for x in units[lo:k]); post = "".join(x["u"] for x in units[k + 1:hi + 1])
        sc = score(seg, [pre + SUB[r][0] + post for r in rs])
        best = rs[int(np.argmax(sc))]
        rows.append({"i": L["i"], "t": round(u["s"], 2), "ctx": say[max(0, ustart[k] - 4):ustart[k] + 5], "ch": c, "expect": e, "heard": best,
                     "margin": round(max(sc) - sc[rs.index(e)], 2), "s0": s0, "s1": s1,
                     "cands": {r: round(v, 2) for r, v in zip(rs, sc)}})

O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
with open(O/"report.tsv", "w", encoding="utf-8") as f:
    f.write("句\t秒\t上下文\t字\t该念\t听起来像\t分差\t各读法得分\n")
    for r in rows:
        f.write(f"{r['i']}\t{r['t']}\t{r['ctx']}\t{r['ch']}\t{r['expect']}\t{r['heard']}\t{r['margin']}\t{json.dumps(r['cands'], ensure_ascii=False)}\n")
bad = [r for r in rows if r["heard"] != r["expect"] and r["margin"] >= a.margin]
# 试听：每处前后各多留 0.5 秒，中间 0.25 秒提示音
beep = (0.15 * np.sin(2 * np.pi * 880 * np.arange(int(0.12 * 16000)) / 16000)).astype(np.float32)
gap = np.zeros(int(0.25 * 16000), np.float32)
parts, notes = [], []
for r in bad:
    s0, s1 = max(0, r["s0"] - 0.5), r["s1"] + 0.5
    parts += [beep, gap, wav[int(s0 * 16000):int(s1 * 16000)], gap]
    notes.append(f"第 {r['i']} 句 {r['t']}s「{r['ctx']}」的「{r['ch']}」该念 {r['expect']}，听起来像 {r['heard']}（分差 {r['margin']}）")
if parts: sf.write(O/"flagged.wav", np.concatenate(parts), 16000)
(O/"flagged.txt").write_text("\n".join(notes) + "\n", "utf-8")
print(f"查了 {len(rows)} 处多音字，疑似读错 {len(bad)} 处")
for n in notes: print("  " + n)

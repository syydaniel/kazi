// 字幕：字按真实念到的时间由浅变深，关键词念到时变色。
// 长句按标点切成不超过 16 个字的小段（Netflix 简体中文规范：每行最多 16 字）；逗号句号去掉，问号叹号保留。
import React from 'react';
import {lines, Cap} from '../lib/time';
import {E, clamp, inv, lerp} from '../lib/anim';

type Chunk = {chars: Cap[]; show: number; hide: number};
const MAX = 16;
const BREAK = new Set('，。？！；：、,.?!;:');
const DROP = new Set('。，、；：,.;:');
const width = (cs: Cap[]) => cs.reduce((a, c) => a + (/[A-Za-z0-9 .\-']/.test(c.c) ? 0.55 : BREAK.has(c.c) ? 0 : 1), 0);

const build = (): Chunk[] => {
  const out: Chunk[] = [];
  for (const l of lines) {
    const phrases: Cap[][] = [];
    let cur: Cap[] = [];
    for (const c of l.cap) {
      cur.push(c);
      if (BREAK.has(c.c)) {
        phrases.push(cur);
        cur = [];
      }
    }
    if (cur.length) phrases.push(cur);
    let acc: Cap[] = [];
    const flush = () => {
      if (acc.length) out.push({chars: acc, show: 0, hide: 0});
      acc = [];
    };
    for (const p of phrases) {
      if (width(acc) + width(p) <= MAX) acc = acc.concat(p);
      else {
        flush();
        if (width(p) <= MAX) acc = p;
        else for (let i = 0; i < p.length; i += MAX) {
          acc = p.slice(i, i + MAX);
          flush();
        }
      }
    }
    flush();
  }
  for (const ch of out) {
    const v = ch.chars.filter((c) => !BREAK.has(c.c));
    ch.show = (v[0] ?? ch.chars[0]).s - 0.1;
    ch.hide = (v[v.length - 1] ?? ch.chars[0]).e + 0.7;
  }
  for (let i = 0; i < out.length - 1; i++) {
    out[i].hide = Math.min(out[i].hide, out[i + 1].show);
    if (out[i + 1].show - out[i].hide < 0.25) out[i].hide = out[i + 1].show; // 空不到 0.25 秒就直接接上，不闪空
  }
  return out;
};
const CHUNKS = build();

export const Caption: React.FC<{t: number; ink?: string; key_?: string; band?: string; font?: string; y?: number}> = ({
  t,
  ink = '#1F1D1A',
  key_ = '#2E49D4',
  band = 'rgba(251,248,241,0.9)',
  font = '"PingFang SC", "Noto Sans SC", sans-serif',
  y = 1006,
}) => {
  const ch = CHUNKS.find((c) => t >= c.show && t < c.hide);
  if (!ch) return null;
  const vis = E.smooth(inv(ch.show, ch.show + 0.1, t));
  const chars = ch.chars.filter((c, i, a) => !(DROP.has(c.c) && i === a.length - 1));
  return (
    <div
      style={{
        position: 'absolute',
        left: 960,
        top: y,
        transform: 'translate(-50%, -50%)',
        opacity: vis,
        padding: '10px 30px 12px',
        borderRadius: 14,
        background: band,
        fontFamily: font,
        fontSize: 44,
        lineHeight: 1.25,
        whiteSpace: 'nowrap',
      }}
    >
      {chars.map((c, i) => {
        if (DROP.has(c.c)) return <span key={i} style={{display: 'inline-block', width: '0.6em'}} />;
        const k = clamp((t - c.s + 0.03) / 0.08);
        return (
          <span key={i} style={{color: c.k && k > 0 ? key_ : ink, opacity: lerp(0.42, 1, k)}}>
            {c.c}
          </span>
        );
      })}
    </div>
  );
};

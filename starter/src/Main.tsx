// 演示：一个细胞先"复制"、再"分开"。画的是两个动词，不是一张细胞插图。
// 每个动作的到位时刻都来自 beats.ts（也就是时间表），不在这里手写秒数。
// 换成你自己的片子时：保留 lib/、comp/ 和 beats.ts 的写法，把这个文件换成你的场景。
import React from 'react';
import {AbsoluteFill, Audio, staticFile, useCurrentFrame} from 'remotion';
import {Caption} from './comp/Caption';
import {B} from './beats';
import {E, inv, lerp, ramp, springTo} from './lib/anim';

const PAPER = '#F4EFE5', INK = '#1F1D1A', BLUE = '#2E49D4';
// 有配音时把 narration.wav 放进 public/，再把这里改成 true
const HAS_AUDIO = false;

export const Main: React.FC = () => {
  const t = useCurrentFrame() / 60;

  // 出场：在第一句开口前就位（进场 0.4 秒，快出慢停）
  const inK = ramp(t, B.cellIn - 0.4, 0.4, E.enter);

  // 复制：第二个核从第一个里滑出来，在 B.copy 这一刻到位（轻微过冲再停稳）
  const copyP = springTo(t, B.copy - B.copyDur, B.copy);
  // 分开：细胞一分为二，在 B.split 到位
  const splitP = springTo(t, B.split - B.splitDur, B.split);
  const gap = 230 * splitP;
  const pinch = Math.sin(Math.PI * Math.min(1, splitP)) * 0.22; // 分开过程中腰部收紧
  const nucOff = 70 * copyP * (1 - Math.min(1, splitP)); // 两个核先在一个细胞里错开

  // 镜头：整段缓慢推近一点，分开那一下略微拉远，给两个细胞留位置
  const zoom = lerp(1.0, 1.05, E.inOut(inv(0, B.split, t))) - 0.04 * Math.min(1, splitP);

  const cell = (dx: number, nuc: number[]) => (
    <g transform={`translate(${dx},0)`}>
      <ellipse cx={0} cy={0} rx={180 * (1 - pinch * 0.4)} ry={180 * (1 - pinch)} fill="#FFFFFF" stroke={INK} strokeWidth={5} />
      {nuc.map((x, i) => (
        <circle key={i} cx={x} cy={0} r={46} fill={i === 1 ? BLUE : INK} opacity={i === 1 ? Math.min(1, copyP * 1.4) : 0.9} />
      ))}
    </g>
  );

  return (
    <AbsoluteFill style={{background: PAPER}}>
      {HAS_AUDIO && <Audio src={staticFile('narration.wav')} />}
      <svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
        <g transform={`translate(960,480) scale(${zoom * lerp(0.85, 1, inK)})`} opacity={inK}>
          {splitP < 0.5
            ? cell(0, [-nucOff, nucOff])
            : [cell(-gap, [0]), cell(gap, [0])].map((c, i) => <g key={i}>{c}</g>)}
        </g>
      </svg>
      <Caption t={t} />
    </AbsoluteFill>
  );
};

// 打点表：每个动作在哪一刻到位。画面（Main.tsx）和音效（scripts/cues.ts）都从这里取，保证声画同一个时间。
// 规矩：到位时刻用 hit()；元素第一次出现用 appearBy() 之前；动作时长 0.25 到 0.5 秒。
import {appearBy, hit, line} from './lib/time';

const L1 = line('复制');
export const B = {
  cellIn: L1.start - 0.25, // 细胞在第一句开口前就已经在了
  copy: hit('复制'), // 复制：第二个核从第一个里滑出来，到位
  split: hit('分开'), // 分开：细胞一分为二，到位
  splitDur: 0.45,
  copyDur: 0.35,
};
// 保险：新元素至少比关键词早 12 帧出现
if (B.cellIn > appearBy('复制')) throw new Error('细胞出现得太晚了');

// 音效打点：t 是声音峰值要落的时刻；kind 对应音效库里的子文件夹
export type Cue = {t: number; kind: string; gain: number; pan?: number};
export const CUES: Cue[] = [
  {t: B.cellIn + 0.15, kind: 'soft_in', gain: -14},
  {t: B.copy, kind: 'pop', gain: -10, pan: 0.2},
  {t: B.split - B.splitDur * 0.5, kind: 'whoosh', gain: -16},
  {t: B.split, kind: 'tick', gain: -12},
];

// 要原样播放的现成声音（比如一段试听），src 是 public/ 下的路径，from 从素材第几秒开始，dur 放多久，gain 在对齐旁白响度之后再加减（dB）。
// 混音时 tools/mix.py --clips out/clips.json --clip-root public 会把它们放进去，放的时候音乐让位。
export type Clip = {t: number; src: string; from?: number; dur?: number; gain?: number};
export const CLIPS: Clip[] = [];

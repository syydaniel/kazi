// 卡字的时间 API：画面里所有和旁白有关的时刻，都从 timeline.json 里查，不按字数估。
// timeline.json 由 tools/timeline.py 生成：每句的起止、每个念法单位（一个汉字，或一个英文/数字词）的起止、关键词列表。
import TL from '../data/timeline.json';

export type Word = {u: string; s: number; e: number};
export type Cap = {c: string; s: number; e: number; k?: number};
export type Line = {i: number; sec: string | null; start: number; end: number; pause: number; say: string; show: string; real: boolean; words: Word[]; cap: Cap[]};
export type Keyword = {line: number; word: string; t: number; e: number};

export const tl = TL as unknown as {
  fps: number;
  total: number;
  lines: Line[];
  sections: Record<string, {start: number; end: number; first: number; last: number}>;
  gaps: {name: string; start: number; end: number}[];
  keywords: Keyword[];
};
export const lines = tl.lines;
export const keywords = tl.keywords ?? [];

// ---- 时间常数（60 帧下）----
// 动作到位的那一帧，放在关键词首字起点前 3 帧（50 毫秒）。
// 依据 ITU-R BT.1359：声音超前约 45 毫秒就会被察觉，画面超前到约 125 毫秒才会被察觉，所以宁早勿晚。
export const LEAD = 3 / 60;
// 允许范围：提前 6 帧到推后 1 帧。
export const LEAD_MIN = -1 / 60, LEAD_MAX = 6 / 60;
// 新元素的第一帧，至少比关键词早 12 帧：眼睛大约 200 毫秒才开始扫过去。
export const APPEAR = 12 / 60;

// 念法单位的切分，和 tools/timeline.py 一致
export const splitUnits = (s: string) => s.match(/[A-Za-z0-9][A-Za-z0-9.\-']*|[㐀-鿿]/g) ?? [];

// 找到含有 text 的第 n 句（从 0 数）。找不到就报错，绝不静默给一个假时间。
export const line = (text: string, n = 0) => {
  const ls = lines.filter((l) => l.say.includes(text) || l.show.includes(text));
  if (!ls[n]) throw new Error(`找不到第 ${n} 句含「${text}」的句子`);
  return ls[n];
};

// 只在某一句里找词：edge='s' 取第一个字开始，'e' 取最后一个字结束
export const atIn = (l: Line, word: string, n = 0, edge: 's' | 'e' = 's') => {
  const units = l.words.map((w) => w.u.toLowerCase());
  const target = splitUnits(word).map((u) => u.toLowerCase());
  let k = 0;
  for (let i = 0; i + target.length <= units.length; i++) {
    if (target.every((u, j) => units[i + j] === u)) {
      if (k === n) return edge === 's' ? l.words[i].s : l.words[i + target.length - 1].e;
      k++;
    }
  }
  throw new Error(`第 ${l.i} 句里找不到第 ${n} 次「${word}」`);
};

// 全片按顺序找第 n 次出现的词
export const at = (word: string, n = 0, edge: 's' | 'e' = 's') => {
  let k = 0;
  for (const l of lines) {
    const units = l.words.map((w) => w.u.toLowerCase());
    const target = splitUnits(word).map((u) => u.toLowerCase());
    for (let i = 0; i + target.length <= units.length; i++) {
      if (target.every((u, j) => units[i + j] === u)) {
        if (k === n) return edge === 's' ? l.words[i].s : l.words[i + target.length - 1].e;
        k++;
      }
    }
  }
  throw new Error(`全片找不到第 ${n} 次「${word}」`);
};

// 动作应该到位的时刻
export const hit = (word: string, n = 0) => at(word, n) - LEAD;
// 新元素最晚什么时候出现
export const appearBy = (word: string, n = 0) => at(word, n) - APPEAR;
// 稿子里标了 *星号* 的关键词
export const kw = (word: string) => {
  const k = keywords.find((x) => x.word === word);
  if (!k) throw new Error(`稿子里没有标成关键词的「${word}」`);
  return k;
};

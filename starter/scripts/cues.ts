// 把打点表导出成 out/cues.json，现成的声音片段导出成 out/clips.json，交给 tools/mix.py
import {mkdirSync, writeFileSync} from 'node:fs';
import {CUES, CLIPS} from '../src/beats';

mkdirSync('out', {recursive: true});
writeFileSync('out/cues.json', JSON.stringify(CUES.slice().sort((a, b) => a.t - b.t), null, 1));
writeFileSync('out/clips.json', JSON.stringify(CLIPS, null, 1));
console.log(`导出 ${CUES.length} 个打点到 out/cues.json，${CLIPS.length} 个声音片段到 out/clips.json`);

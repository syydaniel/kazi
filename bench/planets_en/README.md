# With and without Kazi: English rerun (2026-11-07)

The same test as `bench/README.md` (the eight planets), run again for the English film: English narration, and this time all four runs got **the same art and layout**, so the four videos differ only in how and when things move.

## Setup

- **Narration**: 39.3 s, nine lines, nine keywords (the phrase "eight planets" plus each planet's name), marked with `*stars*` in `script.txt`. The voice is a cloned voice (`narration.wav`).
- **Who builds it**: Claude Code, two models (Opus and Sonnet), each twice: once without Kazi, once with it. Four runs.
- **Isolation**: each run is a subagent in its own empty folder and may only read and write there. The two runs without Kazi have no Kazi files at all.
- **Prompt**: identical for all four (`prompt.txt`). The two Kazi runs get two extra lines saying where the skill is installed (`prompt_kazi_extra.txt`). The prompt never asks to sync anything to the keywords; it only says keywords are marked with stars. The exact prompt each run received is in `runs/<run>/PROMPT.txt` (local paths replaced with `~/kazi-bench`).
- **Shared art**: `kit/make_kit.py` draws the background, Sun, planets and labels that every run was given (it uses the macOS Helvetica Neue font; change `FONT` on other systems).
- **No help along the way**: nobody answered the runs while they worked, and their videos are untouched: `videos/<run>.mp4`.
- **Scoring**: `score_en.py` runs `tools/kazi_check.py` on each video against one shared word timeline (`timeline.json`). A keyword counts as hit when the picture change from 0.35 s before to 0.15 s after the keyword is at least 1.5 times the change in the second before that. The chance baseline applies the same rule at every moment of the narration.
- **The timeline**: `timeline_aligner.json` is the forced aligner's raw output. In it, three names (Earth, Uranus, Neptune) start 0.11 to 0.20 s early, inside the pause before the word. The Opus run with Kazi caught this during its own check and fixed it from the waveform; we confirmed it on the waveform and score all four runs against the corrected `timeline.json`. `tools/timeline.py` now makes this correction automatically.

## Results

| Run | Keywords hit | Chance baseline |
|---|---|---|
| Opus, without Kazi | 5 / 9 | 31% |
| Opus, with Kazi | 9 / 9 | 31% |
| Sonnet, without Kazi | 2 / 9 | 20% |
| Sonnet, with Kazi | 1 / 9 | 22% |

Per-keyword details, offsets and the frame-change curves are in `results.json`; each run's `check.json` is in `runs/<run>/`.

What we saw (checked frame by frame):
- **Both runs without Kazi put every planet in the right place.** Both used speech recognition word timestamps, so whenever a name is spoken, that planet is on screen.
- **The difference is timing.** In the runs without Kazi, 15 of the 18 biggest moves come more than 0.3 s before the name (median 1.8 s early), mostly as each line starts. When the name arrives, little is moving.
- **The Saturn line** ("The one with the rings is Saturn."): without Kazi, Opus slides Saturn in at 26.4 to 27.1 s, a second before the name (27.99 s), and only a small label fades in on the word. With Kazi, Saturn swells and its label pops in at 27.6 to 27.9 s.
- **Sonnet with Kazi** read `SKILL.md` but skipped `timeline.py` and `kazi_check`. Its moves land a little before each name (the biggest change sits about 0.8 s early), and it hit one of nine.
- **Opus with Kazi** ran both tools. Its first draft already scored 9 / 9 on its own timeline (`runs/opus_kazi/check_first_draft.json`); along the way it found the three early names above.

## Limits

- Tiny sample: one run per setup, one narration, nine keywords. A demonstration, not a benchmark. Results will change with model versions.
- The Kazi rate only measures whether motion lands on the keywords, not whether the video looks good. Watch `videos/` yourself.
- The Kazi runs knew one extra fact ("there is a skill here"), which is the thing being tested.
- The hit rule is ours (reasons in `docs/`); change the thresholds and score again.

## Run it again

```bash
python bench/planets_en/score_en.py --out /tmp/results.json
```

Needs ffmpeg and numpy. To rebuild a run from scratch, give a fresh Claude Code session `prompt.txt` (plus `prompt_kazi_extra.txt` and the skill for the Kazi runs), `script.txt`, `narration.wav` and the art from `kit/make_kit.py`.

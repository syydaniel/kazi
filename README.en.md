# kazi (卡字)

**Make AI-generated explainer animations land on the keywords of the voice-over.**

[中文](README.md) · Demo video: on Bilibili (link after release) · Style gallery: [awesome-code-animation](https://github.com/syydaniel/awesome-code-animation)

Beat-sync edits hit the drum; explainer videos should hit the words. An AI writing animation code cannot hear the narration, so it guesses where the viewer should look, and the picture ends up half a beat early or late.
kazi flips the order: **sound first, picture second, and the picture always follows the sound.** A forced aligner gives every character its start and end time; every motion looks its frame up in that table; at the end, every frame is measured.

It is a skill for AI coding assistants: experience docs, tools, and a Remotion starter. Install it, ask for an explainer video, and the assistant follows the method.

Despite the name, it covers more than timing: script, voice, timing, picture, sound, and checking and publishing. The assistant walks through all six stages, measures its own work after each one, and goes back to fix what fails. It also works with the tools you already have: Qwen3-TTS for the voice, ACE-Step for music, Remotion for the picture, FFmpeg for checks, and the [code-animation style gallery](https://github.com/syydaniel/awesome-code-animation) when picking a look.

> The docs are written in Chinese, because the method is tuned for Chinese narration (one keyword per sentence, character-level timing). The tools and the timing rules work for any language the aligner supports.

## With and without

The same 39-second narration about the eight planets (nine keywords), built twice by each model in Claude Code (Opus and Sonnet), once without the skill and once with it:

| | without kazi | with kazi |
|---|---|---|
| Opus | 1 / 9 | 9 / 9 |
| Sonnet | 0 / 9 | 9 / 9 |

Without the skill, every planet was still the right one; the difference is when things move. The biggest motion often came more than a second before the name, so the picture had already settled when the name was spoken. With the skill, the motion lands on the name.

A tiny sample. Method, prompts, the four videos and the caveats are in [bench/](bench/README.md); you can rerun it.

## Six steps

1. **Script**: write for the ear, one keyword per sentence.
2. **Voice**: clone your voice, generate each line four times, let another model listen back, keep the best take.
3. **Timing**: forced alignment gives every character a start and end time.
4. **Picture**: animate the verb, not the noun; motions land 3 frames before the keyword.
5. **Sound**: music ducks under the narration; sound effects are read from the animation cues.
6. **Check and publish**: measure keyword hits, black frames, flashes, loudness; prepare title, cover, description.

The assistant pauses four times (script, voice, sample, final) for your approval, unless you tell it not to.

## Install

```bash
git clone https://github.com/syydaniel/kazi ~/.claude/skills/kazi
cd ~/.claude/skills/kazi
pip install -r tools/requirements.txt
python tools/doctor.py
```

Other assistants that support skills: put the folder in their skills directory under the name `kazi`. Only Claude Code has been tested.

## Models and tools

Models are downloaded once, then everything runs locally:

| Purpose | Default |
|---|---|
| Voice cloning | Qwen3-TTS 1.7B Base (MLX) |
| Listen-back scoring | Whisper large-v3-turbo (MLX) |
| Forced alignment | Qwen3-ForcedAligner 0.6B |
| Music | ACE-Step 1.5 |
| Picture | Remotion 4 (`starter/`), or anything else |
| Check and mix | ffmpeg, `tools/kazi_check.py`, `tools/mix.py` |
| 4K and HDR | Remotion at 2× scale; x265, dovi_tool and MP4Box for Dolby Vision 8.4 |

Remotion is free for individuals and small teams; companies should check its license. Each model is used under its own license.

## Known limits

1. Voice and alignment are only tested on Apple Silicon Macs. Windows is untested; `tools/doctor.py` prints workarounds. Picture, check and mix are cross-platform.
2. Four takes per line is slow: about 15 seconds per line on an Apple Silicon Mac, so a 15-minute film with 150 lines takes 30 to 40 minutes to voice, and several times longer if another model is using the GPU at the same time.
3. The keyword-hit rate measures timing, not taste. Your eyes still decide whether it looks good.

## License

Code (`tools/`, `starter/`, `bench/*.py`): [MIT](LICENSE). Docs (`SKILL.md`, `docs/`, READMEs) and bench media: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

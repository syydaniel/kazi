# 卡字（kazi）

**让 AI 做的讲解动画，画面卡在旁白的关键词上。**

[English](README.en.md) · 演示视频：B 站《卡字》（发布后补链接） · 画风图鉴：[awesome-code-animation](https://github.com/syydaniel/awesome-code-animation)

卡点视频卡鼓点，讲解视频卡字。AI 写动画代码的时候听不到声音，只能估计这一秒观众该看哪里，所以画面总比旁白慢半拍或者快半拍。
卡字把顺序反过来：**先有声音，再有画面，画面始终追随声音**。每个字是第几秒念的，先用对齐模型算出来；动作落在哪一帧，从表里查；做完再逐帧量一遍。

这是一个给 AI 编程助手用的 skill（经验文档 + 工具 + 画面模板）。你装上它，跟 AI 说"做一条讲解视频"，剩下的它照着做。

名字叫卡字，管的却不只是对时间：写稿、配音、对时间、画面、声音、检查和发布，六个环节它都会带着 AI 走一遍，每一步做完自己量一遍，不合格就回去改。它也会和你手边的工具配合：配音交给千问 Qwen3-TTS，配乐交给 ACE-Step，画面交给 Remotion，检查交给 FFmpeg，挑画风时去翻[代码画风图鉴](https://github.com/syydaniel/awesome-code-animation)。

## 装和不装，差多少

同一段讲太阳系八颗行星的旁白（39 秒，九个关键词），Claude Code 里用 Opus 和 Sonnet 各做两遍，一遍不装，一遍装上：

| | 不装卡字 | 装上卡字 |
|---|---|---|
| Opus | 1 / 9 | 9 / 9 |
| Sonnet | 0 / 9 | 9 / 9 |

不装的两遍，行星一颗都没放错，差在动的时机：最大的动作往往比名字早一秒多，名字念出来时画面已经停了。装上的两遍，动作落在念到名字的那一下。
样本很小，测法、提示词、四段原片和局限都在 [bench/](bench/README.md)，可以自己复现。

## 六步

| 步 | 做什么 | 细节 |
|---|---|---|
| 1 写稿 | 照说话写，每句一个关键词 | [docs/01_写稿.md](docs/01_写稿.md) |
| 2 配音 | 克隆你的声音，每句配四遍，另一个 AI 回听，留最好的 | [docs/02_配音.md](docs/02_配音.md) |
| 3 对时间 | 对齐模型给每个字标起止时间 | [docs/03_对时间.md](docs/03_对时间.md) |
| 4 画面 | 画动词；动作落在关键词首字前 3 帧 | [docs/04_画面.md](docs/04_画面.md) |
| 5 声音 | 配乐给旁白让位；音效从动画打点里读出来 | [docs/05_声音.md](docs/05_声音.md) |
| 6 检查和发布 | 逐帧量卡字率、黑帧、闪帧、响度；准备标题、封面、简介 | [docs/06_检查和发布.md](docs/06_检查和发布.md) |

中间有四个停顿：稿子、配音、样片、成片，每一处都会停下来等你点头。你也可以说"不用停，直接做完"。

## 安装

Claude Code：

```bash
git clone https://github.com/syydaniel/kazi ~/.claude/skills/kazi
cd ~/.claude/skills/kazi
pip install -r tools/requirements.txt
python tools/doctor.py        # 看缺什么，缺了怎么装
```

其他支持 skill 的 AI 助手：把整个目录放到它约定的 skills 目录里，名字叫 `kazi`。我们只在 Claude Code 上实测过。

GitHub 打不开的话，网盘里有打包好的版本（链接见视频置顶评论）。

## 怎么用

跟 AI 说就行，比如：

- "用卡字做一条两分钟的讲解视频，讲光为什么会折射。我的声音样本在 ~/voice/ref.wav，逐字稿是 ref.txt。"
- "我已经录好旁白了（narration.wav），用卡字给它配画面。"
- "只帮我做对时间和检查这两步，画面我自己做。"

几个建议：先做 30 秒的样片，没问题再做全片；做出满意的结果，让它把经验记下来，但别覆盖，每一版都留着；做完让它清理中间文件，4K 素材很占地方。

## 用到的模型和工具

第一次用要下载模型，之后都在你自己的电脑上跑，不用调用任何在线接口：

| 用途 | 默认 |
|---|---|
| 配音（克隆声音） | Qwen3-TTS 1.7B Base（MLX 版） |
| 回听打分 | Whisper large-v3-turbo（MLX 版） |
| 逐字对齐 | Qwen3-ForcedAligner 0.6B |
| 配乐 | ACE-Step 1.5 |
| 画面 | Remotion 4（`starter/` 是模板）；也可以换成别的，见画风图鉴 |
| 检查和混音 | ffmpeg + `tools/kazi_check.py`、`tools/mix.py` |
| 4K 和 HDR | Remotion 放大渲染；x265、dovi_tool、MP4Box 出杜比视界 8.4 |

Remotion 对个人和小团队免费，公司使用要看它的许可条款。各个模型按各自的许可证使用。

## 不足，先说清楚

1. 配音和对齐只在苹果芯片的 Mac 上测过，Windows 还没实测。`python tools/doctor.py` 会给出替代办法；画面、检查、混音这几步不受影响。
2. 每句配四遍再挑，会慢不少：在苹果芯片的 Mac 上每句大约 15 秒，一条 15 分钟、150 句的片子，全部配完要半小时到四十分钟；同时跑别的模型、抢显卡时会慢好几倍。
3. 卡字率只量画面落没落在关键词上，好不好看它量不出来，还得靠你的眼睛。

你用卡字做了片子，欢迎发给我，做得好的会放进示例里。有不一样的做法，也欢迎提 issue 或 PR。

## 许可证

代码（`tools/`、`starter/`、`bench/*.py`）用 [MIT](LICENSE)；文档（`SKILL.md`、`docs/`、各 README）用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hans)。`bench/` 里的旁白和测试视频同样按 CC BY 4.0 提供。

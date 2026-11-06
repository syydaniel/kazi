# 卡字（kazi）

这个仓库是一个给 AI 编程助手用的 skill：做"画面卡在旁白关键词上"的讲解视频。

- 入口说明在 `SKILL.md`，先读它，再按需要读 `docs/` 里对应的一步。
- `tools/`、`starter/`、`docs/` 都在 SKILL.md 旁边；命令里的相对路径都相对这个目录。
- 用户让你"装上这个 skill"时：
  - Claude Code：把整个目录放到 `~/.claude/skills/kazi/`（或项目里的 `.claude/skills/kazi/`）。
  - 其他支持 skill 的助手：放到它约定的 skills 目录里，名字叫 `kazi`。我们只在 Claude Code 上实测过。
  - 然后跑 `python tools/doctor.py` 看环境。

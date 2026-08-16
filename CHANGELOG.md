# Changelog

llama.cpp Windows 多模型部署技能（llama-cpp-windows-deployment-skill）版本变更记录。

## [v3.2] - 2026-08-16

### Added
- 新增实战经验沉淀 `references/20260816-session-experience.md`：26B-A4B `--fit` 自动显存分层（实测 10.6→72-93 t/s，约 7 倍）、ctx-shift 与 mmproj 互斥、128K 官方甜点、CPU 工具调用新模型、GBK 乱码 skeleton+LCS 恢复 SOP、MTP 官方 Q8_0 判定法
- 新增参数知识库 `references/model-profiles.json`（17 个模型 profile，含采样/ctx/KV/MTP 规则与 verified 分级）
- 新增 Router Mode 参考脚本 `references/router-mode-preset.bat`（`--models-preset` 版）与 `references/router-mode-simple.bat`（`--models-dir` 版）
- CPU 工具调用脚本扩至 8 菜单项：新增 Qwen3.5-4B-UD、QwenPaw-Flash-9B-heretic-MTP、QwenPaw-Flash-9B、LFM2.5-8B-A1B（含 REASONING 控制块）

### Changed
- SKILL.md 升级至 v3.2：26B-A4B 章节优先 `--fit on --fit-ctx`（手工 `-ngld` 仅后备）、ctx 推荐 32K→128K 官方甜点、Troubleshooting 新增 BAT `for`+`goto` 死循环与 GBK 恢复 SOP
- 许可证由 MIT 切换为 **Apache License 2.0**
- 内容保持脱敏（无个人机器路径 / API Key）

## [v3.1] - 2026-08-05

### Fixed
- 参考脚本转纯 ASCII + 删除 `chcp`（修复中文 Windows cmd 下命令错乱，如 `llama-server.exe` 被截断为 `erver.exe`）
- gemma4 菜单脚本缺失 `goto :menu` 导致启动即报 Empty model path
- `nvidia-smi --format=csv,...` 在 bat 中逗号被 cmd 当参数分隔符（改 `--format="csv,..."`）

### Changed
- 参考脚本参数化：顶部配置区（LLAMA_DIR/CHAT/PORT/API_KEY），端口默认 8080/8083/8086

## [v3.0] - 2026-08-05

### Added
- 新增纯 CPU 工具调用脚本 `references/start-CPU-Toolcall-Launcher.bat`（`-ngl 0`、128K 上下文、4 模型菜单）
- 新增跨平台诊断脚本 `scripts/detect.py`（Python 3.7+，与 `detect.ps1` 功能一致）
- 新增实战经验沉淀文档 `references/20260803-session-experience.md`、`references/20260805-session-experience.md`
- 推理控制参数（`--reasoning` / `--reasoning-budget` / `--reasoning-format`）防止思考/工具调用死循环

### Changed
- SKILL.md 升级至 v3.0：Qwen3.6-27B 深度优化（64K + 高 ngl 提速）、Tool Calling 内置机制、BAT 编码修复经验
- 参考 BAT 脚本统一为 UTF-8 + 全英文（修复中文 Windows 下乱码闪退）
- 内容完成脱敏：个人机器路径、个人化 API Key、会话 ID 替换为通用示例/占位

### Fixed
- BAT 脚本中文乱码闪退问题（GBK/UTF-8 编码，`chcp 65001` 与 BOM 均无效的根因）
- Qwen3.6-27B 128K 配置必 OOM（改 64K + 更高 ngl 反而更快）

## [v2.0] - 2026-07-27

### Added
- 初始版本：Router Mode 多模型路由、Gemma 4 / Qwen / Phi-4 部署、QAT–MTP 推理优化、显存调优、CPU 内存受限场景、WSL2 对接、环境诊断脚本 `scripts/detect.ps1`

# Changelog

llama.cpp Windows 多模型部署技能（llama-cpp-windows-deployment-skill）版本变更记录。

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

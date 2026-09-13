# Changelog

llama.cpp Windows 多模型部署技能（llama-cpp-windows-deployment-skill）版本变更记录。

## [v3.5.1] - 2026-09-13

### Fixed
- `references/mtp-head-grafting.md` §7 改为与 `_mtp_graft.py` 实际行为一致：兼容性是**闸门**
  （不满足即拒绝写入，退出码 1，不落盘），并列出全部拒绝条件；删除已不存在的「锚点层」检查说明
- 补充：**量化类型差异不是拒绝条件** —— 同一基座的两种量化配方（`UD-Q4_K_XL` vs `Q4_K_P`）
  有 381/733 个张量类型不同但 shape 全一致，第一版把它当硬条件会**误拒已知可用的真实组合**

## [v3.5] - 2026-09-13

### Added
- **SKILL.md 新增第五章「MoE 显存预算与卸载」**（此前技能内 `n-cpu-moe` 出现 **0 次**）：
  - §5.1 **`--fit` 与 `--n-cpu-moe` 互斥**（`--n-cpu-moe` 编译成 `-ot` 覆盖 ⇒ `common_fit_params()` 放弃 fit）⇒ MoE 条目上的 `--fit on` **从未生效**，且 **MoE 模型没有自动降级兜底**
  - §5.2 显存预算方程（张量表直读 `W_non`/`E_layer`；专家字节**按层非均匀**，必须用真实累积曲线而非层数×均值）
  - §5.3 **`-b`/`-ub` 必须成对提升**（`n_ubatch` 静默 clamp）；实测 prefill **955 → 1,867 t/s（+95%）**，解码不变
  - §5.4 `--load-mode none` 是**性能修复**而非废弃改名（mmap + `n-cpu-moe` 掉 60% prefill）
  - §5.5 `n-cpu-moe` 阶梯实测参考 + 「结论必须标注测量 ctx」
  - §5.6 **FND-064**：`.bat` 与 `ini` 两套生成路径必须同步
- **新增 `references/mtp-head-grafting.md`（内建 MTP head 嫁接手册）**：完整可照做的 GGUF 张量级手术流程 —— 适用判定、布局与 4 处必改、**四个陷阱**（KV 取 target / 对齐填充 / `nextn_predict_layers` / 写后自检）、7 步验证协议、**显存代价 = 权重字节 × 3**、新目录落地注意事项；参考实现 `plan/_mtp_graft.py`
  - 实测：解码 **64.28 → 86.51 t/s（+34.6%）**，acceptance **0.736**
- **新增 `references/20260913-session-experience.md`**：FND-066 互斥机制、`-ub` clamp 陷阱、`--load-mode` 性能洞、GGUF 张量表解析技巧（按长度 seek 跳 KV / 两种专家命名形态 / 大小校验查不出分类错误）、回归方法论（负向对照 `--self-test`、**变异测试**、`__pycache__` 假失败陷阱、**两个真实点不约束规则 → 补阈值边界用例**）、告警自身会误报、**大体积操作前必须讲清依赖关系**、GitNexus 未索引时的等价影响分析、提交卫生（重组与行为改动分离 / `R100`）
- SKILL.md §一 新增触发词行：内置 MTP head / 嫁接、MoE 卸载、prefill 提速、参数一致性
- SKILL.md §六 新增「内置 MTP head（`nextn_predict_layers`）」小节 + 嫁接手册指引
- SKILL.md 参数速查表新增 `--load-mode` 与 `--batch-size` + `--ubatch-size` 成对行

### Fixed
- SKILL.md 参数速查表 `--no-mmap` 废弃行 → `--load-mode none`，并标注 60% prefill 性能洞
- SKILL.md §七 Troubleshooting 新增 7 行：`--fit` 无效、`-ub` 无效、override 不同步、两路径不一致、MTP 不加速、嫁接结果错、`__pycache__` 假失败

### Notes
- 新增内容中的路径**统一使用占位符**（`<models-dir>` / `<llama-cpp-dir>`），使发布副本与 `.agents` 部署副本**内容一致**，消除历史遗留的脱敏差异维护成本

## [v3.4] - 2026-08-29

### Added
- SKILL.md 新增「故障快速排查（30 秒版）」：端口监听 / WSL 宿主机 IP / apiKey / maxInputTokens / reasoningEffort / silent-fail 六步清单
- SKILL.md 新增「技能版本维护 SOP」（十）：版本行 → CHANGELOG → 同步 .agents → one-line reason → 副本验证
- `scripts/detect.py` / `detect.ps1` 新增 [7/7] 服务连接诊断：端口监听探测 + `/health`（PS）+ WSL 环境提示（宿主机 IP 而非 localhost）
- `update_launchers.py` `make_auto` qwen3.8 生成后新增 REQUIRED_ARGS 断言（缺 `--reasoning-budget 8192`/`--reasoning-format deepseek`/`--chat-template-kwargs`/`--reasoning-preserve`/`--min-p 0.0` 即拒绝写入）——把静默失败变显式失败

## [v3.3] - 2026-08-29

### Added
- 新增 Qwen3.8-27B 专项（Step 3F）：qwen35 架构特性（65 层/内置 MTP/SWA interval=4）、官方采样表、16GB 部署速度矩阵、**128K 手工 ngl 提速方法论**（IQ3_S 128K：fit 16.0 → ngl58+batch512 18.8，+17.5%）、长会话稳定参数（`--reasoning-budget` + `--reasoning-format deepseek`）
- 新增 `references/20260829-session-experience.md`：Qwen3.8 全流程实测、长会话验证 SOP（多轮/跑满/埋针）、更新器工程缺陷 3 连修复、Router fit 多模型限制、WSL/VS Code `ECONNREFUSED` 排查 SOP、参数名与转义坑速查
- Step 4 扩充 VS Code customendpoint 接入要点（maxInputTokens ≤ `-c`、settings 键名、reasoningEffort 取值、apiKey 匹配）与 WSL 宿主机 IP 获取

### Fixed
- 更新器工程缺陷：`make_auto` 生成 group 缺 `"t"` 键致第二次同步 `KeyError: 't'`；多行参数串 CRLF 混合致孤立 LF；字符串 replace 注入参数因行序假设错误静默失败
- 参数名勘误：`--repeat-penalty`（非 `--repetition-penalty`）、`--load-mode mmap`（`--no-mmap` 废弃）、`--cache-reuse` 当前 context 不支持、`--defrag-thold` 废弃
- Troubleshooting 新增 6 行：参数名/转义/思考型 max_tokens/VS Code 连接/废弃参数

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

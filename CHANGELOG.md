# Changelog

llama.cpp Windows 多模型部署技能（llama-cpp-windows-deployment-skill）版本变更记录。

## [未发布 — 目标 v3.7.0] - 2026-09-17

> 项目自洽化改造进行中，方案与进度见 [`docs/merge-plan.md`](./docs/merge-plan.md)。
> 本节只记录**已经落到 `main`** 的内容。**尚未打标签** —— v3.7.0 将在 Phase F 统一发布。

### Added — `scripts/mtp_graft.py`：通用 MTP head 嫁接套件
首个**不依赖任何项目外目录**的嫁接工具。此前手册引用的 `<llama-cpp-dir>\plan\_mtp_graft.py`
`import update_launchers`，脱离 `<llama-cpp-dir>\launcher\` 即 `ModuleNotFoundError`，
而手册却写着「纯标准库，可独立运行」—— **该说法已删除并纠正**。

- **内联**两个原本外部的符号：`GGML_TYPE_BYTES`（34 项，逐条对上游 enum 核过）与
  `read_gguf_tensors` 的尺寸交叉校验（`size_ok = delta < 1%`）
- **头识别改为 block index 区间** `[block_count - nextn, block_count)`。
  按名字过滤会漏掉 `bailingmoe3`（它的头张量用普通后缀 `blk.%d.layer_out_norm`）
- **支持多块头**（`mimo2`=3、`step35`），并在断言单块的架构上**拒绝**写多块
- **新 `--audit`**：列出目标模型全部 per-layer 数组 KV（`compress_ratios` / `shared_kv_layers` /
  `recurrent_layers` / `deepstack_layers` / `layer_types`，外加「长度恰好等于 `block_count`」兜底）。
  加块会与这些数组失配，此前**完全无人审计**
- **12 条闸门**（8 结构 + 4 架构建议）：`granite-switch`（复用 `n_layer_nextn` 作 router）与
  `gemma4-assistant`（头在 `blk.N.*` 之外）直接拒绝；头会被加载但永不执行的架构只警告
- `--preset qwen36-35b-a3b` 一键复现本次嫁接；`--go` **拒绝覆盖已存在的输出**
- **新增 `scripts/tests/test_mtp_graft.py`**：17 例（1 正向 + 16 负向），合成夹具亚秒跑完，不需真模型

### Verified — 迁移等价性（逐字节）
用新工具对 2026-09-13 那次嫁接的**同一对模型**重跑，产物与原工具 **SHA-256 完全相同**：
`5AF97A49D3CC86866CC9C101C72A584803A6FE3E7DD8C5196DEEE15F7072D272`（22.31 GiB）。
即改写是**保真等价**，不是「看起来能用」。

### Added — `scripts/mtp-graft-package/`：本次嫁接的完整过程物料
复盘（含两个 bug 与 `nextn_predict_layers` 的发现过程）、对比分析表、会话导航图、
四个脚本原始副本、A/B 实测数据、配置链路摘录。**已全量脱敏**（凭据 → 环境变量、
本机路径 → 占位符、换行 → CRLF），原始转录 gitignore 仅留本机。

### Added — `docs/`：项目文档（非技能载荷）
`merge-plan.md`（边界、10 条决策、10 条硬伤、阶段与门禁）、`README.md`、Stage 1 门禁报告。

### Fixed — 手册与索引
- `guides/mtp-head-grafting.md` §7 **重写**：指向 `scripts/mtp_graft.py`，列全 12 条闸门、
  block 区间识别、per-layer KV 陷阱，并**保留并纠正**原「可独立运行」的错误说法
- `INDEX.md` 新增 **§5.6 项目内工具**（唯一不指向 `<llama-cpp-dir>` 的路径组）；
  §5.5 移除 `plan/_mtp_graft.py`；**§6 登记其为已废弃**

## [v3.6.0] - 2026-09-13

### Changed — `references/` 目录重整（本次主要变更）
- **重组为三类子目录**（用 `git mv` 迁移，**历史完整保留**，全部以 `R` 状态记录）：
  - `references/sessions/` —— 6 份按日期的会话经验（**历史快照，正文不回改**）
  - `references/guides/` —— 2 份面向操作的专题手册（`mtp-head-grafting.md`、`20260913-moe-offload-community-research.md`）
  - `references/assets/` —— 7 个可复用资源（5 个启动器模板 + 2 个配置模板）
- **新增 `references/INDEX.md`**（快速索引）：三类文档的区别与"什么时候读"、**§5 权威代码路径表**（逐条实测校验）、
  §6 已废弃/已归档路径、§7 维护规范（新文档怎么写、路径改动必须同步哪两张表、双副本怎么同步）
- `SKILL.md` 头部新增 references 索引导航

### Fixed — 代码文件索引路径
- `guides/mtp-head-grafting.md`：`plan/_mtp_graft.py` → 明确为 `<llama-cpp-dir>\plan\_mtp_graft.py`，
  并声明本目录不含副本
- `guides/20260913-moe-offload-community-research.md`：`update_launchers.py` → `launcher/update_launchers.py`
- 6 份会话文档统一加**索引横幅**，指向 `INDEX.md` §5，避免读者照抄历史路径
- `sessions/20260913`：改为指向已移动手册的**相对链接**
- `SKILL.md` 11 条 + `README.md` 16 条链接全部更新（含显示文本）；**两副本链接完整性实测 `broken_links=0`**

### Fixed — 技能副本陈旧（`.agents` 与发布副本）
- `references/assets/model-profiles.json` 从仓库同步：**16 → 18 个 profile**，补入 **3 个结构化 `moe` 块**
  （`gemma4-26b-a4b-qat` / `gemma4-26b-a4b` / `qwen36-35b-a3b`）；保留既有 `_disclaimer`；
  无 BOM、CRLF 一致、无敏感路径
- 顺带修正 `.agents` 副本中 `gemma4-menu-scripts.bat` / `qwen-scripts.bat` 的陈旧版本

### Notes
- **内容零丢失已证明**：用排序多重集比对迁移前后各文档，5 份完全一致，
  另 3 份的差异**仅是本次有意修正的 3 条路径行**
- 双副本仍只有 3 份历史文档存在**有意的脱敏差异**（`20260803` / `20260805` / `20260829`），
  未同步新增差异

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

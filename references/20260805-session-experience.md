# 2026-08-05 实战经验沉淀（推理控制防死循环 + GBK 安全编辑）

> 来源：2026-08-04/05 会话「Gemma 脚本推理防死循环优化 + Hermes 实测调优 + 菜单标注规范 + 预算 8192」
> 硬件：RTX 5060 Ti 16GB + Intel U7 270K + 48GB DDR5
> 本文件为 SKILL.md 的详细数据附录，SKILL.md 保留结论。

## 一、推理控制（防思考死循环 / 工具调用死循环）⭐ 核心

### 参数清单（llama.cpp b10xxx 实测）

| 参数 | 取值 | 说明 |
| ------ | ------ | ------ ----- |
| `-rea, --reasoning` | `on`/`off`/`auto` | 思考开关；`auto`=按模板检测 |
| `--reasoning-budget N` | `-1`不限 / `0`立即结束 / `N>0`预算 | 思考 token **硬上限**，耗尽强制结束 |
| `--reasoning-format` | `none`/`deepseek`/`deepseek-legacy` | 思考内容如何返回 |
| `--reasoning-budget-message` | 字符串 | 预算耗尽时注入的消息（强制收敛/换思路钩子） |
| `--reasoning-preserve` | on/off | 保留思考轨迹到完整历史（模板支持时） |

### 🔥 关键实测发现：`--reasoning-format none` 会禁用预算

- `none` → llama.cpp 不解析思考 → **`--reasoning-budget` 失效** → 思考无上限（实测 3000+ token 仍在思考 = 正是要防的死循环）
- `deepseek` → 思考解析进 `message.reasoning_content` → **预算生效**（实测 budget=64 时思考被硬截断于 ~64 token，注入收敛消息后转入回答，`finish_reason: stop`）
- Hermes / VS Code 等 OpenAI 兼容客户端原生支持 `reasoning_content`，无兼容问题

### 推荐配置模板

**主模型（复杂任务/Agent，如 Gemma 12B/26B）**：

```bat
set "REASONING=on"
set "REASONING_BUDGET=8192"        REM 硬上限；Hermes 复杂任务留足空间仍防失控
set "REASONING_FORMAT=deepseek"    REM 关键：none 会让 budget 失效
set "REASONING_BUDGET_MSG=思考预算已用尽，请立即停止思考，基于已有信息直接给出最终答案。"
set "REASONING_ARG=--reasoning %REASONING% --reasoning-budget %REASONING_BUDGET% --reasoning-budget-message "%REASONING_BUDGET_MSG%" --reasoning-format %REASONING_FORMAT%"
```

**工具模型（思考关，防工具循环，如 Qwen3.5-2B CPU）**：

```bat
set "REASONING=off"
set "REASONING_ARG=--reasoning %REASONING%"
```

### 客户端每请求覆盖（无需改服务端）

| 字段 | 用途 |
| ------ | ------ |
| `reasoning_budget_tokens`（别名 `thinking_budget_tokens`） | 升级预算（如 8192） |
| `reasoning_budget_message` | 覆盖耗尽注入消息 |
| `reasoning_control: true` | 立即强制结束思考（工具信号/收敛时） |
| `chat_template_kwargs: {"enable_thinking": "false"}` | 重试降级：关思考 |
| `reasoning_effort: "none"` | 同上 |

### 验证方法

1. `/props` 的 `reasoning_format` 字段**不可靠**（实测可能显示 none 但 deepseek 实际生效），以 API 行为为准
2. 决定性测试：发请求看 `message.reasoning_content` 是否出现、`content` 是否为空
3. 预算测试：请求体带 `reasoning_budget_tokens: 64`，看思考是否被硬截断 + 收敛消息是否注入 + `finish_reason: stop`

### Gemma 4 思考标记

- Gemma 4（含 E4B-QAT）思考标记是 **`<|channel>thought`**，不是 `<think>`；`--reasoning-format none` 时思考留在 content

## 二、BAT 脚本 GBK 安全编辑（本会话 2 次踩坑）

### 🔥 核心教训：编辑工具会破坏 GBK 文件

- **VS Code Copilot 编辑工具会把 GBK .bat 读作 UTF-8 → 写回 UTF-8 + U+FFFD 替换字符（不可逆）**，中文全毁（本会话发生 2 次）
- 即使只改一行 ASCII，工具也会重写整个文件 → 整个文件中文被破坏（778/954 个 U+FFFD）

### 正确做法：PowerShell 显式 GBK 读写

```powershell
$p="C:\llama.cpp\start-Gemma4-Launcher.bat"
$s=[System.IO.File]::ReadAllText($p,[System.Text.Encoding]::GetEncoding(936))
# ... 字符串替换（改行/菜单/配置）...
$strict=[System.Text.Encoding]::GetEncoding(936,[System.Text.EncoderFallback]::ExceptionFallback,[System.Text.DecoderFallback]::ExceptionFallback)
[System.IO.File]::WriteAllBytes($p,$strict.GetBytes($s))
```

- 严格编码器（ExceptionFallback）可在写入前发现无法用 GBK 表示的字符
- **emoji（❌⚠️→ 等）GBK 无法表示**，需先替换为 `[X]`/`[!]`/`->`
- cmd 中 echo 的 `%` 会被吞（如 `+35%` 显示为 `+35`）→ 写 `%%`
- GBK 脚本不要 `chcp 65001`（会让中文乱码/闪退）；中文 Windows 默认代码页 936 即可

### 恢复被破坏的文件（VS Code 本地历史）

- 位置：`%APPDATA%\Code\User\History\<hash>\`（每文件一目录）
- `entries.json` 记录每份快照对应的原始文件（`resource` 字段），快照按时间排序
- 找"0 个 U+FFFD"的快照即可恢复；本会话某 .bat 文件就是编辑后完好的 UTF-8 版
- 恢复后仍需按 GBK 转码 + 去 emoji + 删 chcp 65001

## 三、菜单标注规范（2026-08-05 定稿）

- **B 参数前置**：`12B Q5 ...`（12B 放 Q5/Q8 前）
- **量化标注**：`[Q5_K_M]` / `[Q8_K_XL]` / `[Q4_K_XL]`（QAT-UD）/ `[Q4_K_M]`（破限版）——防新增量化版本时混淆
- **破限版**：Uncensored 用 `破限版` 标注
- **速度标注**：实测值 `[113t/s]` / `[160t/s]` / `[78t/s]`
- **已知问题**：15 项 MTP 入口 `[!]已知失败`（llama.cpp bug：invalid vector subscript）

## 四、启动脚本变量工程化

- 集中配置 + 变量拼接参数：`TOOLS_ARG` / `REASONING_ARG` / `REPEAT_ARG`，一处可调、全部启动项生效
- `REPEAT_PENALTY=1.0`（默认关），>1.0 时生成 `--repeat-penalty N`（token 级防思考重复的粗粒度兜底；llama.cpp 无 Jaccard 语义去重）
- 菜单项用 `goto` 复用启动例程（16/17 Agent 项 → 5/8），标注改动只需改一处

## 五、实测验证记录（2026-08-04/05）

- Gemma 脚本 15 处启动行全部带 `%REASONING_ARG%`，菜单 16/17 自动继承
- E4B 真实启动：`reasoning_content` 出现（578/415 chars）、content 为空 → deepseek 生效
- budget=64 每请求覆盖：思考硬截断 + 中文收敛消息注入 + `finish_reason: stop` ✅
- CPU 工具模型 `--reasoning off`：响应无 reasoning_content、工具调用正常（`finish_reason: tool_calls`）
- 菜单解析：GBK 脚本 exit 0，中文显示正常，`+35%%` → `+35%` 正确渲染
- `/props` 显示 `reasoning_format=none` 但 API 实测 deepseek 生效 → **不要信 /props 该字段**

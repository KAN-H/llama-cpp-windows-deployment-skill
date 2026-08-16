# 2026-08-06 实战经验沉淀（26B+MTP 启动失败 invalid vector subscript 排查与修复）

> 来源：2026-08-06 会话「26B-QAT + MTP 菜单 8/17 启动报 invalid vector subscript 排查修复」
> 硬件：RTX 5060 Ti 16GB + Intel U7 270K + 48GB DDR5 | llama.cpp b10158 (f87067841)
> 结论一句话：**26B MTP draft 必须用官方 Q8_0；第三方 Q4_0 在 llama-server 的 draft 加载路径必崩，但 llama-cli 单独加载正常（极具迷惑性）。**

## 一、现象
- 菜单 8（26B-QAT + MTP 64K）与 17（+Agent 工具）启动即崩，日志末尾：
  ```
  I common_speculative_init_result: loading draft model '...\gemma4_mtp\mtp-gemma-4-26B-A4B-it-Q4_0.gguf'
  W load: control-looking token: 50 '<|tool_response>' ...
  E llama_model_load: error loading model: invalid vector subscript
  E common_speculative_init_result: failed to load draft model
  ```
- 主模型（26B-QAT UD）加载正常，只有无害警告；崩溃点在 **draft 模型加载**。
- 日志开头的 `Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)` 是**预期警告**，非根因。

## 二、根因（三层证据链）
1. **官方 MTP 从未有 Q4_0**：查 `unsloth/gemma-4-26B-A4B-it-GGUF` 全部历史提交（6/5→7/17），`MTP/` 目录始终只有 **Q8_0 / BF16 / F16**，官方推荐 Q8_0（root 镜像 `mtp-gemma-4-26B-A4B-it.gguf`）。命名演变：`gemma-4-26B-A4B-it-{Q8_0,BF16,F16}-MTP.gguf`（6/9-7/9）→ `mtp-gemma-4-26B-A4B-it-{Q8_0,BF16,F16}.gguf`（7/9 起）。
2. **本地文件被替换/误标**：`Get-Item` 显示官方 Q8_0 应为 ~440MB，而本机 `mtp-gemma-4-26B-A4B-it.gguf` 只有 240MB（实为 Q4_0）；两个 Q4_0 文件均为 240MB 第三方版。
3. **llama-cli 能加载 ≠ server 能加载**：对 4 个候选文件逐个 `llama-cli -m <f> -c 256 -p hi -n 8`（含 `-ngl 24` 复现 server 参数）→ **全部正常解析**（只有预期 `ctx_other` 提示）；但 llama-server 加载 Q4_0 时必报 `invalid vector subscript`。**必须用真实 server 命令做加载测试，llama-cli 测试会误判**。

## 三、修复
1. `start-Gemma4-Launcher.bat`：`DRAFT_TYPE=q4`→`q8`；`D26_Q4`、`D26_UNCENS` 全部改指官方 `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`（GBK 936 安全编辑，见下）。
2. **验证方式**：用菜单 8 同款参数但换 Q8_0 draft 跑 `llama-server`，`/health` 返回 `200 {"status":"ok"}` ✅。
3. 清理：3 个 240MB 的 Q4_0/误标文件移入 `gemma4_mtp\_unused_thirdparty_Q4_0\`（防止 `generate_ini.bat` 再扫进 Router 配置）；同步删除 `models-config.ini` 中引用它们的 26B MTP 段。

## 四、可复用经验
- **Gemma 4 MTP draft 来源判定**：官方仅 `unsloth/gemma-4-26B-A4B-it-GGUF`（及 12B 对应仓库）的 `MTP/`；文件名含 `Q4_0` 的一律非官方。`mtp-gemma-4-26B-A4B-it.gguf` 官方应为 ~440MB（Q8_0），若只有 ~240MB 说明被 Q4_0 覆盖。
- **诊断命令**：
  ```powershell
  Get-Item <draft>.gguf | Select Name,Length,LastWriteTime   # 440MB=官方Q8_0, 240MB=Q4_0
  llama-cli -m <draft> -c 256 -p hi -n 8                      # 只能验证解析，不能验证 server 可用
  # 决定性测试：llama-server + --model-draft <draft> ... ; 查 /health
  ```
- **GBK 安全编辑**：启动脚本含中文，必须 PowerShell 936 读写（`[Text.Encoding]::GetEncoding(936)` + `EncoderFallback::ExceptionFallback` 写回），编辑工具会把 GBK 转 UTF-8 损坏中文（U+FFFD），已多次踩坑。
- **旧经验修正**：菜单 15 曾标注"26B Uncensored + MTP 因 llama.cpp bug 失败（invalid vector subscript）"——**并非 llama.cpp bug**，而是用了第三方 Q4_0/误标 draft。换官方 Q8_0 后应可正常。

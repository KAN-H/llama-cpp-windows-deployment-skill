# Qwen3.8-27B-GSQ-RCO 调研 · 实测 · 落地 报告

**日期**：2026-10-12 ｜ **执行**：GitHub Copilot（llama-cpp-windows-deployment 技能工作流）
**模型**：`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp`（ISTA-DASLab / ModelScope，11.28 GiB + MTP 头 332 MiB + mmproj 0.87 GiB）
**硬件**：RTX 5060 Ti 16GB ｜ 48GB DDR5 ｜ Intel U7 270K ｜ llama.cpp **b10713**

---

## 一、结论（终选配置）

> **终选依据** = 全部实测数据的支配关系（详见 §二/§三）：b256 胜出、MTP 必开、全层 GPU 硬约束、视觉档必须无 MTP。双档同时落地到启动器菜单，用户按场景选择。
>
> **落地状态（2026-10-12 已完成）**：`launcher-models.json` + `preset-overrides.json` 已写入 → `start-Qwen-Launcher.bat`（6 项菜单：第 5 项 Agent / 第 6 项 Vision）+ `models-config.ini` 已生成 → 回归 `--strict` **全绿**（FND-070/071/021 全部修复验证）。

**双档设计**（启动器菜单）：

| 档 | 菜单 | 配置要点 | 45K 上下文实测 |
|---|---|---|---|
| 纯文本 Agent | Qwen 启动器 | ctx 141312 / 全层 GPU / KV q4_0 / **MTP b256** | decode **28.10 t/s** / prefill 690 t/s / 余量 **555 MiB** |
| 视觉 mmproj | Qwen 启动器 | 同 + mmproj、**无 MTP** | decode **20.66 t/s** / prefill 744 t/s / 余量 **1,009 MiB** |

---

## 二、实测数据总表（45000 token 长 prompt，除注明外）

| 配置 | decode t/s | prefill t/s | VRAM/余量 MiB | 备注 |
|---|---|---|---|---|
| q8_0 KV / fit | 14.35 | — | 14,326 / 1,985 | 无 MTP |
| K8V4 KV / fit | 22.01 | — | 14,372 / 1,939 | 无 MTP |
| q4_0 KV / fit（≡ngl64） | 26.3 | — | 14,390 / 1,921 | 无 MTP |
| q4_0 / ngl64 / b2048 | 26.2（37tok 短） | — | 15,264 / 1,047 | b2048 无收益 |
| **q4_0 / ngl64 / MTP / b256** | **28.10** | 690 | **15,756 / 555** | **性能档（新基线）** |
| q4_0 / ngl64 / MTP / b512 / p-min 0.75 | 26.07 | 716 | 15,948 / 363 | acc 0.865 但 -7.2% |
| q4_0 / ngl64 / MTP / b512 | 27.45 | 717 | 15,950 / 361 | 原基线（b256 胜出） |
| q4_0 / ngl61 / MTP | 14.39 | 691 | 15,476 / 835 | CPU 断崖 |
| q4_0 / ngl58 / MTP | 11.46 | 664 | 14,932 / 1,379 | CPU 断崖 |
| q4_0 / n-cpu-ffn 8 / MTP | 19.19 | 695 | 15,374 / 937 | 劣质杠杆 |
| q4_0 / ngl64 / **无 MTP** | 20.87 | 745 | 14,414 / 1,897 | 稳定档 |
| 90K 检索（no-think） | 16.4* | 586 | — | acc=1.0 / mean len 4.0 |
| 4K 冒烟 无/有 MTP | 28.4 / **48.0** | — | +934 MiB | acc 0.667 |
| 视觉 ngl57+mmproj+MTP | 11.06 | 655 | 15,666 / 645 | 双杀组合（弃） |
| **视觉 ngl64+mmproj 无MTP** | **20.66** | 744 | 15,302 / **1,009** | **视觉档（成立！）** |

\* 该点在 ngl61（断崖配置）下测得，非终选配置。

---

## 三、关键发现（Top 7）

1. **★ CPU 参与断崖**：dense+MTP 模型下，任意层卸载到 CPU 都是断崖——3 层 -48%、6 层 -58%、n-cpu-ffn 8 -30%。**与 MoE 模型经验完全相反**（MoE 的 CPU 专家路径是批量搬运，可卸几十层）。⇒ 本模型必须全层 GPU；省显存只能靠 KV 类型与 ctx。
2. **MTP 实测**：短上下文 +69%（28.4→48.0，acc 0.667）；45K +31.5%（20.87→27.45，acc 0.507）；显存代价 **+1,536 MiB @141K**（是头权重 332 MiB 的 4.6 倍）；结构化短输出（no-think 列清单）acc=1.0/mean len 4.0（完美利用）。**p-min 0.75 反直觉**：acceptance 升至 0.865 (+64%)、mean len 2.81，**但 decode 反降 7.2%**（26.07）——每验证步成本增加 > 接受量收益 ⇒ 弃用（保留默认 p-min）。
3. **KV 阶梯**：q8_0→K8V4→q4_0 = 14.35→22.01→26.3 t/s（无 MTP）——fit 会把 KV 省下的显存自动换成更多 GPU 层；q4_0 时 fit 已给到全 64 层（与手工 -ngl 64 完全同值）。
4. **dense 不吃 batch 杠杆**：b512→b2048 的 prefill 无变化（745→738）、decode 无变化，但多占 938 MiB（MoE 的 `-ub 2048` +95% 经验不适用）。**反向发现：b256/u256 比 b512 省 194 MiB 且不掉速（28.10 vs 27.45 t/s）**——小 ubatch 让 MTP 草稿验证的临时缓冲更小，**终选采用 b256**。
5. **`--n-cpu-ffn` 是劣质杠杆**：省 1 MiB 显存的 decode 代价约为 ngl 整层卸载的 5 倍——淘汰。
6. **GSQ-RCO 量化兼容良好**：12 种张量类型混配（IQ3_S/IQ4_XS/IQ3_XXS/Q4_K/IQ2_S/Q6_K/…），b10713 全部正常加载运行；大小交叉校验 0.000%。
7. **视觉档成立（推翻早期误判）**：全层+mmproj+无 MTP = **20.66 t/s / 余量 1,009 MiB**（对比无视觉 20.87——视觉塔几乎零 decode 代价，仅 +888 MiB 显存）。早期 f2 的 11 t/s 失败是 **ngl57 断崖 × MTP 双杀**所致，与视觉能力无关。

---

## 四、上游风险（调研结论）

| Issue | 风险 | 本机处理 |
|---|---|---|
| [#30302](https://github.com/ggml-org/llama.cpp/issues/30302) | 同款 GSQ-RCO-MTP：Win+CUDA 开 draft-mtp **首推理蓝屏**（5090M+616/617 驱动） | b10713 已分级冒烟 + 10+ 次实跑**未复现**；新 build/新驱动用户仍应先冒烟 |
| [#30357](https://github.com/ggml-org/llama.cpp/issues/30357) | Qwen3.5+ 空白 round-trip 破坏 → hybrid 模型全量 re-prefill | 90K 实测 `reprefill=0`；长 agent 多轮持续观察日志 |
| [#30307](https://github.com/ggml-org/llama.cpp/issues/30307) | qwen4exp+MTP+图像 abort（b11548+） | b10713 不受影响；视觉+MTP 组合在本机弃用 |
| [#30368](https://github.com/ggml-org/llama.cpp/issues/30368) | draft-mtp + kv-unified + np>1 崩溃 | -np 1 不触发 |

---

## 五、落地清单（已写入/待执行）

**修复的生成器缺陷**（`launcher/update_launchers.py`）：
- **FND-070**：`make_auto` 从不生成 `_MM` 变量定义 → 所有带 mmproj 的自动菜单项（Q5/Q6）在 `:check_file` 空路径处死亡。修复：vars 循环补 `vmm` 定义 + 回归断言 **E6**。
- **FND-071**：qwen38 分支 `mtp_seg` 携带 `-fit off` 覆盖前面的 `--fit on`（后者生效）→ 菜单项实际以 fit off 运行（潜在纯 CPU）。修复：删除 + 回归断言 **E7**。
- 其余：qwen38 `--load-mode mmap→none`（与实测一致）、模板 batch/ub 统一 **256**（g3 实测最优）。
- **FND-021（顺带修复）**：`launcher/models-config.bat` 的 `--api-key` 尾部与共享 key 不一致（`-abc` vs `-001`）→ 已统一为共享 key。

**数据文件**（`plan/_q6_apply.py` 一次性写入，自动备份）：
- `launcher/launcher-models.json`：region 补 `_MM`（Q6+Q5 两处）、Q6 body 重写、新增视觉档 entry。
- `launcher/preset-overrides.json`：Q6 实测参数 + tuning 记录。

**生成物**（跑 `update_launchers.py --yes` 后）：`start-Qwen-Launcher.bat`（Q6 双档菜单）、`models-config.ini`。

**技能沉淀**：`references/guides/qwen38-gsqrco-27b-tuning.md` + INDEX 登记。

---

## 六、使用说明 & 密钥（重要）

- **菜单**：`llama-hub.bat → 4) Qwen 启动器 →` Q6 两个新档（端口 8084）。
- **API key：无需手动修改任何密钥**。全部沿用现有共享 key（与其它启动器一致）。
  - 如需更换：改 `launcher\launcher-models.json` 中 qwen 的 `head` 里 `set "API_KEY=..."`（**1 处**），然后跑 `update-launchers.bat --yes` 重新生成。改动会同时影响 Qwen 启动器全部菜单项。
- Hermes/VS Code 接入：`http://<宿主机IP>:8084/v1`，key 同上。

---

## 七、遗留事项与建议

1. **余量 555 MiB（性能档 b256）的边界**：纯文本 API（Hermes）稳态下实测稳定；若同时开浏览器/游戏抢显存，可能触发失败或需要降档（建议出问题就用"稳定档"参数：去掉 `--spec-type draft-mtp`，余量回到 ~1,900）。
2. **单发超大 prompt 的性能悬崖**：**135K** 单发在性能档（全层+MTP，余 361）下 >10 分钟未完成 prefill（无报错）；**90K** 在同样的全层+MTP 配置下也观察到异常缓慢（f1b 实验 >5 分钟未完成，已终止）；而 **90K@ngl61（余 835）与 45K@全层 均正常**。⇒ **性能档（全层+MTP）的舒适单发区间保守取 ≤45K**；**会话累计 KV 到 138K 不受此限**（逐轮增长；90K 检索 acc=1.0 / reprefill=0）。需要单发大 prompt 时：改用视觉档配置（无 MTP、余 1,009+）或临时去掉 `--spec-type draft-mtp`（稳定档余 1,897）。
3. 若遇到 #30302 类蓝屏（未来升级 llama.cpp 后）：先用 Vulkan 后端或退回 b10713。
4. Q5（35B）的 `_MM` 变量本次一并修复（同一 bug），其菜单项此前同样不可用。
5. **n-max=4 未完成（遗留研究）**：全层+MTP+b512+`--spec-draft-n-max 4` 在 45K prompt 下 >8 分钟未完成（参数映射已核实合法；疑似 MTP 在 n-max>3 时的退化路径）——**维持 n-max 3**。若上游修复可再试。
6. **多轮压力测试未完成（发布前）**：3 轮会话测试（45K 素材 + 思考模式 + MTP，全层）首轮 >4 分钟未响应（与单发 90K/135K 的"慢路径"同型），已终止。**单轮 45K（74s 完整跑完、reprefill=0）与 90K 检索（acc=1.0）不受影响**；长 agent 多轮场景建议首次使用小步验证（观察每轮 prefill 时间与日志 `forcing full prompt re-processing`）。

---

## 八、Router（8082）集成（2026-10-12 完成）

- **`[Qwen3.8-27B-138K-Agent]` 已进 `models-config.ini`**：q4_0 KV / b256 / ctx 141312 / `spec-type = draft-mtp` / **`no_mmproj`**（生成器新开关，纯文本节不载视觉塔）。
- **踩坑与修复（重要）**：官方 preset 优先级为 **命令行 > 模型节 > `[*]` 全局**——`models-config.bat` 命令行的 `--cache-type-k/v q8_0` **覆盖**了模型节的 q4_0 → 141K 全层爆显存 → 跌至 **2.9-5.2 t/s**（fallback 慢路径）。**删除命令行那两行 KV** 后 → **prompt 136 t/s / decode 41.5 t/s**（短上下文实测，MTP 生效）✓。
- 其它模型仍从 `[*]` 继承 q8_0（行为不变）；该文件是**手工维护**（非生成物），修改须 GBK 编码。
- 新增生成器能力：`no_mmproj: true`（preset-overrides 模型条目顶层键）——`gen_ini_v2()` 两处 mmproj 输出均跳过。

---

*原始数据：`plan/_q6_results.json` ｜ 探针：`plan/_q6_probe.py` ｜ 元数据：`plan/_q6_metadata.json`*

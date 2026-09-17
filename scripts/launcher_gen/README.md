# launcher_gen — 启动器生成器

把 `launcher-models.json`（注册表）＋ `preset-overrides.json`（参数）
渲染成 3 个菜单启动器 `.bat` 和 `models-config.ini`。

```
update_launchers.py    生成器（3,892 行）：扫描 → 渲染 → 校验
llama_hub.py           交互式门面：菜单 / --guide / --diagnose / --write-docs
update-launchers.bat   非交互包装（转发 %* 给 update_launchers.py）
llama-hub.bat          门面包装
_sanitize_hub.py       一次性：清除 llama_hub.py 里 3 处硬编码 key（已完成，保留作凭证）
```

纯标准库。**刻意不含任何本机路径** —— 见下面的两个环境变量。

---

## 1. 两个环境变量

| 变量 | 含义 | 默认 |
|---|---|---|
| `LAUNCHER_DIR` | **数据目录**：`launcher-models.json`、`preset-overrides.json`、`model-profiles.json`、`backup/`、`docs/` | 本文件所在目录 |
| `CHAT_DIR` | **模型目录**（扫描对象） | **无** —— 不设就报错退出 2 |

```powershell
$env:LAUNCHER_DIR = '<launcher-dir>'
$env:CHAT_DIR     = '<models-dir>'
python update_launchers.py --check
```

> **为什么 `CHAT_DIR` 没有默认值**：模型目录是机器属性，不是工具属性。
> 旧版把**模型目录**写死在源码里，于是工具只在这一台机器上说得通。
> 现在缺参时它会明确告诉你，而不是猜一个路径。
>
> **`LAUNCHER_DIR` 的默认值是"本文件所在目录"**，所以把工具放在 `launcher\` 里
> 原地运行**行为完全不变** —— 迁移没有破坏任何现有安装。

---

## 2. 数据文件各自是什么

| 文件 | 角色 | 缺失时 |
|---|---|---|
| `launcher-models.json` | **注册表**：3 个启动器的模板 + 每个模型的菜单条目 | `--extract` 重建；其余命令报错退出 |
| `preset-overrides.json` | **唯一参数事实源**：每个模型的调参与实测记录 | 取内置骨架 |
| `model-profiles.json` | 参数知识库（含 `moe` 块） | 返回 `[]`，走通用默认 |
| `draft-blacklist.json` / `draft-health.json` | MTP 草稿实测结论与黑名单 | 视为空 |
| `.llama-server-keys.json` | ⚠️ **名字有误导** —— 它是 **332 个 CLI 参数名的白名单**，不是密钥库 | 跳过参数名校验 |

---

## 3. ★ 冷启动（全新机器）

**实测评注：`--extract` 是从"既有的启动器 `.bat`"反推注册表的**，
所以它要求机器上**至少已经有一个菜单形态的 `.bat`**。全新机器没有，
于是这条引导链是**循环**的 —— 这正是本项目要消除的那类问题。

### 3.1 可用的引导链（已实测走通）

技能自带的 `references/assets/` 里有现成模板，**其中"菜单形态"的那些可以直接当种子**：

```powershell
# 1. 建目录，放入两个脚本
mkdir <launcher-dir>; cd <launcher-dir>
copy <skill>\scripts\launcher_gen\*.py .

# 2. 用技能模板当种子 —— 注意文件名必须匹配生成器的 SPECS
copy <skill>\references\assets\gemma4-menu-scripts.bat         start-Gemma4-Launcher.bat
copy <skill>\references\assets\start-CPU-Toolcall-Launcher.bat start-CPU-Toolcall-Launcher.bat

# 3. 从种子反推注册表
$env:LAUNCHER_DIR = $PWD; $env:CHAT_DIR = '<models-dir>'
python update_launchers.py --extract
#   [OK] extracted start-Gemma4-Launcher.bat  (9)
#   [OK] extracted start-CPU-Toolcall-Launcher.bat  (8)
#   [OK] registry written: launcher-models.json

# 4. 用真实磁盘内容校正参数（会写 preset-overrides.json）
python update_launchers.py --derive-params

# 5. 干跑，看它打算改什么
python update_launchers.py --check

# 6. 应用
python update_launchers.py --yes
```

> ⚠️ 第 2 步的**文件名必须精确匹配**，它们在源码里是常量：
> `start-Gemma4-Launcher.bat` / `start-Qwen-Launcher.bat` / `start-CPU-Toolcall-Launcher.bat`。
> 名字不对就是"文件不存在"。

### 3.2 ⚠️ 不是每个技能模板都能当种子

| 模板 | 形态 | 可作种子 |
|---|---|---|
| `gemma4-menu-scripts.bat` | 有 `goto :menu`、`:menu`、`set /p c=`、`:RUN_*` | ✅ 实测提取 9 条 |
| `start-CPU-Toolcall-Launcher.bat` | 同上 | ✅ 实测提取 8 条 |
| **`qwen-scripts.bat`** | **只有 `goto :eof`，没有菜单循环** —— 它是"顺序跑三套配置"的脚本 | ❌ **`no 'goto :menu' found`** |

提取器需要三件东西，缺一即拒：

1. **至少一个** `set "VAR=%CHAT%\...\xxx.gguf"` 形式的模型路径变量（起始锚点）
2. **`goto :menu`**（菜单区结束位置）
3. 菜单区的 `set /p c=` 与 `if "%c%"=="N" goto RUN_x` 结构

`qwen-scripts.bat` 缺第 2 条。**要给它做种子，得先把它改造成菜单形态**，
或干脆手工写一个最小的菜单 `.bat`（照抄 `gemma4-menu-scripts.bat` 的骨架即可）。

### 3.3 注册表结构（给手工编写者）

```jsonc
{
  "version": 1,
  "chat_dir": "<models-dir>",
  "mtp_dir_name": "gemma4_mtp",
  "family": { "gemma": "gemma4", "qwen": "qwen", "lfm": "cpu", "default": "qwen" },
  "launchers": {
    "gemma4": {
      "script": "start-Gemma4-Launcher.bat",
      "encoding": "gbk",              // cpu 用 "ascii"；gbk 是 ASCII 的超集
      "head": "...",                  // 第一个模型变量之前的全部内容（\r\n 连接）
      "menu_open": "...", "menu_lines": "...", "item_tpl": "...",
      "dispatch_tpl": "...", "prompt_tpl": "...",
      "exit_line_tpl": "...", "exit_dispatch_tpl": "...",
      "post_dispatch": "...", "trailing_newline": true,
      "entries": [
        {
          "label": "12B-QAT [Q4_K_XL] 260K 裸跑 (+mmproj) [推荐 Agent]",
          "item_tpl": "echo  {N}) {LABEL}",
          "goto": "RUN_12B",
          "banner": [""],
          "body": "...",              // 该菜单项的完整批处理正文
          "dirs": ["gemma-4-12B-it-qat-UD-Q4_K_XL"],
          "drafts": [],
          "auto": false
        }
      ]
    }
  }
}
```

**`head` / `menu_*` / `*_tpl` 都是"从既有 .bat 里原样抠出来的结构"** ——
所以最省事的做法永远是：**先有一个能跑的菜单 `.bat`，再 `--extract`**，
而不是从零手写这份 JSON。

---

## 4. 常用命令

| 命令 | 作用 |
|---|---|
| `--paths` | 打印每个数据文件解析到哪、是否存在（排查环境变量用） |
| `--check` | **干跑**：渲染结果写到 `backup/preview/`，不动线上文件 |
| `--yes` | 应用（每次覆盖前自动备份到 `backup/`） |
| `--extract` | 从既有 `.bat` 重建注册表 |
| `--derive-params` | 为磁盘上每个模型补 `preset-overrides.json` 条目（不覆盖已有值） |
| `--ini-diff` | 只看 preset 相对 `models-config.ini` 的变化（只读） |
| `--audit` | 检查注册表 ↔ `model-profiles.json` 一致性 + 显存 |
| `--audit-drafts` / `--validate-drafts` | MTP 草稿健康度（后者会真加载模型） |
| `--tune-sweep` | `n-cpu-moe` 阶梯实测 |
| `--fix-mmproj` | 把静态 mmproj 引用同步为磁盘上最新的 |
| `--fix-bodies` | 修 `^` 续行里的空行（cmd 会静默丢参数） |
| `--fix-moe` | 修与 `preset-overrides.json` 不一致的 MoE 菜单项 |
| `--no-scan` | 不扫描磁盘，逐字渲染注册表（**回归模式**，总是写预览） |

---

## 5. 回归模式 —— 改生成器时的第一道防线

```powershell
python update_launchers.py --no-scan     # 总是渲染，总是写 backup/preview/
```

它**跳过磁盘扫描**、逐字渲染注册表，因此输出是**确定的**。
拿改动前后的两次输出做逐字节对比，就能判断改动是否影响渲染 ——
不需要真实模型，也不会碰线上文件。本次迁移就是用这个方法验证的
（3 个 `.bat` 逐字节相同）。

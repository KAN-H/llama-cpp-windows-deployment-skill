# llama.cpp Windows Deployment Skill 🚀

> An AI Agent skill for deploying, optimizing, and managing llama.cpp on Windows — covering Router Mode multi-model routing, Gemma 4 / Qwen / Phi-4 deployment, QAT–MTP inference optimization, GPU VRAM tuning, CPU memory-constrained scenarios, and WSL2 agent integration.

[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 📦 What's Included

| File | Description |
|------|-------------|
| [`SKILL.md`](./SKILL.md) | Main skill file — activated by Agent on demand (v3.2) |
| [`references/gemma4-menu-scripts.bat`](./references/gemma4-menu-scripts.bat) | Full 10-option Gemma 4 launch menu (Q5/Q8, QAT, MTP, 26B-A4B) |
| [`references/qwen-scripts.bat`](./references/qwen-scripts.bat) | Qwen 3-tier deployment scripts (bare / external draft / built-in MTP) |
| [`references/preset-templates.json`](./references/preset-templates.json) | Preset templates for 7 deployment scenarios |
| [`references/start-CPU-Toolcall-Launcher.bat`](./references/start-CPU-Toolcall-Launcher.bat) | CPU-only tool-calling launcher (`-ngl 0`, 128K ctx, 8-model menu) |
| [`references/router-mode-preset.bat`](./references/router-mode-preset.bat) | Router Mode launcher — `--models-preset` variant (per-model overrides) |
| [`references/router-mode-simple.bat`](./references/router-mode-simple.bat) | Router Mode launcher — `--models-dir` simple variant |
| [`references/model-profiles.json`](./references/model-profiles.json) | Curated model parameter knowledge base (17 profiles, verified levels) |
| [`references/20260803-session-experience.md`](./references/20260803-session-experience.md) | Benchmark data: Qwen3.6-27B 64K tuning, CPU toolcall, 11/11 tool matrix, BAT encoding |
| [`references/20260805-session-experience.md`](./references/20260805-session-experience.md) | Reasoning control (anti-deadloop, `--reasoning-budget`) & GBK safe editing |
| [`references/20260806-session-experience.md`](./references/20260806-session-experience.md) | 26B MTP draft: official Q8_0 vs third-party Q4_0 diagnosis |
| [`references/20260816-session-experience.md`](./references/20260816-session-experience.md) | `--fit` auto-offload (7x speedup), CPU toolcall new models, GBK recovery SOP |
| [`scripts/detect.ps1`](./scripts/detect.ps1) | Automated environment diagnostics (PowerShell) |
| [`scripts/detect.py`](./scripts/detect.py) | Cross-platform diagnostics (Python 3.7+, Windows/Linux/macOS) |

## 🎯 When to Use

- Deploying `llama.cpp` as a service on Windows
- Configuring **Router Mode** for multi-model switching
- Optimizing **GPU inference** (RTX 5060 Ti 16GB / Blackwell)
- Running models in **CPU-only** memory-constrained environments
- Setting up **MTP speculative decoding** (Gemma 4 / Qwen3.6)
- Migrating between **Gemma 4 ↔ Qwen** model families
- Troubleshooting **OOM**, connection issues, or parameter mismatches

## 🧰 Prerequisites

- **Windows 10/11** (64-bit) with or without WSL2
- **llama.cpp** precompiled package (`b10056 – b10158+`)
- **NVIDIA GPU** with driver ≥ 610.47 (for GPU offloading)
- **GGUF models** — standard, QAT-UD, or MTP variants
- **PowerShell + CMD** environment

## 🚀 Quick Start

```batch
:: 1. Set your paths
set LLAMA_CPP_DIR=C:\path\to\llama.cpp
set MODELS_CHAT_DIR=C:\path\to\models\chat

:: 2. Start Router Mode
cd /d %LLAMA_CPP_DIR%
llama-server.exe --models-dir %MODELS_CHAT_DIR% --host 0.0.0.0 --port 8080 -fa on -c 8192 -np 1 -t 16
```

Or run the diagnostics script first:
```powershell
.\scripts\detect.ps1 -llamaDir "C:\llama.cpp" -modelsDir "C:\models"
```

## 📖 Agent Integration

This skill is designed as a **VS Code Agent Skill** (`.github/skills/<name>/SKILL.md`). Place it in your project's skill directory, and the Agent will auto-discover it for relevant tasks.

**Place at**: `.github/skills/llama-cpp-windows-deployment/SKILL.md`
Apache License 2.0 — See [LICENSE](./LICENSE) for details.

## 🏷️ Topics

This repo is tagged with topics such as `llama-cpp`, `local-llm`, `agent-skills`, `claude-skills`, `llm-inference`, `llama-cpp-mtp`, `speculative-decoding`, `ai-agents`, `local-ai`, `windows` — see the GitHub repo page for the full list.

---

**Built from real-world RTX 5060 Ti 16GB + Intel U7 270K deployment experience.**
For version history, see [CHANGELOG](./CHANGELOG.md).
MIT — See [LICENSE](./LICENSE) for details.

---

**Built from real-world RTX 5060 Ti 16GB + Intel U7 270K deployment experience.**

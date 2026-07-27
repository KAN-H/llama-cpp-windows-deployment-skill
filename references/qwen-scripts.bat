@echo off
:: ============================================================
:: Qwen 系列部署脚本 — 引用自 SKILL.md Step 3B
:: 包含三档脚本变体，按需取消注释使用
:: 依赖: llama-server.exe 在工作目录或 PATH 中
:: ============================================================
chcp 65001 >nul
cd /d %LLAMA_CPP_DIR%

:: ============================================================
:: 脚本 B-1：Qwen3-30B-A3B 裸跑（无 MTP，最稳档）
:: MoE 30.5B/3.3B active, 无 MTP heads
:: ============================================================
:RUN_QWEN30B_BARE
echo [B-1] Qwen3-30B-A3B bare run, ctx=48K
set "MAIN=%MODELS_CHAT_DIR%\Qwen3-30B-A3B\Qwen3-30B-A3B-Q4_K_XL.gguf"

llama-server.exe ^
  -m "%MAIN%" ^
  -c 49152 ^
  -ngl 99 ^
  -fa on ^
  -np 1 ^
  -t 10 ^
  --batch-size 1024 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --no-mmap ^
  --host 0.0.0.0 ^
  --port 8083 ^
  --api-key sk-local-qwen30b ^
  --chat-template-kwargs "{\"enable_thinking\":false}" ^
  --temp 0.7 ^
  --top-p 0.8 ^
  --top-k 20 ^
  --repeat-penalty 1.05 ^
  --metrics ^
  --timeout 300
pause
goto :eof

:: ============================================================
:: 脚本 B-2：Qwen3-30B-A3B + 外挂小 draft（投机解码）
:: --spec-type draft（非 draft-mtp），外挂 Qwen3-8B Q8_0
:: ============================================================
:RUN_QWEN30B_DRAFT
echo [B-2] Qwen3-30B-A3B + external draft (speculative)
set "MAIN=%MODELS_CHAT_DIR%\Qwen3-30B-A3B\Qwen3-30B-A3B-Q4_K_XL.gguf"
set "DRAFT=%MODELS_CHAT_DIR%\Qwen3-8B\Qwen3-8B-Q8_0.gguf"

llama-server.exe ^
  -m "%MAIN%" ^
  --model-draft "%DRAFT%" ^
  --spec-type draft ^
  --spec-draft-n-max 4 ^
  -c 32768 ^
  -ngl 99 ^
  --gpu-layers-draft 99 ^
  -fa on ^
  -np 1 -t 10 ^
  --batch-size 1024 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --no-mmap ^
  --host 0.0.0.0 ^
  --port 8083 ^
  --api-key sk-local-qwen30b ^
  --chat-template-kwargs "{\"enable_thinking\":false}" ^
  --temp 0.7 ^
  --top-p 0.8 ^
  --top-k 20 ^
  --repeat-penalty 1.05 ^
  --metrics ^
  --timeout 300
pause
goto :eof

:: ============================================================
:: 脚本 B-3：Qwen3.6-35B-A3B-MTP（内置 heads）
:: 注意：无 --model-draft、无 --gpu-layers-draft
:: ============================================================
:RUN_QWEN35B_MTP
echo [B-3] Qwen3.6-35B-A3B-MTP (built-in MTP heads)
set "MAIN=%MODELS_CHAT_DIR%\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"

llama-server.exe ^
  -m "%MAIN%" ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 3 ^
  -c 32768 ^
  -ngl 99 ^
  -fa on ^
  -np 1 ^
  -t 10 ^
  --batch-size 1024 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --no-mmap ^
  --host 0.0.0.0 ^
  --port 8083 ^
  --api-key sk-local-qwen35b-mtp ^
  --chat-template-kwargs "{\"enable_thinking\":false}" ^
  --temp 0.7 ^
  --top-p 0.8 ^
  --top-k 20 ^
  --repeat-penalty 1.05 ^
  --metrics ^
  --timeout 300
pause
goto :eof

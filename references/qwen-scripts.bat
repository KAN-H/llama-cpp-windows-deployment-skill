@echo off
:: ============================================================
:: Qwen deployment scripts - reference from SKILL.md Step 3B
:: Three variants (B-1/B-2/B-3), uncomment as needed
:: Requires: llama-server.exe in cwd or PATH
:: ============================================================

:: ========== Generic config (edit for your machine) ==========
:: Change these variables to adapt to any machine
set "LLAMA_DIR=C:\llama.cpp"
set "CHAT=C:\models\chat"
:: Port: change if conflicting (8083 Qwen default)
set "PORT=8083"
set "API_KEY=sk-local-qwen30b"
cd /d "%LLAMA_DIR%"

:: ============================================================
:: Script B-1: Qwen3-30B-A3B bare run (no MTP, most stable)
:: MoE 30.5B/3.3B active, no MTP heads
:: ============================================================
:RUN_QWEN30B_BARE
echo [B-1] Qwen3-30B-A3B bare run, ctx=48K
set "MAIN=%CHAT%\Qwen3-30B-A3B\Qwen3-30B-A3B-Q4_K_XL.gguf"

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
  --port %PORT% ^
  --api-key %API_KEY% ^
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
:: Script B-2: Qwen3-30B-A3B + external small draft (speculative)
:: --spec-type draft (not draft-mtp), external Qwen3-8B Q8_0
:: ============================================================
:RUN_QWEN30B_DRAFT
echo [B-2] Qwen3-30B-A3B + external draft (speculative)
set "MAIN=%CHAT%\Qwen3-30B-A3B\Qwen3-30B-A3B-Q4_K_XL.gguf"
set "DRAFT=%CHAT%\Qwen3-8B\Qwen3-8B-Q8_0.gguf"

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
  --port %PORT% ^
  --api-key %API_KEY% ^
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
:: Script B-3: Qwen3.6-35B-A3B-MTP (built-in heads)
:: Note: no --model-draft, no --gpu-layers-draft
:: ============================================================
:RUN_QWEN35B_MTP
echo [B-3] Qwen3.6-35B-A3B-MTP (built-in MTP heads)
set "MAIN=%CHAT%\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf"

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
  --port %PORT% ^
  --api-key %API_KEY% ^
  --chat-template-kwargs "{\"enable_thinking\":false}" ^
  --temp 0.7 ^
  --top-p 0.8 ^
  --top-k 20 ^
  --repeat-penalty 1.05 ^
  --metrics ^
  --timeout 300
pause
goto :eof

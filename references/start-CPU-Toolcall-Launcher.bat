@echo off
setlocal DisableDelayedExpansion

REM ============================================================
REM === CPU Toolcall Launcher (start-CPU-Toolcall-Launcher) ====
REM === HW: Intel U7 270K Plus 24C/24T + 48GB DDR5          ====
REM === Principle: -ngl 0 forces pure CPU, no GPU usage     ====
REM === All models 128K context + tiered KV + memory opt    ====
REM === Default --tools all (built-in agent tools, Web UI)  ====
REM === Measured t/s from 2026-08-03 CPU benchmark           ====
REM ============================================================

REM ============================================================
REM === Environment check ======================================
REM ============================================================
set "LLAMA_DIR=C:\llama.cpp"
set "CHAT=C:\models\chat"
cd /d "%LLAMA_DIR%"
if not exist "llama-server.exe" (
    echo [X] llama-server.exe not found
    pause & exit /b 1
)

REM ============================================================
REM === Config (edit here) =====================================
REM ============================================================
REM --- Port (8080 Gemma / 8083 Qwen / 8086 CPU Toolcall) ---
set "PORT=8086"
REM --- Context size (all 128K) ---
set "CTX_CPU=131072"
REM --- CPU threads (24 physical cores; try 20-24) ---
set "THREADS=16"
REM --- API Key ---
set "API_KEY=sk-local-cpu-001"
REM --- Built-in agent tools (--tools all) ---
REM     Requires --jinja; enables Web UI read_file/grep/exec etc.
REM     Values: all=enabled(default) | off=disabled
set "AGENT_TOOLS=all"
if "%AGENT_TOOLS%"=="all" (
    set "TOOLS_ARG=--tools all"
) else (
    set "TOOLS_ARG="
)

REM ============================================================
REM === Model paths ============================================
REM ============================================================
set "QWEN35_2B=%CHAT%\Qwen3.5-2B\Qwen3.5-2B-Q4_K_M.gguf"
set "QWEN35_2B_MM=%CHAT%\Qwen3.5-2B\mmproj-F16.gguf"
set "PHI4=%CHAT%\Phi-4-mini-instruct-Q4_K_M\Phi-4-mini-instruct-Q4_K_M.gguf"
set "GEMMA12_QAT=%CHAT%\gemma-4-12B-it-qat-UD-Q4_K_XL\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
set "GEMMA12_MM=%CHAT%\gemma-4-12B-it-qat-UD-Q4_K_XL\mmproj-F16.gguf"
set "GEMMA_E4B=%CHAT%\gemma-4-E4B-it-qat-GGUF\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
set "GEMMA_E4B_MM=%CHAT%\gemma-4-E4B-it-qat-GGUF\mmproj-BF16.gguf"

REM ============================================================
REM === Function: file check ===================================
REM ============================================================
goto :menu

:check_file
if "%~1"=="" (
    echo [X] Empty path
    pause & goto menu
)
if not exist "%~1" (
    echo [X] Not found: %~1
    pause & goto menu
)
exit /b 0

REM ============================================================
REM === Menu ====================================================
REM ============================================================
:menu
cls
echo ============================================
echo   CPU Toolcall Launcher - CPU ONLY (no GPU)
echo  Port=[%PORT%]  Threads=[%THREADS%]  AgentTools=[%AGENT_TOOLS%]
echo ============================================
echo  1) Qwen3.5-2B [57 t/s] 128K [KV q4_0 fast] (multimodal)
echo  2) Phi-4-mini [38 t/s] 128K [KV q4_0 fast] [!] NO tool calls
echo  3) gemma-12B-QAT [13 t/s] 128K [KV q8_0 quality]
echo  4) gemma-E4B [30 t/s] 128K [KV q8_0 quality] (multimodal)
echo  5) Exit
echo ============================================
set /p c="Select [1-4]: "

if "%c%"=="1" goto RUN_QWEN35_2B
if "%c%"=="2" goto RUN_PHI4
if "%c%"=="3" goto RUN_GEMMA12
if "%c%"=="4" goto RUN_E4B
if "%c%"=="5" exit
goto menu

REM ============================================================
REM === Launch items (all -ngl 0 force CPU) =====================
REM === Mem opt: KV tiered(q4_0/q8_0) + batch 256 + mmap ======
REM ============================================================

:RUN_QWEN35_2B
call :check_file "%QWEN35_2B%"
call :check_file "%QWEN35_2B_MM%"
echo ============================================
echo Qwen3.5-2B multimodal 128K (CPU)
echo   - 1.5GB weights, ~57 t/s (measured)
echo   - KV q4_0, ~4GB RAM
echo ============================================
llama-server.exe ^
  -m "%QWEN35_2B%" ^
  --mmproj "%QWEN35_2B_MM%" ^
  --jinja ^
  --image-min-tokens 1024 ^
  -c %CTX_CPU% ^
  -ngl 0 ^
  -np 1 ^
  -t %THREADS% ^
  --threads-batch %THREADS% ^
  --batch-size 256 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --keep -1 ^
  --load-mode mmap ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  --temp 0.7 ^
  --top-p 0.95 ^
  --top-k 20 ^
  %TOOLS_ARG% ^
  --timeout 300
pause & goto menu

:RUN_PHI4
call :check_file "%PHI4%"
echo ============================================
echo Phi-4-mini 128K (CPU)
echo   - 2.5GB weights, ~38 t/s (measured)
echo   - KV q4_0, ~7GB RAM
echo   - [!] tool calling NOT supported (template limit)
echo ============================================
llama-server.exe ^
  -m "%PHI4%" ^
  --jinja ^
  -c %CTX_CPU% ^
  -ngl 0 ^
  -np 1 ^
  -t %THREADS% ^
  --threads-batch %THREADS% ^
  --batch-size 256 ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --keep -1 ^
  --load-mode mmap ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  --temp 0.7 ^
  --top-p 0.95 ^
  --top-k 20 ^
  %TOOLS_ARG% ^
  --timeout 300
pause & goto menu

:RUN_GEMMA12
call :check_file "%GEMMA12_QAT%"
call :check_file "%GEMMA12_MM%"
echo ============================================
echo gemma-12B-QAT multimodal 128K (CPU)
echo   - QAT sensitive, KV q8_0 for quality
echo   - 7.6GB weights + ~12GB KV, ~22GB RAM
echo   - [!] measured 13 t/s (bandwidth limit)
echo ============================================
llama-server.exe ^
  -m "%GEMMA12_QAT%" ^
  --mmproj "%GEMMA12_MM%" ^
  --jinja ^
  -c %CTX_CPU% ^
  -ngl 0 ^
  -np 1 ^
  -t %THREADS% ^
  --threads-batch %THREADS% ^
  --batch-size 256 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --keep -1 ^
  --load-mode mmap ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 64 ^
  %TOOLS_ARG% ^
  --timeout 300
pause & goto menu

:RUN_E4B
call :check_file "%GEMMA_E4B%"
call :check_file "%GEMMA_E4B_MM%"
echo ============================================
echo gemma-E4B multimodal 128K (CPU)
echo   - QAT sensitive, KV q8_0 for quality
echo   - 2.6GB weights, ~6GB RAM, ~30 t/s
echo ============================================
llama-server.exe ^
  -m "%GEMMA_E4B%" ^
  --mmproj "%GEMMA_E4B_MM%" ^
  --jinja ^
  -c %CTX_CPU% ^
  -ngl 0 ^
  -np 1 ^
  -t %THREADS% ^
  --threads-batch %THREADS% ^
  --batch-size 256 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --keep -1 ^
  --load-mode mmap ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 64 ^
  %TOOLS_ARG% ^
  --timeout 300
pause & goto menu
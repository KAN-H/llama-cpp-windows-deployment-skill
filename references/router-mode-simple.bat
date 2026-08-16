@echo off
REM ============================================================
REM === Router Mode launcher - simple variant (--models-dir)   ==
REM === Reference from SKILL.md Step 2 / Mode A                ==
REM === Uses --models-dir: llama.cpp scans the dir and loads   ==
REM === any model on demand (no preset.json / ini needed).     ==
REM === Edit LLAMA_DIR / MODELS_DIR / PORT for your machine.   ==
REM ============================================================
title llama.cpp Router Mode - simple variant
set "LLAMA_DIR=C:\llama.cpp"
set "MODELS_DIR=C:\models\chat"
set "PORT=8082"
set "API_KEY=sk-local-001"
cd /d "%LLAMA_DIR%"

if not exist "llama-server.exe" (
    echo [X] llama-server.exe not found in %LLAMA_DIR%
    pause & exit /b 1
)

REM --- --models-dir: no preset, single global config applies ---
llama-server.exe ^
  --models-dir "%MODELS_DIR%" ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  -ngl 99 ^
  -fa on ^
  -c 65536 ^
  -np 1 ^
  -t 10 ^
  --batch-size 1024 ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --load-mode none ^
  --metrics ^
  --timeout 600

pause

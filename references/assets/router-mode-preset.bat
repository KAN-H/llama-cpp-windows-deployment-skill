@echo off
REM ============================================================
REM === Router Mode launcher - preset variant (--models-preset) =
REM === Reference from SKILL.md Step 2 / Mode A                ==
REM === Uses models-config.ini (preset.json equivalent) to      ==
REM === override per-model params; strongest precedence.       ==
REM === Generic template: edit LLAMA_DIR / MODELS_DIR / PORT   ==
REM === (paths are C:\llama.cpp placeholders, replace for your ==
REM ===  machine; the launchers this skill ships are ASCII).   ==
REM ===                                                       ==
REM === NOTE: models-config.ini does NOT ship with this skill - ==
REM === it is a GENERATED file. See the check below.            ==
REM ============================================================
title llama.cpp Router Mode - preset variant
set "LLAMA_DIR=C:\llama.cpp"
set "MODELS_DIR=C:\models\chat"
set "PRESET_FILE=%MODELS_DIR%\models-config.ini"
set "PORT=8082"
set "API_KEY=sk-local-001"
cd /d "%LLAMA_DIR%"

if not exist "llama-server.exe" (
    echo [X] llama-server.exe not found in %LLAMA_DIR%
    pause & exit /b 1
)

REM --- Without this the run dies inside llama-server with a message about the
REM --- preset file, which reads like a llama.cpp bug rather than a missing step.
if not exist "%PRESET_FILE%" (
    echo [X] preset file not found: %PRESET_FILE%
    echo.
    echo     models-config.ini is generated, so a fresh install has none.
    echo     Create it either way:
    echo.
    echo       1. generate it from your own tuning ^(recommended^):
    echo            python scripts\launcher_gen\update_launchers.py --yes
    echo.
    echo       2. or copy the worked example and edit its ^<models-dir^> paths:
    echo            references\assets\models-config.ini.example  -^>  %PRESET_FILE%
    echo.
    pause & exit /b 1
)

REM --- --models-preset: per-model overrides from models-config.ini ---
llama-server.exe ^
  --models-preset "%PRESET_FILE%" ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --api-key %API_KEY% ^
  --models-max 3 ^
  -ngl 99 ^
  -fa on ^
  -c 65536 ^
  -t 10 ^
  --threads-batch 10 ^
  --batch-size 1024 ^
  --jinja ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --metrics ^
  --timeout 600

pause

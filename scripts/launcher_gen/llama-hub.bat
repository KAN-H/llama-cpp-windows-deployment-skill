@echo off
REM ============================================================
REM === llama-hub - the ONLY entry point you need =============
REM ===                                                      ===
REM === Everything else lives in launcher\ :                 ===
REM ===   launcher\llama_hub.py          (this menu, Python) ===
REM ===   launcher\update_launchers.py   (regenerates things) ===
REM ===   launcher\preset-overrides.json (tune models here)  ===
REM ===   launcher\models-config.bat     (router launcher)   ===
REM ===   launcher\docs\                 (guides, Chinese)   ===
REM ===   launcher\backup\               (every overwritten   ===
REM ===                                   file, timestamped) ===
REM ===                                                      ===
REM === Non-interactive (pass straight through):            ===
REM ===   llama-hub.bat --check               dry run        ===
REM ===   llama-hub.bat --yes                 apply          ===
REM ===   llama-hub.bat --audit               vram + params  ===
REM ===   llama-hub.bat --ini-diff            preset diff    ===
REM ===   llama-hub.bat --validate-drafts     MTP health     ===
REM ===   llama-hub.bat --tune-sweep all      n-cpu-moe scan ===
REM ===   llama-hub.bat --write-docs          rebuild guide  ===
REM ===   llama-hub.bat --paths               path self-check ===
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0launcher\llama_hub.py" %*
if errorlevel 1 (
    echo.
    echo [X] llama-hub failed - see messages above.
    pause
    exit /b 1
)

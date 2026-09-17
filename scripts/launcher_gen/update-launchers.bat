@echo off
REM ============================================================
REM === Model Launcher Auto-Updater ===========================
REM === Lives in launcher\ : regenerates the 3 menu launchers ===
REM === and models-config.ini from what is on disk plus the  ===
REM === JSON sources in this folder.                         ===
REM === (source of truth: preset-overrides.json /           ===
REM ===  launcher-models.json / model-profiles.json)        ===
REM ===                                                      ===
REM === Usage: launcher\update-launchers.bat [options]       ===
REM ===   (none)            report, then ask before applying ===
REM ===   --check           dry run -> launcher\backup\preview\
REM ===   --yes             apply without asking
REM ===   --audit           registry vs model-profiles + VRAM
REM ===   --ini-diff        show preset changes (read only)
REM ===   --derive-params   fill preset-overrides.json
REM ===   --tune-sweep DIR  benchmark the n-cpu-moe ladder
REM ===   --validate-drafts load-test every MTP (model, draft)
REM ===   --blacklist-drafts  with the above: blacklist failures
REM ===   --fix-bodies      repair blank-line caret continuations
REM ===   --fix-mmproj      sync mmproj paths to what is on disk
REM ===   --extract         rebuild launcher-models.json
REM ===   --no-scan         render registry verbatim (regression)
REM ============================================================
setlocal
cd /d "%~dp0.."
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0update_launchers.py" %*
if errorlevel 1 (
    echo.
    echo [X] Updater failed - see messages above.
    pause
    exit /b 1
)
echo.
pause

@echo off
setlocal DisableDelayedExpansion
title Gemma 4 Launcher - RTX 5060 Ti (Full Menu)

:: ========== Generic config (edit for your machine) ==========
:: Change these variables to adapt to any machine
set "LLAMA_DIR=C:\llama.cpp"
set "CHAT=C:\models\chat"
:: Port: change if conflicting with existing services
set "PORT=8080"
cd /d "%LLAMA_DIR%"

:: ============================================================
:: Gemma 4 full menu script - reference from SKILL.md Step 3A
:: Usage: run this BAT, select from menu
:: Requires: llama-server.exe in cwd or PATH
:: ============================================================

:: ========== Environment check ==========
where llama-server.exe >nul 2>&1 || (
    echo [ERROR] llama-server.exe not found
    pause & exit /b 1
)

where nvidia-smi >nul 2>&1 || (
    echo [WARNING] nvidia-smi not found, VRAM detection disabled
    set "NO_SMI=1"
)

:: ========== Variable declarations ==========
set "MTP=%CHAT%\gemma4_mtp"

:: 12B model variables
set "Q5=%CHAT%\gemma-4-12b-it-Q5_K_M\gemma-4-12b-it-Q5_K_M.gguf"
set "Q8=%CHAT%\gemma-4-12b-it-UD-Q8_K_XL\gemma-4-12b-it-UD-Q8_K_XL.gguf"
set "Q4_12B=%CHAT%\gemma-4-12B-it-qat-UD-Q4_K_XL\gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
set "MM_12B=%CHAT%\gemma-4-12B-it-qat-UD-Q4_K_XL\mmproj-F16.gguf"

:: 26B model variables
set "Q4_26B=%CHAT%\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
set "MM_26B=%CHAT%\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL\mmproj-F16.gguf"

:: Draft paths (adjust to actual downloads)
set "D12_Q4=%MTP%\mtp-gemma-4-12b-it-Q4_0.gguf"
set "D12_Q8=%MTP%\mtp-gemma-4-12b-it-Q8_0.gguf"
:: NOTE 2026-08-06: official 26B MTP drafter ships Q8_0/BF16/F16 only (NO Q4_0).
:: Third-party Q4_0 draft fails in llama-server with "invalid vector subscript"
:: (loads fine in llama-cli - only a server load test is conclusive).
set "D26_Q4=%MTP%\mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
set "D26_Q8=%MTP%\mtp-gemma-4-26B-A4B-it-Q8_0.gguf"

set "DRAFT_TYPE=q8"
if "%DRAFT_TYPE%"=="q4" ( set "D12=%D12_Q4%" & set "D26=%D26_Q4%"
) else ( set "D12=%D12_Q8%" & set "D26=%D26_Q8%" )

:: ========== Security params ==========
set "API_KEY=sk-local-001"
set "CORS_ORIGINS=http://localhost:* https://localhost:*"
set "TEMP=0.7"
set "TOP_P=0.9"
set "REPEAT_PENALTY=1.1"
set "MAX_TOKENS=4096"

:: ========== Helper functions ==========
goto :menu

:check_file
if "%~1"=="" ( echo [ERROR] Empty model path & pause & exit /b 1 )
if not exist "%~1" ( echo [ERROR] Model not found: %~1 & pause & exit /b 1 )
exit /b 0

:get_vram
set "FREE_MB="
if "%NO_SMI%"=="1" goto :eof
for /f "tokens=1" %%i in (
    'nvidia-smi --query-gpu=memory.free --format="csv,noheader,nounits" -i 0 2^>nul'
) do ( set "FREE_MB=%%i" & goto :eof )
set "FREE_MB=2000"
goto :eof

:: ========== Menu ==========
:menu
cls
echo ============================================
echo Gemma 4 - RTX 5060 Ti 16GB (Full Menu)
echo Draft=[%DRAFT_TYPE%]  API Key=[%API_KEY:~0,4%****]
echo ============================================
echo  1) Q5 + MTP + 64K
echo  2) Q8 + MTP + 64K
echo  3) Q5 bare 64K
echo  4) Q8 bare 64K
echo  5) 12B-QAT 128K bare [Recommended for Hermes]
echo  6) 26B-QAT 128K bare [Extreme Mode]
echo  7) 12B-QAT + MTP 64K [!] NOT recommended: QAT+MTP
echo  8) 26B-QAT + MTP 32K
echo  9) 12B-QAT + MTP 128K [High Performance] [!] NOT recommended: QAT+MTP
echo  10) Exit
echo ============================================
set /p c="Select [1-10]: "

if "%c%"=="1" goto RUN_Q5MTP
if "%c%"=="2" goto RUN_Q8MTP
if "%c%"=="3" goto RUN_Q5B
if "%c%"=="4" goto RUN_Q8B
if "%c%"=="5" goto RUN_12B
if "%c%"=="6" goto RUN_26B
if "%c%"=="7" goto RUN_12B_MTP
if "%c%"=="8" goto RUN_26B_MTP
if "%c%"=="9" goto RUN_12B_MTP128
if "%c%"=="10" exit
goto menu

:RUN_Q5MTP
call :check_file "%Q5%" & call :check_file "%D12%"
llama-server.exe -m "%Q5%" --model-draft "%D12%" --spec-type draft-mtp --spec-draft-n-max 2 -c 32768 -ngl 99 --gpu-layers-draft 60 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 120
pause & goto menu

:RUN_Q8MTP
call :check_file "%Q8%" & call :check_file "%D12%"
llama-server.exe -m "%Q8%" --model-draft "%D12%" --spec-type draft-mtp --spec-draft-n-max 2 -c 16384 -ngl 99 --gpu-layers-draft 50 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 120
pause & goto menu

:RUN_Q5B
llama-server.exe -m "%Q5%" -c 65536 -ngl 99 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --timeout 120
pause & goto menu

:RUN_Q8B
llama-server.exe -m "%Q8%" -c 65536 -ngl 99 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --timeout 120
pause & goto menu

:RUN_12B
call :check_file "%Q4_12B%" & call :check_file "%MM_12B%"
llama-server.exe -m "%Q4_12B%" --mmproj "%MM_12B%" -c 131072 -ngl 99 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --keep -1 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 120
pause & goto menu

:RUN_26B
call :check_file "%Q4_26B%" & call :check_file "%MM_26B%"
llama-server.exe -m "%Q4_26B%" --mmproj "%MM_26B%" -c 131072 -ngl 28 -fa on -np 1 -t 10 --batch-size 512 --cache-type-k q4_0 --cache-type-v q4_0 --keep -1 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 300
pause & goto menu

:RUN_12B_MTP
call :check_file "%Q4_12B%" & call :check_file "%D12%" & call :check_file "%MM_12B%"
llama-server.exe -m "%Q4_12B%" --mmproj "%MM_12B%" --model-draft "%D12%" --spec-type draft-mtp --spec-draft-n-max 2 -c 65536 -ngl 99 --gpu-layers-draft 32 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --keep -1 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 180
pause & goto menu

:RUN_26B_MTP
call :check_file "%Q4_26B%" & call :check_file "%D26%" & call :check_file "%MM_26B%"
llama-server.exe -m "%Q4_26B%" --mmproj "%MM_26B%" --model-draft "%D26%" --spec-type draft-mtp --spec-draft-n-max 2 -c 32768 -ngl 99 --gpu-layers-draft 42 -fa on -np 1 -t 10 --batch-size 512 --cache-type-k q8_0 --cache-type-v q8_0 --keep -1 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 180
pause & goto menu

:RUN_12B_MTP128
call :check_file "%Q4_12B%" & call :check_file "%D12%" & call :check_file "%MM_12B%"
call :get_vram
set /a VM_TEST=%FREE_MB%
if %VM_TEST% lss 2000 (
    echo [ERROR] Need 2GB+ VRAM for 128K MTP, current %FREE_MB% MB
    pause & goto menu
)
llama-server.exe -m "%Q4_12B%" --mmproj "%MM_12B%" --model-draft "%D12%" --spec-type draft-mtp --spec-draft-n-max 2 -c 131072 -ngl 99 --gpu-layers-draft 32 -fa on -np 1 -t 10 --batch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --keep -1 --no-mmap --host 0.0.0.0 --port %PORT% --api-key "%API_KEY%" --cors-origins %CORS_ORIGINS% --temp %TEMP% --top-p %TOP_P% --repeat-penalty %REPEAT_PENALTY% --timeout 180
pause & goto menu

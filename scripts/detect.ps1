<#
.SYNOPSIS
    llama.cpp 部署环境自动诊断脚本
.DESCRIPTION
    一键检测 CUDA 架构、MTP 参数指纹、驱动版本、模型文件状态和显存，
    输出诊断报告供 Agent 或用户参考。
.PARAMETER llamaDir
    llama.cpp 工作目录路径，默认为 C:\llama.cpp
.PARAMETER modelsDir
    模型存放目录路径，默认为 C:\models
.EXAMPLE
    .\detect.ps1
    .\detect.ps1 -llamaDir "C:\llama.cpp" -modelsDir "D:\models"
.NOTES
    编码说明：本脚本为 UTF-8 无 BOM + 中文/emoji 输出。Windows PowerShell 5.1
    默认按 ANSI(GBK) 读取无 BOM 文件，中文与 emoji 可能乱码。建议使用 PowerShell 7+
    （默认 UTF-8）运行；或在 PS 5.1 中先执行
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 再运行。
#>

param(
    [string]$llamaDir = "C:\llama.cpp",
    [string]$modelsDir = "C:\models"
)

$results = @{}
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  llama.cpp 环境自动诊断" -ForegroundColor Cyan
Write-Host "  时间: $timestamp" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------
# [1/6] llama-server.exe 存在性检查
# --------------------------------------------------
Write-Host "[1/6] 检查 llama-server.exe ... " -NoNewline
$svr = Join-Path $llamaDir "llama-server.exe"
if (Test-Path $svr) {
    $svrVersion = (Get-Item $svr).VersionInfo.FileVersionRaw
    Write-Host "✅ 已找到" -ForegroundColor Green
    Write-Host "      路径: $svr"
    if ($svrVersion) { Write-Host "      版本: $($svrVersion.ToString())" }
    $results["llama-server"] = "Found"
} else {
    Write-Host "❌ 未找到" -ForegroundColor Red
    Write-Host "      期望路径: $svr"
    Write-Host "      请确认 llama.cpp 预编译包已解压到正确目录"
    $results["llama-server"] = "Missing"
}

# --------------------------------------------------
# [2/6] CUDA 架构指纹检测
# --------------------------------------------------
Write-Host "[2/6] 检测 CUDA 架构 ... " -NoNewline
if (Test-Path $svr) {
    $help = & $svr --help 2>&1 | Out-String
    if ($help -match "sm_\d+") {
        $arch = [regex]::Match($help, "sm_\d+").Value
        Write-Host "✅ $arch" -ForegroundColor Green
        $results["CUDA_Arch"] = $arch
        if ($arch -eq "sm_89") {
            Write-Host "      ⚠️  sm_89 = CUDA 12.4 编译, Blackwell (RTX 50系) 需 sm_120" -ForegroundColor Yellow
            Write-Host "      建议: 下载 CUDA 13.3 版预编译包 (llama-bXXXX-bin-win-cuda-13.3-x64.zip)"
            $results["CUDA_Arch_Note"] = "sm_89, not sm_120 - Blackwell may have PTX JIT overhead"
        } elseif ($arch -eq "sm_120") {
            Write-Host "      ✅ Blackwell 原生支持" -ForegroundColor Green
            $results["CUDA_Arch_Note"] = "Native Blackwell support"
        } else {
            Write-Host "      架构: $arch (非 Blackwell 系列)"
            $results["CUDA_Arch_Note"] = "Non-Blackwell arch: $arch"
        }
    } else {
        Write-Host "⚠️ 无法识别" -ForegroundColor Yellow
        Write-Host "      llama-server --help 未输出 sm_ 信息，可能为 CPU-only 版本"
        $results["CUDA_Arch"] = "Unknown (CPU-only build?)"
    }
} else {
    Write-Host "⏭️  跳过（llama-server.exe 不存在）" -ForegroundColor DarkYellow
}

# --------------------------------------------------
# [3/6] MTP 参数指纹识别
# --------------------------------------------------
Write-Host "[3/6] 检测 MTP 参数命名规范 ... " -NoNewline
if (Test-Path $svr) {
    $help = & $svr --help 2>&1 | Out-String
    
    if ($help -match "spec-draft-n-max") {
        Write-Host "✅ --spec-* 命名 (b10056 标准)" -ForegroundColor Green
        $results["MTP_Fingerprint"] = "spec-* (b10056)"
        # 检测具体支持的参数
        $specParams = @("spec-type", "spec-draft-n-max", "spec-draft-ngl")
        $foundParams = @()
        foreach ($p in $specParams) {
            if ($help -match $p) { $foundParams += $p }
        }
        Write-Host "      可用参数: $($foundParams -join ', ')"
        
        if ($help -match "spec-type.*draft-mtp") {
            Write-Host "      ✅ draft-mtp 类型可用" -ForegroundColor Green
        } else {
            Write-Host "      ⚠️ draft-mtp 类型不可用" -ForegroundColor Yellow
        }
        
        # 检测已废弃参数
        if ($help -match "spec-draft-buffer") {
            Write-Host "      ⚠️ spec-draft-buffer 仍存在（预期 b10056 已移除）" -ForegroundColor Yellow
        }
        if ($help -match "draft-n") {
            Write-Host "      ℹ️ 旧 --draft-* 参数显示为 removed" -ForegroundColor DarkCyan
        }
        
    } elseif ($help -match "draft-model|draft-mtp-n") {
        Write-Host "⚠️ --draft-* 命名 (较新 build)" -ForegroundColor Yellow
        $results["MTP_Fingerprint"] = "draft-* (newer build)"
        Write-Host "      本技能 MTP 参数需切换到 --draft-* 命名"
        Write-Host "      spec→draft 映射:"
        Write-Host "        --spec-type         → --draft-type"
        Write-Host "        --spec-draft-n-max → --draft-mtp-n"
        Write-Host "        --gpu-layers-draft → --draft-mtp-ngl"
        Write-Host "        --model-draft      → --draft-model"
    } else {
        Write-Host "❌ 无法识别 MTP 参数" -ForegroundColor Red
        $results["MTP_Fingerprint"] = "Unknown"
    }
} else {
    Write-Host "⏭️  跳过（llama-server.exe 不存在）" -ForegroundColor DarkYellow
}

# --------------------------------------------------
# [4/6] NVIDIA 驱动版本检测
# --------------------------------------------------
Write-Host "[4/6] 检测 NVIDIA 驱动 ... " -NoNewline
try {
    $nvidia = nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader 2>$null
    if ($nvidia) {
        $parts = $nvidia.Trim() -split ','
        $driverVer = $parts[0].Trim()
        $totalVRAM = $parts[1].Trim()
        Write-Host "✅ 驱动 $driverVer | 显存 $totalVRAM" -ForegroundColor Green
        $results["NVIDIA_Driver"] = $driverVer
        $results["Total_VRAM"] = $totalVRAM
        
        if ([version]$driverVer -lt [version]"610.47") {
            Write-Host "      ⚠️ Blackwell 最低要求 610.47，当前 $driverVer" -ForegroundColor Yellow
            $results["Driver_Note"] = "Below Blackwell minimum (610.47)"
        } elseif ([version]$driverVer -ge [version]"610.62") {
            Write-Host "      ✅ 驱动版本满足推荐要求" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️ nvidia-smi 无输出" -ForegroundColor Yellow
        $results["NVIDIA_Driver"] = "nvidia-smi returned no data"
    }
} catch {
    Write-Host "⚠️ 检测失败: $_" -ForegroundColor Yellow
    $results["NVIDIA_Driver"] = "Detection failed"
}

# --------------------------------------------------
# [5/6] 模型目录扫描
# --------------------------------------------------
Write-Host "[5/6] 扫描模型目录 ... " -NoNewline
if (Test-Path $modelsDir) {
    $ggufFiles = Get-ChildItem $modelsDir -Recurse -Filter *.gguf -ErrorAction SilentlyContinue
    $ggufCount = $ggufFiles.Count
    Write-Host "✅ 找到 $ggufCount 个 GGUF 文件" -ForegroundColor Green
    $results["GGUF_Count"] = $ggufCount
    
    # 按子目录分组
    $dirGroups = $ggufFiles | Group-Object DirectoryName
    Write-Host "      模型目录分布:"
    foreach ($g in $dirGroups) {
        $dirName = $g.Name -replace [regex]::Escape($modelsDir), ""
        if ($dirName -eq "") { $dirName = "\ (根目录)" }
        $totalSize = ($g.Group | Measure-Object Length -Sum).Sum / 1GB
        Write-Host "        $dirName : $($g.Count) 个文件, $([math]::Round($totalSize,2)) GB"
    }
    
    # 检查常见的模型命名问题
    $noModelGguf = $ggufFiles | Where-Object { $_.Name -ne "model.gguf" -and $_.DirectoryName -notmatch 'mmproj|mtp' }
    if ($noModelGguf.Count -gt 0) {
        Write-Host "      ℹ️ $($noModelGguf.Count) 个非标准命名 GGUF（非 model.gguf）" -ForegroundColor DarkCyan
    }
} else {
    Write-Host "❌ 目录不存在: $modelsDir" -ForegroundColor Red
    $results["GGUF_Count"] = "Directory not found"
}

# --------------------------------------------------
# [6/6] GPU 显存状态（仅 GPU 场景）
# --------------------------------------------------
Write-Host "[6/6] 检测 GPU 显存状态 ... " -NoNewline
try {
    $vramInfo = nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv,noheader,nounits -i 0 2>$null
    if ($vramInfo) {
        $vramParts = $vramInfo.Trim() -split ','
        $freeMB = [int]$vramParts[0].Trim()
        $usedMB = [int]$vramParts[1].Trim()
        $totalMB = [int]$vramParts[2].Trim()
        $usedPct = [math]::Round(($usedMB / $totalMB) * 100, 1)
        Write-Host "✅ 空闲 ${freeMB}MB / 已用 ${usedMB}MB ($usedPct%) / 总共 ${totalMB}MB" -ForegroundColor Green
        $results["VRAM_Free_MB"] = $freeMB
        $results["VRAM_Used_Pct"] = $usedPct
        
        if ($freeMB -lt 2000) {
            Write-Host "      ⚠️ 空闲显存 < 2GB，大模型部署可能受限" -ForegroundColor Yellow
        }
        if ($totalMB -lt 8000) {
            Write-Host "      ⚠️ 总显存 < 8GB，12B+ 模型可能无法加载" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️ 无法获取显存信息" -ForegroundColor Yellow
        $results["VRAM"] = "Unavailable"
    }
} catch {
    Write-Host "⚠️ 检测失败" -ForegroundColor Yellow
    $results["VRAM"] = "Detection failed"
}

# --------------------------------------------------
# 汇总报告
# --------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  诊断汇总" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
$results.Keys | Sort-Object | ForEach-Object {
    Write-Host "  $_ : $($results[$_])"
}
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  诊断完毕" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 返回结果对象供程序化调用
return [PSCustomObject]$results

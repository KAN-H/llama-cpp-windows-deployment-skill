<#
.SYNOPSIS
    Isolated llama-server benchmark harness for MoE models with resource guards.

.DESCRIPTION
    Starts ONE llama-server instance on an isolated port, waits for /health while
    continuously sampling free VRAM, free RAM and commit headroom, then runs a fixed
    workload and records server-side timings (prefill / decode / draft acceptance).

    Guards (all fatal, only the PID started by this script is ever stopped):
      - preflight: free VRAM >= MinFreeGpuStartMiB, free RAM >= MinFreeRamStartMiB, commit >= MinCommitMiB
      - runtime  : free VRAM < GpuFloorMiB or free RAM/commit < 4096 MiB  -> abort

    Why a lower GpuFloorMiB may be legitimate: a production config can genuinely run
    with ~2.1 GiB free. If you lower the floor, state the deviation in your report.

.PARAMETER Mode
    short        : 3 x fixed prompt, fixed output cap (throughput only, may be all reasoning)
    long         : ~12K token prompt -> full prefill, then prefix-cache reuse, then an answer check, plus one image
    capabilities : tool-call round trip, image read, ~4K-token retrieval

.EXAMPLE
    .\moe_bench.ps1 -LlamaDir '<llama-cpp-dir>' -Model '<models-dir>\Qwen3.6-35B-A3B\model.gguf' `
        -Mmproj '<models-dir>\Qwen3.6-35B-A3B\mmproj.gguf' -Context 262144 -CpuExperts 34 -Fit -Mode short

.NOTES
    ASCII only. Refuses to overwrite existing result files.
    Server stderr log is kept next to the results: inspect 'draft acceptance', 'n_slots', 'fit' lines.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LlamaDir,
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Mmproj,
    [string]$Label = 'bench',
    [string]$OutputDir,
    [int]$Port = 18084,
    [int]$Context = 8192,
    [int]$Parallel = 1,
    [int]$CpuExperts = 34,
    [int]$CacheMiB = 0,
    [int]$GpuLayers = 99,
    [int]$Threads = 12,
    [string]$CpuRange = '',
    [int]$BatchSize = 2048,
    [int]$UbatchSize = 2048,
    [string]$CacheTypeK = 'q8_0',
    [string]$CacheTypeV = 'q8_0',
    [string]$LoadMode = 'none',
    [int]$SpecDraftNMax = 4,
    [string]$SpecType = 'draft-mtp',
    [int]$ReasoningBudget = 1024,
    [string]$ReasoningFormat = 'deepseek',
    [int]$GpuFloorMiB = 3072,
    [int]$MinFreeGpuStartMiB = 15000,
    [int]$MinFreeRamStartMiB = 32768,
    [int]$MinCommitMiB = 46080,
    [int]$Requests = 3,
    [switch]$Fit,
    [int]$FitCtx = 0,
    [int]$Verbosity = 0,
    [ValidateSet('short', 'long', 'capabilities')][string]$Mode = 'short',
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

$server = Join-Path $LlamaDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $server)) { throw "llama-server.exe not found under $LlamaDir" }
if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }
if ($Mmproj -and -not (Test-Path -LiteralPath $Mmproj)) { throw "Mmproj not found: $Mmproj" }
if (-not $OutputDir) { $OutputDir = Join-Path ([System.IO.Path]::GetTempPath()) 'moe_bench' }
if (-not (Test-Path -LiteralPath $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

$prefix = Join-Path $OutputDir "$Label-$Mode"
$logOut = "$prefix-stdout.log"
$logErr = "$prefix-stderr.log"
$csvRes = "$prefix-resources.csv"
$jsonOut = "$prefix-results.json"
foreach ($f in @($logOut, $logErr, $csvRes, $jsonOut)) {
    if (Test-Path -LiteralPath $f) { throw "Refusing to overwrite $f" }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use"
}

function Get-Resources {
    $gpuLine = @(& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $gpuLine.Count -ne 1 -or $gpuLine[0] -notmatch '^\s*\d+\s*$') {
        throw 'GPU measurement failed (nvidia-smi)'
    }
    $samples = (Get-Counter @('\Memory\Available MBytes', '\Memory\Commit Limit', '\Memory\Committed Bytes')).CounterSamples
    $ram = ($samples | Where-Object { $_.Path -like '*available mbytes' }).CookedValue
    $limit = ($samples | Where-Object { $_.Path -like '*commit limit' }).CookedValue
    $used = ($samples | Where-Object { $_.Path -like '*committed bytes' }).CookedValue
    if ($null -eq $ram -or $null -eq $limit -or $null -eq $used) { throw 'Memory measurement failed' }
    return @{
        gpu    = [int]$gpuLine[0]
        ram    = [math]::Round($ram)
        commit = [math]::Round(($limit - $used) / 1MB)
    }
}

$argsList = @('-m', $Model)
if ($Mmproj) { $argsList += @('--mmproj', $Mmproj) }
$argsList += @(
    '-c', "$Context", '-ngl', "$GpuLayers", '--n-cpu-moe', "$CpuExperts", '--moe-cache-mib', "$CacheMiB",
    '-fa', 'on', '-np', "$Parallel", '--kv-unified', '-t', "$Threads",
    '-b', "$BatchSize", '-ub', "$UbatchSize", '-ctk', $CacheTypeK, '-ctv', $CacheTypeV,
    '--load-mode', $LoadMode, '--jinja', '--reasoning-budget', "$ReasoningBudget",
    '--reasoning-format', $ReasoningFormat, '--spec-type', $SpecType, '--spec-draft-n-max', "$SpecDraftNMax",
    '--host', '127.0.0.1', '--port', "$Port"
)
if ($CpuRange) { $argsList += @('--cpu-range', $CpuRange) }
if ($Fit) { $argsList += @('--fit', 'on'); if ($FitCtx -gt 0) { $argsList += @('--fit-ctx', "$FitCtx") } }
# the fit verdict ("projected to use ... / will leave ...") is TRACE level in current builds
if ($Verbosity -eq 0 -and $Fit) { $Verbosity = 5 }
if ($Verbosity -gt 0) { $argsList += @('-lv', "$Verbosity") }
if ($ExtraArgs.Count -gt 0) { $argsList += $ExtraArgs }

$pre = Get-Resources
if ($pre.gpu -lt $MinFreeGpuStartMiB -or $pre.ram -lt $MinFreeRamStartMiB -or $pre.commit -lt $MinCommitMiB) {
    throw ("Preflight failed: GPU={0} RAM={1} commit={2} MiB (need >= {3}/{4}/{5})" -f `
            $pre.gpu, $pre.ram, $pre.commit, $MinFreeGpuStartMiB, $MinFreeRamStartMiB, $MinCommitMiB)
}

$proc = Start-Process -FilePath $server -ArgumentList $argsList -WorkingDirectory $LlamaDir `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
Set-Content -LiteralPath $csvRes -Value 'timestamp,phase,gpu_free_mib,ram_available_mib,commit_headroom_mib,server_working_set_mib'

$minGpu = $pre.gpu
$minRam = $pre.ram
$minCommit = $pre.commit
$client = $null

function Assert-Guard([string]$Phase) {
    $script:proc.Refresh()
    if ($script:proc.HasExited) { throw "Server exited with code $($script:proc.ExitCode); see $script:logErr" }
    $r = Get-Resources
    $script:minGpu = [math]::Min($script:minGpu, $r.gpu)
    $script:minRam = [math]::Min($script:minRam, $r.ram)
    $script:minCommit = [math]::Min($script:minCommit, $r.commit)
    $ws = [math]::Round($script:proc.WorkingSet64 / 1MB)
    Add-Content -LiteralPath $script:csvRes -Value "$((Get-Date).ToString('o')),$Phase,$($r.gpu),$($r.ram),$($r.commit),$ws"
    if ($r.gpu -lt $script:GpuFloorMiB -or $r.ram -lt 4096 -or $r.commit -lt 4096) {
        throw ("Resource guard {0}: GPU={1} RAM={2} commit={3} MiB" -f $Phase, $r.gpu, $r.ram, $r.commit)
    }
}

function Invoke-Chat([hashtable]$Body, [string]$Phase, [int]$TimeoutSeconds = 300) {
    $json = $Body | ConvertTo-Json -Depth 30 -Compress
    $content = [System.Net.Http.StringContent]::new($json, [System.Text.Encoding]::UTF8, 'application/json')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $task = $script:client.PostAsync("http://127.0.0.1:$Port/v1/chat/completions", $content)
    while (-not $task.IsCompleted) {
        Assert-Guard $Phase
        if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) { throw "$Phase timed out" }
        Start-Sleep -Milliseconds 250
    }
    $resp = $task.GetAwaiter().GetResult()
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $resp.IsSuccessStatusCode) {
        throw ("{0} HTTP {1}: {2}" -f $Phase, [int]$resp.StatusCode, $text.Substring(0, [math]::Min(400, $text.Length)))
    }
    $watch.Stop()
    return @{ parsed = ($text | ConvertFrom-Json); wall_s = $watch.Elapsed.TotalSeconds }
}

function New-TestImage {
    Add-Type -AssemblyName System.Drawing
    $bmp = [System.Drawing.Bitmap]::new(700, 180)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::White)
    $font = [System.Drawing.Font]::new('Arial', 32)
    $g.DrawString("ORDER: AX-4719`nTOTAL: 238.75", $font, [System.Drawing.Brushes]::Black, 12, 12)
    $ms = [System.IO.MemoryStream]::new()
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $b64 = [Convert]::ToBase64String($ms.ToArray())
    $ms.Dispose(); $g.Dispose(); $font.Dispose(); $bmp.Dispose()
    return $b64
}

try {
    Write-Output ("start label={0} mode={1} port={2} PID={3} | GPU={4} RAM={5} commit={6} MiB" -f `
            $Label, $Mode, $Port, $proc.Id, $pre.gpu, $pre.ram, $pre.commit)

    $ready = $false
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 600) {
        Assert-Guard 'loading'
        $code = & curl.exe --max-time 1 -s -o NUL -w '%{http_code}' "http://127.0.0.1:$Port/health"
        if ($code -eq '200') { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'Health timeout' }
    Assert-Guard 'ready'

    $fitLines = @(Select-String -Path $logErr -Pattern 'projected to use|will leave|no changes needed|failed to fit|n_slots =' |
            ForEach-Object { ($_.Line -replace '^\S+\s+[A-Z]\s+', '').Trim() })
    if ($fitLines.Count -gt 0) { Write-Output '--- fit / slots ---'; $fitLines | ForEach-Object { Write-Output "  $_" } }

    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(600)
    $result = [ordered]@{
        label = $Label; mode = $Mode; llm_server = $server; model = $Model; mmproj = $Mmproj
        context = $Context; parallel = $Parallel; cpu_experts = $CpuExperts; moe_cache_mib = $CacheMiB
        gpu_floor_mib = $GpuFloorMiB; fit = [bool]$Fit; arguments = $argsList
        min_gpu_free_mib = $minGpu; min_ram_free_mib = $minRam; min_commit_free_mib = $minCommit
    }

    if ($Mode -eq 'short') {
        $rows = @()
        for ($i = 0; $i -lt $Requests; $i++) {
            $body = @{
                messages = @(@{ role = 'user'; content = 'Explain in detail how a binary search algorithm handles duplicates and boundary conditions.' })
                max_tokens = 384; seed = 42; stream = $false; cache_prompt = $false
            }
            $r = Invoke-Chat $body "request-$i"
            $rows += @{ iteration = $i; wall_s = [math]::Round($r.wall_s, 3); response = $r.parsed }
            $accept = 'n/a'
            if ($r.parsed.timings.draft_n) {
                $accept = [math]::Round($r.parsed.timings.draft_n_accepted / $r.parsed.timings.draft_n, 3)
            }
            Write-Output ("  iteration={0} finish={1} out={2} decode={3:N2} t/s prefill={4:N1} t/s accept={5}" -f `
                    $i, $r.parsed.choices[0].finish_reason, $r.parsed.usage.completion_tokens,
                $r.parsed.timings.predicted_per_second, $r.parsed.timings.prompt_per_second, $accept)
        }
        $result['rows'] = $rows
    }
    elseif ($Mode -eq 'long') {
        $sb = New-Object System.Text.StringBuilder
        [void]$sb.AppendLine('SERIAL: K7-QX42')
        for ($n = 1; $n -le 500; $n++) {
            [void]$sb.AppendLine("Item $n of the warehouse inventory sheet records a boxed unit with count $( $n % 37 ) stored in aisle $( $n % 12 ).")
        }
        $filler = $sb.ToString()
        $asks = @(
            @{ name = 'long_prefill'; tail = "`n`nReply with the serial code only."; max = 128; cache = $false },
            @{ name = 'long_reuse'; tail = "`n`nReply with the serial code only."; max = 128; cache = $true },
            @{ name = 'long_answer'; tail = "`n`nWhat is 47 * 13 + 8? Reply exactly as: FINAL: <number>"; max = 2048; cache = $true }
        )
        $out = [ordered]@{}
        foreach ($ask in $asks) {
            $body = @{ messages = @(@{ role = 'user'; content = ($filler + $ask.tail) }); max_tokens = $ask.max; seed = 42; cache_prompt = $ask.cache }
            $r = Invoke-Chat $body $ask.name
            $out[$ask.name] = $r.parsed
            Write-Output ("  {0}: prompt={1} cached={2} out={3} prefill={4:N1} t/s decode={5:N1} t/s finish={6}" -f `
                    $ask.name, $r.parsed.usage.prompt_tokens, $r.parsed.usage.prompt_tokens_details.cached_tokens,
                $r.parsed.usage.completion_tokens, $r.parsed.timings.prompt_per_second,
                $r.parsed.timings.predicted_per_second, $r.parsed.choices[0].finish_reason)
        }
        $b64 = New-TestImage
        $img = Invoke-Chat @{
            messages = @(@{ role = 'user'; content = @(
                        @{ type = 'text'; text = 'Read the order and total shown in the image.' },
                        @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$b64" } }) })
            max_tokens = 1536; cache_prompt = $false
        } 'image'
        $out['image'] = $img.parsed
        Write-Output ("  image: out={0} finish={1} text={2}" -f $img.parsed.usage.completion_tokens,
            $img.parsed.choices[0].finish_reason, (([string]$img.parsed.choices[0].message.content) -replace '\s+', ' '))
        $result['results'] = $out
    }
    else {
        $tool = @{ type = 'function'; function = @{
                name = 'get_weather'; description = 'Return weather for a specified city'
                parameters = @{ type = 'object'; properties = @{ city = @{ type = 'string' } }; required = @('city') } } }
        $question = @{ role = 'user'; content = 'Use the get_weather tool to look up Beijing, then tell me the temperature.' }
        $first = Invoke-Chat @{ messages = @($question); tools = @($tool); tool_choice = 'auto'; max_tokens = 1536; seed = 42; cache_prompt = $false } 'tool-call'
        $call = $first.parsed.choices[0].message.tool_calls | Select-Object -First 1
        $second = $null
        if ($null -ne $call) {
            $toolMsg = @{ role = 'tool'; tool_call_id = $call.id; content = '{"city":"Beijing","temperature_c":26}' }
            $second = Invoke-Chat @{
                messages = @($question, $first.parsed.choices[0].message, $toolMsg)
                tools = @($tool); tool_choice = 'auto'; max_tokens = 1536; seed = 42; cache_prompt = $false
            } 'tool-followup'
        }
        $retrieval = Invoke-Chat @{
            messages = @(@{ role = 'user'; content = "Remember this code: MANGO-8147. $(('A small green box is placed on a wooden table. ' * 350)) What code should you remember? Reply with the code." })
            max_tokens = 1536; seed = 42; cache_prompt = $false
        } 'retrieval'
        $callCount = @($first.parsed.choices[0].message.tool_calls).Count
        Write-Output ("  tool_first: calls={0} finish={1}" -f $callCount, $first.parsed.choices[0].finish_reason)
        Write-Output ("  retrieval: prompt={0} answer={1}" -f $retrieval.parsed.usage.prompt_tokens,
            (([string]$retrieval.parsed.choices[0].message.content) -replace '\s+', ' '))
        $result['tool_first'] = $first.parsed
        $result['tool_followup'] = $second
        $result['retrieval'] = $retrieval.parsed
    }

    $result['min_gpu_free_mib'] = $minGpu
    $result['min_ram_free_mib'] = $minRam
    $result['min_commit_free_mib'] = $minCommit
    $result['fit_lines'] = $fitLines
    $result | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $jsonOut -Encoding UTF8
    Write-Output ("done: min GPU={0} RAM={1} commit={2} MiB -> {3}" -f $minGpu, $minRam, $minCommit, $jsonOut)
}
finally {
    if ($null -ne $client) { $client.Dispose() }
    $proc.Refresh()
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    $after = @(& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)
    Write-Output ("stopped test PID={0}; GPU now used/free = {1}" -f $proc.Id, ($after -join ' '))
}

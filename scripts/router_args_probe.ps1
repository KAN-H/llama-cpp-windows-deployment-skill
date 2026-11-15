<#
.SYNOPSIS
    Dump the child-process arguments a running llama-server Router will actually use.

.DESCRIPTION
    Router overlays its own CLI args on top of every model preset, so values in a
    preset file (e.g. `parallel = 3`) can be silently overridden by the launcher
    (e.g. `-np 1`). This script asks the Router for the rendered result instead of
    trusting either file.

    Read-only: it never starts, stops or reloads anything (no `?reload=` is sent).

.PARAMETER Url
    Router base URL, e.g. http://127.0.0.1:8082

.PARAMETER ApiKey
    Bearer token if the router was started with --api-key. Never printed.

.PARAMETER Filter
    Wildcard to select model ids, e.g. 'Q5m*'. Default '*'.

.PARAMETER ShowPreset
    Also print the preset as re-rendered by the router (ini form).

.PARAMETER Raw
    Print the raw /models JSON instead of the summary.

.EXAMPLE
    .\router_args_probe.ps1 -Url http://127.0.0.1:8082 -ApiKey $env:LLAMA_API_KEY

.NOTES
    ASCII only. Exit code 2 when no model matches the filter.
#>
[CmdletBinding()]
param(
    [string]$Url = 'http://127.0.0.1:8082',
    [string]$ApiKey = '',
    [string]$Filter = '*',
    [switch]$ShowPreset,
    [switch]$Raw
)

$ErrorActionPreference = 'Stop'
$Url = $Url.TrimEnd('/')

function Hide-Secrets([string]$text) {
    if (-not $text) { return $text }
    $t = [regex]::Replace($text, '(--api-key\s+)(\S+)', '$1<redacted>')
    $t = [regex]::Replace($t, '(?i)(authorization:\s*bearer\s+)(\S+)', '$1<redacted>')
    if ($ApiKey) { $t = $t.Replace($ApiKey, '<redacted>') }
    return $t
}

$headers = @{}
if ($ApiKey) { $headers['Authorization'] = "Bearer $ApiKey" }

try {
    $resp = Invoke-WebRequest -Uri "$Url/models" -Headers $headers -UseBasicParsing -TimeoutSec 20
}
catch {
    $status = $null
    if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
    if ($status -eq 401) { throw "401 from $Url/models - pass -ApiKey matching the server's --api-key" }
    throw "Cannot reach $Url/models : $($_.Exception.Message)"
}

if ($Raw) { Write-Output (Hide-Secrets $resp.Content); return }

$models = ($resp.Content | ConvertFrom-Json).data
if ($null -eq $models) { throw "Unexpected /models payload (no .data)" }

$selected = @($models | Where-Object { $_.id -like $Filter })
if ($selected.Count -eq 0) {
    Write-Output "No model matched '$Filter'. Available:"
    $models | ForEach-Object { Write-Output ("  - {0}" -f $_.id) }
    exit 2
}

$inRouterMode = $false
foreach ($m in $selected) { if ($m.status -and $m.status.args) { $inRouterMode = $true } }
if (-not $inRouterMode) {
    Write-Output 'This server does NOT look like a Router (no per-model args). Models:'
    $selected | ForEach-Object { Write-Output ("  - {0}" -f $_.id) }
    return
}

foreach ($m in $selected) {
    Write-Output ('=== {0}' -f $m.id)
    Write-Output ('    status: {0}' -f $m.status.value)
    $argsText = Hide-Secrets (@($m.status.args) -join ' ')
    Write-Output '    rendered child args:'
    foreach ($token in $argsText -split ' (?=--)') { Write-Output ("      {0}" -f $token) }
    if ($ShowPreset -and $m.status.preset) {
        Write-Output '    preset (as re-rendered, reserved keys removed):'
        foreach ($line in (Hide-Secrets ([string]$m.status.preset)) -split "`n") {
            if ($line.Trim()) { Write-Output ("      {0}" -f $line.TrimEnd()) }
        }
    }
}

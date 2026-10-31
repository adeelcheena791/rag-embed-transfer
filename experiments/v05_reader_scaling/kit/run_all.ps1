# Runs the whole Phase 4 experiment in the order of README_RUN.md (Steps 3-8, without the optional Gemini step).
# Start it in its own PowerShell window from the kit folder:
#   powershell -NoExit -ExecutionPolicy Bypass -File .\run_all.ps1
# Safe to stop and start again: finished work is skipped. Models already present are never pulled again.

$ErrorActionPreference = "Continue"   # native tools may write progress to stderr; exit codes are checked instead
Set-Location -Path $PSScriptRoot

function Step($label, [string[]]$cmd) {
    Write-Host ""
    Write-Host "=== $label ===" -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "STOPPED at: $label (exit code $LASTEXITCODE). Send a screenshot and results\log.txt to the author." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function EnsureModel($tag) {
    $have = (& ollama list) -join "`n"
    if ($have -notmatch [regex]::Escape($tag)) {
        Step "download $tag" @("ollama", "pull", $tag)
    } else {
        Write-Host "model $tag already present (not pulled again)"
    }
}

Step "check" @("python", "v05.py", "check")
if (-not (Test-Path "prompts\prompts_hp.ok")) { Step "prepare" @("python", "v05.py", "prepare") }

# ---- Part A (required) ----
$partA = @(
    @("qwen3:0.6b-q4_K_M", "core"),
    @("qwen3:1.7b-q4_K_M", "core"),
    @("qwen3:4b-q4_K_M", "core"),
    @("qwen3:8b-q4_K_M", "core"),
    @("qwen2.5:7b-instruct-q4_K_M", "bridge")
)
foreach ($r in $partA) { EnsureModel $r[0] }
EnsureModel "qwen3:14b-q4_K_M"
foreach ($r in $partA) { Step "run $($r[0]) --plan $($r[1])" @("python", "v05.py", "run", $r[0], "--plan", $r[1]) }
Step "determinism" @("python", "v05.py", "determinism")
Step "run qwen3:14b-q4_K_M --plan 14b" @("python", "v05.py", "run", "qwen3:14b-q4_K_M", "--plan", "14b")
Step "pack (Part A)" @("python", "v05.py", "pack")
Write-Host "PART A FINISHED. Send the newest v05_results_*.zip to the author now. Part B starts next." -ForegroundColor Green

# ---- Part B (second model family) ----
$partB = @(
    @("gemma3:1b-it-q4_K_M", "core"),
    @("gemma3:4b-it-q4_K_M", "core"),
    @("gemma3:12b-it-q4_K_M", "large")
)
foreach ($r in $partB) { EnsureModel $r[0] }
foreach ($r in $partB) { Step "run $($r[0]) --plan $($r[1])" @("python", "v05.py", "run", $r[0], "--plan", $r[1]) }
Step "pack (final)" @("python", "v05.py", "pack")
Write-Host "ALL FINISHED. Send the newest v05_results_*.zip to the author." -ForegroundColor Green

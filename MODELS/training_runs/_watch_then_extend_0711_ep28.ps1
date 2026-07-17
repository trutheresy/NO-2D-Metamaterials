# Wait for 0711 ep25 to finish, then resume through epoch 28.
$ErrorActionPreference = "Continue"
$py = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$root = "D:\Research\NO-2D-Metamaterials"
$run = "$root\MODELS\training_runs\NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_L1P1e-08_SS1_G9e-01_ch0u_260711"
$watchLog = "$root\MODELS\training_runs\nmae_l1_0711_watch_ep28.log"
$trainLog = "$root\MODELS\training_runs\nmae_l1_0711_extend_ep26-28.log"
$metrics = "$run\metrics.csv"
$trainPid = 155956

function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Add-Content -Path $watchLog -Value $line
    Write-Host $line
}

function EpochCount {
    if (-not (Test-Path $metrics)) { return 0 }
    return [Math]::Max(0, (Get-Content $metrics | Measure-Object -Line).Lines - 1)
}

function Ep25Done {
    $n = EpochCount
    if ($n -ge 25) { return $true }
    if (Test-Path "$run\train.log") {
        $tail = Get-Content "$run\train.log" -Tail 30 -ErrorAction SilentlyContinue
        if ($tail -match 'epoch=25/25' -and ($tail -match 'Run complete')) { return $true }
    }
    return $false
}

Log "Watcher started; waiting for ep25 (metrics=$(EpochCount), trainPid=$trainPid)"

while (-not (Ep25Done)) {
    $alive = $null -ne (Get-Process -Id $trainPid -ErrorAction SilentlyContinue)
    Log "waiting... metrics_epochs=$(EpochCount) train_alive=$alive gpu check skipped"
    if (-not $alive -and -not (Ep25Done)) {
        # Process died without writing ep25 — abort rather than resume from stale state
        Start-Sleep -Seconds 30
        if (-not (Ep25Done)) {
            Log "ERROR: train PID $trainPid exited before ep25 completed; not launching 26-28"
            exit 1
        }
    }
    Start-Sleep -Seconds 120
}

Log "ep25 complete (metrics=$(EpochCount)); launching epochs 26-28"

# Ensure no leftover trainer
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_from_disk' } |
    ForEach-Object { Log "WARNING: trainer still listed PID=$($_.ProcessId); waiting 60s"; Start-Sleep -Seconds 60 }

$argList = @(
  "$root\train_from_disk_fast.py",
  "--resume-run-dir", $run,
  "--extend-epochs", "3",
  "--loss", "nmae",
  "--l1-penalty", "1e-8",
  "--progress-mode", "plain",
  "--log-every-batches", "100",
  "--batch-size", "520",
  "--num-workers", "2",
  "--prefetch-factor", "3",
  "--hidden-channels", "128",
  "--layers", "4",
  "--modes-height", "32",
  "--modes-width", "32",
  "--learning-rate", "2e-3",
  "--weight-decay", "0",
  "--scheduler", "steplr",
  "--step-size", "1",
  "--gamma", "0.9",
  "--seed", "0",
  "--eigen-ch0-encoding", "uniform",
  "--diagnostic-panels",
  "--diagnostic-samples", "10"
)

$p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root `
    -RedirectStandardOutput $trainLog -RedirectStandardError "$trainLog.err" `
    -PassThru -WindowStyle Hidden
Log "Started ep26-28 PID=$($p.Id) log=$trainLog"
exit 0

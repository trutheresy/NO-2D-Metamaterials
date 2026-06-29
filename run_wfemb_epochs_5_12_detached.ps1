# Detached: L1 epochs 5-8 (same settings as epochs 1-4), then MSE epochs 9-12.
$ErrorActionPreference = "Stop"
$Python = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$Root = "D:\Research\NO-2D-Metamaterials"
$RunDir = "$Root\MODELS\training_runs\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619"
$Log = "$RunDir\train_epochs_5_12_detached.log"

# Matches resolved_config from epochs 1-4 (fresh L1 run).
$Common = @(
    "--resume-run-dir", $RunDir,
    "--progress-mode", "plain",
    "--log-every-batches", "200",
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
    "--amp", "none",
    "--eigen-ch0-encoding", "uniform"
)

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Add-Content -Path $Log -Value $line
}

Log "=== start epochs 5-8 L1 ==="
& $Python "$Root\train_from_disk.py" @Common --loss l1 --extend-epochs 4 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { Log "L1 phase failed exit=$LASTEXITCODE"; exit $LASTEXITCODE }

Log "=== start epochs 9-12 MSE (L2) ==="
& $Python "$Root\train_from_disk.py" @Common --loss mse --extend-epochs 4 2>&1 | Tee-Object -FilePath $Log -Append
$code = $LASTEXITCODE
Log "=== finished exit=$code ==="
exit $code

# Resume wfemb run: 4 more L1 epochs (5-8), then 4 MSE epochs (9-12).
$ErrorActionPreference = "Stop"
$Python = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$Root = "D:\Research\NO-2D-Metamaterials"
$RunDir = "$Root\MODELS\training_runs\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619"
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

Write-Host "=== Phase 1: L1 epochs 5-8 ==="
& $Python "$Root\train_from_disk.py" @Common --loss l1 --extend-epochs 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Phase 2: MSE (L2) epochs 9-12 ==="
& $Python "$Root\train_from_disk.py" @Common --loss mse --extend-epochs 4
exit $LASTEXITCODE

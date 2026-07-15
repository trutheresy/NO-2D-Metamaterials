# Regenerate boundary_length_vs_loss_by_band into *_test with larger fonts,
# slightly shorter figure, and title-case axis labels.
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_boundary_length_vs_loss_by_band_test.log"
Start-Transcript -Path $logPath -Append | Out-Null

$models = @(
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat",
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619_E11_best_MAEMSE",
    "NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626_E11",
    "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
)
$pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt"
$geo = Join-Path $pt "geometries_full.pt"
$tag = "b_test"
$losses = @("mae", "mse", "nmae", "nmse")

function Write-Log($msg) { Write-Output $msg }

function Run-Py($label, [string[]]$pyArgs) {
    Write-Log "`n>> $label START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PY @pyArgs 2>&1 | ForEach-Object { Write-Log $_ }
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($exit -ne 0) { throw "Step failed: $label (exit=$exit)" }
    Write-Log ">> $label DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Log "=== boundary_length_vs_loss_by_band_test regen ==="
if (-not (Test-Path -LiteralPath $geo)) { throw "Missing geometries: $geo" }

foreach ($MODEL in $models) {
    $pred = Join-Path $root "INFERENCE\$MODEL\$tag\predictions_I3O5_$MODEL.pt"
    if (-not (Test-Path -LiteralPath $pred)) {
        Write-Log "SKIP $MODEL (missing predictions)"
        continue
    }
    Write-Log "`n################ MODEL $MODEL ################"
    Run-Py "boundary_by_band_test_$MODEL" @(
        @(
            "scatter_loss_vs_boundary_by_band.py",
            "--dataset-pt-dir", $pt,
            "--inference", $pred,
            "--geometries", $geo,
            "--losses"
        ) + $losses + @(
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--output-subdir", "boundary_length_vs_loss_by_band_test",
            "--device", "cpu",
            "--larger-fonts",
            "--shorter"
        )
    )
}

Write-Log "`nALL_BOUNDARY_LENGTH_VS_LOSS_BY_BAND_TEST_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Stop-Transcript | Out-Null

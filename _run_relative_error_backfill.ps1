# Backfill per-pixel relative-error dataset outputs for completed inference runs.
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_relative_error_backfill.log"

$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)
$models = @(
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat",
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619_E11_best_MAEMSE",
    "NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626_E11"
)

$expectedRelFiles = @(
    "{0}_per_pixel_rel_error_stack.npy",
    "{0}_per_pixel_rel_error_mean_over_samples.npy",
    "{0}_per_pixel_rel_error_mean_over_samples.png",
    "{0}_per_pixel_rel_error_dataset.npz"
)

function Run-Py($label, [string[]]$pyArgs) {
    Write-Output "`n>> $label START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $PY @pyArgs 2>&1; $exit = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if ($exit -ne 0) { throw "Step failed: $label (exit=$exit)" }
    Write-Output ">> $label DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

function Test-RelativeErrorComplete($relDir, $tag) {
    if (-not (Test-Path -LiteralPath $relDir)) { return $false }
    foreach ($pattern in $expectedRelFiles) {
        $path = Join-Path $relDir ($pattern -f $tag)
        if (-not (Test-Path -LiteralPath $path)) { return $false }
    }
    return $true
}

function Write-Log($msg) {
    Write-Output $msg
    Add-Content -Path $logPath -Value $msg
}

Write-Log "=== relative_error_dataset backfill ==="

foreach ($MODEL in $models) {
    $INF = Join-Path $root "INFERENCE\$MODEL"
    foreach ($d in $datasets) {
        $tag = $d.tag
        $pt = $d.pt
        $pred = Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"
        $relDir = Join-Path $INF "$tag\relative_error_dataset"
        if (-not (Test-Path -LiteralPath $pred)) {
            Write-Log "SKIP $MODEL/$tag (missing predictions)"
            continue
        }
        if (Test-RelativeErrorComplete $relDir $tag) {
            Write-Log "SKIP $MODEL/$tag (relative_error_dataset complete)"
            continue
        }
        Run-Py "relative_error_${MODEL}_${tag}" @(
            "plot_per_pixel_relative_error.py",
            "--dataset-mode",
            "--dataset-pt-dir", $pt,
            "--predictions", $pred,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--tag", $tag
        )
    }
}

Write-Log "`n=== structure check ==="
$allModels = $models + @("NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best")
$baseFiles = @(
    "eigenvalue_data_full.pt",
    "eigenvalues_predictions_full.pt",
    "geometries_full.pt",
    "wavevectors_full.pt",
    "loss_comparison_{0}.csv",
    "predictions_I3O5_{1}.pt"
)
foreach ($MODEL in $allModels) {
    Write-Log "`nMODEL $MODEL"
    foreach ($tag in @("c_test", "b_test")) {
        $d = Join-Path $root "INFERENCE\$MODEL\$tag"
        if (-not (Test-Path -LiteralPath $d)) {
            Write-Log "  ${tag}: MISSING"
            continue
        }
        $missing = @()
        foreach ($pattern in $baseFiles) {
            $name = $pattern -f $tag, $MODEL
            if (-not (Test-Path -LiteralPath (Join-Path $d $name))) { $missing += $name }
        }
        $relDir = Join-Path $d "relative_error_dataset"
        $relOk = Test-RelativeErrorComplete $relDir $tag
        $status = if ($relOk) { "OK" } else { "INCOMPLETE" }
        Write-Log "  ${tag}: relative_error_dataset=$status missing=[$($missing -join ', ')]"
        if ($relOk) {
            Get-ChildItem $relDir -File | Sort-Object Name | ForEach-Object {
                Write-Log "    $($_.Name)"
            }
        }
    }
}

Write-Log "`nALL_RELATIVE_ERROR_BACKFILL_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

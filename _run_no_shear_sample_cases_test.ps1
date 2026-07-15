# Regenerate no-shear-mode percentile sample plots with paper styling into *_test folders.
# Reuses existing per_sample_loss_*.npy from *_no_shear_modes (does not recompute losses).
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_no_shear_sample_cases_test.log"
Start-Transcript -Path $logPath -Append | Out-Null

$MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
$INF = Join-Path $root "INFERENCE\$MODEL"
$PLOT = Join-Path $root "PLOTS\$MODEL"
$losses = @("mae", "mse", "rms", "nmae", "nmse", "nrms")
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)

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

Write-Log "=== no-shear sample cases paper-style test regen ==="
Write-Log "Model: $MODEL"
Write-Log "Output subdirs: <LOSS>_sample_case_plots_no_shear_modes_manuscript"

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $pred = Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"
    if (-not (Test-Path -LiteralPath $pred)) {
        Write-Log "SKIP $tag (missing predictions)"
        continue
    }

    foreach ($loss in $losses) {
        $srcSubdir = "$($loss.ToUpper())_sample_case_plots_no_shear_modes"
        $outSubdir = "${srcSubdir}_manuscript"
        $npy = Join-Path $PLOT "$tag\$srcSubdir\per_sample_loss_${loss}_$tag.npy"
        if (-not (Test-Path -LiteralPath $npy)) {
            throw "Missing loss array: $npy"
        }

        Write-Log "`n========== $tag sample_cases $loss -> $outSubdir =========="
        Run-Py "sample_cases_${loss}_no_shear_test_$tag" @(
            "plot_sample_cases.py",
            "--dataset-pt-dir", $pt,
            "--predictions", $pred,
            "--loss-array", $loss, $npy,
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--output-subdir", $outSubdir,
            "--exclude-shear-modes",
            "--no-show-eigfreq",
            "--no-title",
            "--paper-style"
        )
    }
}

Write-Log "`nALL_NO_SHEAR_SAMPLE_CASES_TEST_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Stop-Transcript | Out-Null

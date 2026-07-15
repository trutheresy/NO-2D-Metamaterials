# Displacement histograms + percentile sample cases excluding ky=0 / kx=0 shear-mode lines.
# Writes to *_no_shear_modes subfolders; does not overwrite standard pipeline outputs.
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_0705E12_no_shear_modes.log"
Start-Transcript -Path $logPath -Append | Out-Null

$MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
$INF = Join-Path $root "INFERENCE\$MODEL"
$PLOT = Join-Path $root "PLOTS\$MODEL"

$losses = @("mae", "mse", "rms", "nmae", "nmse", "nrms")
$histSubdir = "disp channel histograms_no_shear_modes"
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)

function Write-Log($msg) {
    Write-Output $msg
}

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

Write-Log "=== 0705 no-shear-modes displacement plots ==="
Write-Log "Model: $MODEL"
Write-Log "Excluding ky=0 row (25 waves) + kx=0 column (13 waves) = 37 wavevectors"

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $pred = Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"
    if (-not (Test-Path -LiteralPath $pred)) {
        Write-Log "SKIP $tag (missing predictions)"
        continue
    }

    Write-Log "`n========== $tag disp histograms (no shear modes) =========="
    Run-Py "hist_disp_no_shear_$tag" @(
        @(
            "plot_loss_histograms.py",
            "--dataset-pt-dir", $pt,
            "--inference", $pred,
            "--losses"
        ) + $losses + @(
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--output-subdir", $histSubdir,
            "--channel-group", "disp_ch",
            "--exclude-shear-modes"
        )
    )

    foreach ($loss in $losses) {
        $subdir = "$($loss.ToUpper())_sample_case_plots_no_shear_modes"
        Write-Log "`n========== $tag per_sample_loss $loss (no shear modes) =========="
        Run-Py "per_sample_loss_${loss}_no_shear_$tag" @(
            "per_sample_loss.py",
            "--dataset-pt-dir", $pt,
            "--inference", $pred,
            "--losses", $loss,
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--category", "plots",
            "--output-subdir", $subdir,
            "--exclude-shear-modes",
            "--device", "cpu",
            "--threads", "4"
        )

        $npy = Join-Path $PLOT "$tag\$subdir\per_sample_loss_${loss}_$tag.npy"
        if (-not (Test-Path -LiteralPath $npy)) {
            throw "Missing loss array: $npy"
        }

        Write-Log "`n========== $tag sample_cases $loss (no shear modes) =========="
        Run-Py "sample_cases_${loss}_no_shear_$tag" @(
            "plot_sample_cases.py",
            "--dataset-pt-dir", $pt,
            "--predictions", $pred,
            "--loss-array", $loss, $npy,
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--output-subdir", $subdir,
            "--no-show-eigfreq",
            "--no-title"
        )
    }
}

Write-Log "`nALL_NO_SHEAR_MODES_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Stop-Transcript | Out-Null

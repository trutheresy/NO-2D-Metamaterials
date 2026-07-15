# Regenerate dispersion_overlay_decoded_truth into *_manuscript with manuscript-leaning figure style:
# square aspect, legend at center-bottom, larger fonts, xlabel "IBZ Contour Wavevector".
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_dispersion_overlay_decoded_truth_manuscript.log"
Start-Transcript -Path $logPath -Append | Out-Null

$MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
$INF = Join-Path $root "INFERENCE\$MODEL"
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

Write-Log "=== dispersion_overlay_decoded_truth_manuscript regen ==="
Write-Log "Model: $MODEL"

foreach ($d in $datasets) {
    $tag = $d.tag
    $truthDir = Join-Path $root "INFERENCE\_decoded_truth\$tag"
    $pred = Join-Path $INF "$tag\eigenvalues_predictions_full.pt"
    if (-not (Test-Path -LiteralPath $truthDir)) { throw "Missing decoded truth: $truthDir" }
    if (-not (Test-Path -LiteralPath $pred)) { throw "Missing predictions: $pred" }

    Write-Log "`n========== $tag overlay_decoded_truth_manuscript =========="
    Run-Py "dispersion_overlay_decoded_truth_manuscript_$tag" @(
        "2d-dispersion-py\plot_dispersions_true_vs_pred.py",
        "--true", $truthDir,
        "--pred", $pred,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--output-subdir", "dispersion_overlay_decoded_truth_manuscript",
        "--square",
        "--larger-fonts",
        "--legend-loc", "lower center",
        "--xlabel", "IBZ Contour Wavevector"
    )
}

Write-Log "`nALL_DISPERSION_OVERLAY_DECODED_TRUTH_MANUSCRIPT_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Stop-Transcript | Out-Null

# Histograms + percentile sample cases excluding degenerate pivot wavevectors
# (ky=0, kx=0, TRIM / k≡-k including M). Writes to *_no_degenerate_pivot subfolders.
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$logPath = Join-Path $root "INFERENCE\_run_no_degenerate_pivot_plots.log"
Start-Transcript -Path $logPath -Append | Out-Null

$models = @(
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401_best_fno2d_compat",
    "NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619_E11_best_MAEMSE",
    "NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626_E11",
    "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
)
$losses = @("mae", "mse", "rms", "nmae", "nmse", "nrms")
$histGroups = @(
    @{ key = "all_ch";  subdir = "all channel histograms_no_degenerate_pivot" },
    @{ key = "disp_ch"; subdir = "disp channel histograms_no_degenerate_pivot" },
    @{ key = "freq_ch"; subdir = "freq channel histograms_no_degenerate_pivot" }
)
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)
$stageSubdir = "per_sample_loss_no_degenerate_pivot"

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

Write-Log "=== no_degenerate_pivot plots for prior inference runs ==="
Write-Log "Excluding ky=0 + kx=0 + TRIM (39 wavevectors)"
Write-Log "Suffix: *_no_degenerate_pivot"

foreach ($MODEL in $models) {
    $INF = Join-Path $root "INFERENCE\$MODEL"
    $PLOT = Join-Path $root "PLOTS\$MODEL"
    Write-Log "`n################ MODEL $MODEL ################"

    foreach ($d in $datasets) {
        $tag = $d.tag
        $pt = $d.pt
        $pred = Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"
        if (-not (Test-Path -LiteralPath $pred)) {
            Write-Log "SKIP $MODEL/$tag (missing predictions)"
            continue
        }

        Write-Log "`n========== $tag histograms =========="
        foreach ($g in $histGroups) {
            Run-Py "hist_$($g.key)_${MODEL}_$tag" @(
                @(
                    "plot_loss_histograms.py",
                    "--dataset-pt-dir", $pt,
                    "--inference", $pred,
                    "--losses"
                ) + $losses + @(
                    "--tag", $tag,
                    "--model-name", $MODEL,
                    "--dataset", $tag,
                    "--output-subdir", $g.subdir,
                    "--channel-group", $g.key,
                    "--exclude-degenerate-pivot-cases",
                    "--device", "cpu"
                )
            )
        }

        Write-Log "`n========== $tag per_sample_loss (all losses, one pass) =========="
        Run-Py "per_sample_loss_${MODEL}_$tag" @(
            @(
                "per_sample_loss.py",
                "--dataset-pt-dir", $pt,
                "--inference", $pred,
                "--losses"
            ) + $losses + @(
                "--tag", $tag,
                "--model-name", $MODEL,
                "--dataset", $tag,
                "--category", "plots",
                "--output-subdir", $stageSubdir,
                "--exclude-degenerate-pivot-cases",
                "--device", "cpu",
                "--threads", "4"
            )
        )

        foreach ($loss in $losses) {
            $caseSubdir = "$($loss.ToUpper())_sample_case_plots_no_degenerate_pivot"
            $stageNpy = Join-Path $PLOT "$tag\$stageSubdir\per_sample_loss_${loss}_$tag.npy"
            if (-not (Test-Path -LiteralPath $stageNpy)) {
                throw "Missing staged loss array: $stageNpy"
            }
            $caseDir = Join-Path $PLOT "$tag\$caseSubdir"
            New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
            Copy-Item -LiteralPath $stageNpy -Destination (Join-Path $caseDir (Split-Path $stageNpy -Leaf)) -Force

            Write-Log "`n========== $tag sample_cases $loss =========="
            Run-Py "sample_cases_${loss}_${MODEL}_$tag" @(
                "plot_sample_cases.py",
                "--dataset-pt-dir", $pt,
                "--predictions", $pred,
                "--loss-array", $loss, $stageNpy,
                "--tag", $tag,
                "--model-name", $MODEL,
                "--dataset", $tag,
                "--output-subdir", $caseSubdir,
                "--exclude-degenerate-pivot-cases",
                "--no-show-eigfreq",
                "--no-title"
            )
        }
    }
}

Write-Log "`nALL_NO_DEGENERATE_PIVOT_PLOTS_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Stop-Transcript | Out-Null

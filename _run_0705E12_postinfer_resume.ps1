# Resume post-inference from histograms (c_test per_sample_loss already complete).
$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"
$MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260705_E12_best"
$INF = Join-Path $root "INFERENCE\$MODEL"
$PLOT = Join-Path $root "PLOTS\$MODEL"
$losses = @("mae", "mse", "rms", "nmae", "nmse", "nrms")
$histGroups = @(
    @{ key = "all_ch"; subdir = "all channel histograms" },
    @{ key = "disp_ch"; subdir = "disp channel histograms" },
    @{ key = "freq_ch"; subdir = "freq channel histograms" }
)
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt"; skipPerSample = $true },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt"; skipPerSample = $false }
)

function Run-Py($label, [string[]]$pyArgs) {
    Write-Output "`n>> $label START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $PY @pyArgs 2>&1; $exit = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if ($exit -ne 0) { throw "Step failed: $label (exit=$exit)" }
    Write-Output ">> $label DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

foreach ($d in $datasets) {
    $tag = $d.tag; $pt = $d.pt
    $pred = Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"
    Write-Output "`n========== $tag (resume) =========="
    if (-not $d.skipPerSample) {
        foreach ($loss in $losses) {
            $subdir = "$($loss.ToUpper())_sample_case_plots"
            Run-Py "per_sample_loss_${loss}_$tag" @(
                "per_sample_loss.py", "--dataset-pt-dir", $pt, "--inference", $pred,
                "--losses", $loss, "--tag", $tag, "--model-name", $MODEL, "--dataset", $tag,
                "--category", "plots", "--output-subdir", $subdir, "--device", "cpu", "--threads", "4"
            )
        }
    }
    foreach ($g in $histGroups) {
        Run-Py "hist_$($g.key)_$tag" @(
            @("plot_loss_histograms.py", "--dataset-pt-dir", $pt, "--inference", $pred, "--losses") + $losses + @(
                "--tag", $tag, "--model-name", $MODEL, "--dataset", $tag,
                "--output-subdir", $g.subdir, "--channel-group", $g.key
            )
        )
    }
    foreach ($loss in $losses) {
        $subdir = "$($loss.ToUpper())_sample_case_plots"
        $npy = Join-Path $PLOT "$tag\$subdir\per_sample_loss_${loss}_$tag.npy"
        Run-Py "sample_cases_${loss}_$tag" @(
            "plot_sample_cases.py", "--dataset-pt-dir", $pt, "--predictions", $pred,
            "--loss-array", $loss, $npy, "--tag", $tag, "--model-name", $MODEL, "--dataset", $tag,
            "--output-subdir", $subdir, "--no-show-eigfreq", "--no-title"
        )
    }
    $nmaeNpy = Join-Path $PLOT "$tag\NMAE_sample_case_plots\per_sample_loss_nmae_$tag.npy"
    $nmseNpy = Join-Path $PLOT "$tag\NMSE_sample_case_plots\per_sample_loss_nmse_$tag.npy"
    Run-Py "high_loss_$tag" @(
        "plot_high_loss_samples.py", "--dataset-pt-dir", $pt, "--predictions", $pred,
        "--loss-array", "nmae", $nmaeNpy, "--loss-array", "nmse", $nmseNpy,
        "--tag", $tag, "--model-name", $MODEL, "--dataset", $tag,
        "--output-subdir", "high_loss_analysis", "--no-show-eigfreq"
    )
}

$bpt = ($datasets | Where-Object { $_.tag -eq "b_test" })[0]
$bpred = Join-Path $INF "b_test\predictions_I3O5_$MODEL.pt"
$bgeo = Join-Path $bpt.pt "geometries_full.pt"
Run-Py "scatter_boundary" @(
    @("scatter_loss_vs_boundary.py", "--dataset-pt-dir", $bpt.pt, "--inference", $bpred,
      "--geometries", $bgeo, "--losses") + $losses + @(
        "--tag", "b_test", "--model-name", $MODEL, "--dataset", "b_test",
        "--output-subdir", "boundary_length_vs_loss", "--device", "cpu"
    )
)
Run-Py "scatter_boundary_by_band" @(
    @("scatter_loss_vs_boundary_by_band.py", "--dataset-pt-dir", $bpt.pt, "--inference", $bpred,
      "--geometries", $bgeo, "--losses") + $losses + @(
        "--tag", "b_test", "--model-name", $MODEL, "--dataset", "b_test",
        "--output-subdir", "boundary_length_vs_loss_by_band", "--device", "cpu"
    )
)

$origBase = Join-Path $root "INFERENCE\_orig_npy"
foreach ($d in $datasets) {
    $tag = $d.tag; $pt = $d.pt
    $orig = Join-Path $origBase $tag
    New-Item -ItemType Directory -Force -Path $orig | Out-Null
    Run-Py "orig_npy_$tag" @("-c", "import torch,numpy as np; d=torch.load(r'$pt\eigenvalue_data_full.pt',map_location='cpu',weights_only=True); np.save(r'$orig\eigenvalue_data.npy', d.numpy()); print('wrote', d.shape)")
    Run-Py "dispersion_overlay_$tag" @(
        "2d-dispersion-py\plot_dispersions_true_vs_pred.py", "--true", $pt,
        "--pred", (Join-Path $INF "$tag\eigenvalues_predictions_full.pt"),
        "--model-name", $MODEL, "--dataset", $tag, "--output-subdir", "dispersion_overlay"
    )
    Run-Py "dispersion_plots_$tag" @(
        "2d-dispersion-py\plot_dispersion_infer_eigenfrequencies.py", $pt, $orig,
        "-n", "1000", "--no-infer", "--model-name", $MODEL, "--dataset", $tag, "--output-subdir", "dispersion_plots"
    )
    $designRaw = Join-Path $PLOT "$tag\dispersion_plots\design_raw"
    if (Test-Path -LiteralPath $designRaw) { Remove-Item -LiteralPath $designRaw -Recurse -Force }
    Run-Py "relative_error_$tag" @(
        "plot_per_pixel_relative_error.py",
        "--dataset-mode",
        "--dataset-pt-dir", $pt,
        "--predictions", (Join-Path $INF "$tag\predictions_I3O5_$MODEL.pt"),
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--tag", $tag
    )
}
Write-Output "`nALL_0705_POSTINFER_RESUME_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

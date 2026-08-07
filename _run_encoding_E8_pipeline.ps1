# Full inference + analysis pipeline for encoding ablation E8 checkpoints.
# Usage: edit $Job below, or call via _run_encoding_E8_both.ps1.
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$CkptName,
    [Parameter(Mandatory = $true)][string]$ModelName,
    [Parameter(Mandatory = $true)][ValidateSet("wavelet", "sinusoidal", "uniform", "constant")][string]$InputEncoding,
    [string]$IoCase = "",
    [string]$EigenEncoding = "uniform"
)

$ErrorActionPreference = "Stop"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"

$ckpt = Join-Path $RunDir $CkptName
if (-not (Test-Path -LiteralPath $ckpt)) { throw "Missing checkpoint: $ckpt" }

if (-not $IoCase) {
    $IoCase = if ($InputEncoding -in @("uniform", "constant")) { "I4O5" } else { "I3O5" }
}
$MODEL = $ModelName
$INF = Join-Path $root "INFERENCE\$MODEL"
$PLOT = Join-Path $root "PLOTS\$MODEL"

$losses = @("mae", "mse", "rms", "nmae", "nmse", "nrms")
$histGroups = @(
    @{ key = "all_ch"; subdir = "all channel histograms" },
    @{ key = "disp_ch"; subdir = "disp channel histograms" },
    @{ key = "freq_ch"; subdir = "freq channel histograms" }
)
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)
$refFiles = @("eigenvalue_data_full.pt", "geometries_full.pt", "wavevectors_full.pt")

function Run-Py($label, [string[]]$pyArgs) {
    Write-Output "`n>> $label START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Stringify ErrorRecords so tqdm/progress on stderr never becomes terminating.
        $output = & $PY @pyArgs 2>&1
        $exit = $LASTEXITCODE
        foreach ($line in $output) {
            if ($line -is [System.Management.Automation.ErrorRecord]) {
                Write-Output ($line.ToString())
            } else {
                Write-Output $line
            }
        }
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($exit -ne 0) { throw "Step failed: $label (exit=$exit)" }
    Write-Output ">> $label DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Output "=== Encoding E8 pipeline ==="
Write-Output "Checkpoint     : $ckpt"
Write-Output "Model folder   : $MODEL"
Write-Output "IO case        : $IoCase"
Write-Output "Input encoding : $InputEncoding"
Write-Output "Eigen encoding : $EigenEncoding"

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $outDir = Join-Path $INF $tag
    $pred = Join-Path $outDir "predictions_${IoCase}_$MODEL.pt"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Run-Py "infer_gpu_$tag" @(
        "run_model_inference_gpu.py",
        "--model_path", $ckpt,
        "--input_dataset_path", $pt,
        "--output_path", $pred,
        "--batch_size", "1024",
        "--input-encoding", $InputEncoding
    )

    foreach ($f in $refFiles) {
        $src = Join-Path $pt $f
        $dst = Join-Path $outDir $f
        if (-not (Test-Path -LiteralPath $src)) { throw "Missing reference file: $src" }
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
    Write-Output "Copied reference files into $outDir"
}

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $pred = Join-Path $INF "$tag\predictions_${IoCase}_$MODEL.pt"

    Run-Py "compare_$tag" @(
        "compare_inference_to_truth.py",
        "--predictions", $pred,
        "--dataset-pt-dir", $pt,
        "--eigen-encoding", $EigenEncoding,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--device", "cpu"
    )

    Run-Py "decode_$tag" @(
        "decode_predicted_eigenvalues.py",
        "--input-dir", (Join-Path $INF $tag),
        "--reference-pt-dir", $pt,
        "--eigen-encoding", $EigenEncoding
    )

    Run-Py "relative_error_$tag" @(
        "plot_per_pixel_relative_error.py",
        "--dataset-mode",
        "--dataset-pt-dir", $pt,
        "--predictions", $pred,
        "--eigen-encoding", $EigenEncoding,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--tag", $tag
    )

    Run-Py "second_peak_$tag" @(
        "analyze_second_peak_waves.py",
        "--dataset-pt-dir", $pt,
        "--predictions", $pred,
        "--eigen-encoding", $EigenEncoding,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--tag", $tag
    )
}

Run-Py "second_peak_ibz_map" @(
    "plot_ibz_second_peak_waves.py",
    "--model-name", $MODEL,
    "--datasets", "c_test", "b_test"
)

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $pred = Join-Path $INF "$tag\predictions_${IoCase}_$MODEL.pt"
    Write-Output "`n========== $tag loss/plots =========="

    foreach ($loss in $losses) {
        $subdir = "$($loss.ToUpper())_sample_case_plots"
        Run-Py "per_sample_loss_${loss}_$tag" @(
            @("per_sample_loss.py",
              "--dataset-pt-dir", $pt,
              "--inference", $pred,
              "--eigen-encoding", $EigenEncoding,
              "--losses", $loss,
              "--tag", $tag,
              "--model-name", $MODEL,
              "--dataset", $tag,
              "--category", "plots",
              "--output-subdir", $subdir,
              "--device", "cpu",
              "--threads", "4")
        )
    }

    foreach ($g in $histGroups) {
        Run-Py "hist_$($g.key)_$tag" (
            @("plot_loss_histograms.py",
              "--dataset-pt-dir", $pt,
              "--inference", $pred,
              "--eigen-encoding", $EigenEncoding,
              "--losses") + $losses + @(
              "--tag", $tag,
              "--model-name", $MODEL,
              "--dataset", $tag,
              "--output-subdir", $g.subdir,
              "--channel-group", $g.key)
        )
    }

    foreach ($loss in $losses) {
        $subdir = "$($loss.ToUpper())_sample_case_plots"
        $npy = Join-Path $PLOT "$tag\$subdir\per_sample_loss_${loss}_$tag.npy"
        Run-Py "sample_cases_${loss}_$tag" @(
            "plot_sample_cases.py",
            "--dataset-pt-dir", $pt,
            "--predictions", $pred,
            "--eigen-encoding", $EigenEncoding,
            "--input-encoding", $InputEncoding,
            "--loss-array", $loss, $npy,
            "--tag", $tag,
            "--model-name", $MODEL,
            "--dataset", $tag,
            "--output-subdir", $subdir,
            "--no-show-eigfreq",
            "--no-title"
        )
    }

    $nmaeNpy = Join-Path $PLOT "$tag\NMAE_sample_case_plots\per_sample_loss_nmae_$tag.npy"
    $nmseNpy = Join-Path $PLOT "$tag\NMSE_sample_case_plots\per_sample_loss_nmse_$tag.npy"
    Run-Py "high_loss_$tag" @(
        "plot_high_loss_samples.py",
        "--dataset-pt-dir", $pt,
        "--predictions", $pred,
        "--eigen-encoding", $EigenEncoding,
        "--input-encoding", $InputEncoding,
        "--loss-array", "nmae", $nmaeNpy,
        "--loss-array", "nmse", $nmseNpy,
        "--tag", $tag,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--output-subdir", "high_loss_analysis",
        "--no-show-eigfreq"
    )
}

$bpt = $datasets | Where-Object { $_.tag -eq "b_test" } | Select-Object -First 1
$bpred = Join-Path $INF "b_test\predictions_${IoCase}_$MODEL.pt"
$bgeo = Join-Path $bpt.pt "geometries_full.pt"

Run-Py "scatter_boundary_b_test" (
    @("scatter_loss_vs_boundary.py",
      "--dataset-pt-dir", $bpt.pt,
      "--inference", $bpred,
      "--geometries", $bgeo,
      "--eigen-encoding", $EigenEncoding,
      "--losses") + $losses + @(
      "--tag", "b_test",
      "--model-name", $MODEL,
      "--dataset", "b_test",
      "--output-subdir", "boundary_length_vs_loss",
      "--device", "cpu")
)

Run-Py "scatter_boundary_by_band_b_test" (
    @("scatter_loss_vs_boundary_by_band.py",
      "--dataset-pt-dir", $bpt.pt,
      "--inference", $bpred,
      "--geometries", $bgeo,
      "--eigen-encoding", $EigenEncoding,
      "--losses") + $losses + @(
      "--tag", "b_test",
      "--model-name", $MODEL,
      "--dataset", "b_test",
      "--output-subdir", "boundary_length_vs_loss_by_band",
      "--device", "cpu")
)

$origBase = Join-Path $root "INFERENCE\_orig_npy"
foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $orig = Join-Path $origBase $tag
    New-Item -ItemType Directory -Force -Path $orig | Out-Null
    Run-Py "orig_npy_$tag" @(
        "-c",
        "import torch,numpy as np; d=torch.load(r'$pt\eigenvalue_data_full.pt',map_location='cpu',weights_only=True); np.save(r'$orig\eigenvalue_data.npy', d.numpy()); print('wrote', d.shape)"
    )

    Run-Py "dispersion_overlay_$tag" @(
        "2d-dispersion-py\plot_dispersions_true_vs_pred.py",
        "--true", $pt,
        "--pred", (Join-Path $INF "$tag\eigenvalues_predictions_full.pt"),
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--output-subdir", "dispersion_overlay"
    )

    Run-Py "dispersion_plots_$tag" @(
        "2d-dispersion-py\plot_dispersion_infer_eigenfrequencies.py",
        $pt,
        $orig,
        "-n", "1000",
        "--no-infer",
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--output-subdir", "dispersion_plots"
    )

    $designRaw = Join-Path $PLOT "$tag\dispersion_plots\design_raw"
    if (Test-Path -LiteralPath $designRaw) {
        Remove-Item -LiteralPath $designRaw -Recurse -Force
    }
}

Write-Output "`nALL_ENCODING_E8_PIPELINE_DONE model=$MODEL $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

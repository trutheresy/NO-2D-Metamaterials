# Loss-only inference for 0711 E12 (MAE/MSE/NMAE/NMSE on c_test + b_test).
$ErrorActionPreference = "Continue"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$PY = "C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe"

$RunDir = Join-Path $root "MODELS\training_runs\NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_L1P1e-08_SS1_G9e-01_ch0u_260711"
$Ckpt = Join-Path $RunDir "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_L1P1e-08_SS1_G9e-01_ch0u_260711_E12.pth"
$MODEL = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_L1P1e-08_SS1_G9e-01_ch0u_260711_E12"
$INF = Join-Path $root "INFERENCE\$MODEL"
$EigenEncoding = "uniform"
$InputEncoding = "wavelet"
$IoCase = "I3O5"
$losses = @("mae", "mse", "nmae", "nmse")
$datasets = @(
    @{ tag = "c_test"; pt = "$root\DATASETS\c_test\continuous_2026-03-05_20-07-34_pt" },
    @{ tag = "b_test"; pt = "$root\DATASETS\b_test\binarized_2026-03-08_16-34-27_pt" }
)
$refFiles = @("eigenvalue_data_full.pt", "geometries_full.pt", "wavevectors_full.pt")

function Run-Py($label, [string[]]$pyArgs) {
    Write-Output "`n>> $label START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $output = & $PY @pyArgs 2>&1
    $exit = $LASTEXITCODE
    foreach ($line in $output) {
        if ($line -is [System.Management.Automation.ErrorRecord]) {
            Write-Output ($line.ToString())
        } else {
            Write-Output $line
        }
    }
    if ($exit -ne 0) { throw "Step failed: $label (exit=$exit)" }
    Write-Output ">> $label DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

if (-not (Test-Path -LiteralPath $Ckpt)) { throw "Missing checkpoint: $Ckpt" }
Write-Output "=== 0711 E12 loss-only pipeline ==="
Write-Output "Checkpoint: $Ckpt"
Write-Output "Model: $MODEL"

foreach ($d in $datasets) {
    $tag = $d.tag
    $pt = $d.pt
    $outDir = Join-Path $INF $tag
    $pred = Join-Path $outDir "predictions_${IoCase}_$MODEL.pt"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    if (-not (Test-Path -LiteralPath $pred)) {
        Run-Py "infer_gpu_$tag" @(
            "run_model_inference_gpu.py",
            "--model_path", $Ckpt,
            "--input_dataset_path", $pt,
            "--output_path", $pred,
            "--case", $IoCase,
            "--input-encoding", $InputEncoding
        )
    } else {
        Write-Output "Skip infer_gpu_$tag (exists): $pred"
    }

    foreach ($f in $refFiles) {
        $src = Join-Path $pt $f
        $dst = Join-Path $outDir $f
        if (-not (Test-Path -LiteralPath $dst)) {
            Copy-Item -LiteralPath $src -Destination $dst
        }
    }

    Run-Py "compare_$tag" @(
        "compare_inference_to_truth.py",
        "--predictions", $pred,
        "--dataset-pt-dir", $pt,
        "--eigen-encoding", $EigenEncoding,
        "--model-name", $MODEL,
        "--dataset", $tag,
        "--device", "cpu"
    )

    foreach ($loss in $losses) {
        $subdir = "$($loss.ToUpper())_sample_case_plots"
        Run-Py "per_sample_loss_${loss}_$tag" @(
            "per_sample_loss.py",
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
            "--threads", "4"
        )
    }
}

Write-Output "`nALL_0711_E12_LOSS_PIPELINE_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

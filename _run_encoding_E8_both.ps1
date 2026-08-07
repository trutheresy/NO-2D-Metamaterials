# Run full inference pipelines for sinusoidal E8 and uniform/constant E8, sequentially.
# Uses Start-Process redirects so Python/tqdm stderr cannot abort the parent under $ErrorActionPreference=Stop.
$ErrorActionPreference = "Continue"
$root = "d:\Research\NO-2D-Metamaterials"
Set-Location $root
$pipeline = Join-Path $root "_run_encoding_E8_pipeline.ps1"
$logRoot = Join-Path $root "INFERENCE"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$jobs = @(
    @{
        RunDir = "$root\MODELS\training_runs\NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_insin_260723"
        CkptName = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_insin_260723_E8.pth"
        ModelName = "NO_I3O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_insin_260723_E8"
        InputEncoding = "sinusoidal"
        IoCase = "I3O5"
        Log = "$logRoot\_run_insin_E8_pipeline.log"
        Err = "$logRoot\_run_insin_E8_pipeline.err.log"
    },
    @{
        RunDir = "$root\MODELS\training_runs\NO_I4O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_inconst_260725"
        CkptName = "NO_I4O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_inconst_260725_E8.pth"
        ModelName = "NO_I4O5_BCF16_NMAE_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_inconst_260725_E8"
        InputEncoding = "constant"
        IoCase = "I4O5"
        Log = "$logRoot\_run_inconst_E8_pipeline.log"
        Err = "$logRoot\_run_inconst_E8_pipeline.err.log"
    }
)

foreach ($j in $jobs) {
    Write-Output "`n################ $($j.ModelName) ################"
    Write-Output "Log: $($j.Log)"
    Write-Output "Err: $($j.Err)"
    if (Test-Path -LiteralPath $j.Log) { Remove-Item -LiteralPath $j.Log -Force }
    if (Test-Path -LiteralPath $j.Err) { Remove-Item -LiteralPath $j.Err -Force }

    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $pipeline,
        "-RunDir", $j.RunDir,
        "-CkptName", $j.CkptName,
        "-ModelName", $j.ModelName,
        "-InputEncoding", $j.InputEncoding,
        "-IoCase", $j.IoCase,
        "-EigenEncoding", "uniform"
    )
    $p = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $argList `
        -WorkingDirectory $root `
        -RedirectStandardOutput $j.Log `
        -RedirectStandardError $j.Err `
        -PassThru -Wait -NoNewWindow

    $code = $p.ExitCode
    if ($null -eq $code) { $code = -1 }
    if ($code -ne 0) {
        Write-Output "--- last 40 lines of $($j.Err) ---"
        if (Test-Path -LiteralPath $j.Err) {
            Get-Content -LiteralPath $j.Err -Tail 40 | ForEach-Object { Write-Output $_ }
        }
        throw "Pipeline failed for $($j.ModelName) (exit=$code). See $($j.Log) / $($j.Err)"
    }
    Write-Output "DONE $($j.ModelName)"
}

Write-Output "`nALL_BOTH_ENCODING_E8_PIPELINES_DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

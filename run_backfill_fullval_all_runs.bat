@echo off
setlocal
set PY=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set SCRIPT=D:\Research\NO-2D-Metamaterials\backfill_val_dual_loss.py
set RUNS=D:\Research\NO-2D-Metamaterials\MODELS\training_runs

echo === 260401 epochs 2-32 ===
"%PY%" "%SCRIPT%" --run-dir "%RUNS%\NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401" --mse-start-epoch 11 --epochs 2-32
if errorlevel 1 exit /b 1

echo === 260619 epochs 1-16 ===
"%PY%" "%SCRIPT%" --run-dir "%RUNS%\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619" --mse-start-epoch 9
if errorlevel 1 exit /b 2

echo === 260622 epochs 1-12 ===
"%PY%" "%SCRIPT%" --run-dir "%RUNS%\NO_I3O5_BCF16_L1_HC128_LR4e-03_WD0e+00_SS1_G8e-01_ch0u_260622"
if errorlevel 1 exit /b 3

echo === ALL DONE ===

echo === report + Huber training ===
call D:\Research\NO-2D-Metamaterials\run_after_backfill_report_and_huber.bat

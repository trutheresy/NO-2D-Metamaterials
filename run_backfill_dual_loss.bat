@echo off
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
cd /d %ROOT%

echo [%date% %time%] Starting backfill 260619 epochs 2-12 >> "%ROOT%\backfill_dual_loss.log"
"%PYTHON%" backfill_val_dual_loss.py --run-dir "MODELS/training_runs/NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619" --epochs 2-12 --mse-start-epoch 9 >> "%ROOT%\backfill_dual_loss.log" 2>&1

echo [%date% %time%] Starting backfill 260401 all epochs >> "%ROOT%\backfill_dual_loss.log"
"%PYTHON%" backfill_val_dual_loss.py --run-dir "MODELS/training_runs/NO_I3O5_BCF16_L1&L2_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260401" --mse-start-epoch 11 >> "%ROOT%\backfill_dual_loss.log" 2>&1

echo [%date% %time%] Backfill complete >> "%ROOT%\backfill_dual_loss.log"

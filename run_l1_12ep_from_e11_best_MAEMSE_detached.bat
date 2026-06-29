@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set INIT=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619_E11_best_MAEMSE.pth
set LOG=%ROOT%\MODELS\training_runs\run_l1_12ep_LR4e-03_G8e-01_from_e11_best_MAEMSE_detached.log

echo %date% %time% ^| === start new L1 12-epoch run from E11 best_MAEMSE === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --loss l1 --epochs 12 ^
  --init-checkpoint "%INIT%" ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 4e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.8 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

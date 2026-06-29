@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set RUNDIR=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_L1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260619
set LOG=%RUNDIR%\train_epochs_5_12_detached.log

echo %date% %time% ^| === start epochs 5-8 L1 === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --resume-run-dir "%RUNDIR%" ^
  --loss l1 --extend-epochs 4 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 2e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.9 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
if errorlevel 1 (
  echo %date% %time% ^| L1 phase failed >> "%LOG%"
  exit /b 1
)

echo %date% %time% ^| === start epochs 9-12 MSE === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --resume-run-dir "%RUNDIR%" ^
  --loss mse --extend-epochs 4 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 2e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.9 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

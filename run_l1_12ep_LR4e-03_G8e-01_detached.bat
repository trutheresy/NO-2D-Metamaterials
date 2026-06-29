@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set LOG=%ROOT%\MODELS\training_runs\run_l1_12ep_LR4e-03_G8e-01_detached.log

echo %date% %time% ^| === start new L1 12-epoch run (random init) === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --loss l1 --epochs 12 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 4e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.8 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

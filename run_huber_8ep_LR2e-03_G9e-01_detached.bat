@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set LOG=%ROOT%\MODELS\training_runs\run_huber_8ep_LR2e-03_G9e-01_detached.log

echo %date% %time% ^| === start Huber/SmoothL1 8-epoch run (random init, same as 260619 except loss) === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --loss smoothl1 ^
  --epochs 8 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 2e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.9 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

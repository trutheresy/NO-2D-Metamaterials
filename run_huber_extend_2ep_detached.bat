@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set RUN_DIR=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626
set LOG=%ROOT%\MODELS\training_runs\run_huber_extend_2ep_detached.log

echo %date% %time% ^| === resume Huber 260626: +2 epochs (15-16) === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --resume-run-dir "%RUN_DIR%" ^
  --extend-epochs 2 ^
  --loss smoothl1 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 2e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.9 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform ^
  --diagnostic-panels --diagnostic-samples 10 >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

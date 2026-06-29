@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set RUNDIR=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_L1_HC128_LR4e-03_WD0e+00_SS1_G8e-01_ch0u_260622
set LOG=%RUNDIR%\train_resume_ep11_reset_lr_detached.log

echo %date% %time% ^| === resume from E10 checkpoint, L1, reset lr/gamma to e1 === >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --resume-run-dir "%RUNDIR%" ^
  --resume-from-epoch 10 ^
  --loss l1 --extend-epochs 2 ^
  --reset-optimizer-scheduler ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --learning-rate 4e-3 --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.8 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

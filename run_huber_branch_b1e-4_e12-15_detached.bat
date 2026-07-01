@echo off
setlocal
set PYTHON=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set SRC=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260626
set OUT=%ROOT%\MODELS\training_runs\NO_I3O5_BCF16_SL1_HC128_LR2e-03_WD0e+00_SS1_G9e-01_ch0u_260630
set LOG=%ROOT%\MODELS\training_runs\run_huber_branch_b1e-4_e12-15_detached.log

if exist "%OUT%" (
  echo %date% %time% ^| ERROR: output run dir already exists: %OUT% >> "%LOG%"
  exit /b 1
)

echo %date% %time% ^| === branch from 0626 E11, beta=1e-4, ep 12-15, LR=epoch-12 (5.649e-4) === >> "%LOG%"
echo %date% %time% ^| prior run (epochs 1-11): %SRC% >> "%LOG%"
"%PYTHON%" "%ROOT%\train_from_disk.py" ^
  --resume-run-dir "%SRC%" ^
  --output-run-dir "%OUT%" ^
  --resume-from-epoch 11 ^
  --extend-epochs 4 ^
  --loss smoothl1 ^
  --huber-beta 1e-4 ^
  --learning-rate 5.64859072962e-4 ^
  --progress-mode plain --log-every-batches 200 ^
  --batch-size 520 --num-workers 2 --prefetch-factor 3 ^
  --hidden-channels 128 --layers 4 --modes-height 32 --modes-width 32 ^
  --weight-decay 0 ^
  --scheduler steplr --step-size 1 --gamma 0.9 ^
  --seed 0 --amp none --eigen-ch0-encoding uniform ^
  --diagnostic-panels --diagnostic-samples 10 >> "%LOG%" 2>&1
echo %date% %time% ^| === finished exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

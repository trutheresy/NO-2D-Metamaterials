@echo off
setlocal
set PY=C:\ProgramData\anaconda3\envs\NO_2D_Metamaterials\python.exe
set ROOT=D:\Research\NO-2D-Metamaterials
set LOG=%ROOT%\MODELS\training_runs\after_backfill_report_and_huber.log

echo %date% %time% ^| === full-val metrics report === >> "%LOG%"
"%PY%" "%ROOT%\report_fullval_metrics_tables.py" >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

echo %date% %time% ^| === start Huber training === >> "%LOG%"
call "%ROOT%\run_huber_8ep_LR2e-03_G9e-01_detached.bat" >> "%LOG%" 2>&1
echo %date% %time% ^| === after_backfill pipeline done exit=%errorlevel% === >> "%LOG%"
exit /b %errorlevel%

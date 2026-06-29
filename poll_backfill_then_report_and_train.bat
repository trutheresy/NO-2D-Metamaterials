@echo off
setlocal
set ROOT=D:\Research\NO-2D-Metamaterials
set LOG=%ROOT%\MODELS\training_runs\backfill_fullval_all_runs.log
set POLL=%ROOT%\MODELS\training_runs\poll_backfill_then_report_and_train.log

echo %date% %time% ^| polling for backfill completion... >> "%POLL%"
:wait_loop
findstr /C:"=== ALL DONE ===" "%LOG%" >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait_loop
)
echo %date% %time% ^| backfill complete, starting report + Huber train >> "%POLL%"
call "%ROOT%\run_after_backfill_report_and_huber.bat"
echo %date% %time% ^| poll pipeline exit=%errorlevel% >> "%POLL%"
exit /b %errorlevel%

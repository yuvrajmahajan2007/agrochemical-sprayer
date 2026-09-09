@echo off
REM AI-Driven Precision Agrochemical Sprayer - launch training detached (logs to logs\training.log)
REM Uses %~dp0 (this script's folder) as the project root.
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
set "SCRIPT=%ROOT%training\train_model.py"
set "LOG=%ROOT%logs\training.log"
set "ERR=%ROOT%logs\training.err"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
cd /d "%ROOT%"
>"%LOG%" 2>"%ERR%" "%VENV%" "%SCRIPT%" --epochs 14 --batch-size 24

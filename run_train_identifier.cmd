@echo off
REM AI-Driven Precision Agrochemical Sprayer - train plant identifier detached
REM Uses %~dp0 (this script's folder) as the project root.
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
set "APP=%ROOT%training\train_plant_identifier.py"
set "LOG=%ROOT%logs\train_identifier.log"
set "ERR=%ROOT%logs\train_identifier.err"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
cd /d "%ROOT%"
>"%LOG%" 2>"%ERR%" "%VENV%" "%APP%" --epochs 14 --batch-size 24

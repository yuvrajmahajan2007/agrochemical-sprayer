@echo off
REM AI-Driven Precision Agrochemical Sprayer - train Real Plant/Leaf vs Non-Plant validator
REM Builds the real dataset, trains a real MobileNetV2, evaluates + picks the threshold.
REM Uses %~dp0 (this script's folder) as the project root.
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
cd /d "%ROOT%"
"%VENV%" -u "%ROOT%training\build_validator_dataset.py" --include-mobile
if %errorlevel% neq 0 exit /b %errorlevel%
"%VENV%" -u "%ROOT%training\train_validator.py" --epochs 15 --batch-size 32
if %errorlevel% neq 0 exit /b %errorlevel%
"%VENV%" -u "%ROOT%training\evaluate_validator.py"

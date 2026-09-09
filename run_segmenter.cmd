@echo off
REM AI-Driven Precision Agrochemical Sprayer - train the Real leaf-health U-Net segmenter
REM Builds real pixel masks from real leaf photos, then trains the 3-class U-Net.
REM Uses %~dp0 (this script's folder) as the project root.
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
cd /d "%ROOT%"
"%VENV%" -u "%ROOT%training\build_segmentation_masks.py" --n 1500
if %errorlevel% neq 0 exit /b %errorlevel%
"%VENV%" -u "%ROOT%training\train_segmenter.py" --epochs 12 --batch-size 16 --img-size 128

@echo off
REM AI-Driven Precision Agrochemical Sprayer - launch Flask backend (foreground)
REM Works from anywhere: uses %~dp0 (the folder this script lives in) as the project root.
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
set "APP=%ROOT%backend\app.py"
"%VENV%" --version >nul 2>&1 || (
  echo ERROR: virtual env not found at %VENV%
  echo Create it first:  python -m venv venv
  echo Then:             pip install -r requirements.txt
  exit /b 1
)
cd /d "%ROOT%"
"%VENV%" "%APP%"

@echo off
REM ================================================================
REM AI-Driven Precision Agrochemical Sprayer - 24/7 Host Launcher
REM
REM - Runs Flask backend as a persistent process
REM - Auto-restarts on crash (10s delay)
REM - Logs stdout/stderr to logs/ with timestamps
REM - Designed for Windows Task Scheduler (runs at system startup)
REM
REM Uses %~dp0 (this script's folder) as the project root so it works
REM after a git clone anywhere on the machine.
REM ================================================================
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%venv\Scripts\python.exe"
set "APP=%ROOT%backend\app.py"
set "LOGDIR=%ROOT%logs"
set "ERRLOG=%LOGDIR%\service.err"
set "LOGFILE=%LOGDIR%\service.log"
set "LOCK=%LOGDIR%\service.lock"

REM ensure logs dir
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM prevent duplicate instances
if exist "%LOCK%" (
    set /p LOCKPID=<"%LOCK%"
    tasklist /FI "PID eq %LOCKPID%" 2>nul | find /i "%LOCKPID%" >nul
    if not errorlevel 1 (
        echo [%date% %time%] Another instance running (PID %LOCKPID%). Exiting. >> "%ERRLOG%"
        exit /b 0
    )
)

REM write own PID as lock
echo %~dp0 > "%LOCK%"

echo [%date% %time%] ==================================================== >> "%LOGFILE%"
echo [%date% %time%] 24/7 SERVICE STARTING >> "%LOGFILE%"
echo [%date% %time%] ==================================================== >> "%LOGFILE%"

set RESTART_COUNT=0

:RESTART
set /a RESTART_COUNT+=1
echo [%date% %time%] Starting backend (attempt #%RESTART_COUNT%)... >> "%LOGFILE%"

REM kill any stale python on port 5000
for /f "tokens=5" %%p in ('netstat -aon ^| findstr :5000 ^| findstr LISTENING') do taskkill /PID %%p /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM run Flask (no debug, no reloader)
cd /d "%ROOT%"
"%VENV%" -u "%APP%" >> "%LOGFILE%" 2>> "%ERRLOG%"
set EXITCODE=%ERRORLEVEL%

echo [%date% %time%] Backend exited with code %EXITCODE%. Restarting in 10s... >> "%LOGFILE%"
timeout /t 10 /nobreak >nul
goto RESTART

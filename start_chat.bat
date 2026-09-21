@echo off
chcp 65001 >nul
echo ============================================
echo   AgentRL Chat Server - Starting...
echo ============================================

:: Try to find Python
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python
    goto :start_server
)
where python3 >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python3
    goto :start_server
)
where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=py
    goto :start_server
)

echo [ERROR] Python not found! Please install Python 3.8+ and add to PATH.
echo   Download: https://www.python.org/downloads/
pause
exit /b 1

:start_server
echo Using: %PYTHON_CMD%
echo Starting server on http://127.0.0.1:5000 ...
start "AgentRL Chat Server" /min %PYTHON_CMD% -X utf8 -m uvicorn chat_server:app --host 127.0.0.1 --port 5000
timeout /t 4 >nul
echo Opening browser...
start "" "http://127.0.0.1:5000/chat.html"
echo Done! Server is running in background.

@echo off
:: eScalator launcher — Windows (version: see VERSION file)
cd /d "%~dp0"
setlocal enabledelayedexpansion
set "APP_VERSION=?"
if exist VERSION set /p APP_VERSION=<VERSION

set VENV=.venv
set VENV_PYTHON=%VENV%\Scripts\python.exe

echo =====================================================
echo   eScalator v%APP_VERSION%
echo =====================================================
echo.

:: Find Python 3.10+ — use py launcher if available (version-agnostic),
:: otherwise fall back to 'python' / 'python3' in PATH.
set PYTHON=

:: Try py launcher: iterate its listed versions, pick first that is 3.10+
py --list >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=1" %%V in ('py --list 2^>nul') do (
        if not defined PYTHON (
            py %%V -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
            if not errorlevel 1 set PYTHON=py %%V
        )
    )
)

:: Fallback: 'python' in PATH
if not defined PYTHON (
    python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set PYTHON=python
)

:: Fallback: 'python3' in PATH
if not defined PYTHON (
    python3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set PYTHON=python3
)

if not defined PYTHON (
    echo [ERROR] Python 3.10 or newer not found.
    echo.
    echo   Detected versions:
    py --list 2>nul || echo   ^(Windows py launcher not available^)
    echo.
    echo   If you just installed Python, open a NEW terminal and try again.
    echo   Download Python 3.12: https://www.python.org/downloads/
    echo   During install, check "Add Python to PATH".
    pause & exit /b 1
)

echo [*] Using %PYTHON%

:: Create venv if missing
if not exist "%VENV_PYTHON%" (
    echo [*] Creating virtual environment...
    %PYTHON% -m venv %VENV%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo.
)

:: Always verify base dependencies (catches missing packages in existing venvs)
"%VENV_PYTHON%" -c "import fastapi,uvicorn,multipart,webview,PIL,numpy,packaging,scipy" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing missing base dependencies...
    "%VENV_PYTHON%" -m pip install -q --upgrade pip
    "%VENV_PYTHON%" -m pip install --no-cache-dir --progress-bar on -r requirements-base.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check your internet connection.
        pause & exit /b 1
    )
    echo.
)

echo [*] Launching eScalator...
echo     Missing packages? Open the Settings tab inside the app.
echo.
"%VENV_PYTHON%" app.py
if errorlevel 1 (
    echo.
    echo [ERROR] eScalator exited with an error. Check the output above.
)
pause

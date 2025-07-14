@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM === Setup ===
SET "InputFile=%~1"
IF "%InputFile%"=="" SET "InputFile=example.py"

REM Ensure we're in the root directory (adjust if needed)
CD /D "%~dp0"

echo Installing 'code_analyser' in editable mode...
pip install -e . >nul 2>&1
IF ERRORLEVEL 1 (
    echo Failed to install package in editable mode.
    EXIT /B 1
)

REM Dynamically get Python Scripts path (works for virtual env or global)
FOR /F "delims=" %%I IN ('python -c "import sysconfig; print(sysconfig.get_path('scripts'))"') DO SET "SCRIPT_PATH=%%I"

SET "EXE_PATH=%SCRIPT_PATH%\ast-metrics.exe"

echo Running analysis on %InputFile%...
IF EXIST "%EXE_PATH%" (
    "%EXE_PATH%" --file "%InputFile%" --out metrics.json --verbose
) ELSE (
    echo ast-metrics.exe not found. Falling back to python -m metrics.main...
    python -m metrics.main --file "%InputFile%" --out metrics.json --verbose
)

IF EXIST metrics.json (
    echo Opening metrics.json in Notepad...
    notepad metrics.json
) ELSE (
    echo ERROR: metrics.json not created.
)

ENDLOCAL

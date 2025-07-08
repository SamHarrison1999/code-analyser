@echo off
SETLOCAL

REM Get input argument (file to analyse)
SET "InputFile=%~1"
IF "%InputFile%"=="" SET "InputFile=example.py"

echo Installing 'code-analyser' in editable mode...
pip install -e .

REM Set the script path (adjust Python version as needed)
SET "SCRIPT_PATH=%APPDATA%\Python\Python313\Scripts"

echo Running ast-metrics on %InputFile%...
"%SCRIPT_PATH%\ast-metrics.exe" --file "%InputFile%" --out metrics.json --verbose

echo Opening metrics.json in Notepad...
notepad metrics.json

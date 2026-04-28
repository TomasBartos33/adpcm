@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    py -3 -m venv venv
    if errorlevel 1 (
        python -m venv venv
    )
)

call venv\Scripts\activate.bat

python -c "import PyQt6, numpy, matplotlib, scipy" >nul 2>nul
if errorlevel 1 (
    pip install -r requirements.txt
)

python -m adpcm_py.main %*

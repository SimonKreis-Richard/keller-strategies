@echo off
rem Keller Strategies — double-click launcher.
rem First run creates the venv and installs dependencies; later runs start instantly.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo First run: creating virtual environment and installing dependencies...
    py -3 -m venv venv || python -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    rem Optional: native desktop window (falls back to browser without it)
    venv\Scripts\python.exe -m pip install pywebview
)

venv\Scripts\python.exe app.py
if errorlevel 1 pause

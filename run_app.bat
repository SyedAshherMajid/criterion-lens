@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe was not found.
    echo Create the project environment and install the app dependencies first.
    exit /b 1
)

".venv\Scripts\python.exe" -m app %*

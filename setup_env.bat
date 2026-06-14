@echo off
REM Setup virtual environment and install project in editable mode

SET "VENV_DIR=.venv"

REM Allow passing a python executable as first arg
IF NOT "%1"=="" (
    SET "PYEXE=%~1"
) ELSE (
    SET "PYEXE=python"
)

REM Check Python availability
%PYEXE% --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python was not found. Please install Python 3.10+ and retry.
    exit /b 1
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment already exists; skipping creation.
) else (
    echo Creating virtual environment in %VENV_DIR%...
    %PYEXE% -m venv "%VENV_DIR%"
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

REM Do not rely on activating the venv in this script (shell-specific).
echo Using venv python executable to run pip commands so installation is reproducible.
echo Upgrading pip, setuptools and wheel inside the venv...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

echo Installing project in editable mode into the venv...
"%VENV_DIR%\Scripts\python.exe" -m pip install -e .

echo Installation complete.
echo To activate the venv in this shell, run:
echo     %VENV_DIR%\Scripts\activate.bat
echo To run tests inside the venv:
echo     %VENV_DIR%\Scripts\python.exe -m pytest

exit /b 0

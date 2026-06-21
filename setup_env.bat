@echo off
REM Setup virtual environment, install the project, and activate the venv

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

REM Use the venv Python directly for installation so this is reproducible.
echo Using venv python executable to run pip commands so installation is reproducible.
echo Upgrading pip, setuptools and wheel inside the venv...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

echo Installing project in editable mode into the venv...
"%VENV_DIR%\Scripts\python.exe" -m pip install --no-build-isolation -e .[dev]
IF ERRORLEVEL 1 (
    echo Failed to install project into the virtual environment.
    exit /b 1
)

echo Activating the virtual environment in this shell...
call "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Installation complete.
echo The virtual environment is now active.
echo To run an example inside the venv, use:
echo     python examples.py render_api
echo To run tests inside the venv:
echo     python -m pytest

exit /b 0

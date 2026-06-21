@echo off
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
set "PYTEST_BASETEMP=%TEMP%\sdf_ui_pytest"

%PYTHON% -m ruff format src
%PYTHON% -m sphinx -M html docs docs\_build
%PYTHON% -m pytest --basetemp="%PYTEST_BASETEMP%"

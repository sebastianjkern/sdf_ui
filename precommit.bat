@echo off
call "./setup_env.bat"

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
set "PYTEST_BASETEMP=.pytest"

%PYTHON% -m ruff format src build_tools setup.py
%PYTHON% -m ruff check src build_tools setup.py
if exist ".venv\Scripts\mypy.exe" %PYTHON% -m mypy src build_tools
%PYTHON% -m sphinx -M html docs docs\_build
%PYTHON% -m pytest --basetemp="%PYTEST_BASETEMP%"

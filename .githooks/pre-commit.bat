@echo off
:: Windows hook that calls the repository precommit.bat
setlocal enabledelayedexpansion
set HOOK_DIR=%~dp0
:: remove trailing backslash
if "%HOOK_DIR:~-1%"=="\" set HOOK_DIR=%HOOK_DIR:~0,-1%
:: REPO_ROOT is parent of hooks dir
for %%I in ("%HOOK_DIR%\..") do set REPO_ROOT=%%~fI
call "%REPO_ROOT%\precommit.bat"
if errorlevel 1 exit /b 1
exit /b 0
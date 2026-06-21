@echo off
git config core.hooksPath .githooks
if errorlevel 1 exit /b 1
echo core.hooksPath set to .githooks
exit /b 0
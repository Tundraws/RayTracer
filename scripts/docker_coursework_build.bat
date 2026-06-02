@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0docker_coursework_build.ps1" %*
exit /b %ERRORLEVEL%

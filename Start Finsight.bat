@echo off
setlocal
cd /d "%~dp0"
rem Prefer a supported Python if several versions are installed.
where py >nul 2>nul
if errorlevel 1 goto use_python
for %%V in (3.13 3.12 3.11) do (
    py -%%V --version >nul 2>nul
    if not errorlevel 1 (
        py -%%V "%~dp0launch.py" %*
        if errorlevel 1 goto launch_error
        exit /b 0
    )
)
py -3 "%~dp0launch.py" %*
if errorlevel 1 goto launch_error
exit /b 0

:use_python
where python >nul 2>nul
if errorlevel 1 goto missing_python
python "%~dp0launch.py" %*
if errorlevel 1 goto launch_error
exit /b 0

:missing_python
echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
echo Enable "Add Python to PATH" if offered, then open this launcher again.
echo Extract the entire Finsight ZIP before launching.
pause
exit /b 1

:launch_error
echo.
echo Read the error above. Setup help is in docs\INSTALL.md.
pause
exit /b 1

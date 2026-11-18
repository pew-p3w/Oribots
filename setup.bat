@echo off
rem Create .venv with Python 3.11 and install the pinned dependencies (Windows).
rem
rem   setup.bat
rem
rem Needs the Python 3.11 launcher entry: install it with `py install 3.11` if
rem `py -3.11 --version` fails. `noise` is skipped on Windows (see requirements.txt).
setlocal
cd /d "%~dp0"

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is not installed. Run:  py install 3.11
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv with Python 3.11 ...
    py -3.11 -m venv .venv || exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -c constraints.txt -r requirements.txt || exit /b 1

echo.
echo Setup done. Next:  tests\run_tests.bat
endlocal

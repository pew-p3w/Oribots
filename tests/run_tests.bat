@echo off
rem Run the test suite on Windows (the Makefile's `make check`).
rem
rem   tests\run_tests.bat            every check, including the ~30 s training workflow
rem   tests\run_tests.bat quick      every check except the training workflow
rem   tests\run_tests.bat physics    one check by name (tests\test_physics.py)
rem   tests\run_tests.bat brain ea   several checks by name
rem
rem Prints PASS/FAIL per check and a summary; exits 1 if any check failed.
rem Output of each check is kept in tests\logs\<name>.log.
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No .venv found. Run setup.bat first.
    exit /b 1
)

rem Same order as `make check`; new checks go after the originals.
set "ALL=environment bodies brain genotype evaluator head configs ea cmaes run jobs pbs bench physics fast_trial workflow"
set "QUICK=environment bodies brain genotype evaluator head configs ea cmaes run jobs pbs bench physics fast_trial"

if "%~1"=="" (
    set "CHECKS=%ALL%"
) else if /i "%~1"=="quick" (
    set "CHECKS=%QUICK%"
) else (
    set "CHECKS=%*"
)

if not exist "tests\logs" mkdir "tests\logs"
set /a PASSED=0
set /a FAILED=0
set "FAILED_NAMES="

for %%T in (%CHECKS%) do (
    if not exist "tests\test_%%T.py" (
        echo   ????  %%T  ^(no tests\test_%%T.py^)
        set /a FAILED+=1
        set "FAILED_NAMES=!FAILED_NAMES! %%T"
    ) else (
        "%PY%" "tests\test_%%T.py" > "tests\logs\%%T.log" 2>&1
        if errorlevel 1 (
            echo   FAIL  %%T   ^(see tests\logs\%%T.log^)
            set /a FAILED+=1
            set "FAILED_NAMES=!FAILED_NAMES! %%T"
        ) else (
            echo   PASS  %%T
            set /a PASSED+=1
        )
    )
)

echo.
echo %PASSED% passed, %FAILED% failed.
if %FAILED% gtr 0 (
    echo Failed:%FAILED_NAMES%
    exit /b 1
)
endlocal
exit /b 0

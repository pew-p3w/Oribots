@echo off
rem Train the same config twice and compare: the original code and physics
rem (RK4 1 ms) against both speed-ups (fast trial loop + implicitfast 5 ms).
rem Both runs share the config's SEED, so they start identically and train on
rem the same balls. Both best controllers are then re-scored under the original
rem physics on the training balls and on 8 unseen balls.
rem
rem   bench\compare_training.bat                            spider in the walled arena
rem   bench\compare_training.bat config\other_cmaes.py      another CMA-ES config (needs SEED)
rem
rem Results: output\speed_compare_<stamp>\report.txt, plus both runs' snapshots
rem (watch one with: .venv\Scripts\python run.py -t output\speed_compare_<stamp>\fast\gen30.pkl)
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No .venv found. Run setup.bat first.
    exit /b 1
)
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config\spider_arena_cmaes.py"
"%PY%" bench\compare_training.py --config "%CONFIG%"
endlocal

@echo off
rem Benchmark the speed changes against the original code and physics.
rem
rem   bench\run_bench.bat                 full sweep (several minutes, uses all workers)
rem   bench\run_bench.bat quick           tiny sweep to check everything runs (~1 minute)
rem   bench\run_bench.bat code            only the code changes (fast trial + kept workers),
rem                                       same physics: the results must be identical
rem   bench\run_bench.bat breakdown       only the per-trial time breakdown
rem
rem Environment overrides (optional):
rem   set BENCH_CONFIG=config/spider_cmaes.py
rem   set BENCH_WORKERS=8          simulator processes (default: what the config picks)
rem   set BENCH_CANDIDATES=24
rem   set BENCH_BALLS=2
rem   set BENCH_SIMTIME=60         simulated seconds per trial
rem   set BENCH_GENERATIONS=3      repeats of the generation (workers kept between them)
rem
rem Results: bench\results\<label>.json; the comparison tables are printed and
rem saved to bench\results\summary.txt (physics) and bench\results\code.txt.
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No .venv found. Run setup.bat first.
    exit /b 1
)

if not defined BENCH_CONFIG set "BENCH_CONFIG=config/spider_cmaes.py"
if not defined BENCH_CANDIDATES set "BENCH_CANDIDATES=24"
if not defined BENCH_BALLS set "BENCH_BALLS=2"
if not defined BENCH_SIMTIME set "BENCH_SIMTIME=60"
if not defined BENCH_GENERATIONS set "BENCH_GENERATIONS=3"
if /i "%~1"=="quick" (
    set "BENCH_CANDIDATES=4"
    set "BENCH_BALLS=1"
    set "BENCH_SIMTIME=10"
    set "BENCH_GENERATIONS=2"
)
set "WORKERS_ARG="
if defined BENCH_WORKERS set "WORKERS_ARG=--workers %BENCH_WORKERS%"
set "COMMON=--config %BENCH_CONFIG% --candidates %BENCH_CANDIDATES% --balls %BENCH_BALLS% --sim-time %BENCH_SIMTIME% --generations %BENCH_GENERATIONS% %WORKERS_ARG%"
rem The code exactly as it was: original trial loop, a fresh worker pool every generation.
set "ORIGINAL=--set FAST_TRIAL=False --set PERSISTENT_WORKERS=False"
rem Configs now default to implicitfast at 5 ms, so every run names its physics.
set "RK1=--set SIM_INTEGRATOR=RK4 --set SIM_TIMESTEP=0.001"
set "IF5=--set SIM_INTEGRATOR=implicitfast --set SIM_TIMESTEP=0.005"

if /i "%~1"=="code" goto :code
if /i "%~1"=="breakdown" goto :breakdown

:breakdown
echo === Per-trial breakdown: where one trial's time goes (original trial loop) ===
for %%V in ("RK4 0.001" "implicitfast 0.005") do (
    for /f "tokens=1,2" %%A in (%%V) do (
        echo.
        "%PY%" bench\breakdown.py --config %BENCH_CONFIG% --sim-time 20 --integrator %%A --timestep %%B || exit /b 1
    )
)
if /i "%~1"=="breakdown" goto :end

:code
echo.
echo === Code changes, same physics: %BENCH_CANDIDATES% candidates x %BENCH_BALLS% balls x %BENCH_SIMTIME% s x %BENCH_GENERATIONS% generations ===
"%PY%" bench\bench.py --label original_rk4_1ms          %COMMON% %ORIGINAL% %RK1% || exit /b 1
"%PY%" bench\bench.py --label fast_rk4_1ms              %COMMON% %RK1% || exit /b 1
"%PY%" bench\bench.py --label original_implicitfast_5ms %COMMON% %ORIGINAL% %IF5% || exit /b 1
"%PY%" bench\bench.py --label fast_implicitfast_5ms     %COMMON% %IF5% || exit /b 1
echo.
echo --- RK4 1 ms: original code vs fast trial (max diff must be 0) ---
"%PY%" bench\compare.py bench\results\original_rk4_1ms.json bench\results\fast_rk4_1ms.json > bench\results\code.txt
echo --- implicitfast 5 ms: original code vs fast trial (max diff must be 0) --- >> bench\results\code.txt
"%PY%" bench\compare.py bench\results\original_implicitfast_5ms.json bench\results\fast_implicitfast_5ms.json >> bench\results\code.txt
type bench\results\code.txt
if /i "%~1"=="code" goto :end

echo.
echo === Physics options, with the fast trial ===
"%PY%" bench\bench.py --label fast_rk4_2ms           %COMMON% --set SIM_INTEGRATOR=RK4 --set SIM_TIMESTEP=0.002 || exit /b 1
"%PY%" bench\bench.py --label fast_euler_1ms         %COMMON% --set SIM_INTEGRATOR=Euler --set SIM_TIMESTEP=0.001 || exit /b 1
"%PY%" bench\bench.py --label fast_implicitfast_1ms  %COMMON% --set SIM_INTEGRATOR=implicitfast --set SIM_TIMESTEP=0.001 || exit /b 1
"%PY%" bench\bench.py --label fast_implicitfast_2ms  %COMMON% --set SIM_INTEGRATOR=implicitfast --set SIM_TIMESTEP=0.002 || exit /b 1
"%PY%" bench\bench.py --label fast_implicitfast_2_5ms %COMMON% --set SIM_INTEGRATOR=implicitfast --set SIM_TIMESTEP=0.0025 || exit /b 1

echo.
echo === Everything against the original code and physics ===
"%PY%" bench\compare.py bench\results\original_rk4_1ms.json bench\results\fast_rk4_1ms.json bench\results\fast_rk4_2ms.json bench\results\fast_euler_1ms.json bench\results\fast_implicitfast_1ms.json bench\results\fast_implicitfast_2ms.json bench\results\fast_implicitfast_2_5ms.json bench\results\original_implicitfast_5ms.json bench\results\fast_implicitfast_5ms.json > bench\results\summary.txt
type bench\results\summary.txt

:end
endlocal

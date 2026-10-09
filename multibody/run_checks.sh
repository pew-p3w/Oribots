#!/bin/bash
# Run every multibody test script and report. Driven by multibody/Makefile.
#
# Auto-discovers tests/test_*.py, runs each under RUN_PYTHON with its own
# CHECK_TIMEOUT-second watchdog (D-12), and exits 0 only when every script it
# ran exited 0. A script named in MULTIBODY_SKIP (space-separated base names) is
# skipped; `make check-fast` uses this to skip the slow isolation test.
#
# Each test script is expected to print "<NAME> CHECK PASSED" / "FAILED" itself;
# this runner adds a one-line per-script result and a final summary. A script
# killed by the watchdog is reported as a timeout failure (the per-process limit
# of requirement 22.9 / D-12).

set -u

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

run_python="${RUN_PYTHON:-../.venv.nosync/bin/python}"
timeout_s="${CHECK_TIMEOUT:-1800}"
skip="${MULTIBODY_SKIP:-}"

# Portable watchdog: run the command in the background, and a sleeper that kills
# the process group if it outlives the limit. macOS has no coreutils `timeout`.
run_with_timeout() {
    local limit="$1"; shift
    # Start the test in its own process group so a timeout kills its children
    # (worker processes) too.
    set -m
    "$@" &
    local cmd_pid=$!
    set +m
    (
        sleep "$limit"
        # Still alive? Kill the whole group.
        if kill -0 "$cmd_pid" 2>/dev/null; then
            kill -TERM -"$cmd_pid" 2>/dev/null
            sleep 3
            kill -KILL -"$cmd_pid" 2>/dev/null
        fi
    ) &
    local watcher_pid=$!
    local status=0
    wait "$cmd_pid" || status=$?
    # Test finished on its own: stop the watcher.
    kill "$watcher_pid" 2>/dev/null
    wait "$watcher_pid" 2>/dev/null
    return "$status"
}

shopt -s nullglob
scripts=(tests/test_*.py)
shopt -u nullglob

if [ "${#scripts[@]}" -eq 0 ]; then
    echo "multibody check: no test scripts in tests/ yet (nothing to run)."
    exit 0
fi

if [ ! -x "$run_python" ] && ! command -v "$run_python" >/dev/null 2>&1; then
    echo "multibody check: interpreter not found: $run_python" >&2
    echo "Set RUN_PYTHON to a Python 3.11 that has the Oribots deps installed." >&2
    exit 1
fi

failed=()
skipped=()
passed=()

for script in "${scripts[@]}"; do
    name="$(basename "$script")"
    case " $skip " in
        *" $name "*)
            echo "SKIP  $name"
            skipped+=("$name")
            continue
            ;;
    esac
    start=$(date +%s)
    run_with_timeout "$timeout_s" "$run_python" "$script"
    status=$?
    elapsed=$(( $(date +%s) - start ))
    if [ "$status" -eq 0 ]; then
        echo "PASS  $name  (${elapsed}s)"
        passed+=("$name")
    elif [ "$elapsed" -ge "$timeout_s" ]; then
        echo "FAIL  $name  (timed out after ${timeout_s}s; D-12)"
        failed+=("$name")
    else
        echo "FAIL  $name  (exit $status, ${elapsed}s)"
        failed+=("$name")
    fi
done

echo "----"
echo "multibody check: ${#passed[@]} passed, ${#failed[@]} failed, ${#skipped[@]} skipped"
if [ "${#failed[@]}" -ne 0 ]; then
    echo "failed: ${failed[*]}"
    exit 1
fi
exit 0

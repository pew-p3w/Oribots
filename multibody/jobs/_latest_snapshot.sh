#!/bin/bash
# Shared by the multibody jobs (sourced, not run): print the latest snapshot of a run folder.
#
#     source multibody/jobs/_latest_snapshot.sh
#     latest="$(latest_snapshot output/<run>)"     # empty if the folder has none
#
# "Latest" is the gen<N>.pkl with the largest N compared as an integer, so gen10
# comes after gen9 (a plain glob sort would order them wrong).

latest_snapshot() {
    local folder="$1" best="" best_n=-1 snapshot name number
    for snapshot in "$folder"/gen*.pkl; do
        [ -e "$snapshot" ] || continue
        name="$(basename "$snapshot" .pkl)"
        number="${name#gen}"
        case "$number" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "$((10#$number))" -gt "$best_n" ]; then
            best_n="$((10#$number))"
            best="$snapshot"
        fi
    done
    printf '%s' "$best"
}

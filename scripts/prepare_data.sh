#!/usr/bin/env bash
#
# Build every cohort and prompt file from the raw MIMIC tables.
#
#   scripts/prepare_data.sh --mimic3 /path/to/mimic-iii/1.4 \
#                           --mimic4 /path/to/mimic-iv/3.0/hosp
#
# Either database may be omitted. Both require PhysioNet credentialing; see
# docs/data_preparation.md.
#
# Unlike the script this replaces, a failing step aborts the run instead of
# printing a message and carrying on with stale inputs.

set -euo pipefail

MIMIC3=""
MIMIC4=""
DATA_ROOT="data"

usage() {
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mimic3)    MIMIC3="$2"; shift 2 ;;
        --mimic4)    MIMIC4="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$MIMIC3" && -z "$MIMIC4" ]]; then
    echo "error: pass --mimic3 and/or --mimic4" >&2
    usage 1
fi

for path in "$MIMIC3" "$MIMIC4"; do
    if [[ -n "$path" && ! -d "$path" ]]; then
        echo "error: not a directory: $path" >&2
        exit 1
    fi
done

build_args=()
[[ -n "$MIMIC3" ]] && build_args+=(--mimic3_path "$MIMIC3")
[[ -n "$MIMIC4" ]] && build_args+=(--mimic4_path "$MIMIC4")

echo "==> 1/3 reading MIMIC tables into per-task sample files"
python -m clinicalbench.data.build_cohort "${build_args[@]}" --data_root "$DATA_ROOT"

echo "==> 2/3 rendering prompts (plain and few-shot)"
for task in length_pred mortality_pred readmission_pred; do
    [[ -n "$MIMIC3" ]] && python -m clinicalbench.data.make_prompts \
        --task "$task" --dataset mimic3 --mimic_path "$MIMIC3" \
        --data_root "$DATA_ROOT" --both
    [[ -n "$MIMIC4" ]] && python -m clinicalbench.data.make_prompts \
        --task "$task" --dataset mimic4 --mimic_path "$MIMIC4" \
        --data_root "$DATA_ROOT" --both
done

echo "==> 3/3 regenerating cohort splits"
# The repository already ships these; regenerating overwrites them with
# byte-identical files, and `pytest tests/test_config.py` checks the sizes.
python -m clinicalbench.data.make_splits --all --data_root "$DATA_ROOT"

echo "==> done"

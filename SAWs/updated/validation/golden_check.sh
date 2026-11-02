#!/usr/bin/env bash
# Golden-output check: run the seeded cases with a BASELINE commit and with the
# current working tree, then require bit-identical outputs (golden_report.py).
#
#   bash SAWs/updated/validation/golden_check.sh [-b BASE_REF] [-c CASES] [-o WORKDIR] [-p PYTHON]
#
#   -b  baseline git ref (default: origin/main)
#   -c  comma-separated case patterns for run_cases.py (default: all)
#   -o  work directory (default: ${TMPDIR:-/tmp}/golden_<timestamp>)
#   -p  python interpreter (default: python)
#
# Run it in the environment being validated (on Midway3: source
# midway3_common.sh first, inside sinteractive -- the full set takes ~25 min on
# one core).  Only *.py / *.json files of the baseline ref are extracted, with
# `git archive`, so no worktree or large data files are needed; input data are
# read from this working tree for both runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
BASE_REF="origin/main"
CASES=""
WORK="${TMPDIR:-/tmp}/golden_$(date +%Y%m%d_%H%M%S)"
PY="python"
while getopts "b:c:o:p:" opt; do
    case "$opt" in
        b) BASE_REF="$OPTARG" ;;
        c) CASES="$OPTARG" ;;
        o) WORK="$OPTARG" ;;
        p) PY="$OPTARG" ;;
        *) sed -n '2,16p' "$0"; exit 2 ;;
    esac
done

mkdir -p "$WORK"
echo "baseline ref : $BASE_REF ($(git -C "$REPO" rev-parse --short "$BASE_REF"))"
echo "candidate    : working tree of $REPO ($(git -C "$REPO" rev-parse --short HEAD), $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ') uncommitted paths)"
echo "work dir     : $WORK"
"$PY" -c 'import sys, numpy; print("python", sys.version.split()[0], "numpy", numpy.__version__)'

# 1. inputs shared by both runs
"$PY" "$HERE/make_inputs.py" --out "$WORK/inputs" > /dev/null

# 2. baseline code (python + json only)
rm -rf "$WORK/baseline_src" && mkdir -p "$WORK/baseline_src"
git -C "$REPO" ls-tree -r -z --name-only "$BASE_REF" -- SAWs/updated \
    | grep -z -E '\.(py|json)$' \
    | xargs -0 git -C "$REPO" archive --format=tar "$BASE_REF" -- \
    | tar -x -i -C "$WORK/baseline_src"   # -i: xargs may emit several archives

# 3. run both
CASE_ARGS=()
[[ -n "$CASES" ]] && CASE_ARGS=(--cases "$CASES")
echo "== baseline run"
"$PY" "$HERE/run_cases.py" --code-root "$WORK/baseline_src/SAWs/updated" \
    --data-root "$REPO/SAWs/updated" --inputs "$WORK/inputs" --out "$WORK/baseline" \
    --python "$PY" ${CASE_ARGS[@]+"${CASE_ARGS[@]}"}
echo "== candidate run"
"$PY" "$HERE/run_cases.py" --code-root "$REPO/SAWs/updated" \
    --data-root "$REPO/SAWs/updated" --inputs "$WORK/inputs" --out "$WORK/candidate" \
    --python "$PY" ${CASE_ARGS[@]+"${CASE_ARGS[@]}"}

# 4. compare
echo "== comparison"
"$PY" "$HERE/golden_report.py" --baseline "$WORK/baseline" --candidate "$WORK/candidate" \
    --baseline-code "$WORK/baseline_src/SAWs/updated" --candidate-code "$REPO/SAWs/updated" \
    --report "$WORK/golden_report.txt"

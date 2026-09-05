#!/usr/bin/env bash
# Driver for Phase 0's concurrency-crash reproduction.
#
# Why a shell loop instead of a Python loop: a CUDA device-side assert poisons
# the CUDA context for the entire process, so every repetition needs a FRESH
# process. See scripts/repro_concurrent_crash.py's docstring.
#
# Usage:
#   bash scripts/run_phase0_repro.sh baseline
#   bash scripts/run_phase0_repro.sh A [reps]      # clean repro, tests H2
#   bash scripts/run_phase0_repro.sh B             # CUDA_LAUNCH_BLOCKING, 1 rep
#   bash scripts/run_phase0_repro.sh threadlocal [reps]   # tests H1
#   bash scripts/run_phase0_repro.sh lock [reps]          # tests decision (b)
set -u

cd /workspace/GPU_Profiling
PY=python3.12          # plain `python3` on this pod is 3.10 and has no torch
OUT=/tmp/phase0
mkdir -p "$OUT"

EXP="${1:-A}"
REPS="${2:-10}"

echo "commit=$(git rev-parse --short HEAD)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "free GPU mem check (rule 9 -- you should be the only one on it):"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

run_one () {   # run_one <tag> <rep> <extra args...>
  local tag="$1"; local rep="$2"; shift 2
  "$PY" scripts/repro_concurrent_crash.py \
      --mode concurrent --seed "$rep" \
      --out "$OUT/${tag}_rep${rep}.json" \
      "$@" > "$OUT/${tag}_rep${rep}.log" 2>&1
  echo "$?"
}

summarise () {  # summarise <tag> <reps>
  local tag="$1"; local reps="$2"
  local crash=0 mismatch=0 clean=0 other=0
  for i in $(seq 1 "$reps"); do
    case "$(cat "$OUT/${tag}_rep${i}.exit")" in
      0) clean=$((clean+1)) ;;
      1) crash=$((crash+1)) ;;
      2) mismatch=$((mismatch+1)) ;;
      *) other=$((other+1)) ;;
    esac
  done
  echo
  echo "=== $tag over $reps reps: crash=$crash  silent-mismatch=$mismatch  clean=$clean  setup-fail=$other"
  echo "    device-side asserts seen: $(grep -l 'Indexing.cu' "$OUT/${tag}_rep"*.log 2>/dev/null | wc -l) / $reps runs"
}

case "$EXP" in
  baseline)
    echo ">>> Recording the sequential ground truth. Nothing concurrent happens here."
    "$PY" scripts/repro_concurrent_crash.py --mode baseline \
        --baseline "$OUT/baseline.json" --out "$OUT/baseline_meta.json"
    ;;
  A|clean)
    echo ">>> Experiment A: clean repro, no profiler, no monkey-patches. Tests H2."
    for i in $(seq 1 "$REPS"); do
      code=$(run_one A "$i" --fix none --baseline "$OUT/baseline.json")
      echo "$code" > "$OUT/A_rep${i}.exit"
      echo "  rep $i exit=$code"
    done
    summarise A "$REPS"
    ;;
  B|blocking)
    echo ">>> Experiment B: CUDA_LAUNCH_BLOCKING=1, one rep, for the TRUE launch site."
    echo "    If this does NOT reproduce, that is a result: it means the bug depends on"
    echo "    launch overlap, i.e. a genuine race rather than a deterministic OOB."
    CUDA_LAUNCH_BLOCKING=1 "$PY" scripts/repro_concurrent_crash.py \
        --mode concurrent --seed 1 --fix none \
        --baseline "$OUT/baseline.json" --out "$OUT/B_blocking.json" \
        2>&1 | tee "$OUT/B_blocking.log"
    echo "--- first Python frame inside our code (this is the number that matters) ---"
    grep -n "vis_pruner_copy\|scripts/repro" "$OUT/B_blocking.log" | head -20
    ;;
  threadlocal|lock)
    echo ">>> Experiment C: --fix $EXP"
    for i in $(seq 1 "$REPS"); do
      code=$(run_one "$EXP" "$i" --fix "$EXP" --baseline "$OUT/baseline.json")
      echo "$code" > "$OUT/${EXP}_rep${i}.exit"
      echo "  rep $i exit=$code"
    done
    summarise "$EXP" "$REPS"
    ;;
  *)
    echo "unknown experiment: $EXP" >&2; exit 64 ;;
esac

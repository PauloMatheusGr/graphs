#!/usr/bin/env bash
# Parallel runner for encoding_D plan. Skips jobs whose summary already has
# the required model_keys. Safe to re-launch after interrupt.
#
#   ./run_encoding_D_parallel.sh 2>&1 | tee logs/encoding_D_par_$(date +%Y%m%d_%H%M%S).log
#   JOBS=4 ./run_encoding_D_parallel.sh

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=modules

CLAIM="${CLAIM:-48m_6m}"
LONG="${LONG:-t1_ols}"
JOBS="${JOBS:-4}"
SKIP_R10="${SKIP_R10:-0}"
SKIP_4ALGO="${SKIP_4ALGO:-0}"
SKIP_LATE="${SKIP_LATE:-0}"
LATE_GRID="${LATE_GRID:-full}"

COMMON=(--tasks smci_pmci --selection l1_stable --combat false --repeats 10 --seed 42
  --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0)

COHORTS_R10=(36m_6m 36m_12m 48m_6m 48m_12m 48m_6m_soft_False)
MODS=(vol shape texture firstorder disp)
REPS_4ALGO=(t1_only t1_r10 t1_r10_r21 t1_ols)
MODELS_4="svm,rf,elasticnet,xgb"

rep_dir() {
  case "$1" in
    t1_only) echo ablation_results_t1_only ;;
    t1_r10) echo ablation_results_r10 ;;
    t1_r10_r21) echo ablation_results_r10r21 ;;
    t1_ols) echo ablation_results_ols ;;
    *) echo "unknown_rep_$1"; return 1 ;;
  esac
}

# Return 0 if summary already has all required models for smci_pmci.
has_models() {
  local cohort="$1" rep="$2" mod="$3" need="$4"
  local summ="csvs/cohorts/${cohort}/$(rep_dir "$rep")/${mod}/ablation_summary.csv"
  [[ -f "$summ" ]] || return 1
  python - "$summ" "$need" <<'PY'
import sys
import pandas as pd
path, need = sys.argv[1], set(sys.argv[2].split(","))
df = pd.read_csv(path)
m = df[df["task"].astype(str) == "smci_pmci"]
if "with_combat" in m.columns:
    m = m[~m["with_combat"].astype(bool)]
have = set(m["model_key"].astype(str))
sys.exit(0 if need <= have else 1)
PY
}

run_mono() {
  local cohort="$1" rep="$2" mod="$3" models="$4"
  if has_models "$cohort" "$rep" "$mod" "$models"; then
    echo "SKIP $cohort | $rep | $mod | have $models"
    return 0
  fi
  echo "=== MONO $cohort | $rep | $mod | $models $(date -Is) ==="
  python 5_ablation.py --cohort "$cohort" --representation "$rep" \
    --modality "$mod" --models "$models" "${COMMON[@]}"
}

export -f has_models rep_dir run_mono
export CLAIM LONG COMMON
# bash arrays don't export; pass via env strings
export _MODS="${MODS[*]}"
export _COMMON_STR="${COMMON[*]}"

# rebuild COMMON inside exported function via re-parse
run_mono_job() {
  # args: cohort rep mod models
  local cohort="$1" rep="$2" mod="$3" models="$4"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PYTHONPATH=modules
  COMMON=(--tasks smci_pmci --selection l1_stable --combat false --repeats 10 --seed 42
    --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0)
  if has_models "$cohort" "$rep" "$mod" "$models"; then
    echo "SKIP $cohort | $rep | $mod | have $models"
    return 0
  fi
  echo "=== MONO $cohort | $rep | $mod | $models $(date -Is) ==="
  python 5_ablation.py --cohort "$cohort" --representation "$rep" \
    --modality "$mod" --models "$models" "${COMMON[@]}"
}
export -f run_mono_job

TASKFILE=$(mktemp)
trap 'rm -f "$TASKFILE"' EXIT

if [[ "$SKIP_R10" != "1" ]]; then
  for c in "${COHORTS_R10[@]}"; do
    for m in "${MODS[@]}"; do
      echo "run_mono_job $c t1_r10 $m svm" >>"$TASKFILE"
    done
  done
fi

if [[ "$SKIP_4ALGO" != "1" ]]; then
  for rep in "${REPS_4ALGO[@]}"; do
    for m in "${MODS[@]}"; do
      echo "run_mono_job $CLAIM $rep $m $MODELS_4" >>"$TASKFILE"
    done
  done
fi

echo "Queued $(wc -l <"$TASKFILE") mono jobs | JOBS=$JOBS"
if [[ -s "$TASKFILE" ]]; then
  # xargs -P parallel; each line is a shell command
  <"$TASKFILE" xargs -P "$JOBS" -I{} bash -c '{}'
fi

if [[ "$SKIP_LATE" != "1" ]]; then
  SPEC_ALL_T1="vol:t1_only,shape:t1_only,texture:t1_only,disp:t1_only,firstorder:t1_only"
  SPEC_ALL_D="vol:${LONG},shape:${LONG},texture:${LONG},disp:${LONG},firstorder:${LONG}"
  SPECS=("$SPEC_ALL_T1" "$SPEC_ALL_D")
  if [[ "$LATE_GRID" == "full" ]]; then
    mapfile -t GRID < <(LONG="$LONG" python -c "
import os, sys
sys.path.insert(0, 'modules')
from ablation_representation import iter_late_fusion_grid
print('\n'.join(iter_late_fusion_grid(longitudinal=os.environ['LONG'])))
")
    SPECS+=("${GRID[@]}")
  fi
  declare -A SEEN=()
  for spec in "${SPECS[@]}"; do
    [[ -n "${SEEN[$spec]:-}" ]] && continue
    SEEN[$spec]=1
    fp=$(LONG="$LONG" SPEC="$spec" python -c "
import os, sys
sys.path.insert(0, 'modules')
from ablation_representation import parse_fusion_spec, fusion_fingerprint
print(fusion_fingerprint(parse_fusion_spec(os.environ['SPEC'])))
")
    out="csvs/cohorts/${CLAIM}/ablation_results_late_fusion/${fp}/ablation_summary.csv"
    if [[ -f "$out" ]]; then
      # late OLS grid may share all-T1 dirs already present from Q4 era; only skip
      # if fingerprint has no d21 (always) — always re-run if ols in fp and missing? 
      # Skip if summary exists (idempotent).
      echo "SKIP LATE $fp"
      continue
    fi
    echo "=== LATE $CLAIM | $spec $(date -Is) ==="
    python 5_ablation_late_fusion.py --cohort "$CLAIM" --fusion "$spec" \
      --models svm --reuse-disk "${COMMON[@]}"
  done
fi

echo "DONE encoding_D_parallel $(date -Is)"

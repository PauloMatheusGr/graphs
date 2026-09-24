#!/usr/bin/env bash
# Encoding temporal v2: t1_r10 (2v) + claim 4-algo + late grid com LONG=t1_ols.
# Não apaga B0 abs. Requer monos t1_ols/t1_only já no disco para late --reuse-disk.
#
# Uso:
#   ./run_encoding_D.sh 2>&1 | tee logs/encoding_D_$(date +%Y%m%d_%H%M%S).log
#   SKIP_R10=1 SKIP_4ALGO=1 ./run_encoding_D.sh   # só late
#   SKIP_LATE=1 ./run_encoding_D.sh                # só mono

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=modules

CLAIM="${CLAIM:-48m_6m}"
LONG="${LONG:-t1_ols}"
SKIP_R10="${SKIP_R10:-0}"
SKIP_4ALGO="${SKIP_4ALGO:-0}"
SKIP_LATE="${SKIP_LATE:-0}"
LATE_GRID="${LATE_GRID:-full}"  # full|paper

COMMON='--tasks smci_pmci --selection l1_stable --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0'

COHORTS_R10=(36m_6m 36m_12m 48m_6m 48m_12m 48m_6m_soft_False)
MODS=(vol shape texture firstorder disp)
REPS_4ALGO=(t1_only t1_r10 t1_r10_r21 t1_ols)
MODELS_4="svm,rf,elasticnet,xgb"

run_mono() {
  local cohort="$1" rep="$2" mod="$3" models="$4"
  echo "=== MONO $cohort | $rep | $mod | $models $(date -Is) ==="
  python 5_ablation.py --cohort "$cohort" --representation "$rep" \
    --modality "$mod" --models "$models" $COMMON
}

if [[ "$SKIP_R10" != "1" ]]; then
  for c in "${COHORTS_R10[@]}"; do
    for m in "${MODS[@]}"; do
      run_mono "$c" t1_r10 "$m" svm
    done
  done
fi

if [[ "$SKIP_4ALGO" != "1" ]]; then
  for rep in "${REPS_4ALGO[@]}"; do
    for m in "${MODS[@]}"; do
      run_mono "$CLAIM" "$rep" "$m" "$MODELS_4"
    done
  done
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
  # dedupe preserving order
  declare -A SEEN=()
  for spec in "${SPECS[@]}"; do
    [[ -n "${SEEN[$spec]:-}" ]] && continue
    SEEN[$spec]=1
    echo "=== LATE $CLAIM | $spec $(date -Is) ==="
    python 5_ablation_late_fusion.py --cohort "$CLAIM" --fusion "$spec" \
      --models svm --reuse-disk $COMMON
  done
fi

echo "DONE encoding_D $(date -Is)"

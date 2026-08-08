#!/usr/bin/env bash
# Mono (4 cohorts × t1_only/t1_deltas/wide) → late (72 specs, 4 cohorts) → early (claim).
# Claim multimodal = 48m_12m. ComBat só no mono (sensibilidade); late/early = nocombat.
# Pré-requisito: features + 4_run_post_extract nas 4 coortes.
#
# Uso:
#   ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log

set -euo pipefail
cd "$(dirname "$0")"

CLAIM="${CLAIM:-48m_12m}"
COMMON_MONO='--tasks smci_pmci --selection l1_stable --models svm --combat both --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
COMMON_FUSION='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'

COHORTS=(36m_6m 36m_12m 48m_6m 48m_12m)
MODS=(vol shape texture disp)
REPS=(t1_only t1_deltas)
REPS_MONO=(t1_only t1_deltas wide)

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

# --- 1) MONO ---
for C in "${COHORTS[@]}"; do
  for R in "${REPS_MONO[@]}"; do
    echo "=== MONO $C $R ==="
    # shellcheck disable=SC2086
    "$PY" 5_ablation.py --cohort "$C" --representation "$R" \
      --modality vol,shape,texture,disp $COMMON_MONO \
      || { echo "FAIL mono $C $R"; exit 1; }
  done
done

# --- 2) LATE (grade completa; reusa mono) ---
SPECS=()
for ((i = 0; i < ${#MODS[@]}; i++)); do
  for ((j = i + 1; j < ${#MODS[@]}; j++)); do
    for R1 in "${REPS[@]}"; do
      for R2 in "${REPS[@]}"; do
        SPECS+=("${MODS[i]}:${R1},${MODS[j]}:${R2}")
      done
    done
  done
done
for ((i = 0; i < ${#MODS[@]}; i++)); do
  for ((j = i + 1; j < ${#MODS[@]}; j++)); do
    for ((k = j + 1; k < ${#MODS[@]}; k++)); do
      for R1 in "${REPS[@]}"; do
        for R2 in "${REPS[@]}"; do
          for R3 in "${REPS[@]}"; do
            SPECS+=("${MODS[i]}:${R1},${MODS[j]}:${R2},${MODS[k]}:${R3}")
          done
        done
      done
    done
  done
done
for R1 in "${REPS[@]}"; do
  for R2 in "${REPS[@]}"; do
    for R3 in "${REPS[@]}"; do
      for R4 in "${REPS[@]}"; do
        SPECS+=("vol:${R1},shape:${R2},texture:${R3},disp:${R4}")
      done
    done
  done
done
echo "n_specs_late=${#SPECS[@]}"

for C in "${COHORTS[@]}"; do
  for F in "${SPECS[@]}"; do
    echo "=== LATE $C | $F ==="
    # shellcheck disable=SC2086
    "$PY" 5_ablation_late_fusion.py --cohort "$C" --fusion "$F" \
      --combine mean --reuse-disk $COMMON_FUSION \
      || { echo "FAIL late $C $F"; exit 1; }
  done
done

# --- 3) EARLY: tabela early vs late (claim, nocombat) ---
EARLY_SPECS=(
  'shape:t1_only,texture:t1_deltas'
  'shape:t1_only,vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,texture:t1_deltas'
  'shape:t1_only,vol:t1_deltas'
  'vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,vol:t1_deltas'
)

echo "CLAIM=$CLAIM"
for F in "${EARLY_SPECS[@]}"; do
  echo "=== EARLY $CLAIM | $F ==="
  # shellcheck disable=SC2086
  "$PY" 5_ablation_early_fusion.py --cohort "$CLAIM" --fusion "$F" \
    $COMMON_FUSION \
    || { echo "FAIL early $F"; exit 1; }
done

echo "DONE"

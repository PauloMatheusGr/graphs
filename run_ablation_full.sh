#!/usr/bin/env bash
# Único bash de ablação (GitHub).
# Mono só firstorder (vol/shape/texture/disp já no disco) → 3 late × 4 coortes → 3 early claim.
# Longitudinal oficial = t1_d21_d32 (T1+D21+D32). t1_deltas no disco = sensibilidade.
# Extra (cn_ad / clínica / clinic+img / leaky): SKIP_EXTRA=1 default — não pisa mono claim.
# Sens RF/EN só se SENS_REPS estiver set (reescreve mono).
#
# Uso:
#   ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
#   SKIP_MONO=1 ./run_ablation_full.sh          # firstorder já correu; só late/early
#   SKIP_EXTRA=0 ./run_ablation_full.sh         # + cn_ad, clínica, leaky
#   SENS_REPS=t1_d21_d32 SKIP_EXTRA=0 ./run_ablation_full.sh  # RF/EN na claim (pisa Q4 svm)

set -euo pipefail
cd "$(dirname "$0")"

CLAIM="${CLAIM:-48m_12m}"
PRIMARY="${PRIMARY:-$CLAIM}"
LONG="${LONG:-t1_d21_d32}"
SKIP_MONO="${SKIP_MONO:-0}"
SKIP_EXTRA="${SKIP_EXTRA:-1}"
COMMON_MONO='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
COMMON_FUSION='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
# ponytail: sem --tasks no COMMON_EXTRA — argparse fica com o último --tasks; cn_ad precisa do seu
COMMON_EXTRA='--selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'

COHORTS=(36m_6m 36m_12m 48m_6m 48m_12m)
REPS_MONO=(t1_only "$LONG")
FUSION_MOD="${FUSION_MOD:-shape}"
FUSION_REP="${FUSION_REP:-t1_only}"

SPEC_ALL_T1="vol:t1_only,shape:t1_only,texture:t1_only,disp:t1_only,firstorder:t1_only"
SPEC_ALL_Q4="vol:${LONG},shape:${LONG},texture:${LONG},disp:${LONG},firstorder:${LONG}"
SPEC_ANCORA="shape:t1_only,vol:${LONG},texture:${LONG},disp:${LONG},firstorder:${LONG}"
SPECS=("$SPEC_ALL_T1" "$SPEC_ALL_Q4" "$SPEC_ANCORA")

root_for_rep() {
  case "$1" in
    t1_only) echo "ablation_results_t1_only" ;;
    t1_deltas) echo "ablation_results_deltas" ;;
    t1_d21_d32) echo "ablation_results_d21d32" ;;
    t1_ma) echo "ablation_results_ma" ;;
    wide|abs) echo "ablation_results" ;;
    *) echo "erro: representation desconhecida: $1" >&2; return 1 ;;
  esac
}

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

echo "LONG=$LONG SKIP_MONO=$SKIP_MONO SKIP_EXTRA=$SKIP_EXTRA CLAIM=$CLAIM"

# --- 1) MONO só firstorder (não toca vol/shape/texture/disp) ---
if [[ "$SKIP_MONO" != "1" ]]; then
  for C in "${COHORTS[@]}"; do
    for R in "${REPS_MONO[@]}"; do
      echo "=== MONO firstorder $C $R ==="
      # shellcheck disable=SC2086
      "$PY" 5_ablation.py --cohort "$C" --representation "$R" \
        --modality firstorder $COMMON_MONO \
        || { echo "FAIL mono firstorder $C $R"; exit 1; }
    done
  done
else
  echo "SKIP_MONO=1 — pula seção 1"
fi

# --- 2) LATE: 3 specs pré-especificadas (--reuse-disk) ---
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

# --- 3) EARLY: mesmos 3 specs, só claim ---
echo "CLAIM=$CLAIM"
for F in "${SPECS[@]}"; do
  echo "=== EARLY $CLAIM | $F ==="
  # shellcheck disable=SC2086
  "$PY" 5_ablation_early_fusion.py --cohort "$CLAIM" --fusion "$F" \
    $COMMON_FUSION \
    || { echo "FAIL early $F"; exit 1; }
done

# --- 4+) EXTRA (default off) ---
if [[ "$SKIP_EXTRA" == "1" ]]; then
  echo "SKIP_EXTRA=1 — pula cn_ad/clínica/leaky"
  echo "DONE"
  exit 0
fi

ROOT="$(root_for_rep "$LONG")"
echo "EXTRA PRIMARY=$PRIMARY REP=$LONG → $ROOT"

if [[ -n "${SENS_REPS:-}" ]]; then
  COMMON_SENS="--tasks smci_pmci --selection l1_stable --models svm,rf,elasticnet --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"
  # shellcheck disable=SC2086
  for R in $SENS_REPS; do
    echo "=== SENS mono $PRIMARY $R svm,rf,elasticnet ==="
    # shellcheck disable=SC2086
    "$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$R" \
      --modality vol,shape,texture,disp,firstorder $COMMON_SENS \
      || { echo "FAIL sens $R"; exit 1; }
  done
fi

echo "=== CN_AD $PRIMARY $LONG vol → ${ROOT}/vol_cn_ad ==="
# shellcheck disable=SC2086
"$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$LONG" \
  --modality vol --tasks cn_ad \
  --results-dir "csvs/cohorts/${PRIMARY}/${ROOT}/vol_cn_ad" \
  $COMMON_EXTRA \
  || { echo "FAIL cn_ad"; exit 1; }

echo "=== CLINICAL $PRIMARY ==="
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set clinical \
  --tasks smci_pmci --models svm \
  --repeats 10 --tuner optuna --optuna-trials 10 \
  || { echo "FAIL clinical"; exit 1; }

echo "=== FUSION clinic+img $PRIMARY $FUSION_MOD $FUSION_REP ==="
# shellcheck disable=SC2086
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set fusion \
  --modality "$FUSION_MOD" --representation "$FUSION_REP" --tasks smci_pmci $COMMON_EXTRA \
  || { echo "FAIL fusion"; exit 1; }

echo "=== LEAKY $PRIMARY $LONG vol ==="
# shellcheck disable=SC2086
"$PY" 5_ablation_leaky.py --cohort "$PRIMARY" --representation "$LONG" \
  --inflate "" --modality vol --tasks smci_pmci $COMMON_EXTRA \
  || { echo "FAIL leaky"; exit 1; }

echo "DONE"

#!/usr/bin/env bash
# Ablação paper: uniclasse 5 famílias × {t1_only, Q4} × 4 coortes → 3 late.
# Longitudinal = t1_d21_d32. Sem early, sem --modality all.
# WIPE=1 (default) apaga CSVs allowlist nessas pastas — senão late --reuse-disk lê o espaço velho.
# Long CSVs (ablation/*.csv) não se tocam.
#
# Uso:
#   ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
#   WIPE=0 SKIP_MONO=1 ./run_ablation_full.sh   # mono novo já no disco; só late
#   SKIP_EXTRA=0 ./run_ablation_full.sh         # + cn_ad, clínica, leaky (fora do paper)

set -euo pipefail
cd "$(dirname "$0")"

CLAIM="${CLAIM:-48m_12m}"
PRIMARY="${PRIMARY:-$CLAIM}"
LONG="${LONG:-t1_d21_d32}"
SKIP_MONO="${SKIP_MONO:-0}"
SKIP_EXTRA="${SKIP_EXTRA:-1}"
WIPE="${WIPE:-1}"
MODS_MONO="${MODS_MONO:-vol,shape,texture,disp,firstorder}"
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

echo "LONG=$LONG SKIP_MONO=$SKIP_MONO SKIP_EXTRA=$SKIP_EXTRA WIPE=$WIPE MODS=$MODS_MONO CLAIM=$CLAIM"

if [[ "$WIPE" == "1" ]]; then
  echo "WIPE=1 — apaga resultados t1_only / ${LONG} / late_fusion (não apaga *_long.csv)"
  for C in "${COHORTS[@]}"; do
    for R in "${REPS_MONO[@]}"; do
      rm -rf "csvs/cohorts/${C}/$(root_for_rep "$R")"
    done
    rm -rf "csvs/cohorts/${C}/ablation_results_late_fusion"
  done
elif [[ "$SKIP_MONO" == "1" ]]; then
  echo "WIPE=0 SKIP_MONO=1 — late reusa o que estiver no disco"
fi

# --- 1) MONO: 5 famílias × {t1_only, Q4} × 4 coortes ---
if [[ "$SKIP_MONO" != "1" ]]; then
  for C in "${COHORTS[@]}"; do
    for R in "${REPS_MONO[@]}"; do
      echo "=== MONO $MODS_MONO $C $R ==="
      # shellcheck disable=SC2086
      "$PY" 5_ablation.py --cohort "$C" --representation "$R" \
        --modality "$MODS_MONO" $COMMON_MONO \
        || { echo "FAIL mono $C $R"; exit 1; }
    done
  done
else
  echo "SKIP_MONO=1 — pula uniclasse"
fi

# --- 2) LATE: 3 specs, --reuse-disk só depois dos mono novos ---
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

# --- extra (default off): não é o paper ---
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
      --modality "$MODS_MONO" $COMMON_SENS \
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

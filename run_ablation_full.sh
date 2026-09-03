#!/usr/bin/env bash
# Ablação paper: uniclasse × {t1_only, t1_d21, t1_d21_d32} → late.
# Longitudinal 3 visitas = t1_d21_d32. Duas visitas = t1_d21 (T1+D21, sem i3).
# LATE_GRID=full (default na CLAIM): 232 uniões k≥2, cada fam. T1 ou Q4.
# LATE_GRID=paper: só all-T1, all-Q4, âncora (shape T1 ∪ resto Q4).
# Sem early, sem --modality all. Sem grelha 3^k (D21 no mix).
# WIPE=1 apaga só roots em REPS_MONO + late_fusion das COHORTS. Default WIPE=0.
# Long CSVs (ablation/*.csv) não se tocam.
#
# Uso:
#   ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
#   WIPE=0 SKIP_MONO=1 ./run_ablation_full.sh
#   SKIP_EXTRA=0 ./run_ablation_full.sh
#   COHORTS="48m_6m" TWO_VISIT=1 SKIP_MONO=0 SKIP_EXTRA=0 ./run_ablation_full.sh
#   REPS_MONO="t1_d21" COHORTS="48m_6m" SKIP_MONO=0 ./run_ablation_full.sh
# SKIP_EXTRA=0 na PRIMARY: 4 modelos (svm,rf,elasticnet,xgb), cn_ad no Q4 (5 fam.),
# depois late (reuse), clínica, leaky. HM (sem pareamento) NÃO entra — store à parte.
# Mono SVM já no disco: SKIP_EXISTING=1 não regrava no bloco 1; o extra 4 modelos SIM regrava.

set -euo pipefail
cd "$(dirname "$0")"

CLAIM="${CLAIM:-48m_6m}"
PRIMARY="${PRIMARY:-$CLAIM}"
LONG="${LONG:-t1_d21_d32}"
SKIP_MONO="${SKIP_MONO:-0}"
SKIP_EXTRA="${SKIP_EXTRA:-1}"
WIPE="${WIPE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODS_MONO="${MODS_MONO:-vol,shape,texture,disp,firstorder}"
COMMON_MONO='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
COMMON_FUSION='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
# ponytail: sem --tasks no COMMON_EXTRA — argparse fica com o último --tasks; cn_ad precisa do seu
COMMON_EXTRA='--selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'

if [[ -n "${COHORTS:-}" ]]; then
  # shellcheck disable=SC2206
  COHORTS=($COHORTS)
else
  COHORTS=(36m_6m 36m_12m 48m_6m 48m_12m)
fi

TWO_VISIT="${TWO_VISIT:-}"
if [[ -z "$TWO_VISIT" ]]; then
  if [[ ${#COHORTS[@]} -eq 1 && "${COHORTS[0]}" == "$CLAIM" ]]; then
    TWO_VISIT=1
  else
    TWO_VISIT=0
  fi
fi

if [[ -n "${REPS_MONO:-}" ]]; then
  # shellcheck disable=SC2206
  REPS_MONO=($REPS_MONO)
else
  REPS_MONO=(t1_only "$LONG")
  if [[ "$TWO_VISIT" == "1" ]]; then
    REPS_MONO=(t1_only t1_d21 "$LONG")
  fi
fi

FUSION_MOD="${FUSION_MOD:-shape}"
FUSION_REP="${FUSION_REP:-t1_only}"

SPEC_ALL_T1="vol:t1_only,shape:t1_only,texture:t1_only,disp:t1_only,firstorder:t1_only"
SPEC_ALL_Q4="vol:${LONG},shape:${LONG},texture:${LONG},disp:${LONG},firstorder:${LONG}"
SPEC_ANCORA="shape:t1_only,vol:${LONG},texture:${LONG},disp:${LONG},firstorder:${LONG}"
SPECS_PAPER=("$SPEC_ALL_T1" "$SPEC_ALL_Q4" "$SPEC_ANCORA")
LATE_GRID="${LATE_GRID:-full}"

has_rep() {
  local r
  for r in "${REPS_MONO[@]}"; do
    [[ "$r" == "$1" ]] && return 0
  done
  return 1
}

if has_rep t1_d21 && [[ "$LATE_GRID" == "paper" ]]; then
  SPEC_ALL_D21="vol:t1_d21,shape:t1_d21,texture:t1_d21,disp:t1_d21,firstorder:t1_d21"
  SPEC_ANCORA_D21="shape:t1_only,vol:t1_d21,texture:t1_d21,disp:t1_d21,firstorder:t1_d21"
  SPECS_PAPER+=("$SPEC_ALL_D21" "$SPEC_ANCORA_D21")
fi

emit_late_grid() {
  LONG="$LONG" "$PY" -c "
import os, sys
sys.path.insert(0, 'modules')
from ablation_representation import iter_late_fusion_grid
print('\\n'.join(iter_late_fusion_grid(longitudinal=os.environ['LONG'])))
"
}

late_done() {
  local c="$1" spec="$2"
  local fp
  fp="$(
    LONG="$LONG" "$PY" -c "
import sys
sys.path.insert(0, 'modules')
from ablation_representation import fusion_fingerprint, parse_fusion_spec
print(fusion_fingerprint(parse_fusion_spec(sys.argv[1])))
" "$spec"
  )"
  [[ -f "csvs/cohorts/${c}/ablation_results_late_fusion/${fp}/ablation_results_all.csv" ]]
}

root_for_rep() {
  case "$1" in
    t1_only) echo "ablation_results_t1_only" ;;
    t1_deltas) echo "ablation_results_deltas" ;;
    t1_d21) echo "ablation_results_d21" ;;
    t1_d21_d32) echo "ablation_results_d21d32" ;;
    t1_ma) echo "ablation_results_ma" ;;
    wide|abs) echo "ablation_results" ;;
    *) echo "erro: representation desconhecida: $1" >&2; return 1 ;;
  esac
}

mono_done() {
  local c="$1" r="$2"
  local root
  root="$(root_for_rep "$r")"
  [[ -f "csvs/cohorts/${c}/${root}/shape/ablation_results_all.csv" ]]
}

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

echo "LONG=$LONG SKIP_MONO=$SKIP_MONO SKIP_EXTRA=$SKIP_EXTRA WIPE=$WIPE SKIP_EXISTING=$SKIP_EXISTING TWO_VISIT=$TWO_VISIT LATE_GRID=$LATE_GRID MODS=$MODS_MONO CLAIM=$CLAIM COHORTS=${COHORTS[*]} REPS=${REPS_MONO[*]}"

if [[ "$WIPE" == "1" ]]; then
  echo "WIPE=1 — apaga REPS_MONO + late_fusion (não apaga *_long.csv). Backup antes."
  for C in "${COHORTS[@]}"; do
    for R in "${REPS_MONO[@]}"; do
      rm -rf "csvs/cohorts/${C}/$(root_for_rep "$R")"
    done
    rm -rf "csvs/cohorts/${C}/ablation_results_late_fusion"
  done
elif [[ "$SKIP_MONO" == "1" ]]; then
  echo "WIPE=0 SKIP_MONO=1 — late reusa o que estiver no disco"
fi

# --- 1) MONO ---
if [[ "$SKIP_MONO" != "1" ]]; then
  for C in "${COHORTS[@]}"; do
    for R in "${REPS_MONO[@]}"; do
      if [[ "$SKIP_EXISTING" == "1" ]] && mono_done "$C" "$R"; then
        echo "=== SKIP MONO $C $R (já existe) ==="
        continue
      fi
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

# --- 2) 4 modelos + CN×AD na PRIMARY (antes do late: reuse lê estes CSVs) ---
if [[ "$SKIP_EXTRA" != "1" ]]; then
  SENS_REPS="${SENS_REPS:-${REPS_MONO[*]}}"
  COMMON_SENS="--selection l1_stable --models svm,rf,elasticnet,xgb --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"
  echo "EXTRA PRIMARY=$PRIMARY SENS_REPS=$SENS_REPS"
  # shellcheck disable=SC2086
  for R in $SENS_REPS; do
    if [[ "$R" == "$LONG" ]]; then
      TASKS_SENS="cn_ad,smci_pmci"
    else
      TASKS_SENS="smci_pmci"
    fi
    echo "=== SENS $PRIMARY $R $TASKS_SENS svm,rf,elasticnet,xgb ==="
    # shellcheck disable=SC2086
    "$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$R" \
      --modality "$MODS_MONO" --tasks "$TASKS_SENS" $COMMON_SENS \
      || { echo "FAIL sens $R"; exit 1; }
  done
fi

# --- 3) LATE (depois do SENS para scores SVM alinhados) ---
# CLAIM + LATE_GRID=full → 232 (T1|Q4 por família, k≥2). Outras coortes: 3 paper.
SPECS_FULL=()
if [[ "$LATE_GRID" == "full" ]]; then
  mapfile -t SPECS_FULL < <(emit_late_grid)
  echo "n_specs_late_claim=${#SPECS_FULL[@]} (grid T1×Q4 k≥2)"
fi
echo "n_specs_late_paper=${#SPECS_PAPER[@]}"
for C in "${COHORTS[@]}"; do
  if [[ "$C" == "$CLAIM" && "$LATE_GRID" == "full" && ${#SPECS_FULL[@]} -gt 0 ]]; then
    SPECS_RUN=("${SPECS_FULL[@]}")
  else
    SPECS_RUN=("${SPECS_PAPER[@]}")
  fi
  echo "=== LATE cohort=$C n=${#SPECS_RUN[@]} ==="
  for F in "${SPECS_RUN[@]}"; do
    if [[ "$SKIP_EXISTING" == "1" ]] && late_done "$C" "$F"; then
      echo "=== SKIP LATE $C | $F ==="
      continue
    fi
    echo "=== LATE $C | $F ==="
    # shellcheck disable=SC2086
    "$PY" 5_ablation_late_fusion.py --cohort "$C" --fusion "$F" \
      --combine mean --reuse-disk $COMMON_FUSION \
      || { echo "FAIL late $C $F"; exit 1; }
  done
done

if [[ "$SKIP_EXTRA" == "1" ]]; then
  echo "SKIP_EXTRA=1 — pula clínica/leaky"
  echo "DONE"
  exit 0
fi

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

if has_rep t1_d21; then
  echo "=== LEAKY $PRIMARY t1_d21 vol ==="
  # shellcheck disable=SC2086
  "$PY" 5_ablation_leaky.py --cohort "$PRIMARY" --representation t1_d21 \
    --inflate "" --modality vol --tasks smci_pmci $COMMON_EXTRA \
    || { echo "FAIL leaky t1_d21"; exit 1; }
fi

echo "DONE"

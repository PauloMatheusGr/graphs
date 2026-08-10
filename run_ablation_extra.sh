#!/usr/bin/env bash
# Extras na claim (default 48m_12m), depois do full + teto SVM confirmado.
# 0) sensibilidade uniclasse: t1_only vs t1_deltas × svm,rf,elasticnet
# 1) CN×AD  2) clínica  3) fusion clinic+img  4) leaky
#
# Pré-requisito: full acabou; features + 4_run_post_extract em PRIMARY.
#
# Encoding:
#   REP          → cn_ad + leaky (default t1_deltas = claim longitudinal)
#   FUSION_MOD   → clinic+img (default shape = teto unimodal)
#   FUSION_REP   → clinic+img (default t1_only = teto shape; NÃO deltas)
#
# Nota: cn_ad NÃO entra em cohort_results.csv (filtro `_cn_ad` em cohort_compare).
# Ver em 6_results.ipynb / pasta vol_cn_ad/.
#
# Uso:
#   ./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
#   PRIMARY=48m_12m REP=t1_deltas FUSION_MOD=shape FUSION_REP=t1_only ./run_ablation_extra.sh

set -euo pipefail
cd "$(dirname "$0")"

PRIMARY="${PRIMARY:-48m_12m}"
REP="${REP:-t1_deltas}"                 # cn_ad / leaky (encoding longitudinal)
FUSION_MOD="${FUSION_MOD:-shape}"       # clinic+img: teto unimodal
FUSION_REP="${FUSION_REP:-t1_only}"     # clinic+img: teto shape (não deltas)
# ponytail: sem --tasks no COMMON — argparse fica com o último --tasks; cn_ad precisa do seu
COMMON="--selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"
COMMON_SENS="--tasks smci_pmci --selection l1_stable --models svm,rf,elasticnet --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"

root_for_rep() {
  case "$1" in
    t1_only) echo "ablation_results_t1_only" ;;
    t1_deltas) echo "ablation_results_deltas" ;;
    wide|abs) echo "ablation_results" ;;
    *) echo "erro: representation desconhecida: $1" >&2; return 1 ;;
  esac
}

ROOT="$(root_for_rep "$REP")"
FUSION_ROOT="$(root_for_rep "$FUSION_REP")"

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

echo "PRIMARY=$PRIMARY"
echo "  cn_ad/leaky: REP=$REP → $ROOT"
echo "  clinic+img:  FUSION_MOD=$FUSION_MOD FUSION_REP=$FUSION_REP → $FUSION_ROOT"

# --- 0) Sensibilidade: baseline vs longitudinal × 3 modelos (reescreve mono claim) ---
for R in t1_only t1_deltas; do
  echo "=== SENS mono $PRIMARY $R svm,rf,elasticnet ==="
  # shellcheck disable=SC2086
  "$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$R" \
    --modality vol,shape,texture,disp $COMMON_SENS \
    || { echo "FAIL sens $R"; exit 1; }
done

# --- 1) Sanity CN×AD (pasta separada vol_cn_ad; não sobrescreve vol/) ---
echo "=== CN_AD $PRIMARY $REP vol → ${ROOT}/vol_cn_ad ==="
# shellcheck disable=SC2086
"$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$REP" \
  --modality vol --tasks cn_ad \
  --results-dir "csvs/cohorts/${PRIMARY}/${ROOT}/vol_cn_ad" \
  $COMMON \
  || { echo "FAIL cn_ad"; exit 1; }

# --- 2) Clínica / demográfico ---
echo "=== CLINICAL $PRIMARY ==="
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set clinical \
  --tasks smci_pmci --models svm \
  --repeats 10 --tuner optuna --optuna-trials 10 \
  || { echo "FAIL clinical"; exit 1; }

# --- 3) Fusion clínica + imagem (teto shape @ t1_only) ---
echo "=== FUSION clinic+img $PRIMARY $FUSION_MOD $FUSION_REP ==="
# shellcheck disable=SC2086
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set fusion \
  --modality "$FUSION_MOD" --representation "$FUSION_REP" --tasks smci_pmci $COMMON \
  || { echo "FAIL fusion"; exit 1; }

# --- 4) Leaky (encoding longitudinal; suplemento) ---
echo "=== LEAKY $PRIMARY $REP vol ==="
# shellcheck disable=SC2086
"$PY" 5_ablation_leaky.py --cohort "$PRIMARY" --representation "$REP" \
  --inflate "" --modality vol --tasks smci_pmci $COMMON \
  || { echo "FAIL leaky"; exit 1; }

echo "DONE extra"

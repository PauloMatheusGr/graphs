#!/usr/bin/env bash
# Extras na claim (default 48m_12m), depois do full + teto SVM confirmado.
# 0) sensibilidade uniclasse: t1_only vs t1_deltas × svm,rf,elasticnet
# 1) CN×AD  2) clínica  3) fusion clinic+img  4) leaky
#
# Pré-requisito: full acabou; features + 4_run_post_extract em PRIMARY.
# Ajustar FUSION_MOD/REP ao teto uniclasse (SVM, nocombat) da claim.
#
# Uso:
#   FUSION_MOD=shape REP=t1_deltas ./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log

set -euo pipefail
cd "$(dirname "$0")"

PRIMARY="${PRIMARY:-48m_12m}"
REP="${REP:-t1_deltas}"                 # rep da fusion / CN×AD / leaky
FUSION_MOD="${FUSION_MOD:-shape}"       # teto uniclasse (SVM nocombat)
COMMON="--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"
COMMON_SENS="--tasks smci_pmci --selection l1_stable --models svm,rf,elasticnet --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

echo "PRIMARY=$PRIMARY REP=$REP FUSION_MOD=$FUSION_MOD"

# --- 0) Sensibilidade: baseline vs longitudinal × 3 modelos (reescreve mono claim) ---
for R in t1_only t1_deltas; do
  echo "=== SENS mono $PRIMARY $R svm,rf,elasticnet ==="
  # shellcheck disable=SC2086
  "$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$R" \
    --modality vol,shape,texture,disp $COMMON_SENS \
    || { echo "FAIL sens $R"; exit 1; }
done

# --- 1) Sanity CN×AD (vol; protocolo = COMMON) ---
echo "=== CN_AD $PRIMARY $REP vol ==="
# shellcheck disable=SC2086
"$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$REP" \
  --modality vol --tasks cn_ad $COMMON \
  || { echo "FAIL cn_ad"; exit 1; }

# --- 2) Clínica / demográfico ---
echo "=== CLINICAL $PRIMARY ==="
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set clinical \
  --tasks smci_pmci --models svm \
  --repeats 10 --tuner optuna --optuna-trials 10 \
  || { echo "FAIL clinical"; exit 1; }

# --- 3) Fusion clínica + imagem (teto) ---
echo "=== FUSION clinic+img $PRIMARY $FUSION_MOD $REP ==="
# shellcheck disable=SC2086
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set fusion \
  --modality "$FUSION_MOD" --representation "$REP" $COMMON \
  || { echo "FAIL fusion"; exit 1; }

# --- 4) Leaky (mesmo encoding; suplemento) ---
echo "=== LEAKY $PRIMARY $REP vol ==="
# shellcheck disable=SC2086
"$PY" 5_ablation_leaky.py --cohort "$PRIMARY" --representation "$REP" \
  --inflate "" --modality vol $COMMON \
  || { echo "FAIL leaky"; exit 1; }

echo "DONE extra"

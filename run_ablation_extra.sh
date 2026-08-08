#!/usr/bin/env bash
# Extras na coorte de claim (default 48m_12m): CN×AD, clínica, fusion clinic+img, leaky.
# Separado do full — falha no full não bloqueia estes.
#
# Pré-requisito: features + 4_run_post_extract em PRIMARY.
# Imagem = t1_deltas; fusion clinic+img = shape (teto unimodal do claim), não vol.
# Protocolo: SVM, L1, nocombat, Optuna 10 trials / 10 repeats (igual ao full fusion).
#
# Uso:
#   ./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log

set -euo pipefail
cd "$(dirname "$0")"

PRIMARY="${PRIMARY:-48m_12m}"
REP="${REP:-t1_deltas}"
FUSION_MOD="${FUSION_MOD:-shape}"
COMMON="--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1"

mkdir -p logs
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "erro: não achei $PY" >&2
  exit 1
fi

echo "PRIMARY=$PRIMARY REP=$REP FUSION_MOD=$FUSION_MOD"

# --- 1) Sanity CN×AD (vol, mesma rep; protocolo = COMMON) ---
echo "=== CN_AD $PRIMARY $REP vol ==="
# shellcheck disable=SC2086
"$PY" 5_ablation.py --cohort "$PRIMARY" --representation "$REP" \
  --modality vol --tasks cn_ad $COMMON \
  || { echo "FAIL cn_ad"; exit 1; }

# --- 2) Clínica / demográfico (SEX AGE MMSE ADAS FAQ) ---
echo "=== CLINICAL $PRIMARY ==="
"$PY" 5_clinic_img.py --cohort "$PRIMARY" --feature-set clinical \
  --tasks smci_pmci --models svm \
  --repeats 10 --tuner optuna --optuna-trials 10 \
  || { echo "FAIL clinical"; exit 1; }

# --- 3) Fusion clínica + imagem (shape × t1_deltas; teto unimodal) ---
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

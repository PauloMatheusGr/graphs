#!/usr/bin/env bash
# Rerun shape pós-ICV homotetia. Não toca vol/texture/disp/firstorder mono.
# Longs: já gerados por 4_run_post_extract.py — este script NÃO chama 4_.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PWD}/.venv/bin/python"
COMMON='--selection l1_stable --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'
CLAIM=48m_6m
GRAD=(36m_6m 36m_12m 48m_12m 48m_6m_soft_False)

echo "=== 1) SVM shape uniclasse — claim T1 / D21 / Q4 ==="
for R in t1_only t1_d21 t1_d21_d32; do
  echo "=== SVM $CLAIM $R ==="
  "$PY" 5_ablation.py --cohort "$CLAIM" --representation "$R" \
    --modality shape --tasks smci_pmci --models svm $COMMON
done

echo "=== 2) SVM shape — gradiente + soft_False (T1 + Q4) ==="
for C in "${GRAD[@]}"; do
  for R in t1_only t1_d21_d32; do
    echo "=== SVM $C $R ==="
    "$PY" 5_ablation.py --cohort "$C" --representation "$R" \
      --modality shape --tasks smci_pmci --models svm $COMMON
  done
done

echo "=== 3) EXTRA 4 modelos shape — só claim ==="
for R in t1_only t1_d21 t1_d21_d32; do
  if [[ "$R" == t1_d21_d32 ]]; then T=cn_ad,smci_pmci; else T=smci_pmci; fi
  echo "=== EXTRA $CLAIM $R tasks=$T ==="
  "$PY" 5_ablation.py --cohort "$CLAIM" --representation "$R" \
    --modality shape --tasks "$T" --models svm,rf,elasticnet,xgb $COMMON
done

echo "=== 4) Late fusion reuse (reescreve late; mono shape já novo) ==="
# paper nas 3; full (232) na claim. SKIP_EXISTING=0 força regrava late.
SKIP_MONO=1 SKIP_EXTRA=1 SKIP_EXISTING=0 WIPE=0 \
  COHORTS="36m_6m 36m_12m 48m_12m 48m_6m" LATE_GRID=full \
  ./run_ablation_full.sh

# Opcional — só se paper usa clinic+img com shape:
echo "=== 5) clinic+img shape T1 claim ==="
"$PY" 5_clinic_img.py --cohort "$CLAIM" --feature-set fusion \
  --modality shape --representation t1_only --tasks smci_pmci $COMMON

echo "DONE shape rerun $(date -Is)"
#!/usr/bin/env bash
# Análise secundária (adendo 06/10/2026 em 0_todo.md): família só-jacobiano
# {jac_det_mean, logjac_mean, logjac_rel_mean} × L/R, mesmo protocolo do Gate B.
# Não altera o veredito do Gate B. Rodar só depois de `run_dvf_oasis_pilot.sh gate-b` terminar.
#   bash run_dvf_oasis_secondary.sh
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
mkdir -p logs

[[ -f csvs/pilot/gateB_summary.csv ]] || { echo "Gate B ainda não terminou (falta csvs/pilot/gateB_summary.csv)" >&2; exit 1; }

COHORTS=(${COHORTS:-48m_6m 48m_6m_soft_False})
ROIS=(${ROIS:-hippocampus_d2 hippocampus})
MODS=(disp_oasis_jac disp_oasis_ad_jac)
LOG="logs/dvf_oasis_secondary_$(date +%Y%m%d_%H%M%S).log"

# Idêntico a COMMON_MONO de run_dvf_oasis_pilot.sh (mesmos folds → comparação pareada válida).
COMMON_MONO=(
  --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
)

{
"$PY" pilot_oasis_gate.py qc
for C in "${COHORTS[@]}"; do
  for R in "${ROIS[@]}"; do
    for M in "${MODS[@]}"; do
      echo "=== MONO $M $C $R t1_only ==="
      "$PY" 5_ablation.py --cohort "$C" --modality "$M" --representation t1_only \
        --roi "$R" --tasks cn_ad,smci_pmci \
        --results-dir "csvs/cohorts/${C}/ablation_results_oasis/${R}/t1_only/${M}" \
        "${COMMON_MONO[@]}"
    done
  done
done
"$PY" pilot_oasis_gate.py secondary
echo "log: $LOG"
} 2>&1 | tee "$LOG"

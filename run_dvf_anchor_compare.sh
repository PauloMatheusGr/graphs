#!/usr/bin/env bash
# Compara âncoras disp: CN (disp) vs AD (disp_ad) vs CN+AD (disp_cnad).
# Mono only — late-fusion fica com disp CN (run_dvf_v4.sh).
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
mkdir -p logs backups

CLAIM="${CLAIM:-48m_6m}"
COHORTS=(${COHORTS:-48m_6m 48m_6m_soft_False})
REPS=(t1_only t1_d21 t1_d21_d32)
MODS=(disp disp_ad disp_cnad)
STAMP=$(date +%Y%m%d_%H%M%S)
BAK="backups/disp_anchor_compare_${STAMP}"
LOG="logs/ablation_disp_anchor_compare_${STAMP}.log"

COMMON_MONO=(
  --tasks smci_pmci --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
)

root_for_rep() {
  case "$1" in
    t1_only) echo ablation_results_t1_only ;;
    t1_d21) echo ablation_results_d21 ;;
    t1_d21_d32) echo ablation_results_d21d32 ;;
    *) echo "rep desconhecida: $1" >&2; return 1 ;;
  esac
}

{
echo "=== GATE: disp / disp_ad / disp_cnad longs ==="
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("csvs/cohorts/48m_6m/ablation/hippocampus")
cn = pd.read_csv(base / "disp_long.csv", nrows=2)
ad = pd.read_csv(base / "disp_ad_long.csv", nrows=2)
cnad = pd.read_csv(base / "disp_cnad_long.csv", nrows=2)
assert any(c.startswith("jac_det_") for c in cn.columns), list(cn.columns)[:20]
assert any(c.startswith("jac_det_") for c in ad.columns), list(ad.columns)[:20]
assert any(c.startswith("cn_mag_") for c in cnad.columns), list(cnad.columns)[:30]
assert any(c.startswith("ad_mag_") for c in cnad.columns), list(cnad.columns)[:30]
print("ok gate", base)
PY

echo "=== BACKUP pastas mono que serão reescritas → $BAK ==="
mkdir -p "$BAK"
for C in "${COHORTS[@]}"; do
  for R in "${REPS[@]}"; do
    for M in "${MODS[@]}"; do
      src="csvs/cohorts/${C}/$(root_for_rep "$R")/${M}"
      if [[ -d "$src" ]]; then
        mkdir -p "$BAK/${C}/$(root_for_rep "$R")"
        cp -a "$src" "$BAK/${C}/$(root_for_rep "$R")/${M}"
        echo "backed $src"
      fi
    done
  done
done

echo "=== MONO: ${MODS[*]} × ${REPS[*]} ==="
for C in "${COHORTS[@]}"; do
  for R in "${REPS[@]}"; do
    for M in "${MODS[@]}"; do
      echo "=== MONO $M $C $R ==="
      "$PY" 5_ablation.py --cohort "$C" --modality "$M" --representation "$R" \
        "${COMMON_MONO[@]}"
    done
  done
done

echo "=== DONE ==="
echo "backup: $BAK"
echo "log:    $LOG"
echo "claim:  comparar AUC mono disp vs disp_ad vs disp_cnad"
} 2>&1 | tee "$LOG"

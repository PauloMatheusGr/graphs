#!/usr/bin/env bash
# disp v4: mono + late (só uniões com disp). Não WIPE roots; não roda vol/shape/texture.
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
mkdir -p logs backups

CLAIM="${CLAIM:-48m_6m}"
# claim + sensibilidade soft; acrescenta 36m_* / 48m_12m se tabela de coortes precisar
COHORTS=(${COHORTS:-48m_6m 48m_6m_soft_False})
REPS=(t1_only t1_d21 t1_d21_d32)
LONG=t1_d21_d32
STAMP=$(date +%Y%m%d_%H%M%S)
BAK="backups/disp_v4_rerun_${STAMP}"
LOG="logs/ablation_disp_v4_${STAMP}.log"

COMMON_MONO=(
  --tasks smci_pmci --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
)
COMMON_FUSION=(
  --tasks smci_pmci --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
  --combine mean --reuse-disk
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
echo "=== GATE: disp_long = v4 ==="
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd
p = Path("csvs/cohorts/48m_6m/ablation/hippocampus/disp_long.csv")
d = pd.read_csv(p, nrows=2)
assert any(c.startswith("jac_det_") for c in d.columns), list(d.columns)[:25]
print("ok", p, "jac_det present")
PY

echo "=== BACKUP só paths que serão reescritos → $BAK ==="
mkdir -p "$BAK"
for C in "${COHORTS[@]}"; do
  for R in "${REPS[@]}"; do
    src="csvs/cohorts/${C}/$(root_for_rep "$R")/disp"
    if [[ -d "$src" ]]; then
      mkdir -p "$BAK/${C}/$(root_for_rep "$R")"
      cp -a "$src" "$BAK/${C}/$(root_for_rep "$R")/disp"
      echo "backed $src"
    fi
  done
  # late fingerprints que contêm disp (não mexe nos sem disp)
  late_root="csvs/cohorts/${C}/ablation_results_late_fusion"
  if [[ -d "$late_root" ]]; then
    mkdir -p "$BAK/${C}/ablation_results_late_fusion"
    "$PY" - <<PY
import os, shutil
from pathlib import Path
import sys
sys.path.insert(0, "modules")
from ablation_representation import iter_late_fusion_grid, fusion_fingerprint, parse_fusion_spec
late = Path("csvs/cohorts/${C}/ablation_results_late_fusion")
bak = Path("$BAK/${C}/ablation_results_late_fusion")
n = 0
for spec in iter_late_fusion_grid(longitudinal="$LONG"):
    if "disp:" not in spec:
        continue
    fp = fusion_fingerprint(parse_fusion_spec(spec))
    src = late / fp
    if src.is_dir():
        dst = bak / fp
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n += 1
print(f"backed {n} late-with-disp dirs for ${C}")
PY
  fi
done

echo "=== 1) MONO disp (só pasta disp/; irmãos intactos) ==="
for C in "${COHORTS[@]}"; do
  for R in "${REPS[@]}"; do
    echo "=== MONO disp $C $R ==="
    "$PY" 5_ablation.py --cohort "$C" --modality disp --representation "$R" \
      "${COMMON_MONO[@]}"
  done
done

echo "=== 2) LATE só specs com disp: (sem disp: intactos) ==="
"$PY" - <<PY > "/tmp/late_specs_with_disp_${STAMP}.txt"
import sys
sys.path.insert(0, "modules")
from ablation_representation import iter_late_fusion_grid
specs = [s for s in iter_late_fusion_grid(longitudinal="$LONG") if "disp:" in s]
for s in specs:
    print(s)
print(f"# n={len(specs)}", file=sys.stderr)
PY

n_specs=$(grep -cv '^#' "/tmp/late_specs_with_disp_${STAMP}.txt" || true)
echo "n_late_with_disp=$n_specs"

while IFS= read -r F; do
  [[ -z "$F" || "$F" == \#* ]] && continue
  for C in "${COHORTS[@]}"; do
    echo "=== LATE $C | $F ==="
    "$PY" 5_ablation_late_fusion.py --cohort "$C" --fusion "$F" \
      "${COMMON_FUSION[@]}"
  done
done < "/tmp/late_specs_with_disp_${STAMP}.txt"

echo "=== DONE ==="
echo "backup: $BAK"
echo "log:    $LOG"
echo "intactos: vol/shape/texture/firstorder mono; late SEM disp:"
} 2>&1 | tee "$LOG"
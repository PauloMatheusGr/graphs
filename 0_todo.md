

```bash
cd /mnt/study-data/pgirardi/graphs
.venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules')
from cohort_compare import save_cohort_comparison
root = Path('csvs/cohorts')
cohorts = [c for c in ('36m_6m','36m_12m','48m_6m','48m_12m')
           if (root / c / 'ablation_results').is_dir()]
p_res, p_feat, *_ = save_cohort_comparison(
    cohorts, Path('csvs/cohort_comparison'), cohorts_root=root, n_boot=2000)
print(p_res, p_feat)
"
```

cd /mnt/study-data/pgirardi/graphs
mkdir -p logs
PY="${PWD}/.venv/bin/python"

CLAIM="${CLAIM:-48m_12m}"
COMMON_FUSION='--tasks smci_pmci --selection l1_stable --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1'

COHORTS=(36m_6m 36m_12m 48m_6m 48m_12m)
MODS=(vol shape texture disp)
REPS=(t1_only t1_deltas)

# --- LATE ---
SPECS=()
for ((i = 0; i < ${#MODS[@]}; i++)); do
  for ((j = i + 1; j < ${#MODS[@]}; j++)); do
    for R1 in "${REPS[@]}"; do
      for R2 in "${REPS[@]}"; do
        SPECS+=("${MODS[i]}:${R1},${MODS[j]}:${R2}")
      done
    done
  done
done
for ((i = 0; i < ${#MODS[@]}; i++)); do
  for ((j = i + 1; j < ${#MODS[@]}; j++)); do
    for ((k = j + 1; k < ${#MODS[@]}; k++)); do
      for R1 in "${REPS[@]}"; do
        for R2 in "${REPS[@]}"; do
          for R3 in "${REPS[@]}"; do
            SPECS+=("${MODS[i]}:${R1},${MODS[j]}:${R2},${MODS[k]}:${R3}")
          done
        done
      done
    done
  done
done
for R1 in "${REPS[@]}"; do
  for R2 in "${REPS[@]}"; do
    for R3 in "${REPS[@]}"; do
      for R4 in "${REPS[@]}"; do
        SPECS+=("vol:${R1},shape:${R2},texture:${R3},disp:${R4}")
      done
    done
  done
done
echo "n_specs_late=${#SPECS[@]}"

for C in "${COHORTS[@]}"; do
  for F in "${SPECS[@]}"; do
    echo "=== LATE $C | $F ==="
    "$PY" 5_ablation_late_fusion.py --cohort "$C" --fusion "$F" \
      --combine mean --reuse-disk $COMMON_FUSION \
      || { echo "FAIL late $C $F"; exit 1; }
  done
done

# --- EARLY (só claim) ---
EARLY_SPECS=(
  'shape:t1_only,texture:t1_deltas'
  'shape:t1_only,vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,texture:t1_deltas'
  'shape:t1_only,vol:t1_deltas'
  'vol:t1_deltas,texture:t1_deltas'
  'shape:t1_deltas,vol:t1_deltas'
)

echo "CLAIM=$CLAIM"
for F in "${EARLY_SPECS[@]}"; do
  echo "=== EARLY $CLAIM | $F ==="
  "$PY" 5_ablation_early_fusion.py --cohort "$CLAIM" --fusion "$F" \
    $COMMON_FUSION \
    || { echo "FAIL early $F"; exit 1; }
done

echo "DONE late+early"

#!/usr/bin/env bash
# Piloto DVF OASIS: templates OASIS em MNI, SyNRA CC r4, mesmos atributos disp da ADNI.
#   bash run_dvf_oasis_pilot.sh templates   # 2.3: OASIS-3 hist-match → rígido MNI → groupwise por estrato (+ qc)
#   bash run_dvf_oasis_pilot.sh gate-a      # 80 CN/AD: registro, atributos, gate A
#   bash run_dvf_oasis_pilot.sh gate-b      # baselines 48m_6m(+soft_False): registro, 4_, ablação t1_only, gate B
# ROIs: hippocampus_d2 (núcleo + esfera de 2 voxels, principal) e hippocampus (núcleo, controle).
# SHARDS processos por âncora (CN e AD em paralelo), 1 thread ITK cada = determinístico.
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
mkdir -p logs

STAGE="${1:?uso: $0 templates|gate-a|gate-b}"
SHARDS="${SHARDS:-12}"
COHORTS=(${COHORTS:-48m_6m 48m_6m_soft_False})
ROIS=(${ROIS:-hippocampus_d2 hippocampus})
MODS=(disp_oasis disp_oasis_ad disp_oasis_cnad)
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/dvf_oasis_${STAGE}_${STAMP}.log"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

COMMON_MONO=(
  --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
)

register_and_extract() {
  local ids="$1"
  for D in CN AD; do
    for ((k = 0; k < SHARDS; k++)); do
      extra=()
      [[ $k -eq 0 ]] && extra=(--verbose-first)
      "$PY" 3.1_feat_gen_dvf.py --src oasis --diag "$D" --ids-csv "$ids" \
        --shard "$k/$SHARDS" --threads 1 "${extra[@]}" \
        > "logs/dvf_oasis_reg_${D}_${k}of${SHARDS}_${STAMP}.log" 2>&1 &
    done
  done
  wait
  grep -h "\[DONE\]\|\[ERROR\]" logs/dvf_oasis_reg_*_"${STAMP}".log || true
  for D in CN AD; do
    "$PY" 3.2_feat_dvf.py --src oasis --diag "$D" --ids-csv "$ids" &
  done
  wait
}

{
case "$STAGE" in
  templates)
    "$PY" modules/oasis_refs.py
    "$PY" 2.3_oasis_templates_mni.py select
    for ((k = 0; k < SHARDS; k++)); do
      "$PY" 2.3_oasis_templates_mni.py prep --shard "$k/$SHARDS" \
        > "logs/oasis_prep_${k}of${SHARDS}_${STAMP}.log" 2>&1 &
    done
    wait
    for f in csvs/oasis/selected_DIAG-*.csv; do
      IFS=_ read -r _ d s a _ <<< "$(basename "$f" .csv)"
      BUILD_THREADS="${BUILD_THREADS:-2}" "$PY" 2.3_oasis_templates_mni.py build "${d#DIAG-}" "${s#SEX-}" "${a#AGE-}" \
        > "logs/oasis_build_${d#DIAG-}_${s#SEX-}_${a#AGE-}_${STAMP}.log" 2>&1 &
    done
    wait
    grep -h "\[OK\]\|Error\|Traceback" logs/oasis_build_*_"${STAMP}".log || true
    "$PY" 2.3_oasis_templates_mni.py qc
    ;;
  gate-a)
    "$PY" 3.2_feat_dvf.py --src oasis --self-check
    "$PY" pilot_oasis_gate.py ids-a
    register_and_extract csvs/pilot/oasis_gateA_ids.csv
    "$PY" pilot_oasis_gate.py gate-a
    ;;
  gate-b)
    "$PY" pilot_oasis_gate.py ids-b
    register_and_extract csvs/pilot/oasis_gateB_ids.csv
    "$PY" 4_run_post_extract.py --oasis-only
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
    "$PY" pilot_oasis_gate.py gate-b
    ;;
  *) echo "estágio desconhecido: $STAGE" >&2; exit 1 ;;
esac
echo "log: $LOG"
} 2>&1 | tee "$LOG"

#!/usr/bin/env bash
# Rodada completa DVF OASIS (depois do Gate B): 2 visitas = S0,R10 (t1_r10), 3 visitas = D (t1_ols).
# 1 visita (t1_only): 48m_6m e soft_False já saem do gate-b (pulados aqui); 48m_12m (treino da
# validação externa ADNI-3/4) é feita aqui.
#   bash run_dvf_oasis_full.sh register   # todas as visitas t0/t1/t2: registro, atributos, 4_ --oasis-only
#   bash run_dvf_oasis_full.sh refs       # disp_ad / disp_cnad ADNI que faltam no disco
#   bash run_dvf_oasis_full.sh ablate     # disp_oasis* × ROIs × {t1_only, t1_r10, t1_ols} + compare pareado
#   bash run_dvf_oasis_full.sh all        # register → refs → ablate
# Retomável: registro pula warps completos, 3.2 usa done_keys, ablações existentes são puladas.
# SHARDS processos de registro por âncora (CN e AD em paralelo); JOBS ablações em paralelo.
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
mkdir -p logs

STAGE="${1:?uso: $0 register|refs|ablate|all}"
SHARDS="${SHARDS:-12}"
JOBS="${JOBS:-4}"
COHORTS=(${COHORTS:-48m_6m 48m_6m_soft_False 48m_12m})  # = FULL_COHORTS do pilot_oasis_gate.py
ROIS=(${ROIS:-hippocampus_d2 hippocampus})
REPS=(${REPS:-t1_only t1_r10 t1_ols})
MODS=(disp_oasis disp_oasis_ad disp_oasis_cnad)
IDS=csvs/pilot/oasis_full_ids.csv
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/dvf_oasis_full_${STAGE}_${STAMP}.log"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

COMMON_MONO=(
  --tasks smci_pmci --selection l1_stable --models svm --combat false
  --repeats 10 --seed 42 --tuner optuna --optuna-trials 10
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
  --stable-bootstrap 50 --stable-l1-c 0.1
)

rep_root() {
  "$PY" -c "import sys; sys.path.insert(0, 'modules'); from ablation_representation import RESULTS_ROOT_BY_PROTOCOL as R; print(R['abs']['$1'])"
}

throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
}

require() {
  local miss=0 f
  for f in "$@"; do [[ -f "$f" ]] || { echo "[FALTA] $f (ver logs/ablation_*_${STAMP}.log)"; miss=1; }; done
  return "$miss"
}

# ponytail: cópia do register_and_extract do run_dvf_oasis_pilot.sh (aquele não pode ser editado
# enquanto o Gate A/B roda: bash lê o script aos poucos). Unificar quando o piloto acabar.
register() {
  "$PY" pilot_oasis_gate.py ids-full
  for D in CN AD; do
    for ((k = 0; k < SHARDS; k++)); do
      extra=()
      [[ $k -eq 0 ]] && extra=(--verbose-first)
      "$PY" 3.1_feat_gen_dvf.py --src oasis --diag "$D" --ids-csv "$IDS" \
        --shard "$k/$SHARDS" --threads 1 "${extra[@]}" \
        > "logs/dvf_oasis_full_reg_${D}_${k}of${SHARDS}_${STAMP}.log" 2>&1 &
    done
  done
  wait
  grep -h "\[DONE\]\|\[ERROR\]" logs/dvf_oasis_full_reg_*_"${STAMP}".log || true
  for D in CN AD; do
    "$PY" 3.2_feat_dvf.py --src oasis --diag "$D" --ids-csv "$IDS" &
  done
  wait
  "$PY" 4_run_post_extract.py --oasis-only
}

refs() {
  local outs=()
  for C in "${COHORTS[@]}"; do
    for R in "${REPS[@]}"; do
      root=$(rep_root "$R")
      for M in disp_ad disp_cnad; do
        out="csvs/cohorts/${C}/${root}/${M}/ablation_results_all.csv"
        outs+=("$out")
        if [[ -f "$out" ]]; then echo "=== SKIP REF $C $R $M ==="; continue; fi
        throttle
        echo "=== REF $C $R $M ==="
        "$PY" 5_ablation.py --cohort "$C" --modality "$M" --representation "$R" "${COMMON_MONO[@]}" \
          --log-file "logs/ablation_ref_${C}_${R}_${M}_${STAMP}.log" > /dev/null &
      done
    done
  done
  wait
  require "${outs[@]}"
}

ablate() {
  local outs=()
  for C in "${COHORTS[@]}"; do
    for R in "${REPS[@]}"; do
      for ROI in "${ROIS[@]}"; do
        for M in "${MODS[@]}"; do
          dir="csvs/cohorts/${C}/ablation_results_oasis/${ROI}/${R}/${M}"
          outs+=("${dir}/ablation_results_all.csv")
          if [[ -f "${dir}/ablation_results_all.csv" ]]; then echo "=== SKIP $C $R $ROI $M ==="; continue; fi
          throttle
          echo "=== MONO $C $R $ROI $M ==="
          "$PY" 5_ablation.py --cohort "$C" --modality "$M" --representation "$R" --roi "$ROI" \
            --results-dir "$dir" "${COMMON_MONO[@]}" \
            --log-file "logs/ablation_oasis_${C}_${R}_${ROI}_${M}_${STAMP}.log" > /dev/null &
        done
      done
    done
  done
  wait
  require "${outs[@]}"
  for R in "${REPS[@]}"; do  # compare_t1_only.csv repete o gate-b nas coortes 6m e acrescenta 48m_12m
    "$PY" pilot_oasis_gate.py compare --rep "$R"
  done
}

{
case "$STAGE" in
  register) register ;;
  refs) refs ;;
  ablate) ablate ;;
  all) register; refs; ablate ;;
  *) echo "estágio desconhecido: $STAGE" >&2; exit 1 ;;
esac
echo "log: $LOG"
} 2>&1 | tee "$LOG"

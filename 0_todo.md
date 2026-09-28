# tmux a -t 1
Ordem após groupwise AD terminar
python 3.1_feat_gen_dvf.py --diag AD
python 3.2_feat_dvf.py --diag AD
# CN já feito; se precisar re-rodar:
# python 3.1_feat_gen_dvf.py --diag CN
# python 3.2_feat_dvf.py --diag CN
python 4_run_post_extract.py
bash run_dvf_anchor_compare.sh

# tmux a -t 0
# Ablation encoding D — fechar figuras artigo v2

## Feito
- [x] SVM smci: 5 coortes × {t1_only,t1_r10,t1_r10_r21,t1_ols} × 5 fam
- [x] late_ols: 232 specs (all-OLS AUC 0.791)
- [x] claim_vol4: vol × {t1_only,t1_ols} × 4 modelos (só smci)
- [x] rebuild cohort_compare

## Único run pendente — claim_core4
48m_6m × {t1_only,t1_ols} × 5 fam × {svm,rf,elasticnet,xgb} × {cn_ad,smci_pmci}.
Refaz vol (falta cn_ad). smci sai igual (seed 42). Manter svm em --models (CSV é reescrito).

```bash
tmux new -s claim_core4
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate && export PYTHONPATH=modules
mkdir -p logs
COMMON='--tasks cn_ad,smci_pmci --selection l1_stable --combat false --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0'

for rep in t1_only t1_ols; do
  for mod in vol shape texture firstorder disp; do
    echo "=== $rep | $mod $(date -Is) ==="
    python 5_ablation.py --cohort 48m_6m --representation "$rep" \
      --modality "$mod" --models svm,rf,elasticnet,xgb $COMMON \
      2>&1 | tee logs/claim_core4_${rep}_${mod}.log
  done
done
echo DONE claim_core4 $(date -Is)
```

- sem outros jobs pesados em paralelo (XGB n_jobs=-1 → contenção)
- ~1–2 dias

## Depois de DONE claim_core4
- [ ] `PYTHONPATH=modules python modules/cohort_compare.py --include-soft --n-boot 2000`
- [ ] `6_results` — adaptar: ROC × 4 modelos (T1 + D), heatmap modelos×fam (T1/D), cn_ad T1 vs D
- [ ] `6_results` V1: fig A, claim, ROC encodings, four_ceilings, soft, heatmaps, ROC modelos, cn_ad
- [ ] `7_stats` §§3–5
- [ ] tex v2: renomear `*_q4_*` → `*_ols_*` (unimodal, heatmap, roc_models)
- [ ] tex v2: inserir `trajectories_vol_48m6m.pdf` (cell `v1_fig_traj`, já pronta) no **suplemento**
- [ ] tex v2: legendas Fig A / ROC vol / tetos → "3 visitas" = OLS; tetos "vol baseline ∪ forma 3 visitas" = `late__t1_vol__t1_ols_shape`; atualizar tabelas L715/L764
- [ ] tex v2: `fig_a_encoding_4cohorts.pdf` → grelha 2×2 `fig_a_encoding_{36m6m,36m12m,48m6m,48m12m}.pdf` (`height=` igual; legenda só 36m12m)
- [ ] tex v2: `soft_true_vs_false_48m6m.pdf` → 2 subfigures `…_baseline.pdf` (sem legenda) + `…_3visits.pdf` (com legenda)
- [ ] conferir se `diagrama.pdf` desenha Δ21/Δ32 → trocar por β̂₁ OLS
- [ ] tex v2: B₀ (Q4 abs) + grelha ABCD só no suplemento (sensibilidade); corpo = A vs D

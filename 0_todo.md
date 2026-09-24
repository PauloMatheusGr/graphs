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
# Ablation encoding D — próximos passos

## Em curso (paralelo)
- [ ] **claim4** (`tmux a -t claim4`) — 4 modelos × 5 mods × {t1_only,t1_r10,t1_r10_r21,t1_ols} em `48m_6m`
  - logs: `logs/claim4_${rep}_${mod}.log`
  - DONE: `DONE claim4 …`
- [ ] **late_ols** (`tmux a -t late_ols`) — grelha `--grid` baseline=t1_only × long=t1_ols + all-T1/all-OLS
  - logs: `logs/late_ols_grid_*.log`
  - ~234 specs; DONE quando o processo termina (exit 0)

## Depois de claim4 + late_ols DONE
- [ ] Rebuild compare:

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
PYTHONPATH=modules python modules/cohort_compare.py --include-soft --n-boot 2000
```

- [ ] `6_results.ipynb` — secção **V1**: Fig A, claim T1 vs D, ROC volume (4 curvas), ROC × 4 algos, four_ceilings, soft
- [ ] `7_stats.ipynb` — §§**3–5**: D/R10/R10R21 vs T1 + gradient + four_ceilings (best_late)
- [ ] Revisar tabelas em `Artigo 1 pgirardi/tables/` + PDFs em `figures/`

# CLI — runs principais (`smci_pmci`, `--repeats 10`)

Modalidades de imagem: **`vol,shape,texture,disp,all`** (5) em abs, deltas, fusion e global.

Pool dos testes rápidos (abs, deltas, fusion, global):

```bash
--stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2
```

Modelos: `svm,rf,logreg_l1,elasticnet` (sem `xgb` até GPU/driver ok + `xgboost-cpu`).

Rodar **um experimento por vez** (não paralelizar vários `GridSearchCV`).

Cada protocolo de imagem = **5 modalidades × 4 modelos × 10 repeats = 200 jobs** (por task).

---

## 1. Clínica

Sem modalidade de imagem — só atributos clínicos baseline. Stable pool não se aplica (`selection_mode=none`).

```bash
python 2_run_baseline_comparison.py --feature-set clinical \
  --tasks smci_pmci --models svm,rf,logreg_l1,elasticnet \
  --repeats 10
```

---

## 2. Clínica + imagem

Uma modalidade por arquivo de saída (`fusion_{mod}_...`).

```bash
python 2_run_baseline_comparison.py --feature-set fusion \
  --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection mrmr_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2
```

---

## 3. Abs

```bash
python 2_run_ablation.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection mrmr_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2
```

---

## 4. Deltas

```bash
python 2_run_ablation_deltas.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection mrmr_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2
```

---

## 5. Global (leaky)

Sem `--inflate` → só leaky baseline. Para inflar mais: `--inflate max` (não comparar com abs/deltas).

```bash
python 2_run_ablation_leaky.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection mrmr_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2
```

---

## Notas

| Item | Detalhe |
|------|---------|
| Comparar modalidades | ordenar `*_summary.csv` por `auc_pooled` dentro de cada protocolo |
| `all` | merge vol+shape+texture+disp; costuma ser o job mais lento |
| `global` + `--repeats 10` | Exploratório; se tempo apertar, `--repeats 0` só no global |
| `xgb` | Adicionar depois: reboot GPU + `pip install xgboost-cpu` |
| Ordem sugerida | clínica → fusion → abs → deltas → global |
| Saídas | `csvs/longitudinal_4_groups/ablation_results_*` |
| ROC / testes | usar `*_results_all.csv` (não só `*_summary.csv`) |

# CLI — população optimo (paper)

**Base:** `csvs/longitudinal_optimo_4_groups`  
**Task principal:** `smci_pmci` | **Modelos:** `svm,rf,elasticnet` | **Repeats:** 10  
**Seleção:** `l1_stable` | **pct 70** | **timepoints 0** | **Optuna 10 trials**

Rodar **um experimento por vez**. Análise: `3_results.ipynb` (`BASE = csvs/longitudinal_optimo_4_groups`).

### Pré-requisito

```bash
# paths em 1_run_post_extract.py → longitudinal_optimo_4_groups + adnimerged_longitudinal_optimo.csv
python 1_run_post_extract.py
```

Gera `ablation/hippocampus/{vol,rad,shape,disp,merge}_long.csv`.

### Já feitos (não repetir)

| Protocolo | Modalidade | Pasta |
|-----------|------------|--------|
| wide | vol | `ablation_results/vol/` |
| t1_only | vol | `ablation_results_t1_only/vol/` |

### Paths

Defaults = `longitudinal_optimo_4_groups`. `2_run_ablation.py` aceita `--base-dir`.

---

## Flags comuns (imagem / fusion / leaky)

```bash
--selection l1_stable \
--models svm,rf,elasticnet \
--repeats 10 \
--tuner optuna --optuna-trials 10 \
--stable-pool-min-pct 70 \
--stable-pool-min-timepoints 0 \
--stable-bootstrap 50 \
--stable-l1-c 0.1
```

---

## 1. Wide (abs) — shape, texture, disp, all

```bash
python 5_run_ablation.py \
  --base-dir csvs/longitudinal_optimo_4_groups/ablation/hippocampus \
  --representation wide \
  --modality shape,texture,disp,all \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm,rf,elasticnet \
  --combat both \
  --repeats 10 \
  --tuner optuna \
  --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_optimo_4_groups/ablation_results/{modality}/`

---

## 2. T1-only — shape, texture, disp, all

```bash
python 5_run_ablation.py \
  --base-dir csvs/longitudinal_optimo_4_groups/ablation/hippocampus \
  --representation t1_only \
  --modality shape,texture,disp,all \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm,rf,elasticnet \
  --combat both \
  --repeats 10 \
  --tuner optuna \
  --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_optimo_4_groups/ablation_results_t1_only/{modality}/`

---

## 3. Clínica (sem imagem)

Stable pool / modality / combat **não** se aplicam. Não passar `--combat both`.

```bash
python 5_run_baseline_comparison.py \
  --feature-set clinical \
  --tasks smci_pmci \
  --models svm,rf,elasticnet \
  --repeats 10 \
  --tuner optuna \
  --optuna-trials 10
```

Saída: `csvs/longitudinal_optimo_4_groups/ablation_results_clinic/`

---

## 4. Fusion wide — vol

`--combat` só `false` | `true` (não `both`).

```bash
python 5_run_baseline_comparison.py \
  --feature-set fusion \
  --modality vol \
  --tasks smci_pmci \
  --representation wide \
  --selection l1_stable \
  --models svm,rf,elasticnet \
  --combat false \
  --repeats 10 \
  --tuner optuna \
  --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_optimo_4_groups/ablation_results_clinic_img/`

---

## 5. Global (leaky) — opcional / suplemento

Pré-processamento **global** antes do CV. **Não** é endpoint principal; AUC otimista — não misturar com wide/t1 leak-free.

```bash
python 5_run_ablation_leaky.py \
  --representation wide \
  --inflate "" \
  --modality vol \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm,rf,elasticnet \
  --combat both \
  --repeats 10 \
  --tuner optuna \
  --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

`--inflate ""` = só `leaky_global` (mais defensável). Outros: `pseudo`, `fulltune`, `testthr`, `max`.

Saída: `csvs/longitudinal_optimo_4_groups/ablation_results_leaky/{modality}/`

---

## Ordem sugerida

1. `4_run_post_extract.py`  
2. Clínica (rápido)  
3. Wide shape/texture/disp/all  
4. T1-only shape/texture/disp/all  
5. Fusion vol  
6. Leaky vol (se quiser suplemento)  
7. `3_results.ipynb`

---

## Referência rápida — seleção / Optuna

| Flag | Valor paper | Nota |
|------|-------------|------|
| `--selection` | `l1_stable` | bootstrap × L1 |
| `--stable-pool-min-pct` | `70` | frequência mínima |
| `--stable-pool-min-timepoints` | `0` | só frequência (sem ≥2 tempos) |
| `--stable-bootstrap` | `50` | |
| `--stable-l1-c` | `0.1` | |
| `--tuner` | `optuna` | |
| `--optuna-trials` | `10` | |

`mrmr_stable` = legado. `--stable-pool-n` só afeta mRMR.

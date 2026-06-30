# CLI — runs principais (`smci_pmci`, `--repeats 10`)

Modalidades de imagem: **`vol,shape,texture,disp,all`** (5) em abs, deltas, fusion e global.

Modelos recomendados: `svm,rf,logreg_l1,elasticnet` (sem `xgb` até GPU/driver ok + `xgboost-cpu`).

Rodar **um experimento por vez** (não paralelizar vários tuners / `GridSearchCV`).

Cada protocolo de imagem ≈ **5 modalidades × 4 modelos × 10 repeats = 200 jobs** (por task).

---

## Bloco recomendado (abs / deltas / fusion / leaky)

Combinação atual: pool estável **L1 bootstrap** + tuning **Optuna**:

```bash
--selection l1_stable \
--tuner optuna --optuna-trials 30 \
--stable-pool-n 200 \
--stable-pool-min-pct 50 \
--stable-pool-min-timepoints 2 \
--stable-bootstrap 50 \
--stable-l1-c 0.1
```

> `--stable-pool-min-pct` default no código = **70**; nos exemplos abaixo usamos **50** para testes mais rápidos.

---

## Seleção de atributos

### Modos (`--selection`)

| Modo | Pool estável | Pré-seleção final | Notas |
|------|--------------|-------------------|-------|
| **`l1_stable`** | bootstrap × L1 (após corr/var) | corr + var | **recomendado** |
| `mrmr_stable` | inner CV × mRMR | corr + var + MRMR | legado (runs antigos) |
| `mrmr` | — | corr + var + MRMR | sem pool |
| `filters` | — | corr + var | |
| `raw` | — | nenhuma | baseline sem seleção |

### Pipeline `l1_stable`

1. **Outer fold** — no treino externo, pool estável:
   - `N` bootstraps (`--stable-bootstrap`, default 50)
   - cada bootstrap: corr/var → `LogisticRegression` L1 (`--stable-l1-c`, default 0.1)
   - agrega frequência por biomarcador; filtro temporal opcional
2. **Inner CV** — no pool restrito: corr/var → classificador tunado (`grid` ou `optuna`)

### Flags do pool estável

| Flag | Default código | Significado |
|------|----------------|-------------|
| `--stable-bootstrap` | `50` | Réplicas bootstrap no outer train |
| `--stable-l1-c` | `0.1` | `C` da L1 em cada bootstrap |
| `--stable-pool-min-pct` | `70` | Biomarcador em ≥ N% dos bootstraps (ou inner folds no legado) |
| `--stable-pool-min-timepoints` | `2` | Mín. visitas T1/T2/T3 estáveis; **`0` = só frequência, sem filtro temporal** |
| `--stable-pool-n` | `50` | Só **`mrmr_stable`**: top-K mRMR por inner fold |

### Exemplos de seleção

Com filtro temporal (default):

```bash
--selection l1_stable \
--stable-pool-min-pct 50 --stable-pool-min-timepoints 2 \
--stable-bootstrap 50 --stable-l1-c 0.1
```

Sem filtro temporal:

```bash
--selection l1_stable \
--stable-pool-min-timepoints 0 \
--stable-bootstrap 50 --stable-l1-c 0.1
```

`logreg_l1` / `elasticnet`: seleção embarcada no classificador (sem estágio `preselect`); pool estável ainda restringe colunas de entrada.

---

## Tuning (`--tuner`)

Substitui `GridSearchCV` no **inner CV** (5 folds, `roc_auc`). Implementação: `ablation_optuna.py`. **Sem pruning.**

| Flag | Default | Significado |
|------|---------|-------------|
| `--tuner` | `grid` | `grid` = grade fixa; `optuna` = TPE adaptativo |
| `--optuna-trials` | `30` | Trials por outer fold (só com `--tuner optuna`) |

### Espaço Optuna (resumo)

| Modelo | Parâmetros tunados |
|--------|-------------------|
| `logreg_l1` | `clf__C` log [1e-4, 1e4], `class_weight` |
| `elasticnet` | `clf__C` log, `l1_ratio` [0.05, 0.95], `class_weight` |
| `svm` | `clf__C` log, `kernel` linear/rbf, `gamma` se rbf, `class_weight` |
| `rf` | `n_estimators` 50–500, `max_depth`, `class_weight` |
| `mlp` | `hidden_layer_sizes`, `alpha` log |
| `mrmr` / `mrmr_stable` | + `preselect__n_features_total` 10–50 |

Com `--tuner grid`, usa `PARAM_GRIDS` em `ablation_runner.py` (grade discreta).

**CSV:** coluna `tuner` (`grid` | `optuna`); `best_params` no formato sklearn (`clf__C`, …).

---

## 1. Clínica

Sem imagem — stable pool não se aplica (`selection_mode=none`).

```bash
python 2_run_baseline_comparison.py --feature-set clinical \
  --tasks smci_pmci --models svm,rf,logreg_l1,elasticnet \
  --tuner optuna --optuna-trials 30 \
  --repeats 10
```

Saída: `csvs/longitudinal_4_groups/ablation_results_clinic/`

---

## 2. Clínica + imagem (fusion)

Uma modalidade por arquivo (`fusion_{mod}_...`).

```bash
python 2_run_baseline_comparison.py --feature-set fusion \
  --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection l1_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --tuner optuna --optuna-trials 30 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

Sem filtro temporal:

```bash
python 2_run_baseline_comparison.py --feature-set fusion \
  --modality disp --tasks smci_pmci \
  --selection l1_stable --models logreg_l1,elasticnet \
  --combat false --repeats 10 --tuner optuna \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_4_groups/ablation_results_clinic_img/`

---

## 3. Abs

Default do script: `--selection raw,l1_stable` (dois modos numa execução). Nos runs abaixo: só `l1_stable`.

```bash
python 2_run_ablation.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection l1_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --tuner optuna --optuna-trials 30 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

Grade fixa (reproduzir runs antigos):

```bash
python 2_run_ablation.py --modality disp --tasks smci_pmci \
  --selection l1_stable --models logreg_l1 \
  --combat false --repeats 10 --tuner grid \
  --stable-pool-min-timepoints 2 --stable-bootstrap 50 --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_4_groups/ablation_results/{modality}/`

---

## 4. Deltas

T1 + deltas relativos (D21, D31, SLOPE). Mesmas flags de seleção/tuning.

```bash
python 2_run_ablation_deltas.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection l1_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --tuner optuna --optuna-trials 30 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

Saída: `csvs/longitudinal_4_groups/ablation_results_deltas/{modality}/`

---

## 5. Global (leaky)

Pré-processamento **global** antes do CV (z-score, ComBat se ligado, pool no dataset inteiro). **Não comparar AUC com abs/deltas** — protocolo otimista.

### `--inflate` (infladores extras)

Valores separados por vírgula. Vazio (default) = só `leaky_global` — o mais defensável.

| Valor | O que faz | Quando usar |
|-------|-----------|-------------|
| *(vazio)* | 1 linha/paciente; split ≈ por paciente; tune/threshold só no treino | **Run principal** |
| `pseudo` | ~3 linhas/paciente; split por visita | Isolar leak de pseudo-replicação |
| `fulltune` | Tuning no dataset inteiro | Isolar leak de tuning global |
| `testthr` | Threshold Youden no **teste** | Isolar leak de threshold |
| `univariate` | `SelectKBest` global + grid de K | Seleção univariada global |
| `max` | `pseudo` + `fulltune` + `testthr` + `univariate` | Pior caso exploratório (~AUC 1.0) |

Combinar: `--inflate pseudo,fulltune`. Tag `protocol` no CSV: `leaky_global` + sufixos.

```bash
# leaky baseline (recomendado)
python 2_run_ablation_leaky.py --modality vol,shape,texture,disp,all --tasks smci_pmci \
  --selection l1_stable --models svm,rf,logreg_l1,elasticnet \
  --combat false --repeats 10 \
  --tuner optuna --optuna-trials 30 \
  --stable-pool-n 200 --stable-pool-min-pct 50 --stable-pool-min-timepoints 2 \
  --stable-bootstrap 50 --stable-l1-c 0.1

# exploratório
# python 2_run_ablation_leaky.py ... --inflate pseudo
# python 2_run_ablation_leaky.py ... --inflate max
```

Saída: `csvs/longitudinal_4_groups/ablation_results_leaky/{modality}/`

---

## Defaults por script

| Script | `--selection` default | `--tuner` default |
|--------|----------------------|-------------------|
| `2_run_ablation.py` | `raw,l1_stable` | `grid` |
| `2_run_ablation_deltas.py` | `l1_stable` | `grid` |
| `2_run_ablation_leaky.py` | `l1_stable` | `grid` |
| `2_run_baseline_comparison.py` (fusion) | `l1_stable` | `grid` |

Para runs novos com pool L1 + Optuna, passar explicitamente `--selection l1_stable --tuner optuna` (como nos exemplos acima).

---

## Notas

| Item | Detalhe |
|------|---------|
| Comparar modalidades | `*_summary.csv` → `auc_pooled` dentro de cada protocolo |
| `all` | merge vol+shape+texture+disp; job mais lento |
| `global` + `--repeats 10` | exploratório; `--repeats 0` se tempo apertar |
| `mrmr_stable` | legado; manter só para comparar com CSVs antigos |
| `xgb` | adicionar depois: reboot GPU + `pip install xgboost-cpu` |
| Ordem sugerida | clínica → fusion → abs → deltas → global |
| Análise | `3_results.ipynb` (`RESULTS_ROOT` conforme protocolo) |
| ROC / testes estatísticos | `*_results_all.csv` (não só `*_summary.csv`) |


TODO

--stable-l1-c
0.1
0.01, 0.05, 0.5, 1.0
C baixo = pool mais esparsa; alto = mais features no pool
--stable-bootstrap
10
30–50 (produção)
Estabilidade do pool; 10 é só smoke
--stable-pool-min-pct
50
60–70
Pool mais restrito → menos ruído, risco de perder sinal
--stable-pool-min-timepoints
2
0 ou 2
0 = mais features; 2 = mais estável longitudinalmente
--stable-pool-n
200
ignorar em l1_stable
Só afeta mrmr_stable
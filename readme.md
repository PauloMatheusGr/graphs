# Plano experimental enxuto (paper 36m_6m)

**Base:** `csvs/cohorts/36m_6m`  
**Trocar cohort:** `--cohort` nos `4_`/`5_*` (ou `COHORT` em `6_results.ipynb`).  
**Análise:** `6_results.ipynb` → `all_protocols_summary.csv` (1 cohort) e `csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv` (multi-cohort)  
**Stats:** `7_stats.ipynb`  
**Definição de coorte:** `1_dataset1.ipynb` (protocolo actual). Legado só-mínimo / extremos → pastas `*_old`.

Rodar **um experimento por vez**. Não expandir a grade sem necessidade.

---

## Gap entre imagens: ±2 meses (não ±3)

Coortes `6m` / `12m` usam **banda min/máx** e selecção **forward a partir do baseline** (`i1` = baseline; `i2`/`i3` = próximas elegíveis na banda), não extremos do follow-up com só gap mínimo.

| Nominal | Banda adoptada |
|---------|----------------|
| 6 m | **[4, 8]** (±2) |
| 12 m | **[10, 14]** (±2) |

**Porquê ±2 e não ±3:** com ±3 fechado, 6±3 = `[3, 9]` e 12±3 = `[9, 15]` partilham o bordo **9 meses** (gap=9 cairia nas duas bandas). Mesmo com semiaberto `[3, 9)` + `[9, 15]`, as bandas ficam coladas. **±2** deixa a faixa 8–10 “órfã”, evita ambiguidade no 9 e mantém contraste limpo 6 vs 12 com perda pequena de *n*.

Protocolo antigo (só `gap ≥ 6` ou `≥ 12`, extremos + meio equidistante) ficou em `csvs/cohorts/{cohort}_old/`.

---

## Primary endpoint (pré-especificado)

Única análise “oficial” do estudo — definida **antes** de olhar rankings:

| Item | Valor |
|------|--------|
| Task | `smci_pmci` |
| Representação | `wide` (protocolo `abs`) |
| Modalidade | `vol` |
| Modelo | `svm` |
| ComBat | `false` |
| Seleção | `l1_stable`, pct `70`, timepoints `0`, bootstrap `50`, L1 `C=0.1` |
| Tuner | Optuna, `10` trials, `10` repeats |
| Métrica | patient-level AUC (OOF subject-averaged) |

Resto = ablação / sensibilidade / controle — **não** substitui o primary.

---

## Matriz enxuta (o que rodar)

| Bloco | Task | Protocolo | Modalidade | Modelos | ComBat | Obrigatório? |
|-------|------|-----------|------------|---------|--------|--------------|
| **Primary** | `smci_pmci` | wide | `vol` | `svm` | `false` | sim (já feito) |
| **A. Wide** | `smci_pmci` | wide | `vol,shape,texture,disp,all` | `svm,rf,elasticnet` | `both` | sim |
| **B. T1-only** | `smci_pmci` | t1_only | `vol,shape,texture,disp,all` | **`svm` só** | `both` | sim (claim wide vs T1) |
| **C. Clínica** | `smci_pmci` | clinic | — | `svm,rf,elasticnet` | — | sim |
| **D. Fusion** | `smci_pmci` | fusion wide | **`vol` só** | **`svm` só** (rf/en opcional) | `false` | sim |
| **E. Sanity** | `cn_ad` | wide | **`vol` só** | **`svm` só** | `false` | recomendado |
| **F. Leaky** | `smci_pmci` | leaky wide | **`vol` só** | **`svm` só** | `false` | opcional / suplemento |

### Cortar (não rodar no paper)

- Tasks: `cn_smci`, `cn_pmci`, `smci_ad`, `pmci_ad`
- Fusion / leaky / deltas em shape|texture|disp|all
- T1-only com rf/elasticnet (svm basta)
- Deltas (só se sobrar tempo)
- Optuna/repeats acima de 10

### Já feitos (não repetir)

| Protocolo | Modalidade | Pasta |
|-----------|------------|--------|
| wide | vol | `ablation_results/vol/` |
| t1_only | vol | `ablation_results_t1_only/vol/` |

---

## Flags comuns (imagem)

```bash
--tasks smci_pmci \
--selection l1_stable \
--repeats 10 \
--tuner optuna --optuna-trials 10 \
--stable-pool-min-pct 70 \
--stable-pool-min-timepoints 0 \
--stable-bootstrap 50 \
--stable-l1-c 0.1
```

---

## Pré-requisito

```bash
python 4_run_post_extract.py
```

Gera `ablation/hippocampus/{vol,rad,shape,disp,merge}_long.csv`.

---

## CLIs (ordem sugerida)

### 0. Completar wide — shape, texture, disp, all  
*(vol já feito; incluir vol de novo só se quiseres re-rodar)*

```bash
python 5_ablation.py \
  --cohort 36m_6m \
  --representation wide \
  --modality shape,texture,disp,all \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm,rf,elasticnet \
  --combat both \
  --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `ablation_results/{modality}/`

### 1. Clínica

```bash
python 5_clinic_img.py \
  --cohort 48m_12m \
  --feature-set clinical \
  --tasks smci_pmci \
  --models svm \
  --repeats 10 \
  --tuner optuna --optuna-trials 10
```

Saída: `ablation_results_clinic/`

Ou tudo extra de uma vez (CN×AD + clínica + fusion + leaky, rep=`t1_deltas`):

```bash
./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
# PRIMARY=36m_6m REP=t1_deltas ./run_ablation_extra.sh
```

### 2. T1-only — 5 modalidades, só SVM  
*(vol já feito)*

```bash
python 5_ablation.py \
  --cohort 36m_6m \
  --representation t1_only \
  --modality shape,texture,disp,all \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm \
  --combat both \
  --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `ablation_results_t1_only/{modality}/`

### 3. Fusion — vol + SVM (`t1_deltas`)

```bash
python 5_clinic_img.py \
  --cohort 48m_12m \
  --feature-set fusion \
  --modality vol \
  --tasks smci_pmci \
  --representation t1_deltas \
  --selection l1_stable \
  --models svm \
  --combat false \
  --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `ablation_results_clinic_img_deltas/` (ver `default_fusion_results_dir`)

### 4. Sanity CN×AD — vol + SVM

```bash
python 5_ablation.py \
  --cohort 48m_12m \
  --representation t1_deltas \
  --modality vol \
  --tasks cn_ad \
  --selection l1_stable \
  --models svm \
  --combat false \
  --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

### 5. Leaky (opcional) — vol + SVM

```bash
python 5_ablation_leaky.py \
  --cohort 48m_12m \
  --representation t1_deltas \
  --inflate "" \
  --modality vol \
  --tasks smci_pmci \
  --selection l1_stable \
  --models svm \
  --combat false \
  --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 \
  --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 \
  --stable-l1-c 0.1
```

Saída: `ablation_results_leaky_deltas/vol/` — **não** misturar com leak-free no abstract.

### 6. Análise

1. `6_results.ipynb` — célula consolidação → `all_protocols_summary.csv` (coluna `protocol`)  
2. `7_stats.ipynb` — testes pareados / FDR  

---

## Como reportar (abstrair)

No texto do artigo **não** listar dezenas de AUCs:

1. Primary (wide vol SVM no ComBat)  
2. Ranking modalidades (wide, 3 modelos ou foco SVM)  
3. Wide vs T1 (SVM; Δ + bootstrap)  
4. Clínica vs imagem vs fusion  
5. CN×AD sanity + ComBat/leaky em suplemento se útil  

---

## Referência rápida — flags

| Flag | Valor paper |
|------|-------------|
| `--selection` | `l1_stable` |
| `--stable-pool-min-pct` | `70` |
| `--stable-pool-min-timepoints` | `0` |
| `--stable-bootstrap` | `50` |
| `--stable-l1-c` | `0.1` |
| `--tuner` / `--optuna-trials` | `optuna` / `10` |
| `--repeats` | `10` |

Modos `mrmr` / `mrmr_stable` removidos; seleção = `l1_stable` (+ corr/var).

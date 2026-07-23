# Plano de ablações e discussão do artigo

**Primary:** `csvs/cohorts/36m_6m`  
**Endpoint:** `smci_pmci` · wide (abs) · `vol` · `svm` · ComBat `false` · `l1_stable`  
**Análise:** `6_results.ipynb` + `7_stats.ipynb` (stats só no primary)  
**Pré-requisito:** `4_run_post_extract.py` nas 6 cohorts

---

## Racional geral

| Decisão | Porquê |
|---------|--------|
| **Uma** coorte primary (`36m_6m`) | Hipótese pré-especificada; evita pesca de significância |
| Matriz completa só no primary | Modalidades/modelos/ComBat respondem *o quê medir* |
| Outros cohorts = só `vol` wide | Respondem *em quem* (definição de follow-up); sensibilidade descritiva |
| Kernel = **wide vs T1** | Claim central: valor longitudinal vs baseline único (só imagem) |
| Stats formais só em `36m_6m` | Cohorts aninhados — não testar “qual cohort é melhor” |
| Sem deltas / tasks periféricas | Plano enxuto; YAGNI |

**Mensagem do artigo (1 frase):**  
Em MCI, representação longitudinal de atributos de imagem (esp. volumétricos) melhora a discriminação sMCI×pMCI face ao baseline único, de forma robusta a definições alternativas de follow-up e com controlos metodológicos (nested CV, estabilidade, leaky, clínico).

---

## Checklist operacional

- [x] Extração features (`all_population`: vol, rad, DVF)
- [x] `4_run_post_extract.py` × 6 cohorts
- [x] A1 Wide `36m_6m` (tmux `36m_6m_wide`)
- [x] A2 T1-only `36m_6m` (tmux `36m_6m_t1`)
- [ ] A3 Clínica
- [ ] A4 Fusion
- [ ] A5 Sanity CN×AD
- [ ] A6 Leaky (opcional)
- [ ] B Sensibilidade vol × 5 cohorts
- [ ] `6_results` → `cohort_comparison/`
- [ ] `7_stats` (primary)

---

## 0. Flags comuns

```bash
source .venv/bin/activate
cd /mnt/study-data/pgirardi/graphs

COMMON="--tasks smci_pmci --selection l1_stable --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1"
```

`$COMMON` / `$C` = variáveis bash (shell expande antes do Python).

---

## 1. CLIs das ablações

### A. Primary `36m_6m` — matriz paper

#### A1. Wide (abs) — todas as modalidades

**Racional:** ranking de famílias de atributos + modelos + ComBat no endpoint longitudinal.

```bash
C=36m_6m
python 5_run_ablation.py --cohort $C --representation wide \
  --modality vol,shape,texture,disp,all \
  --models svm,rf,elasticnet --combat both $COMMON
```

Saída: `csvs/cohorts/36m_6m/ablation_results/{modality}/`

#### A2. T1-only — claim longitudinal (kernel)

**Racional:** isola o ganho de t1/t2 vs só baseline; sustenta narrativa “imagem longitudinal”.

```bash
C=36m_6m
python 5_run_ablation.py --cohort $C --representation t1_only \
  --modality vol,shape,texture,disp,all \
  --models svm --combat both $COMMON
```

Saída: `csvs/cohorts/36m_6m/ablation_results_t1_only/{modality}/`

#### A3. Clínica

**Racional:** piso demográfico/clínico; imagem precisa superar ou complementar escalas.

Nota: `5_run_baseline_comparison.py` usa constante `COHORT` (sem `--cohort`). Garantir `36m_6m`.

```bash
python 5_run_baseline_comparison.py --feature-set clinical \
  --tasks smci_pmci --models svm,rf,elasticnet \
  --repeats 10 --tuner optuna --optuna-trials 10
```

Saída: `ablation_results_clinic/`

#### A4. Fusion (imagem + clínico)

**Racional:** valor incremental da RM quando o clínico já está no modelo.

```bash
python 5_run_baseline_comparison.py --feature-set fusion \
  --modality vol --tasks smci_pmci --representation wide \
  --selection l1_stable --models svm --combat false $COMMON
```

Saída: `ablation_results_clinic_img/`

#### A5. Sanity CN × AD

**Racional:** ceiling do pipeline em tarefa fácil; contraste com sMCI×pMCI.

```bash
python 5_run_ablation.py --cohort 36m_6m --representation wide \
  --modality vol --tasks cn_ad --models svm --combat false $COMMON
```

#### A6. Leaky (opcional / suplemento)

**Racional:** controlo metodológico — selecção global infla AUC?

```bash
python 5_run_ablation_leaky.py --cohort 36m_6m --representation wide \
  --inflate "" --modality vol --models svm --combat false $COMMON
```

Saída: `ablation_results_leaky/vol/` — **não** misturar com wide/t1 no abstract.

#### Fora do plano

- Deltas (`5_run_ablation_deltas.py`)
- Tasks: `cn_smci`, `cn_pmci`, `smci_ad`, `pmci_ad`
- Fusion/leaky em shape|texture|disp|all
- T1-only com rf/elasticnet

---

### B. Sensibilidade — outros 5 cohorts (só vol)

**Racional:** mesma medida (primary vol), muda só a definição de população (`t_janela` × `t_imagens`). Sem FDR multi-cohort; tabela descritiva AUC ± IC.

```bash
for C in 36m_9m 36m_12m 48m_6m 48m_9m 48m_12m; do
  python 5_run_ablation.py --cohort $C --representation wide \
    --modality vol --models svm --combat false $COMMON
done
```

Opcional (gap wide–T1 na sensibilidade):

```bash
for C in 36m_9m 36m_12m 48m_6m 48m_9m 48m_12m; do
  python 5_run_ablation.py --cohort $C --representation t1_only \
    --modality vol --models svm --combat false $COMMON
done
```

**Porquê não 5 modalidades nos outros cohorts?**  
Modalidades exploram *feature family* (já respondido no primary). Cohorts exploram *definição demográfica*. Cruzar as duas grelhas = custo ×6 e pesca de p-values.

---

### C. Pós-experimento

1. `6_results.ipynb` → `all_protocols_summary.csv` + `csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv`
2. `7_stats.ipynb` → **só** `36m_6m`

| Ficheiro | Conteúdo |
|----------|----------|
| `cohort_results.csv` | Métricas × cohort × config; `n_features_mean` |
| `cohort_features_long.csv` | Atributos estáveis (`anatomical_key`, `pct`, freq T1/T2/T3) |
| `stats_*.csv` | ΔAUC pareado / FDR no primary |

---

## 2. Ordem tmux

| Fase | Sessões | Jobs |
|------|---------|------|
| 1 | `36m_6m_wide`, `36m_6m_t1` | A1 + A2 em paralelo |
| 2 | 1 sessão | A3 → A4 → A5 (+ A6) |
| 3 | 5 tmux leves | Loop B (só vol) |
| 4 | — | `6_results` + `7_stats` |

Não lançar 6 matrizes full em paralelo.

---

## 3. Contagem

| Item | Contagem |
|------|----------|
| Configs `36m_6m` (A1–A5, sem leaky) | ~45 |
| + leaky | +1 |
| Sensibilidade 5× vol | +5 |
| **Invocações CLI** | ~10–11 |

Um CLI com várias mods/modelos/`combat both` = várias configs no mesmo processo.

---

## 4. Testes / análises por bloco

| Bloco | Pergunta | Análise |
|-------|----------|---------|
| Primary wide/vol/svm/nocombat | Separar sMCI×pMCI com imagem longitudinal? | AUC patient-level ± IC |
| Wide × modalidades | Qual família carrega o sinal? | Ranking AUC; features estáveis |
| Wide × modelos | Sinal depende do classificador? | svm vs rf vs elasticnet |
| ComBat on/off | Harmonização ajuda ou apaga sinal? | Δ AUC |
| **Wide vs T1** | Visitas extras > baseline só? | ΔAUC pareado + FDR (`7_stats`) |
| Clínica | Escalas sozinhas separam? | AUC vs imagem |
| Fusion | RM acrescenta ao clínico? | clinic vs img vs fusion |
| Demografia | Grupos balanceados? Idade/sexo preveem? | MW, χ², permutação |
| CN×AD | Pipeline OK em tarefa fácil? | Sanity |
| Leaky | Selecção global infla? | abs vs leaky |
| Sensibilidade cohorts | Depende de janela/gap? | Tabela descritiva |

### Balanceamento (igual em todos os cohorts)

- Sem SMOTE / undersampling
- `StratifiedKFold` (outer + inner)
- `class_weight ∈ {None, "balanced"}` (hiperparâmetro)
- Tuning / primary metric: ROC AUC

---

## 5. Discussão do artigo

### Núcleo

1. Valor longitudinal (wide > T1) — ou honestidade se Δ≈0
2. Modalidade vencedora (vol vs radiomics vs DVF)
3. Atributos estáveis (interpretabilidade)

### Metodologia

4. Nested CV + patient-level AUC  
5. `l1_stable`  
6. ComBat (over-correction?)  
7. Controlo leaky  
8. Desbalanceamento via estratificação + pesos + AUC  

### Clínica

9. Imagem vs clínico vs fusion  
10. CN×AD como ceiling  

### Limitações

11. Soft-pMCI / definição de conversão  
12. Robustez multi-cohort (sem overclaim em `*12m`)  
13. ADNI / generalização  
14. ROI hipocampo  

### Framing sugerido

| Secção | Conteúdo |
|--------|----------|
| Resultados 1 | Primary AUC + features estáveis |
| Resultados 2 | **Wide vs T1** (claim principal) |
| Resultados 3 | Modalidades / modelos / ComBat |
| Resultados 4 | Clínico / fusion |
| Resultados 5 | Sensibilidade 6 cohorts |
| Suplemento | Leaky, cn_ad |

---

## 6. Flags CLI (`--cohort`)

| Script | `--cohort`? |
|--------|-------------|
| `5_run_ablation.py` | sim |
| `5_run_ablation_leaky.py` | sim |
| `5_run_ablation_deltas.py` | sim (fora do plano) |
| `4_run_post_extract.py` | não — constante `COHORT` |
| `5_run_baseline_comparison.py` | não — constante `COHORT` |

Ver também `readme.md` (plano experimental enxuto).

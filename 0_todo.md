# O que falta para fechar o artigo

Actualizado 2026-07-25. Primary = `36m_6m`.

```bash
source .venv/bin/activate
cd /mnt/study-data/pgirardi/graphs

COMMON="--tasks smci_pmci --selection l1_stable --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1"

COMMON_NOTASK="--selection l1_stable --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1"
```

---

## 1. Já feito 🟠

| # | Experiência | Pasta / nota |
|---|-------------|--------------|
| 1 | Features vol / rad / DVF CN-template | `features_displacement.csv` · `images/displacement_field/` |
| 2 | Wide `36m_6m` todas modalidades | `ablation_results/{vol,shape,texture,disp,all}/` |
| 3 | T1 `36m_6m` vol,shape,texture,disp,all | `ablation_results_t1_only/` |
| 4 | Backup wide DVF CN | `ablation_results/disp_cn_backup/` |
| 5 | `*_long.csv` hipocampo nos 6 cohorts | `disp` = CN-template (`DISP_FEATURES`) |
| 6 | Clínica | `ablation_results_clinic/` |
| 7 | Fusion clinic+vol | `ablation_results_clinic_img/` |
| 8 | Sanity CN×AD | `ablation_results/vol_cn_ad/` |
| 9 | Leaky baseline (`--inflate ""`) | `ablation_results_leaky/vol/` |

**DVF follow-up→baseline (intra-sujeito): descartado.** AUC ~chance; só 2 TPs (sem linha baseline); `t1_only` deixa de ser baseline clínico. Scripts/artefactos long removidos. Oficial = visita → template CN.

---

## 2. A correr agora

Nada. Pipeline DVF long cancelado; `disp_long.csv` dos 6 cohorts regenerado com legado.

---

## 3. Ainda falta

### B — Leaky com outros `--inflate` (suplemento)

Já tens `--inflate ""` (só leaky_global). Para o suplemento completo:

```bash
# pseudo-replicação
python 5_run_ablation_leaky.py --cohort 36m_6m --representation wide \
  --inflate pseudo --modality vol --models svm --combat false $COMMON

# pseudo + fulltune
python 5_run_ablation_leaky.py --cohort 36m_6m --representation wide \
  --inflate pseudo,fulltune --modality vol --models svm --combat false $COMMON

# pior caso (opcional)
python 5_run_ablation_leaky.py --cohort 36m_6m --representation wide \
  --inflate max --modality vol --models svm --combat false $COMMON
```

### C — Gradiente temporal (hipótese do artigo)

`disp` oficial = CN-template em todos os cohorts. Ablação nos 5:

```bash
for C in 36m_9m 36m_12m 48m_6m 48m_9m 48m_12m; do
  python 5_run_ablation.py --cohort $C --representation wide \
    --modality vol,shape,texture,disp,all \
    --models svm --combat false $COMMON

  python 5_run_ablation.py --cohort $C --representation t1_only \
    --modality vol,shape,texture,disp,all \
    --models svm --combat false $COMMON
done
```

(Mínimo CPU: só `--modality vol`.)

### D — Planilhas

1. `6_results.ipynb` — consolidação + comparação cohorts  
2. `7_stats.ipynb` — FDR no primary

---

## Checklist curta

- 🟠[x] Wide + T1 primary (incl. T1 disp smci_pmci)
- 🟠[x] Clínica + Fusion + Sanity + Leaky (`inflate ""`)
- 🟠[x] DVF oficial = CN-template (long descartado; `4` × 6 com legado)
- [ ] Leaky `pseudo` / `pseudo,fulltune` / (`max`)
- [ ] Gradiente 5 cohorts
- [ ] `6_results` + `7_stats`

---

# Pipeline do estudo e plano experimental

Documento de referência: **o quê** fazemos, **como**, e **porquê** — do ADNIMERGED bruto até às ablações e à discussão do artigo.

Lê isto até conseguires explicar cada etapa sem consultar o código. Os CLIs estão no final (secção 10).

## Mapa mental (1 frase cada)

| Bloco | Pergunta |
|-------|----------|
| Wide vs T1 no `36m_6m` | Longitudinal ajuda no primary? |
| Clínica / fusion / sanity / leaky | Piso clínico, RM acrescenta, ceiling, leakage? |
| Modalidade `disp` (CN-template) | Deformação vs CN ajuda? (comparável wide↔T1) |
| 5 cohorts (Passo 2) | Gap maior → Δ(wide−T1) maior? |
| `6` + `7` | Tabelas + testes formais |

---

## Mensagem do artigo (1 frase)

Em MCI, a representação **longitudinal** de atributos de imagem (várias visitas) deve discriminar sMCI×pMCI **melhor** que o baseline único (análise transversal), e esse ganho deve **aumentar** quando o intervalo entre imagens (`t_imagens`) e/ou a janela clínica (`t_janela`) crescem — porque há mais tempo para diferenças anatómicas se pronunciarem.

**Primary formal (pré-especificado):** `36m_6m` · `smci_pmci` · wide · `vol` · `svm` · ComBat `false` · `l1_stable`.  
**Kernel científico:** wide vs T1 em **6 cenários** (`t_janela` ∈ {36,48} × `t_imagens` ∈ {6,9,12}).

---

## Mapa mental do pipeline

```
ADNI brutos
    → analysis_adni.ipynb          [QC + merge MRI↔DX↔scores]
    → csvs/adnimerged.csv
    → 1_dataset.ipynb              [rótulos clínicos + escolha de 3 imagens]
    → csvs/cohorts/{Xm_Ym}/ + all_population/
    → 2_resample.py                [espaço comum MNI 1 mm]
    → 3_feat_vol / rad / gen_dvf / dvf   [atributos na store união]
    → 4_run_post_extract.py        [filtra cohort → *_long.csv hipocampo]
    → 5_run_ablation*.py           [nested CV, wide/t1/clinic/fusion/leaky]
    → 6_results + 7_stats          [tabelas, ΔAUC, FDR só no primary]
```

---

## 1. Filtragem upstream — `../datasets/analysis_adni.ipynb`

**Papel:** construir uma tabela MRI–clínica limpa **antes** das definições de coorte do paper.  
**Saída consumida pelo graphs:** `graphs/csvs/adnimerged.csv` (cópia/uso de `datasets/output/adni/adnimerged.csv`).

### 1.1 Merge MRI ↔ diagnóstico ↔ escalas

| Passo | Regra | Racional |
|-------|--------|----------|
| Sexo | Remove `SEX == 'X'` | Sexo inválido |
| Outra demência | Remove `DXOTHDEM == 1` | Foco em trajectória AD / MCI / CN |
| DX / datas vazias | Remove | Sem rótulo temporal não há coorte |
| Reversão AD | Exclui paciente se CN/MCI **depois** do 1.º AD | Trajectória clínica incoerente |
| Reversão MCI | Exclui se CN **depois** do 1.º MCI | Idem |
| Escalas | ADAS, CDR, MMSE, FAQ sem buracos no merge | Precisamos de clínico para baseline/fusion |
| Proximidade MRI–DX | `merge_asof` nearest; \|MRI−DX\| ≤ **3 meses** (`time_diff_mri_diag=3`, ≈90 dias) | Score clínico “da mesma visita” |
| Campo magnético | Valores >2.5 → 3.0 | Normaliza 2.9 T ≈ 3 T |
| Outliers | Remove `ID_IMG` listados em `outliers_adni.txt` | QC de imagem (pipeline de detecção à parte) |

**Porquê esta etapa existe:** sem QC e alinhamento temporal MRI–diagnóstico, qualquer definição de sMCI/pMCI a jusante herda lixo.

### 1.2 Critérios “legado” no mesmo notebook

Há um bloco CRITERIA antigo (janelas deslizantes, abordagem 4, etc.) que gera `image_data.txt` / combinações.  
**Não é** o que alimenta `csvs/cohorts/` do paper. As coortes de análise vêm do redefine em `1_dataset.ipynb` (secção 2).

---

## 2. Definição populacional — `1_dataset.ipynb`

**Entrada:** `csvs/adnimerged.csv` (já filtrado).  
**Saída:** `csvs/cohorts/{t_janela}m_{t_imagens}m/adnimerged_longitudinal.csv` + união `all_population/all_population.csv`.

### 2.1 Ideia central (duas fases)

1. **Rótulo clínico** com horizonte `t_janela` a partir do primeiro MRI (`t0`).  
2. **Escolha de exactamente `qtd_imagens` (=3) MRIs** com espaçamento mínimo `t_imagens`, para predictores.

Slots `t0/t1/t2` = ordem no conjunto escolhido (**não** visitas ADNI fixas m12/m24).

### 2.2 Parâmetros

| Símbolo | Primary | Grelha de sensibilidade | Significado |
|---------|---------|-------------------------|-------------|
| `T_JANELA` | **36** m | 36, 48 | Horizonte clínico desde `t0` |
| `T_IMAGENS` | **6** m | 6, 9, 12 | Gap mínimo entre imagens consecutivas |
| `QTD_IMAGENS` | **3** | 3 | Sempre triplo longitudinal |
| `SOFT_PMCI` | **True** | True | Permite 1.º AD só no **último** slot se faltar MCI pré-AD |

**Racional `t_janela≈36`:** tempo MCI→AD no ADNIMERGED tem mediana ~19–20 meses; 36 m cobre a maioria das conversões.  
**Racional `t_imagens`:** gap curto (6 m) ≈ quase transversal; gaps maiores → mais mudança estrutural esperada → wide deve afastar-se mais de T1.  
**Racional soft-pMCI:** recupera conversores com poucas MRI pré-AD **sem** reescrever o DIAG observado (fica AD; `DIAG_EFFECTIVE=MCI_soft`).

### 2.3 Regras de GROUP (`classify_patient`)

`t0` = data do 1.º MRI do paciente (após dedupe). `t_end = t0 + T_JANELA`.

| GROUP | Regra (linguagem clara) | Racional |
|-------|-------------------------|----------|
| **CN** | Só CN na trajectória; tem exame ≥ `t_end` ainda CN | Confirma estabilidade no horizonte |
| **AD** | Começa e permanece AD | Grupo de sanity / contraste |
| **pMCI** | Começa MCI; 1.º AD em `(t0, t_end]`; sem reversões; predictores **antes** do AD | Conversor *dentro* da janela |
| **sMCI** | Começa MCI; **não** converte até `t_end`; confirmação pós-janela ainda MCI | *Não-conversor na janela* — **não** “MCI eterno” |

Exclusões típicas: trajectória mista, reversão diagnóstica, sem confirmação pós-janela, imagens insuficientes.

### 2.4 Selecção das 3 imagens (`select_images`)

1. Pool de predictores no intervalo correcto (pMCI: até antes do AD).  
2. Escolhe extremos (mais cedo / mais tarde) + ponto(s) intermédios equidistantes.  
3. Gaps consecutivos ≥ `T_IMAGENS`.  
4. Soft: se falhar, último slot = 1.º AD (só pMCI).

**Racional:** mesmo protocolo temporal para todos os pacientes → comparáveis; extremos + equidistantes maximizam cobertura do intervalo sem cherry-pick manual.

### 2.5 Seis cohorts em disco

`36m_6m` (primary), `36m_9m`, `36m_12m`, `48m_6m`, `48m_9m`, `48m_12m`.

`all_population/` = união de `ID_IMG` (extração **uma vez**; análise filtra por cohort).

### 2.6 Hipótese multi-cohort (porquê 6, não 1)

| Expectativa | Motivo |
|-------------|--------|
| Em `36m_6m`, wide ≈ T1 é **plausível** | Gap 6 m: pouca mudança estrutural |
| Δ(wide−T1) **sobe** com `t_imagens` | Mais tempo entre scans → anatomia diverge |
| `t_janela` 36 vs 48 | Definição de conversão / follow-up mais longa muda quem é pMCI |

**Confounds a declarar:** N cai com critérios mais duros; % soft-pMCI sobe; quem tem 3 MRI com gap 12 m ≠ população com gap 6 m (viés de retenção). Análise = tendência descritiva + IC; FDR formal só no primary.

---

## 3. Pré-processamento de imagem — `2_resample.py`

| Item | Valor | Racional |
|------|--------|----------|
| Entrada | `all_population.csv` + T1 bias-corrected | Lista união |
| Registo | ANTs **Rigid** → MNI 152 1 mm | Espaço comum sem warping não-linear agressivo na anatomia nativa da feature vol/rad |
| Labels | Nearest-neighbor | Preserva IDs de ROI |
| Saídas | `images/resampled_1.0mm/`, regions, seg, brain_mask | Base de vol/rad/DVF |

**Porquê:** volumetria, radiomics e DVF precisam da mesma geometria.

---

## 4. Extração de atributos — `3_feat_*.py`

Tudo grava em `csvs/cohorts/all_population/`. Resume/skip IDs já feitos.

### 4.1 Volumetria — `3_feat_vol.py` → `features_volumetric.csv`

Por ROI (20 estruturas L/R, hipocampo…ínsula) + linha `__global__`:  
`mask/gm/wm/csf/tissues` em mm³ e `*_norm` (fracções).

**Racional:** atrofia / composição tecidual = âncora clínica clássica em AD.

### 4.2 Radiomics — `3_feat_rad.py` → `features_radiomic.csv`

PyRadiomics, `binCount=64`, imagem Original: firstorder, shape, glcm, glrlm, glszm, ngtdm, gldm.

**Racional:** textura/forma além do volume; standard IBSI-like via PyRadiomics (não LBP).

### 4.3 DVF — `3_feat_gen_dvf.py` + `3_feat_dvf.py` → `features_displacement.csv`

| Item | Valor |
|------|--------|
| Âncora | visita → template CN (sexo/idade do baseline do paciente) |
| Warps | `images/displacement_field/` |
| Features | `csvs/cohorts/all_population/features_displacement.csv` |
| `4_run_post_extract.py` | `DISP_FEATURES = "features_displacement.csv"` |

Fluxo: cada `ID_IMG` (T1/T2/T3) tem warp → CN; domínio = imagem clínica; `ref_tag` identifica o template.  
**Racional:** deformação vs CN = “afastamento do envelhecimento típico”; 3 visitas → wide vs `t1_only` comparáveis.

**Descartado:** follow-up→baseline do paciente (intra-sujeito). Só 2 TPs (sem i1→i1); `t1_only` deixava de ser baseline; AUC fraca no primary.

Ordem: `python 3_feat_gen_dvf.py` → `python 3_feat_dvf.py` → `python 4_run_post_extract.py`.

---

## 5. Pós-extração por cohort — `4_run_post_extract.py`

| Item | Detalhe | Racional |
|------|---------|----------|
| Features | Lê **sempre** `all_population/features_*.csv` | Não reextrair 6 vezes |
| Meta | Filtra `ID_IMG` do `adnimerged_longitudinal` do cohort | Só pacientes da definição |
| ICV | Normaliza tamanhos radiomics/vol com ICV global | Comparabilidade de tamanho craniano |
| Batch ComBat | `MANUFACTURER` + `FIELD_STRENGTH` | Scanner como batch |
| ROI ablação | **hipocampo** L/R | Foco anatómico AD; reduz dimensionalidade |
| Saídas | `ablation/hippocampus/{vol,shape,rad,disp,merge}_long.csv` | Uma linha por imagem×ROI |

**Nota CLI:** hoje **sem** `--cohort` — editar constante `COHORT` no script.

---

## 6. O que entra no classificador (modalidades)

O CSV long pode ser “gordo”; o modelo só vê **allowlists** (`modules/ablation_prep.py`):

| Modalidade | Sufixos / features | Racional do corte |
|------------|--------------------|-------------------|
| **vol** | `gm_norm`, `wm_norm`, `csf_norm` | Frações teciduais; evita mm³ brutos redundantes com ICV |
| **shape** | MeshVolume, SurfaceArea, SurfaceVolumeRatio, Sphericity, Elongation, Flatness | Forma clássica; corta diâmetros redundantes |
| **texture** | GLCM Contrast, Correlation, Idm, JointEntropy | GLCM canónico; corta GLRLM/GLSZM/… (enxuto 2026-07) |
| **disp** | logjac + strain_fro (mean/std/skewness/kurtosis) | Momentos do artigo; corta ux/uy/uz, percentis, etc. |
| **all** | união das acima | Multimodalidade imagem |

Wide pivot: colunas tipo `hippocampus_L_T1_gm_norm`, `…_T2_…`, `…_T3_…`.

---

## 7. Representações (protocolos)

| Protocolo | Features temporais | Pasta resultados | Papel |
|-----------|--------------------|------------------|-------|
| **wide** (abs) | T1+T2+T3 | `ablation_results/` | Análise **longitudinal** |
| **t1_only** | só T1 (baseline do conjunto) | `ablation_results_t1_only/` | Análise **transversal** de controlo |
| clinic | SEX, AGE, MMSE, ADAS, FAQ (t0) | `ablation_results_clinic/` | Piso clínico |
| fusion | clinic + imagem wide vol | `ablation_results_clinic_img/` | Incremental RM |
| leaky | wide com ops globais | `ablation_results_leaky/` | Controlo de vazamento (suplemento) |
| deltas | fora do plano paper | — | YAGNI |

**Comparação justa wide vs T1:** mesma task, seleção, repeats, Optuna, **mesmo modelo**, **mesmo ComBat**, mesma modalidade.  
O Δ reportado no artigo = **svm ↔ svm** (e combat emparelhado).

---

## 8. Aprendizagem / CV / desbalanceamento / modelos

### 8.1 Nested CV (`modules/ablation_runner.py`)

| Peça | Valor | Racional |
|------|--------|----------|
| Unidade de split | **paciente** (`ID_PT`) | Evita leak entre visitas do mesmo sujeito |
| Outer / inner | StratifiedKFold **5×5** | Proporção de classes nos folds |
| Repeats | **10** | Estabilidade da estimativa |
| Seed base | 42 | Reprodutibilidade |
| Métrica de tuning | ROC **AUC** | Ranking; robusta a desbalanceamento |
| Métrica primary reportada | **`auc_patient_mean`** (média OOF por paciente → AUC) | Alinhada a decisão clínica por sujeito |
| Limiar | Youden no inner OOF | Operacionaliza sens/spec (secundário) |

### 8.2 Desbalanceamento (ponderação, não reamostragem)

**Não** há SMOTE / undersampling.

1. **Estratificação** — folds com a mesma proporção sMCI/pMCI do cohort.  
2. **`class_weight ∈ {None, "balanced"}`** — hiperparâmetro (grid/Optuna) em svm/rf/elasticnet: se `"balanced"`, a classe minoritária pesa mais no treino (`n / (n_classes × n_c)`).  
3. **AUC** como alvo — não optimizamos accuracy enviesada para a maioria.

**Como explicar em 15 s:**  
*Não mexemos na amostra; mantemos a proporção nos folds; o modelo pode pesar mais a classe rara se isso melhorar a AUC.*

**Porquê igual em todos os cohorts:** diferenças de AUC reflectem população/tempo, não mudança de receita de balanceamento.

### 8.3 Selecção de atributos — `l1_stable`

No **outer train** apenas:

1. Bootstrap ×50  
2. Filtro correlação/variância → Logistic L1 `C=0.1`  
3. Mantém features em ≥ **70%** dos boots (`--stable-pool-min-pct 70`)  
4. `--stable-pool-min-timepoints 0` (não exige estabilidade em T1 e T2 e T3 em simultâneo)  
5. Pipeline: scaler → classificador (Optuna, 10 trials)

**Racional:** reduz overfitting e dá lista de atributos **estáveis** interpretáveis (não só AUC).

### 8.4 ComBat

Opcional por fold: fit no train, aplica no test; batch = fabricante × campo.  
Paper primary = **false**. Ablação = `both` no wide/t1 do `36m_6m` para ver se harmonização ajuda ou apaga sinal.

### 8.5 Modelos

| Modelo | Onde | Racional |
|--------|------|----------|
| **svm** | Primary + todos os contrastes temporais | Endpoint pré-especificado; linear/RBF + `class_weight` |
| **rf**, **elasticnet** | **Só** wide `36m_6m` (ablacão de algoritmo) | “O sinal depende do classificador?” — uma vez, no endpoint rico |
| Não no T1 / outros cohorts | — | Fixar svm isola a pergunta temporal; evita misturar algoritmo × tempo |

Clínica / fusion / sanity / leaky / sensibilidade: **svm** (alinhado ao primary).

### 8.6 Clínica — variáveis

`SEX`, `AGE`, `MMSE_SCORE`, `ADAS_SCORE`, `FAQ_SCORE` (CDR comentado no código).  
Baseline = visita `t0` do conjunto.

---

## 9. Camada de análise

| Artefacto | Conteúdo | Racional |
|-----------|----------|----------|
| `6_results.ipynb` | Heatmaps, ROC, summary 1 cohort | Figuras do primary |
| `cohort_results.csv` | AUC ± IC × cohort × config | Gradiente temporal descritivo |
| `cohort_features_long.csv` | Features estáveis (`pct`, T1/T2/T3) | Monitorar *quais* atributos |
| `7_stats.ipynb` | ΔAUC pareado, bootstrap, FDR-BH | Inferência **só** `36m_6m` |

Stats multi-cohort formais: **não** (amostras aninhadas + pesca).

---

## 10. Plano experimental e CLIs

### Flags comuns

```bash
source .venv/bin/activate
cd /mnt/study-data/pgirardi/graphs

COMMON="--tasks smci_pmci --selection l1_stable --repeats 10 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1"
```

### Checklist

Ver **topo deste ficheiro** — fonte de verdade operacional.

- 🟠[x] Extração `all_population` (vol, rad, DVF CN-template)
- 🟠[x] `4_run_post_extract` × 6 cohorts (`disp` = CN-template)
- 🟠[x] A1 Wide `36m_6m`
- 🟠[x] A2 T1 `36m_6m` (vol/shape/texture/disp/all)
- 🟠[x] A3 Clínica · A4 Fusion · A5 Sanity · A6 Leaky (`inflate ""`)
- 🟠[x] DVF follow-up→baseline descartado
- [ ] B Leaky `pseudo` / `pseudo,fulltune` / (`max`)
- [ ] C Gradiente 5 cohorts × wide + t1
- [ ] `6_results` / `cohort_comparison`
- [ ] `7_stats`

---

### A. Primary `36m_6m` — matriz rica

#### A1. Wide — longitudinal + ablação modelo/ComBat

**Racional:** endpoint longitudinal completo; 3 modelos = sensibilidade ao algoritmo; ComBat both = efeito da harmonização.  
**Δ vs T1 usa apenas células svm** deste run.

```bash
C=36m_6m
python 5_run_ablation.py --cohort $C --representation wide \
  --modality vol,shape,texture,disp,all \
  --models svm,rf,elasticnet --combat both $COMMON
```

#### A2. T1-only — transversal (espelho do primary para o Δ)

**Racional:** mesmo pipeline, **só svm** (modelo do endpoint). Não precisa de rf/en para o claim wide vs T1 ser justo.  
`combat both` → pares nocombat–nocombat e combat–combat.

```bash
C=36m_6m
python 5_run_ablation.py --cohort $C --representation t1_only \
  --modality vol,shape,texture,disp,all \
  --models svm --combat both $COMMON
```

**Porquê A1 ≠ A2 nos modelos?** Wide é **superset**. Comparação reportada = svm. RF/EN no wide respondem outra pergunta.

#### A3. Clínica — piso (svm alinhado)

**Racional:** mesmo classificador que a imagem primary → comparação clinic vs img vs fusion sem misturar algoritmos.

```bash
# Garantir COHORT = "36m_6m" em 5_run_baseline_comparison.py (sem --cohort ainda)
python 5_run_baseline_comparison.py --feature-set clinical \
  --tasks smci_pmci --models svm \
  --repeats 10 --tuner optuna --optuna-trials 10
```

#### A4. Fusion

**Racional:** RM acrescenta ao clínico?

```bash
python 5_run_baseline_comparison.py --feature-set fusion \
  --modality vol --tasks smci_pmci --representation wide \
  --selection l1_stable --models svm --combat false $COMMON
```

#### A5. Sanity CN×AD

**Racional:** ceiling do pipeline em tarefa fácil.  
**Importante:** pasta separada + `COMMON_NOTASK` (senão `$COMMON` sobrescreve `--tasks` com smci_pmci).

```bash
python 5_run_ablation.py --cohort 36m_6m --representation wide \
  --modality vol --models svm --combat false $COMMON_NOTASK \
  --tasks cn_ad \
  --results-dir csvs/cohorts/36m_6m/ablation_results/vol_cn_ad
```

#### A6. Leaky (opcional)

**Racional:** controlo metodológico — selecção/normalização global infla AUC?

```bash
python 5_run_ablation_leaky.py --cohort 36m_6m --representation wide \
  --inflate "" --modality vol --models svm --combat false $COMMON
```

---

### B. Gradiente temporal — outros 5 cohorts

**Racional (hipótese forte):** mesma análise longitudinal vs transversal em cada célula `(t_janela, t_imagens)`.  
Protocolo **fixo:** 5 modalidades · **svm** · **nocombat** · wide **e** t1_only.  
Não repetir 3 modelos (isola o efeito do tempo).

```bash
for C in 36m_9m 36m_12m 48m_6m 48m_9m 48m_12m; do
  python 5_run_ablation.py --cohort $C --representation wide \
    --modality vol,shape,texture,disp,all \
    --models svm --combat false $COMMON

  python 5_run_ablation.py --cohort $C --representation t1_only \
    --modality vol,shape,texture,disp,all \
    --models svm --combat false $COMMON
done
```

**Mínimo se CPU apertar:** só `--modality vol` nos dois comandos (ainda testa a hipótese temporal).

**Figura-chave:** ΔAUC(wide−T1) vs `t_imagens`, facet por `t_janela` (começar em vol; depois por modalidade).

---

### C. Pós-run

1. `6_results.ipynb` → summary + `csvs/cohort_comparison/`  
2. `7_stats.ipynb` → demografia + Δ wide vs T1 + FDR (**só** `36m_6m`)

---

## 11. Ordem tmux sugerida

| Fase | Sessões | Jobs |
|------|---------|------|
| 1 | leaky | B — `pseudo` / `pseudo,fulltune` (/ `max`) |
| 2 | até 5 tmux | C — gradiente (wide+t1 por cohort) |
| 3 | — | `6_results` + `7_stats` |

---

## 12. O que discutir no artigo

1. **Δ wide−T1** no primary (FDR nas modalidades) — claim kernel  
2. **Gradiente** Δ vs `t_imagens` / `t_janela` — relevância do desenho longitudinal  
3. Ranking modalidades + features estáveis (vol/csf_norm, etc.)  
4. SVM vs RF/EN só no wide primary  
5. ComBat on/off  
6. Clínica / fusion  
7. Soft-pMCI, N↓, viés de retenção — limitações honestas  
8. Leaky / CN×AD — suplemento metodológico  

---

## 13. Flags `--cohort`

| Script | `--cohort`? |
|--------|-------------|
| `5_run_ablation.py` | sim |
| `5_run_ablation_leaky.py` | sim |
| `5_run_ablation_deltas.py` | sim (fora do plano) |
| `4_run_post_extract.py` | não — constante `COHORT` |
| `5_run_baseline_comparison.py` | não — constante `COHORT` |

---

## 14. FAQ rápido (para a defesa / oral)

**“Porquê 36m_6m como primary?”**  
Pré-especificado; janela cobre a maioria das conversões; gap 6 m é o extremo “quase transversal” da grelha — âncora + ponto de partida do gradiente.

**“Porquê não 3 modelos em todo o lado?”**  
Ablação de algoritmo uma vez (wide primary). No contraste temporal fixamos SVM.

**“T1 com menos modelos é injusto?”**  
Não: o Δ usa sempre svm↔svm.

**“Porquê hipocampo só?”**  
Âncora AD; controla dimensionalidade; outras ROIs existem na extração mas a ablação paper foca hipocampo.

**“O que é soft-pMCI?”**  
Última imagem pode ser o 1.º AD se faltarem MCIs com o gap exigido; DIAG observado mantém-se AD.

**“Balanceamento?”**  
Estratificação + `class_weight` opcional + AUC; sem SMOTE.

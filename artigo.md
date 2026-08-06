Pergunta principal:

No hipocampo, para sMCI→pMCI, o quanto a representação longitudinal com 3 visitas (t0 + deltas) agrega vs baseline transversal, em atributos unimodais e em fusão late multimodal, e como isso se comporta em janelas de tempo de observação com intervalos entre imagens diferentes?

## Definição e seleção populacional

A população de análise deriva do ADNI. Antes de qualquer definição de coorte sMCI/pMCI, constrói-se uma tabela longitudinal limpa a partir dos dados brutos. Cada exame de RM é alinhado ao diagnóstico clínico e às escalas cognitivas/funcionais (ADAS, CDR, MMSE, FAQ) por correspondência temporal, exigindo distância absoluta ≤ 3 meses entre a data da MRI e a data do diagnóstico, de modo que o rótulo clínico corresponda, na prática, à mesma visita.

Removem-se registos com sexo inválido, demência diferente de AD, diagnóstico ou datas em falta, e pacientes com regressão diagnóstica nas fases contínuas da AD (retorno a CN ou MCI após o primeiro AD; retorno a CN após o primeiro MCI). Imagens de repetição são excluídas. Outliers de qualidade de imagem identificados por MRQy são retirados.

Para cada paciente, a data do primeiro MRI define a imagem de baseline $i_1$ no tempo $t_0$. O conjunto longitudinal usa as três primeiras imagens elegíveis

$$
i=\{i_{1,t_0},\,i_{2,t_1},\,i_{3,t_2}\},
$$

sempre a partir do mesmo baseline $i_{1,t_0}$, respeitando um intervalo mínimo entre aquisições (nominalmente próximo de 6 ou 12 meses, conforme a coorte). A janela de observação clínica inicia-se em $t_0$ e é de **36 meses** ou **48 meses**.

Definiram-se quatro grupos: CN, sMCI, pMCI e AD.

- **CN:** todos os diagnósticos na janela são CN.
- **sMCI:** todos os diagnósticos na janela são MCI, com pelo menos uma avaliação pós-janela ainda MCI. A definição não implica que o paciente nunca converterá; exige apenas ausência de conversão no horizonte pré-especificado, com confirmação na primeira imagem pós-janela.
- **pMCI:** baseline MCI e, dentro da janela, pelo menos um diagnóstico de AD. Composições admitidas (respeitando janela e gap de imagem): $\{\mathrm{MCI},\mathrm{MCI},\mathrm{MCI}\}$ com ≥1 AD na janela, ou $\{\mathrm{MCI},\mathrm{MCI},\mathrm{AD}\}$ tratando esse AD como rótulo de conversão no conjunto. Não se admite $\{\mathrm{MCI},\mathrm{AD},\mathrm{AD}\}$: apenas o primeiro AD classifica o grupo, e um segundo AD afastaria o último MCI do limite de 12 meses, com alterações estruturais adicionais.
- **AD:** todos os diagnósticos na janela são AD.

**Coortes de estudo (contagens sMCI/pMCI em conjuntos):**

| Janela | Gap imagens | sMCI / pMCI |
|--------|-------------|-------------|
| 36 m | ~6 m | 153 / 90 |
| 36 m | ~12 m | 69 / 21 |
| 48 m | ~6 m | 96 / 109 |
| 48 m | ~12 m | **71 / 40** |

A tarefa principal é a classificação **sMCI vs pMCI**. 
A coorte inicial foi **36m6m**, porém foi expandida para **36m6m**,**36m12m**,**48m6m**,**48m12m** para diferentes cenários.

A coorte **48m_12m** para analise multimodal, visto que em vol e texture houve maior agrgação longitudinal.

Comparações 36m_6m vs 48m_12m para constraste de gap entre as imagens de 6 vs 12 meses.

---

## Pré-processamento e atributos

Após tratamento das imagens, alinhamento temporal e filtro de ROI hipocampal, extraem-se quatro famílias de atributos por visita: 
* volume (frações teciduais normalizadas); 
* shape (descritores geométricos);
* texture (GLCM); 
* Deslocamento/DVF. 

Representações temporais avaliadas:

| Representação | Definição |
|---------------|-----------|
| `t0_only` | apenas baseline |
| `t0_deltas` | baseline + deltas absolutos entre as 3 visitas |
| `abs` | visitas concatenadas  |


**Dimensionalidade de atributos do hipocampo L/R:**

| Modalidade | Sufixos | t0 (`t0_only`) | Deltas (`t0_deltas`) |
|------------|---------|----------------:|---------------------:|
| vol | gm/wm/csf_norm (3) | 6 | 24 |
| shape | 6 shape | 12 | 48 |
| texture | 4 GLCM | 8 | 32 |
| disp | 4 (mag_mean, strain_fro_*, logjac_var) | 8 | 32 |
| all | soma | 34 | 136 |

Protocolo de modelagem fixo nas ablações oficiais: SVM, seleção `l1_stable` (corte de frequência 70%), sem ComBat, nested CV com 10 repetições, métrica primária **AUC patient-level** (média de scores OOF por paciente).

---

## Principais resultados

A estratégia foi: (i) fixar **SVM + seleção L1 Lasso + sem ComBat** como protocolo principal; (ii) começar pela coorte **`36m_6m`**; (iii) expandir unimodal às demais coortes; (iv) concentrar a análise multimodal em **`48m_12m`** por possuir maior contraste temporal. Outros modelos de ML (RF, MLP) e ComBat ficaram restritos à **abordagem unimodal** como análise de sensibilidade.

### Ablação unimodal: baseline vs longitudinal

**Hipótese inicial (`36m_6m`).** 

Na coorte primária, o `abs` superava `t0_deltas` em vários blocos, por exemplo, texture `abs` **0.757** ≈ `t0_deltas` **0.756** ≫ `t0_only` **0.708**. Isso sugeriu, a princípio, que concatenar as 3 visitas era um longitudinal “suficiente”.

**Expansão (`36m_12m`, `48m_6m`, `48m_12m`).** Com gap **12 m**, o padrão inverte para vol/texture: `t0_deltas` sobe e `abs` deixa de ser o melhor longitudinal. Em `48m_12m`:

| Mod | `t0_only` | `abs` | `t0_deltas` |
|-----|----------:|------:|------------:|
| shape | **0.785** | 0.770 | **0.786** |
| vol | 0.638 | 0.682 | **0.730** |
| texture | 0.637 | 0.706 | **0.743** |
| disp | 0.468 | 0.573 | 0.610 |

**Resumo** 

Em **6 m**, `abs` ≈ baseline ou acrescenta pouco além de redundância (pouca alteração estrutural entre visitas próximas). Em **12 m**, deltas absolutos (`t0_deltas`) capturam melhor a progressão; `abs` piora relativo a deltas. Shape permanece teto estático ($\Delta \approx 0$ entre t0 e deltas). Disp unimodal permanece fraco.

**Melhores resultados por modalidade (svm, nocombat)** — o que o orientador pede: ranking **dentro** de cada família, não um top global (que seria só shape).

**Contraste t0 vs melhor longitudinal — coorte de fusão `48m_12m`:**

| Mod | `t0_only` | Melhor long | Protocol long | Δ (long − t0) | Leitura |
|-----|----------:|------------:|---------------|--------------:|---------|
| shape | **0.785** | **0.786** | `t0_deltas` | +0.001 | teto estático |
| vol | 0.638 | **0.730** | `t0_deltas` | +0.092 | long agrega |
| texture | 0.637 | **0.743** | `t0_deltas` | +0.106 | long agrega |
| disp | 0.468 | 0.610 | `t0_deltas` | +0.142 | sobe, mas continua fraco |

**Top 10 por modalidade (todas as coortes × encodings)** — AUC patient:

| # | shape | vol | texture | disp |
|--:|-------|-----|---------|------|
| 1 | **0.786** `48m_12m` / `t0_deltas` | **0.730** `48m_12m` / `t0_deltas` | **0.757** `36m_6m` / `abs` | **0.647** `36m_6m` / `t0_only` |
| 2 | 0.785 `36m_6m` / `t0_only` | 0.727 `36m_6m` / `t0_only` | 0.756 `36m_6m` / `t0_deltas` | 0.610 `48m_12m` / `t0_deltas` |
| 3 | 0.785 `48m_12m` / `t0_only` | 0.719 `48m_6m` / `abs` | 0.743 `48m_12m` / `t0_deltas` | 0.580 `36m_6m` / `t0_deltas` |
| 4 | 0.781 `48m_6m` / `t0_only` | 0.718 `36m_6m` / `t0_deltas` | 0.739 `36m_12m` / `t0_deltas` | 0.573 `48m_12m` / `abs` |
| 5 | 0.777 `48m_6m` / `abs` | 0.714 `36m_6m` / `abs` | 0.735 `48m_6m` / `abs` | 0.564 `48m_6m` / `t0_only` |
| 6 | 0.770 `48m_12m` / `abs` | 0.710 `48m_6m` / `t0_deltas` | 0.732 `48m_6m` / `t0_deltas` | 0.558 `36m_6m` / `abs` |
| 7 | 0.767 `36m_6m` / `abs` | 0.709 `48m_6m` / `t0_only` | 0.728 `36m_12m` / `abs` | 0.554 `48m_6m` / `abs` |
| 8 | 0.766 `36m_6m` / `t0_deltas` | 0.699 `36m_12m` / `t0_deltas` | 0.708 `36m_6m` / `t0_only` | 0.540 `48m_6m` / `t0_deltas` |
| 9 | 0.762 `48m_6m` / `t0_deltas` | 0.682 `48m_12m` / `abs` | 0.706 `48m_12m` / `abs` | 0.468 `48m_12m` / `t0_only` |
| 10 | 0.757 `36m_12m` / `abs` | 0.678 `36m_12m` / `t0_only` | 0.705 `48m_6m` / `t0_only` | 0.385 `36m_12m` / `t0_only` |

**Leitura para a reunião:**
- **Shape:** melhor bloco absoluto; t0 ≈ deltas — teto do estudo.
- **Vol / texture:** melhores long em gap longo (`t0_deltas` em `48m_12m`); em `36m_6m`, texture ainda favorece `abs`/`t0_deltas` vs t0 (redundância parcial, não ausência de sinal).
- **Disp:** melhor ponto ainda ~0.65 e instável entre os protocolos.
- Hierarquia prática: shape ≫ texture ≳ vol ≫ disp (no melhor de cada uma).

### Ablação multimodal: baseline vs longitudinal em `48m_12m`

**Top 10 união baseline:**

Referência: teto baseline shape = **0.785**

| # | Spec (todos t0) | AUC | vs shape t0 |
|--:|-----------------|----:|------------:|
| 1 | shape ∪ disp | **0.769** | −0.016 |
| 2 | shape ∪ vol | **0.766** | −0.019 |
| 3 | shape ∪ texture | **0.763** | −0.022 |
| 4 | shape ∪ texture ∪ disp | **0.760** | −0.025 |
| 5 | shape ∪ vol ∪ texture | **0.757** | −0.028 |
| 6 | shape ∪ vol ∪ disp | **0.754** | −0.031 |
| 7 | shape ∪ vol ∪ texture ∪ disp | **0.747** | −0.038 |
| 8 | vol ∪ texture | **0.668** | −0.117 |
| 9 | vol ∪ texture ∪ disp | **0.651** | −0.134 |
| 10 | vol ∪ disp | **0.614** | −0.171 |

**Top 10 longitudinal:**

| # | Spec | AUC | Nota |
|--:|------|----:|------|
| 1 | shape t0 ∪ tex Δ ∪ **disp Δ** | **0.837** | exploratório; mono disp fraco |
| 2 | shape t0 ∪ vol Δ ∪ tex Δ ∪ disp Δ | **0.835** | idem |
| 3 | todas 4 mods em Δ | **0.833** | multi-Δ denso |
| 4 | shape t0 ∪ vol Δ ∪ tex Δ | **0.828** | long sem disp |
| 5 | shape t0 ∪ vol Δ ∪ tex Δ ∪ disp t0 | **0.823** | disp só t0 |
| 6 | shape Δ ∪ tex Δ ∪ disp Δ | **0.823** | sem teto t0 shape |
| 7 | **âncora** shape t0 ∪ tex Δ | **0.823** | **claim oficial** |
| 8 | shape Δ ∪ vol Δ ∪ tex Δ | **0.820** | multi-Δ sem disp |
| 9 | shape t0 ∪ vol t0 ∪ tex Δ ∪ disp Δ | **0.819** | misto |
| 10 | shape Δ ∪ vol t0 ∪ tex Δ ∪ disp Δ | **0.818** | misto |


#### Early fusion vs late fusion em `48m_12m` no mesmo protocolo


| Atributos | Early | Late | Δ (late−early) |
|--------------|------:|-----:|---------------:|
| **shape t0 ∪ tex Δ (âncora)** | **0.794** | **0.823** | **+0.029** |
| shape t0 ∪ vol Δ ∪ tex Δ | 0.771 | 0.828 | +0.057 |
| shape Δ ∪ vol Δ ∪ tex Δ | 0.767 | 0.820 | +0.053 |
| shape Δ ∪ tex Δ | 0.777 | 0.814 | +0.037 |
| shape t0 ∪ vol Δ (`deltas_vol`) | 0.779 | 0.805 | +0.026 |
| vol Δ ∪ tex Δ | 0.766 | 0.793 | +0.027 |
| shape Δ ∪ vol Δ | 0.772 | 0.793 | +0.021 |
| shape t0 ∪ vol Δ (`t0_deltas_vol`) | 0.761 | 0.785 | +0.024 |

**Definição da agregação multimodal** 
No early fusion, atributos de diferentes modalidades são concatenados num único espaço de features e submetidos à mesma seleção/classificação, o que pode diluir a contribuição de um bloco forte, por exemplo, o shape perante outro de maior dimensionalidade.

No late fusion, cada modalidade de atributo é modelada de forma independente e as previsões são agregadas a posteriori pela média simples dos scores, preservando melhor o sinal específico de cada família de atributos. 

### Análises adicionais

**Sanidade CN×AD** (`36m_6m`, abs): AUC alta (vol ~0.82; shape ~0.91; texture/all ~0.83–0.86; disp ~0.63) — valida pipeline; a dificuldade é prognóstico sMCI/pMCI.

**Demografia:** idade/sexo equilibrados entre sMCI e pMCI; modelos só demográficos ≈ acaso; imagem > demografia no $\Delta\mathrm{AUC}$ bootstrap (coorte âncora histórica).

**ComBat / outros modelos:** sensibilidade unimodal (não na grade de fusão); primary de imagem permanece SVM sem ComBat. Sem evidência clara de ganho/perda de AUC por ComBat no vol âncora histórica.

**Global/leaky:** diferença NS vs abs; primary justificado por integridade do nested CV.

**Clínico / clínica+imagem:** fusão com clínico supera imagem isolada no desenho antigo — RM hipocampal como **adjunto**, não first-line clínico. Manter como contexto/suplemento se o artigo for centrado em encoding de imagem.

**Sensibilidade** $\tau \in \{50,70,90\}$ (`48m_12m`): performance não monótona; $\tau=70\%$ frequentemente ótimo ou próximo; $\tau=90\%$ não garante melhora.

**Estabilidade de atributos** 

- **Volume:** gm_norm/csf_norm estáveis; wm_norm direito instável no tempo.
- **Shape:** descritores geométricos estáveis em t0–T3.
- **Texture:** Contrast estável; JointEntropy rara (instabilidade de seleção, não de visita).
- **Disp:** pool estável muito fino — alinha com AUC unimodal fraca.
- **All:** herda estabilidade dos blocos dominantes.





Pergunta principal:

No hipocampo, para sMCI→pMCI, o quanto a representação longitudinal com 3 visitas (t0 + deltas) agrega vs baseline transversal, em atributos unimodais e em fusão late multimodal, e como isso se comporta em janelas de tempo de observação com intervalos entre imagens diferentes?

## Definição e seleção populacional

A população de análise deriva do **ADNI**. Também se exploraram as bases **AIBL** e **OASIS** sob a mesma lógica de elegibilidade, mas sem número suficiente de conjuntos longitudinais com três imagens (ver limitação abaixo). Antes de qualquer definição de coorte sMCI/pMCI, constrói-se uma tabela longitudinal limpa a partir dos dados brutos. Cada exame de RM é alinhado ao diagnóstico clínico e às escalas cognitivas/funcionais (ADAS, CDR, MMSE, FAQ) por correspondência temporal, exigindo distância absoluta ≤ 3 meses entre a data da MRI e a data do diagnóstico, de modo que o rótulo clínico corresponda, na prática, à mesma visita.

Removem-se registos com sexo inválido, demência diferente de AD, diagnóstico ou datas em falta, e pacientes com regressão diagnóstica nas fases contínuas da AD (retorno a CN ou MCI após o primeiro AD; retorno a CN após o primeiro MCI). Imagens de repetição são excluídas. Outliers de qualidade de imagem identificados por MRQy são retirados.

Para cada paciente, a data do primeiro MRI define a imagem de baseline $i_1$ no tempo $t_0$. O conjunto longitudinal contém **exactamente três** imagens preditoras

$$
i=\{i_{1,t_0},\,i_{2,t_1},\,i_{3,t_2}\},
$$

seleccionadas pelo protocolo **`forward_band_pm2`**: $i_1$ é o baseline (primeira imagem do pool preditor em $t_0$); $i_2$ é a **próxima** aquisição cujo intervalo face a $i_1$ cai na banda nominal; $i_3$ é a **próxima** após $i_2$ com gap na **mesma** banda. Não se trata de “três imagens *depois* do baseline” (o que seriam quatro visitas): são três no total — baseline + duas seguintes na banda. As bandas adoptadas (±2 meses) são disjuntas e evitam a ambiguidade do bordo aos 9 meses que surgiria com ±3 fechado:

| Gap nominal | Banda (meses) |
|-------------|----------------|
| 6 m | $[4,\ 8]$ |
| 12 m | $[10,\ 14]$ |

A janela de observação clínica inicia-se em $t_0$ e é de **36 meses** ou **48 meses**.

Definiram-se quatro grupos: CN, sMCI, pMCI e AD.

- **CN:** todos os diagnósticos na janela são CN.
- **sMCI:** ausência de AD até $t_{\mathrm{end}}=t_0+$ janela (36 ou 48 meses); existe pelo menos uma visita com data $\ge t_{\mathrm{end}}$; a **primeira** dessas visitas pós-janela deve permanecer **MCI**. A definição não implica que o paciente nunca converterá; exige apenas ausência de conversão no horizonte pré-especificado, com confirmação na primeira avaliação após o fim da janela.
- **pMCI:** baseline MCI e, dentro da janela, pelo menos um diagnóstico de AD. Composições admitidas (respeitando janela e gap de imagem): $\{\mathrm{MCI},\mathrm{MCI},\mathrm{MCI}\}$ com ≥1 AD na janela, ou $\{\mathrm{MCI},\mathrm{MCI},\mathrm{AD}\}$ tratando esse AD como rótulo de conversão no conjunto. Não se admite $\{\mathrm{MCI},\mathrm{AD},\mathrm{AD}\}$: apenas o primeiro AD classifica o grupo, e um segundo AD afastaria o último MCI do limite de 12 meses, com alterações estruturais adicionais.
- **AD:** todos os diagnósticos na janela são AD.

**Confirmação sMCI (rótulo clínico, independente da banda ±2).** O rótulo sMCI/pMCI é decidido em `classify_patient` **antes** da selecção das imagens preditoras:

1. Sem AD até $t_{\mathrm{end}}$.
2. Existe visita com `MRI_DATE` $\ge t_{\mathrm{end}}$.
3. A primeira dessas visitas é MCI → **sMCI**.
4. Se essa primeira visita pós-janela for AD (ou outro diagnóstico não-MCI) → **excluído** (`desfecho_intervalado`).
5. Sem visita pós-janela → **excluído** (`sem_confirmacao_pos_janela`).
6. Se a primeira AD ocorre **dentro** da janela → **pMCI** (não se trata de “sMCI falhou”).

A confirmação **não** usa o range ±2 meses. A banda ±2 aplica-se apenas à escolha das três imagens preditoras (para sMCI: no intervalo até $t_{\mathrm{end}}$, com DIAG MCI). A visita de confirmação pode situar-se fora desse espaçamento entre aquisições.

**Coortes de estudo (contagens sMCI/pMCI em conjuntos):**

| Janela | Gap imagens | sMCI / pMCI |
|--------|-------------|-------------|
| 36 m | 6±2 $[4,8]$ | 125 / 106 |
| 36 m | 12±2 $[10,14]$ | 121 / 33 |
| 48 m | 6±2 $[4,8]$ | 73 / 120 |
| 48 m | 12±2 $[10,14]$ | **72 / 48** |

A tarefa principal é a classificação **sMCI vs pMCI**. 
A coorte inicial foi **36m6m**, porém foi expandida para **36m6m**,**36m12m**,**48m6m**,**48m12m** para diferentes cenários.

A coorte **48m_12m** para analise multimodal, visto que em vol e texture houve maior agrgação longitudinal.

Comparações 36m_6m vs 48m_12m para constraste de gap entre as imagens de 6 vs 12 meses.

**Janela 36 vs 48 no mesmo gap (leitura de `t1_only` / baseline).** Não se deve esperar AUC idêntica em `36m_6m` vs `48m_6m` (nem em `36m_12m` vs `48m_12m`) só porque o encoding usa a visita baseline. Na **intersecção** de pacientes presentes nas duas coortes com o mesmo espaçamento nominal, o `ID_IMG` de $t_0$ coincide (e o triplo $t_0$–$t_1$–$t_2$ também), com o mesmo rótulo de grupo. Contudo as **populações não são as mesmas**: a janela clínica altera a elegibilidade (quem entra ou sai). Exemplos no protocolo actual (banda ±2, selecção forward): sMCI 125 vs 73 (gap 6 m) e 121 vs 72 (gap 12 m), com dezenas de IDs exclusivos de uma das janelas. Logo diferem $n$, balanço sMCI/pMCI, partições de CV e AUC — inclusive em `t1_only` — mesmo com overlap de baselines.

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
| `t0_only` | apenas atributos do baseline |
| `t0_deltas` | atributos do baseline + deltas absolutos entre as 3 visitas |
| `abs` | atributos concatenados das visitas  |


**Dimensionalidade de atributos do hipocampo L/R:**

| Modalidade | Sufixos | baseline (`t0_only`) | Concatenação (`abs`) | Baseline + Deltas (`t0_deltas`) |
|------------|---------|----------------:|------------:|---------------------:|
| vol | gm/wm/csf_norm (3) | 6 | 18 | 24 |
| shape | 6 shape | 12 | 36 | 48 |
| texture | 4 GLCM | 8 | 24 | 32 |
| disp | 4 (mag_mean, strain_fro_*, logjac_var) | 8 | 24 | 32 |
| all | soma | 34 | 102 | 136 |

Protocolo de modelagem fixo nas ablações oficiais: SVM, seleção `l1_stable` (corte de frequência 70%), sem ComBat, nested CV com 10 repetições, métrica primária **AUC patient-level** (média de scores OOF por paciente). AUCs reportadas como média ± desvio padrão (`auc_patient_mean` ± `auc_patient_std`), com 3 casas decimais truncadas.

---

## Principais resultados

A estratégia foi: (i) fixar **SVM + seleção L1 Lasso + sem ComBat** como protocolo principal; (ii) começar pela coorte **`36m_6m`**; (iii) expandir unimodal às demais coortes; (iv) concentrar a análise multimodal em **`48m_12m`** por possuir maior contraste temporal. Outros modelos de ML (RF, MLP) e ComBat ficaram restritos à **abordagem unimodal** como análise de sensibilidade.

### Ablação unimodal: baseline vs longitudinal

**Hipótese inicial (`36m_6m`).**

O `abs` superava `t0_deltas`. Isso sugeriu, a princípio, que concatenar as 3 visitas era um longitudinal “suficiente”.

**Expansão (`36m_12m`, `48m_6m`, `48m_12m`).** Com gap **12 m**, o padrão inverte para vol/texture: `t0_deltas` sobe e `abs` deixa de ser o melhor longitudinal. Em `48m_12m`:

| Mod | `t0_only` | `abs` | `t0_deltas` |
|-----|----------:|------:|------------:|
| shape | **0.785±0.043** | 0.770±0.044 | **0.786±0.044** |
| vol | 0.638±0.055 | 0.682±0.054 | **0.730±0.049** |
| texture | 0.637±0.055 | 0.706±0.052 | **0.743±0.048** |
| disp | 0.468±0.057 | 0.573±0.058 | 0.610±0.060 |

**Resumo** 

Em **6 m**, `abs` ≈ baseline ou acrescenta pouco além de redundância (pouca alteração estrutural entre visitas próximas). Em **12 m**, deltas absolutos (`t0_deltas`) capturam melhor a progressão; `abs` piora relativo a deltas. Shape permanece teto estático ($\Delta \approx 0$ entre t0 e deltas). Disp unimodal permanece fraco.

**Melhores resultados por modalidade (svm, nocombat)**

**Contraste t0 vs melhor longitudinal — coorte de fusão `48m_12m`:**

| Mod | `t0_only` | Melhor long | Protocol long | Δ (long − t0) | Leitura |
|-----|----------:|------------:|---------------|--------------:|---------|
| shape | **0.785±0.043** | **0.786±0.044** | `t0_deltas` | +0.001 | teto estático |
| vol | 0.638±0.055 | **0.730±0.049** | `t0_deltas` | +0.092 | long agrega |
| texture | 0.637±0.055 | **0.743±0.048** | `t0_deltas` | +0.106 | long agrega |
| disp | 0.468±0.057 | 0.610±0.060 | `t0_deltas` | +0.142 | sobe, mas continua fraco |

**Top 10 por modalidade (todas as coortes × encodings)** — AUC patient:

| # | shape | vol | texture | disp |
|--:|-------|-----|---------|------|
| 1 | **0.786±0.044** `48m_12m` / `t0_deltas` | **0.730±0.049** `48m_12m` / `t0_deltas` | **0.757±0.030** `36m_6m` / `abs` | **0.647±0.035** `36m_6m` / `t0_only` |
| 2 | 0.785±0.029 `36m_6m` / `t0_only` | 0.727±0.033 `36m_6m` / `t0_only` | 0.756±0.030 `36m_6m` / `t0_deltas` | 0.610±0.060 `48m_12m` / `t0_deltas` |
| 3 | 0.785±0.043 `48m_12m` / `t0_only` | 0.719±0.034 `48m_6m` / `abs` | 0.743±0.048 `48m_12m` / `t0_deltas` | 0.580±0.036 `36m_6m` / `t0_deltas` |
| 4 | 0.781±0.032 `48m_6m` / `t0_only` | 0.718±0.034 `36m_6m` / `t0_deltas` | 0.739±0.065 `36m_12m` / `t0_deltas` | 0.573±0.058 `48m_12m` / `abs` |
| 5 | 0.777±0.033 `48m_6m` / `abs` | 0.714±0.033 `36m_6m` / `abs` | 0.735±0.035 `48m_6m` / `abs` | 0.564±0.040 `48m_6m` / `t0_only` |
| 6 | 0.770±0.044 `48m_12m` / `abs` | 0.710±0.035 `48m_6m` / `t0_deltas` | 0.732±0.035 `48m_6m` / `t0_deltas` | 0.558±0.038 `36m_6m` / `abs` |
| 7 | 0.767±0.030 `36m_6m` / `abs` | 0.709±0.036 `48m_6m` / `t0_only` | 0.728±0.064 `36m_12m` / `abs` | 0.554±0.041 `48m_6m` / `abs` |
| 8 | 0.766±0.031 `36m_6m` / `t0_deltas` | 0.699±0.070 `36m_12m` / `t0_deltas` | 0.708±0.032 `36m_6m` / `t0_only` | 0.540±0.041 `48m_6m` / `t0_deltas` |
| 9 | 0.762±0.034 `48m_6m` / `t0_deltas` | 0.682±0.054 `48m_12m` / `abs` | 0.706±0.052 `48m_12m` / `abs` | 0.468±0.057 `48m_12m` / `t0_only` |
| 10 | 0.757±0.056 `36m_12m` / `abs` | 0.678±0.067 `36m_12m` / `t0_only` | 0.705±0.036 `48m_6m` / `t0_only` | 0.385±0.067 `36m_12m` / `t0_only` |

**Leitura para a reunião:**
- **Shape:** melhor bloco absoluto; t0 ≈ deltas — teto do estudo.
- **Vol / texture:** melhores long em gap longo (`t0_deltas` em `48m_12m`); em `36m_6m`, texture ainda favorece `abs`/`t0_deltas` vs t0 (redundância parcial, não ausência de sinal).
- **Disp:** melhor ponto ainda ~0.65 e instável entre os protocolos.
- Hierarquia prática: shape ≫ texture ≳ vol ≫ disp (no melhor de cada uma).

### Ablação multimodal: baseline vs longitudinal em `48m_12m`

**Top 10 união baseline:**

Referência: teto baseline shape = **0.785±0.043**

| # | Spec (todos t0) | AUC | vs shape t0 |
|--:|-----------------|----:|------------:|
| 1 | shape ∪ disp | **0.769±0.046** | −0.016 |
| 2 | shape ∪ vol | **0.766±0.045** | −0.019 |
| 3 | shape ∪ texture | **0.763±0.048** | −0.022 |
| 4 | shape ∪ texture ∪ disp | **0.760±0.048** | −0.025 |
| 5 | shape ∪ vol ∪ texture | **0.757±0.047** | −0.028 |
| 6 | shape ∪ vol ∪ disp | **0.754±0.049** | −0.031 |
| 7 | shape ∪ vol ∪ texture ∪ disp | **0.747±0.045** | −0.038 |
| 8 | vol ∪ texture | **0.668±0.052** | −0.117 |
| 9 | vol ∪ texture ∪ disp | **0.651±0.054** | −0.134 |
| 10 | vol ∪ disp | **0.614±0.056** | −0.171 |

**Top 10 longitudinal:**

| # | Spec | AUC | Nota |
|--:|------|----:|------|
| 1 | shape t0 ∪ tex Δ ∪ **disp Δ** | **0.837±0.038** | exploratório; mono disp fraco |
| 2 | shape t0 ∪ vol Δ ∪ tex Δ ∪ disp Δ | **0.835±0.037** | idem |
| 3 | todas 4 mods em Δ | **0.833±0.037** | multi-Δ denso |
| 4 | shape t0 ∪ vol Δ ∪ tex Δ | **0.828±0.038** | long sem disp |
| 5 | shape t0 ∪ vol Δ ∪ tex Δ ∪ disp t0 | **0.823±0.039** | disp só t0 |
| 6 | shape Δ ∪ tex Δ ∪ disp Δ | **0.823±0.041** | sem teto t0 shape |
| 7 | **âncora** shape t0 ∪ tex Δ | **0.823±0.039** | **claim oficial** |
| 8 | shape Δ ∪ vol Δ ∪ tex Δ | **0.820±0.040** | multi-Δ sem disp |
| 9 | shape t0 ∪ vol t0 ∪ tex Δ ∪ disp Δ | **0.819±0.039** | misto |
| 10 | shape Δ ∪ vol t0 ∪ tex Δ ∪ disp Δ | **0.818±0.041** | misto |


#### Early fusion vs late fusion em `48m_12m` no mesmo protocolo


| Atributos | Early | Late | Δ (late−early) |
|--------------|------:|-----:|---------------:|
| **shape t0 ∪ tex Δ (âncora)** | **0.794±0.043** | **0.823±0.039** | **+0.029** |
| shape t0 ∪ vol Δ ∪ tex Δ | 0.771±0.047 | 0.828±0.038 | +0.057 |
| shape Δ ∪ vol Δ ∪ tex Δ | 0.767±0.049 | 0.820±0.040 | +0.053 |
| shape Δ ∪ tex Δ | 0.777±0.048 | 0.814±0.041 | +0.037 |
| shape t0 ∪ vol Δ (`deltas_vol`) | 0.779±0.047 | 0.805±0.046 | +0.026 |
| vol Δ ∪ tex Δ | 0.766±0.046 | 0.793±0.046 | +0.027 |
| shape Δ ∪ vol Δ | 0.772±0.046 | 0.793±0.042 | +0.021 |
| shape t0 ∪ vol Δ (`t0_deltas_vol`) | 0.761±0.049 | 0.785±0.045 | +0.024 |

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

### Limitação — outras bases (AIBL, OASIS)

O estudo usa **ADNI** como única fonte com contagens viáveis sob a definição populacional adoptada: três RM elegíveis a partir do mesmo baseline $t_0$, gaps nominais de ~6 ou ~12 meses, e rótulos sMCI/pMCI na janela clínica de 36 ou 48 meses. **AIBL** e **OASIS** foram analisadas no mesmo pipeline de alinhamento temporal e filtros; contudo, após a exigência de **conjuntos com 3 imagens**, restavam poucos sujeitos — em especial **pMCI**, onde conversão a AD dentro da janela e série tricíclica de RM coincidem raramente. Sem $n$ relevante nessas classes, treino/avaliação multi-coorte ou validação externa nessas bases fica inviável sem relaxar a definição (ex. 1–2 visitas), o que quebraria a comparabilidade com o encoding longitudinal $t_0$+deltas. Trata-se, portanto, de uma **limitação das bases disponíveis face ao desenho**, não de uma falha do protocolo de modelagem: o desenho longitudinal tricíclico restringe a amostra a repositórios com follow-up imagiológico denso o suficiente, hoje sobretudo o ADNI.

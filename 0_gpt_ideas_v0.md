# Anotações importantes

Link chatgpt rferrari: https://chatgpt.com/share/6a96cae5-5d58-83e9-8773-481ad3fbb052

## O problema mais sério: a tarefa não é inteiramente prognóstica

A definição de pMCI permite sequências do tipo (MCI,MCI,AD),

Sugestões, considerar apenas casos MCI,MCI,MCI com 1 AD dentro da janela de observação

| Coorte  | pMCI atual | MCI–MCI–AD | MCI–MCI–MCI → AD | sMCI | Redução de pMCI |
| ------| ---------: | ---------: | ---------------: | ---: | --------------: |
| 36m_6m  |        106 |         46 |           **60** |  125 |      **−43,4%** |
| 36m_12m |         33 |         33 |            **0** |  121 |       **−100%** |
| 48m_6m  |        120 |         46 |           **74** |   73 |      **−38,3%** |
| 48m_12m |         48 |         33 |           **15** |   72 |      **−68,8%** |

A priori a população principal era 48m_12m (soft_pmci=True: 72 sMCI, 48 pMCI| soft_pmci=False: 72 sMCI, 33) pois seguia a hipótese de que maior distancia temporal entre imagens forneceria maior alteração estrutural.

Entretanto, uma possibilidade agora é considerar apenas diagnósticos MCI da coorte 48m_6m (soft_pmci=False: 73 sMCI vs 74 pMCI) como principal pela melhor distribuição dos dados e para realização da análise prognóstica.

## Experimento quase obrigatório

Na coorte 48m_6m rodar o pipeline para soft_pmci=False: 73 sMCI vs 74 pMCI para comparar os resultados com soft_pmci=True: 73 sMCI vs 120 pMCI.

## Resultados

### UNICLASSE:
soft_pmci=true|MCI,MCI,MCI+MCI,MCI,AD: 
    baseline:
        vol: 0.756 
        shape: 0.801
        texture: 0.628
        disp: 0.595
        firstorder: 0.687
    longitudinal:
        vol: 0.763
        shape: 0.791
        texture: 0.674
        disp: 0.577
        firstorder: 0.686

soft_pmci=false|MCI,MCI,MCI: 
    baseline:
        vol: 0.738 
        shape: 0.806 
        texture: 0.626
        disp: 0.502
        firstorder: 0.697   
    longitudinal:
        vol: 0.728
        shape: 0.802
        texture: 0.604
        disp: 0.624
        firstorder: 0.671    

MULTICLASSE:
    tudo baseline
        soft_pmci=True 0.816
        soft_pmci=False 0.813
        Δ +0.01
    tudo longitudinal
        soft_pmci=True 0.819
        soft_pmci=False 0.815
        Δ +0.01
    shape baseline + tudo longitudinal
        soft_pmci=True 0.823
        soft_pmci=False 0.820   
        Δ +0.009

Conclusão: 

**1. Estabilidade Multimodal:**
Ao excluir do treinamento todos os 46 casos de pacientes que já apresentavam Alzheimer na terceira imagem (MCI $\to$ MCI $\to$ AD), o desempenho das arquiteturas multimodais não mostrou diferença relevante em termos de resultados. A fusão em âncora (shape $T_1$ + resto Q4) variou apenas de $0{,}823$ (MCI $\to$ MCI $\to$ AD) para $0{,}820$ (MCI $\to$ MCI $\to$ MCI) ($\Delta = -0{,}003$). 

**2. Superioridade de  *Shape*:**
O melhor preditor individual do sistema obteve desempenho **melhor** no grupo de conversores estritos (MCI $\to$ MCI $\to$ MCI|AUC = $0{,}801$ | $AUC_{longitudinal}$=$0{,}802$) do que no grupo com conversores (MCI $\to$ MCI $\to$ AD | $AUC_{baseline}$=$0{,}806$ | $AUC_{longitudinal}$=$0{,}791$). 

**3. O Efeito Longitudinal Ocorre nos atributos mais fracos**
A exclusão dos conversores impactou no desempenho longitudinal da textura longitudinal que caiu de $0{,}674$ para $0{,}604$, e o volume longitudinal caiu de $0{,}763$ para $0{,}728$). Isso indica que a dinâmica temporal de proporção e intensidade era impulsionada pela agressividade biológica da atrofia desse subgrupo. 

**Conclusão Geral:**
A hipótese de que a terceira aquisição radiológica estaria vazando informações e inflando as métricas do modelo não foi confirmada. 


# analise soft true vs false

Discussão pronta para colar (PT, tom de manuscrito). Ajusta AUCs se os teus CSVs tiverem ±sd.

---

## Discussão — sensibilidade ao critério *soft* pMCI

A definição primária de pMCI nesta coorte (`48m_6m`) admite, quando não há três aquisições MCI pré-conversão na banda temporal, a inclusão do primeiro exame já classificado como AD como terceira visita (sequências MCI–MCI–AD; \(n=46\)). Essa escolha aumenta a amostra de conversores (\(n_{\mathrm{pMCI}}=120\)) à custa de um intervalo em que a morfometria deixa de ser estritamente pré-desfecho. Para testar se o desempenho reportado dependia desse subgrupo — isto é, se o modelo explorava informação já compatível com doença estabelecida no instante \(t_3\) — repetimos o pipeline completo com `soft_pmci=False`, restringindo os conversores a trajetórias MCI–MCI–MCI com conversão posterior na janela de observação (\(73\) sMCI / \(74\) pMCI).

### Teto multimodal e forma estática

Nas três especificações de fusão tardia, a exclusão dos \(46\) casos MCI–MCI–AD não alterou materialmente o desempenho. A âncora shape em \(T_1\) unida ao restante das famílias em representação longitudinal passou de \(0{,}823\) para \(0{,}820\) (\(\Delta=-0{,}003\)); as fusões “tudo baseline” e “tudo longitudinal” variaram na mesma ordem de grandeza (\(\approx 0{,}01\)). Assim, a hipótese de que a terceira aquisição em AD inflaria de forma sistemática o teto multimodal **não** encontra suporte nestes dados: o ganho da combinação de famílias sobre o melhor unimodal permanece após o filtro prognóstico estrito.

O melhor preditor unimodal continua a ser a **forma** hipocampal. No protocolo *soft*, a AUC baseline foi \(0{,}801\) e a longitudinal \(0{,}791\); no protocolo estrito, \(0{,}806\) e \(0{,}802\). Ou seja, a discriminabilidade baseada em shape é estável (e, em \(T_1\), ligeiramente superior no subconjunto estrito), e a representação longitudinal **não** acrescenta sinal a esta família — padrão já observado na análise principal e agora replicado sob a restrição MCI–MCI–MCI. Isto reforça a interpretação de que a informação morfométrica relevante para sMCI *versus* pMCI está largamente presente na anatomia de referência, sem necessidade do salto temporal até um exame pós-conversão.

### Onde o critério *soft* importa

O contraste *soft* versus estrito concentra-se nas famílias com menor desempenho absoluto. Em textura, o protocolo *soft* mostrou ganho longitudinal (\(0{,}628\rightarrow0{,}674\)), que desaparece e se inverte no estrito (\(0{,}626\rightarrow0{,}604\)). Em volume, o pequeno ganho longitudinal *soft* (\(0{,}756\rightarrow0{,}763\)) também se perde no estrito (\(0{,}738\rightarrow0{,}728\)). Esses resultados sugerem que parte da dinâmica temporal em proporções volumétricas e em descritores de intensidade/textura era impulsionada pelo subgrupo MCI–MCI–AD, no qual \(\Delta_{32}\) (e, implicitamente, o arco até \(t_3\)) captura atrofia associada à transição para AD clinicamente manifesto — sinal biologicamente real, porém **não** estritamente prognóstico no sentido de predizer conversão a partir apenas de MCI.

O deslocamento normativo e a primeira ordem não alteram o quadro central: first-order permanece essencialmente estável; disp mostra comportamento mais variável (queda longitudinal no *soft*, subida no estrito), compatível com menor SNR e com a redução amostral, e não deve ser lido como evidência isolada de leakage ou da sua ausência.

Em conjunto, a ablação apoia uma distinção já motivada pela literatura de intervalos de seis meses: biomarcadores **morfométricos** (em especial shape) sustentam o claim principal de forma robusta ao critério de inclusão; biomarcadores **derivados de intensidade** (e, em menor grau, o contraste longitudinal de volume) são mais sensíveis à composição do conjunto de conversores e devem ser tratados como exploratórios na discussão do intervalo inter-exame.

### Implicações para a tarefa e para o desenho do estudo

Dois pontos metodológicos seguem diretamente. Primeiro, a análise primária com `soft_pmci=True` permanece defensável: maximiza \(n\) de pMCI, preserva o gradiente entre coortes (em particular `36m_12m`, inviável no estrito) e, após sensibilidade, não depende do subgrupo MCI–MCI–AD para o teto late nem para o domínio da shape. Segundo, a ablação **não** equivale a provar ausência total de informação pós-desfecho em todas as famílias — apenas a mostrar que essa informação **não** é necessária para as conclusões multimodais e de forma que estruturam o artigo. A formulação adequada é, portanto: *o desempenho claim-defining é robusto à exclusão de MCI–MCI–AD; ganhos longitudinais pontuais em textura/volume no protocolo soft não se replicam no estrito e não devem ser generalizados como evidência prognóstica limpa.*

### Limitações desta sensibilidade

A comparação True *versus* False altera simultaneamente o critério de inclusão e o tamanho/balanço amostral (\(120\) vs \(74\) pMCI). Diferenças unimodais pequenas podem misturar efeito de subgrupo com redução de poder. Uma análise complementar — desempenho do mesmo modelo treinado no protocolo *soft*, avaliado separadamente nos \(74\) estritos e nos \(46\) *soft* — isolaria melhor o contributo do subgrupo sem novo desenho de coorte. Além disso, esta ablação ainda utiliza as três visitas; não responde se o arco \(\approx 12\) meses (\(t_1\rightarrow t_3\)) é necessário face a um único incremento de \(\approx 6\) meses (\(t_1\rightarrow t_2\)), questão deixada para o experimento com apenas \(i_1\) e \(i_2\).

### Síntese

A exclusão dos conversores com AD na terceira imagem não derruba o teto multimodal nem a superioridade da shape. A hipótese de que a terceira aquisição “vazava” doença estabelecida e inflava de forma geral as métricas do estudo **não** se confirma para as arquiteturas e famílias que definem o claim. Confirma-se, contudo, que interpretações de ganho longitudinal em textura (e, em menor medida, volume) no protocolo *soft* devem ser cautelosas: esse sinal é sensível à presença do subgrupo MCI–MCI–AD e não deve ser equiparado ao suporte morfométrico do intervalo de seis meses discutido na introdução.

---

# ablação multiclasse

No pipeline isto **não é classificação multiclasse**. É **late fusion** (união de scores) entre famílias unimodais. Cada família continua um SVM binário sMCI×pMCI; a união média os scores OOF no mesmo paciente / mesmo fold.

### O que se une

Cinco famílias, hipocampo L+R: **vol, shape, texture (GLCM), disp, firstorder**.  
Cada ramo entra com **um** encoding:

- baseline = `t1_only` (só i1)  
- longitudinal = Q4 `t1_d21_d32` (T1 + Δ21 + Δ32)

Não entra `t1_d21` nesta grelha (isso seria 3 opções por família). Não entra early concat (`--modality all`). Não entra clínica nesta grelha (isso é `5_clinic_img.py`).

### Quantas uniões

Família omitida **ou** T1 **ou** Q4, com **pelo menos 2** ramos (1 ramo = unimodal, já medido à parte):

\[
\sum_{k=2}^{5}\binom{5}{k}\,2^{k} = 40+80+80+32 = 232.
\]

Isto **não** é \(5!\). Ordem dos ramos não muda a média. Permutações da mesma união não se repetem.

Por \(k\): 10 pares × 4 encodings; 10 triplos × 8; 5 quádruplos × 16; 1 quíntuplo × 32.

Specs nomeadas (estão dentro das 232):

- all-T1 (5 fam.)  
- all-Q4 (5 fam.)  
- **âncora paper:** shape T1 ∪ vol/texture/disp/firstorder Q4  
- **âncora 2 fam.:** shape T1 ∪ vol Q4  

`LATE_GRID=full` na claim `48m_6m`. Outras 3 coortes: só as 3 specs paper.

### Dados (claim)

Coorte **48m_6m**, `soft_pmci=True`: **n = 193** (73 sMCI / 120 pMCI), **3 visitas** T1-w (~6 meses entre slots; trajectória ~12 meses). Mesmos IDs em T1, D21 e Q4. Split **por paciente**, nunca por imagem.

Nested CV alinhado: **k_out = 5**, **k_in = 5**, **10 repeats**, seed 42, estratificado por classe. Outer ≈ 80/20 (treino ~154, teste ~39; n/fold exacto vem de `test_id_pts` no CSV). SVM + `l1_stable` (var → |ρ|>0.85 → 50× L1 C=0.1 → π≥70%) só no outer-train de **cada ramo**. ComBat off. Tuner Optuna 10 trials no inner.

Métrica: **AUC patient-level** (média do score OOF por `ID_PT`).

### Como a união é feita

1. Corre (ou lê disco) o unimodal de cada slot `família:encoding`.  
2. Late (`5_ablation_late_fusion.py --combine mean --reuse-disk`): para cada `(repeat, fold)` comum, **inner-join** dos `ID_PT` de teste; score união = **média aritmética** dos scores dos ramos.  
3. Não retreina um SVM na concatenação. Dimensão não explode: crescem ramos, não colunas.  
4. Paciente que falte num ramo naquele fold cai fora do join daquele fold.

Saída: `csvs/cohorts/48m_6m/ablation_results_late_fusion/{fingerprint}/` (50 linhas = 10×5 folds por spec, SVM).

Custo extra da grelha: **barato** se os 5×2 unimodais SVM já existem. `SKIP_EXISTING=1` não regrava fingerprint já no disco.

### Como ler o resultado

Teto a bater: **shape T1** (melhor unimodal típico), não “o máximo das 232”. Ranking em `6_results` (argmax `late__*` vs shape T1). As 232 são **exploratórias**: o melhor da grelha está enviesado (winner’s curse). Contraste oficial vs teto = `7_stats` (e FDR se reportares vários specs).

`LATE_GRID=paper` = só all-T1, all-Q4, âncora 5 fam.

# sugestão chatgpt

A sugestão do ChatGPT é **boa para a Introduction/Discussion** — e já está alinhada com o que tens em `0_notes.md`. Ajusta 2–3 pontos ao **lock** do teu paper.

### O que está sólido (usar)

1. **Schuff / Hua / Leung** — justificativa biológica de atrofia ~6 m em MCI/LMCI/conversores.  
2. **Distinção chave:** significância de **grupo** ≠ medição **individual** fiável. Obriga o teu desenho (classificador patient-level) a ser modesto.  
3. **Três visitas ≈ 12 meses de trajectória** — a frase que o ChatGPT propõe em inglês é a formulação certa; evita “substantial change within six months”.  
4. **Morphometric vs intensity-derived** — encaixa perfeitamente no HM e nas famílias fracas (texture/FO).  
5. **Mubeen** — citar com a ressalva multimodal (não como prova de T1 sozinho).

### O que corrigir face ao teu lock

| ChatGPT | Teu paper |
|---------|-----------|
| “Manteria 48m_6m **strict** como principal” | Claim = **soft=True** (73/120). Strict = **sensibilidade**. |
| Ênfase em “strict agora com mais confiança” | Soft maximiza n e o gradiente; strict testa se o claim depende de MCI–MCI–AD. |
| Análise \(\|\Delta_{21}\|,\|\Delta_{31}\|\) + retest | Útil para revisor TMI; **não bloqueia** o fecho experimental actual (`0_todo`: nice-to-have). |

Não mudes a coorte principal por causa deste texto. Usa o texto para **justificar 6 m**, não para trocar soft→strict.

### Como encaixar na pergunta “compensa 2/3 imagens?”

A literatura **não prova** que o teu classificador vai ganhar. Provê:

- é **razoável testar** 6 m (sobretudo morfometria, LMCI/conversores);  
- se o teu ΔAUC for **nulo**, isso **não** contradiz Schuff — contradiz a esperança de que o sinal de grupo vire ganho discriminativo individual com as tuas features/CV;  
- se for **positivo** só em vol/shape, a distinção morphometric vs intensity fica ainda mais forte.

Ou seja: o ChatGPT defende a **pergunta**; os teus experimentos + stats dão a **resposta**.

### O que eu cortaria do draft

- Tom “resposta biológica inequívoca” → mais: *detectável a nível de grupo / potencia em LMCI*.  
- Extrapolação automática para GLCM/FO (ele já avisa — reforça).  
- Prometer noise floor / retest no paper 1 se não fores correr isso agora — menciona como limitação ou trabalho futuro.

### Parágrafo-âncora (conceito)

> Intervalos de ~6 meses foram escolhidos para amostragem temporal densa. Alterações morfométricas hipocampais mensuráveis nesse intervalo estão documentadas em ADNI, sobretudo em MCI avançado e conversores; o desenho de três visitas cobre ~12 meses com dois incrementos sucessivos. A extrapolação para biomarcadores de intensidade (GLCM, first-order) é exploratória, dada a natureza não quantitativa do T1-w.

**Veredito:** usa a sugestão como **fundamentação biológica** do `48m_6m`; mantém claim soft + strict como sensibilidade; morphometric vs intensity como eixo do manuscrito; \(\Delta\)/retest opcional, não obrigatório para submeter.
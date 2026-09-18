https://chatgpt.com/share/6a96cae5-5d58-83e9-8773-481ad3fbb052

# Incorporar ou corrigir no artigo:

## ICV: homotetia, não divisão crua (shape)

Bug antigo: `4_run_post_extract` fazia `x / ICV` para eixos (mm) e `SurfaceArea` (mm²). Unidades arbitrárias (mm⁻², mm⁻¹). Correção:

- comprimento `/ ICV^{1/3}`
- área `/ ICV^{2/3}`
- volume `/ ICV` (`MeshVolume`, mm³; já estava certo)
- `SurfaceVolumeRatio` `× ICV^{1/3}` (já A/V; fica = A′/V′)
- Sphericity / Elongation / Flatness: intocados (adimensionais)
- `gm_norm` etc.: sem ICV

Texto: scaling **geométrico** (invariância de calote), não z-score. `StandardScaler` continua a jusante no SVM.

Pipeline: `4_` escreve em `csvs/cohorts/{COHORT}/` (o que `5_ablation --cohort` lê). Também grava `adnimerged_longitudinal.csv` nessa pasta.

`3_feat_rad.py` **não** reroda. Só 4_ + ablação **shape**.

## Longitudinal ComBat — sensibilidade (Beer et al., 2020)

**Papel no paper:** protocolo primário = **sem** harmonização (`--combat false`). Longitudinal ComBat = **sensibilidade** na coorte principal `48m_6m`, apenas representações com ≥2 visitas (`t1_d21`, `t1_d21_d32`). Não misturar com o contraste de encoding T1 vs 2 vs 3 imagens. Saídas em pastas `*_longcombat` (não pisa claim).

**Citação:** Beer JC, Tustison NJ, Cook PA, Davatzikos C, Sheline YI, Shinohara RT, Linn KA. Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage*. 2020;220:117129. DOI: [10.1016/j.neuroimage.2020.117129](https://doi.org/10.1016/j.neuroimage.2020.117129). Código: `modules/longitudinal_combat.py` + `modules/ablation_harmonize.py`.

### Definição de lote (batch)

Igual ao ComBat transversal antigo — **não** redefinido:

- `batch = MANUFACTURER + "_" + FIELD_STRENGTH` (ex. `GE MEDICAL SYSTEMS_1.5`, `SIEMENS_3.0`)
- **Fora:** site ADNI, `MFG_MODEL`, coil, software
- Motivo: incluir modelo explode micro-lotes no ADNI (n amostral insuficiente no outer-train)

### Tratamento de lotes raros (por fold, só treino)

1. Batch com \(n < 5\) imagens no treino → funde em `OTHER_<Tesla>` (ex. `GE MEDICAL SYSTEMS_3.0` raro → `OTHER_3.0`; **não** mistura 1.5 com 3.0)
2. Se após o pool ainda \(n < 5\) → imagem **excluída** da harmonização (fica raw)
3. Precisa ≥2 batches “grandes” no treino; senão skip (devolve original)
4. Batch só no teste / nunca visto no treino → **não** harmoniza
5. Beer exige ≥2 obs/batch; aqui limiar mais conservador: **≥5** no treino

### O que remove / o que preserva

| Alvo | Remove? | Como |
|---|---|---|
| Fabricante (via batch) | Sim | efeito fixo de lote + EB location/scale |
| Campo 1.5/3 T (via batch) | Sim | no mesmo rótulo `MANUFACTURER_FIELD` |
| Modelo de scanner | Não | deliberadamente fora do batch |
| Biologia entre pacientes | **Não** | intercepto aleatório \((1\mid\mathrm{ID\_PT})\) **preserva** nível do sujeito |
| Mesmo paciente, scanners/Tesla diferentes nas visitas | Corrige salto técnico | cada visita usa o efeito do **seu** batch; \(\eta_j\) partilha nível biológico |
| Diagnóstico / sMCI×pMCI (`GROUP`) | Não | **fora** da fórmula (evita leakage de rótulo) |
| Deltas \(\Delta_{21}/\Delta_{32}\) | Não directamente | harmoniza atributo **por visita**; \(\Delta\) calculado **depois** |

### Passo a passo (nested CV, cada fold externo)

1. Filtra pacientes do fold (treino ∪ teste); se `combat=true`, corta visitas antes do LME (`t1_d21` → T1+T2; `t1_d21_d32` → três). Impede T3 de informar BLUP numa análise D21.
2. Monta tabela **por imagem** (`ID_IMG`): features da família (roi×side×atributo), `batch`, `SEX`, idade basal (AGE da 1ª visita), `time_years` (anos desde `MRI_DATE` da 1ª visita). Uma família por vez (vol com vol, …).
3. Pool / exclusão de batches pequenos (acima).
4. **Fit só no treino.** Para cada feature, MixedLM REML:

\[
y_{ij}(t)=\alpha+\beta_{\mathrm{age}}\,\mathrm{age}_0+\beta_t\,t+\beta_{\mathrm{sex}}\,\mathrm{sex}+\gamma_{\mathrm{batch}}+\eta_j+\varepsilon
\]

com \(\eta_j\sim N(0,\rho^2)\). Sem `GROUP`.

5. Recupera efeitos de batch com \(\sum_i n_i\gamma_i=0\); padroniza resíduos; Empirical Bayes (30 iterações) → \(\gamma^\star\) (média) e \(\delta^{2\star}\) (variância) por batch×feature (Beer REML).
6. **Transform** treino+teste com parâmetros congelados do treino:

\[
y^{\mathrm{ComBat}}=\frac{\sigma}{\delta^\star}(z-\gamma^\star)+\hat y-\gamma_{\mathrm{adj}}
\]

No teste: \(\eta_j\) = BLUP só com as visitas **desse** sujeito; batch desconhecido → linha intacta.

7. Pivot T1/T2/(T3) → \(\Delta\) → SVM / late fusion. Classificador vê encoding já sobre atributos corrigidos.

### Paciente com scanners diferentes ao longo do tempo

Ex.: T1 em `GE_1.5`, T2 em `SIEMENS_3.0`. Remove efeito do lote de cada visita; \(\eta_j\) comum mantém nível biológico; salto técnico entre visitas deixa de entrar no \(\Delta\) (na medida em que o batch o captura). Mudança só de **modelo** GE mantendo `GE_1.5` → mesmo lote (trade-off n amostral).

### Software / saídas

- Python (`statsmodels.MixedLM`); port conceptual do R `longCombat` (Beer), com apply indutivo (pacote R é transductivo).
- CLI: `--combat true` em `5_ablation.py` / `5_ablation_late_fusion.py`.
- Roots: `ablation_results_d21_longcombat/`, `ablation_results_d21d32_longcombat/`, `ablation_results_late_fusion_longcombat/`.
- Meta CSV: `harmonization_method = longitudinal_combat_reml` quando ligado.

### Frase-âncora (Métodos)

> O protocolo primário não aplica harmonização entre aparelhos. Como análise de sensibilidade na coorte principal, nas representações de duas e três visitas, ajustámos Longitudinal ComBat (Beer et al., 2020; REML) aos atributos de cada família ao nível da visita, com lote definido por fabricante × intensidade de campo (1,5/3 T), intercepto aleatório por indivíduo, covariáveis sexo, idade basal e tempo desde a primeira visita, sem o rótulo de classe, estimado apenas no treino de cada dobra externa. Lotes com menos de cinco imagens no treino foram fundidos em `OTHER_<Tesla>` ou excluídos. Os incrementos temporais foram calculados após a correção. A representação restrita à primeira visita não entra neste braço: o modelo misto exige medidas repetidas.

### Limitações (Discussão / Tabela D)

- Sensibilidade ≠ ablação de encoding (não reportar ΔAUC T1 vs Q4 *com* longCombat como prova de 2/3 imagens).
- EB com poucas features (hipocampo L+R × momentos da família) vs 62 espessuras do Beer — prior mais fraco.
- \(\eta=0\) aproximação no teste é BLUP local do sujeito; Beer original é transductivo.
- Sem `GROUP`: se scanner e diagnóstico se associam, parte do sinal de classe pode ir para o lote (trade-off leakage vs preservação).
- Batch grosso (sem modelo): residual de scanner possível.
- Reportar uma linha na Tabela D (ligado vs desligado), não figura grande.

### Resultado empírico (2026-09-17) — `48m_6m` Q4 vs longCombat

**Setup:** coorte `48m_6m` (`soft_pmci=True`), encoding `t1_d21_d32`, SVM, `l1_stable`, Optuna 10 trials, 10×5 folds (50), seed 42. Primário = `ablation_results_d21d32/` (`combat=false`). Sensibilidade = `ablation_results_d21d32_longcombat/` + late `ablation_results_late_fusion_longcombat/` (`harmonization_method=longitudinal_combat_reml`). Métrica: `auc_patient_mean`.

**Veredito:** Longitudinal ComBat **não** melhora o conjunto. Piora famílias fortes e o late fusion; ganhos só em texture/disp (já fracos). Manter primário sem ComBat; Tabela D = sensibilidade negativa.

#### Unimodal Q4

| Família | sem ComBat | longCombat | Δ |
|---|---:|---:|---:|
| vol | 0.763 | 0.726 | **−0.037** |
| shape | 0.721 | 0.724 | +0.003 |
| texture | 0.674 | 0.704 | +0.030 |
| disp | 0.577 | 0.609 | +0.032 |
| firstorder | 0.686 | 0.637 | **−0.049** |

#### Late fusion (5 famílias Q4)

| | `auc_patient_mean` | `auc_mean` | `auc_pooled` |
|---|---:|---:|---:|
| sem ComBat | **0.787** | 0.777 | 0.771 |
| longCombat | 0.761 | 0.749 | 0.741 |
| Δ | **−0.026** | −0.028 | −0.030 |

**Nota:** ramo `disp` neste run ainda = features **v3** (`disp_long` pré-v4). Se promover DVF v4, re-correr só `disp` + late fusion longCombat; vol/shape/texture/firstorder deste run mantêm-se.

**Frase Tabela D (rascunho):** *Na coorte principal, representação de três visitas, Longitudinal ComBat (Beer, REML) não elevou a AUC patient-level face ao protocolo sem harmonização (late fusion 0,787 → 0,761; volume −0,037; first-order −0,049); ganhos pontuais em textura/deslocamento não alteram a conclusão primária.*

# Incorporar ou corrigir no artigo (resto):

## Quantidade de dados após o split dentro do treino/teste externo e interno.

Dados presentes:

Tamanho total por coorte (ex: coorte principal 48m_6m n total, n sMCI / n pMCI)
Razão aproximada: "80% treino, 20% teste" no fold externo
Tabela tab:nfeat mostra cardinalidade média de features, mas não n por fold
Faltam: contagens exatas tipo "fold 1: treino, teste" ou distribuição classe por fold no split externo, e idem para interno. Esse detalhe não está no artigo atual. Seria informação útil para reprodutibilidade — vale adicionar como tabela ou no texto descritivo da seção de validação cruzada.



refazer todas as demais analises com a nova coorte principal (48m_6m) feitas a priori com a coorte 48m_12m e reescrever o artigo com base na nova coorte principal. 

Sim — há suporte médico sólido para a detecção de alterações estruturais em T1-w MRI em intervalos de aproximadamente 6 meses, inclusive no MCI. Contudo, existe uma distinção essencial: a literatura demonstra muito melhor a detectabilidade estatística da alteração em grupos do que a capacidade de medir, com alta confiabilidade, uma alteração biologicamente verdadeira em cada indivíduo ao longo de apenas seis meses.

Isso torna a estratégia 48m_6m defensável, mas eu mudaria ligeiramente a forma de justificá-la.

O trabalho mais diretamente relacionado ao nosso problema

Mubeen et al., no Journal of Neuroradiology (2017), estudaram exatamente a questão: se adicionar uma avaliação longitudinal em aproximadamente 6 meses melhora a predição de conversão sMCI → pMCI. Foram 247 indivíduos com MCI, 162 pMCI e 85 sMCI, usando MRI estrutural T1, variáveis cognitivas e demográficas. O modelo baseline apresentou AUC 0,82, enquanto o modelo incorporando baseline + 6 meses atingiu AUC 0,87, com melhora significativa (\(P<0,05\)). Os autores utilizaram, entre os biomarcadores estruturais, medidas de integridade/atrofia hipocampal e corpo caloso.

Mubeen AM et al. A six-month longitudinal evaluation significantly improves accuracy of predicting incipient Alzheimer's disease in mild cognitive impairment. Journal of Neuroradiology. 2017;44(6):381–387. DOI: 10.1016/j.neurad.2017.05.008.

Há uma ressalva: esse resultado não prova que T1 MRI isoladamente melhora de 0,82 para 0,87, porque o classificador era multimodal. Entretanto, os próprios autores relatam que as alterações estruturais foram particularmente informativas no intervalo curto.

Evidência biológica ainda mais importante: Schuff et al., Brain

Para nossa justificativa, considero este talvez o artigo mais importante.

Schuff et al. analisaram ADNI multicêntrico com 127 CN, 226 MCI e 96 AD, todos examinados em baseline, 6 e 12 meses. Eles mediram diretamente a perda de volume hipocampal em T1-w MRI.

No intervalo 0–6 meses, as taxas anualizadas foram aproximadamente:

Grupo	Taxa anualizada de perda hipocampal
CN	\(-0,9\%\)
MCI	\(-2,0\%\)
AD	\(-3,3\%\)

No MCI, a perda hipocampal já era altamente significativa em 0–6 meses (\(P<0,0001\)).

Isso corresponde aproximadamente, em seis meses, a:

$$ \Delta V_{\rm MCI}^{6m}\approx -1.0\% $$

e, para AD,

$$ \Delta V_{\rm AD}^{6m}\approx -1.65\%. $$

Portanto, a resposta biológica é inequívoca: sim, ocorre alteração macroscópica detectável por T1 MRI nesse intervalo.

Schuff N et al. MRI of hippocampal volume loss in early Alzheimer's disease in relation to ApoE genotype and biomarkers. Brain. 2009;132(4):1067–1077. DOI: 10.1093/brain/awp007.

Há uma nuance importante nesses números

O hipocampo médio dos indivíduos MCI naquele estudo tinha cerca de \(1846~\mathrm{mm^3}\), e a perda anual estimada em 0–6 meses foi aproximadamente

$$ -37.7~\mathrm{mm^3/ano}. $$

Logo, em seis meses estamos falando de apenas aproximadamente

$$ 19~\mathrm{mm^3}. $$

Isso parece pequeno para uma imagem com voxels da ordem de \(1~\mathrm{mm^3}\), mas métodos longitudinais não dependem de detectar “19 voxels que desapareceram”. Registro intraindivíduo, modelos de superfície, BSI, TBM etc. acumulam pequenas alterações de fronteira distribuídas por centenas ou milhares de pontos e podem estimar deslocamentos subvoxel.

Por isso é possível detectar uma variação de ~1% mesmo quando nenhum voxel isolado apresenta uma mudança inequívoca.

Mas aqui está a advertência: no artigo de Schuff, a variabilidade interindividual era muito maior que a alteração média. Portanto,

$$ \text{significância populacional} \;\not\Rightarrow\; \text{medição individual precisa}. $$

Esse ponto deve ser explicitamente reconhecido no nosso artigo.

Evidência especificamente para late MCI

Hua et al., usando 5.738 exames ADNI2, estudaram TBM em T1-w MRI com aquisições em screening, 3, 6, 12 e 24 meses. A conclusão particularmente relevante foi:

para obter potência estatística razoável com biomarcadores MRI-TBM, o intervalo mínimo foi 6 meses para LMCI e AD, mas 12 meses para EMCI.

Hua X et al. MRI-based brain atrophy rates in ADNI phase 2: acceleration and enrichment considerations for clinical trials. Neurobiology of Aging. 2016;37:26–37. DOI: 10.1016/j.neurobiolaging.2015.09.018.

Isso é extremamente relevante para nós porque mostra que a resposta à pergunta

“seis meses é suficiente?”

não é simplesmente sim ou não.

Depende do estágio da doença:

$$ \text{CN/EMCI} \quad\rightarrow\quad \text{sinal menor} $$ $$ \text{LMCI/pMCI próximo da conversão} \quad\rightarrow\quad \text{sinal maior}. $$

E justamente os indivíduos pMCI deveriam estar enriquecidos no segundo cenário.

Outro artigo muito pertinente: Leung et al., Neurology

Leung et al. analisaram scans ADNI em

$$ 0,\;6,\;12,\;18,\;24,\;36~\mathrm{meses} $$

e calcularam mudanças de cérebro inteiro, hipocampo e ventrículos com BSI. Encontraram aceleração significativa da atrofia hipocampal em MCI,

$$ 0.22\%/\mathrm{ano^2}, \qquad p=0.037, $$

e, mais importante, uma análise posterior mostrou que essa aceleração era principalmente determinada pelos MCI que posteriormente converteram para AD, para os quais a aceleração chegou a

$$ 0.50\%/\mathrm{ano^2}, \qquad p=0.003. $$

Leung KK et al. Cerebral atrophy in mild cognitive impairment and Alzheimer disease: rates and acceleration. Neurology. 2013;80(7):648–654. DOI: 10.1212/WNL.0b013e318281ccd3.

Isso fornece precisamente a justificativa biológica que precisamos: os pacientes MCI que caminham para AD são justamente aqueles nos quais esperamos maior dinâmica estrutural longitudinal.

E existe evidência até em indivíduos cognitivamente normais

Murphy et al., NeuroImage (2010), mostraram que mudanças em estruturas do lobo temporal medial entre baseline e 6 meses em T1-w MRI de 142 idosos cognitivamente normais estavam associadas ao declínio cognitivo posterior.

DOI: 10.1016/j.neuroimage.2010.07.016.

Portanto, seis meses não é um intervalo biologicamente absurdo mesmo em CN; naturalmente, o SNR é pior.

Mas existe um ponto muito importante para o nosso próprio desenho

Na realidade, nossa coorte 6m é mais forte do que a expressão “alterações em seis meses” sugere.

Nós temos três imagens:

$$ i_1,\;i_2,\;i_3 $$

com aproximadamente

$$ t_2-t_1\simeq6~\mathrm{meses}, \qquad t_3-t_2\simeq6~\mathrm{meses}. $$

Portanto:

$$ t_3-t_1\simeq12~\mathrm{meses}. $$

Ou seja, a representação longitudinal não está observando apenas um deslocamento em seis meses. Ela contém aproximadamente:

$$ x(t_1), \qquad \Delta x_{21}, \qquad \Delta x_{32}, $$

e implicitamente uma trajetória cobrindo aproximadamente um ano.

Isso muda bastante minha avaliação.

Eu não escreveria:

“we expect substantial hippocampal changes within six months.”

Isso seria forte demais.

Escreveria algo conceitualmente como:

Six-month interscan intervals were selected to provide dense temporal sampling of structural change. Although individual morphometric changes over a single six-month interval can be subtle, previous ADNI studies have demonstrated measurable hippocampal and regional atrophy over this interval, particularly in late MCI and AD. Importantly, the present three-visit design spans approximately 12 months while providing two consecutive short-term change estimates.

Essa formulação é cientificamente muito robusta.

Onde eu ficaria mais cético: nem todos os nossos atributos têm a mesma sustentação

Este é, para mim, o aspecto mais importante da sua pergunta.

A literatura dá sustentação muito forte para mudanças de seis meses em:

volume

$$ V(t) $$

shape/morfometria

cortical thickness

deslocamento/deformação obtidos de registro longitudinal apropriado

ventricular expansion.

Mas eu não estenderia automaticamente essa justificativa para GLCM e first-order intensity features.

T1-w MRI convencional não é uma modalidade quantitativa de intensidade:

$$ I_{\mathrm{T1w}} \neq T_1 $$

e

$$ I_{\mathrm{T1w}}(x) $$

depende de coil sensitivity, ganho, scanner, sequência, bias field, reconstrução, normalização etc.

Portanto, afirmar que uma mudança de textura hipocampal em seis meses representa diretamente alteração microestrutural seria muito mais difícil de defender.

Aliás, o próprio resultado do nosso experimento de histogram matching — com grande alteração do comportamento de GLCM e first-order — é praticamente uma advertência empírica de que essas famílias são bastante sensíveis à transformação fotométrica.

Eu faria uma distinção forte no artigo:

$$ \boxed{\text{morphometric longitudinal biomarkers}} $$

versus

$$ \boxed{\text{intensity-derived exploratory biomarkers}}. $$

Isso aumentaria a credibilidade do estudo.

Portanto, eu manteria 48m_6m strict?

Sim. Depois de verificar a literatura, eu a manteria — e com mais confiança.

Mas com uma mudança conceitual importante:

Eu não justificaria 48m_6m dizendo que:

seis meses necessariamente produzem uma alteração hipocampal grande.

A justificativa seria:

mudanças morfométricas hipocampais já foram demonstradas em MCI em seis meses;
a detectabilidade é maior em estágios mais avançados de MCI e em indivíduos que progridem para AD;
estudos ADNI mostram que 6 meses pode constituir intervalo mínimo útil para LMCI;
já existe inclusive um estudo de classificação sMCI/pMCI mostrando ganho com dados longitudinais de seis meses;
nosso experimento contém três observações e, portanto, cobre aproximadamente 12 meses, com dois incrementos consecutivos de aproximadamente seis meses.

Esse último ponto me parece particularmente forte.

Uma consequência experimental que eu considero importante

Diante da literatura, eu acrescentaria uma análise muito simples aos novos experimentos strict.

Além de comparar

$$ T1 \quad \text{vs.}\quad Q4, $$

calcularia explicitamente para cada atributo longitudinal:

$$ \Delta_{21}=x_2-x_1, \qquad \Delta_{32}=x_3-x_2 $$

e

$$ \Delta_{31}=x_3-x_1. $$

Então verificaríamos a magnitude e distribuição de

$$ |\Delta_{21}|,\;|\Delta_{32}|,\;|\Delta_{31}| $$

em sMCI e pMCI.

Se o sinal for biologicamente real e relativamente consistente, esperaríamos algo aproximadamente como

$$ |\Delta_{31}| > |\Delta_{21}| $$

e/ou

$$ |\Delta_{31}| > |\Delta_{32}|, $$

especialmente para volume e algumas características de shape.

Isso nos permitiria responder empiricamente a uma pergunta que certamente um revisor da TMI poderá fazer:

Are the six-month structural changes larger than the variability of the measurement itself?

E eu iria um pouco além: se existirem scans repetidos muito próximos temporalmente no ADNI que possam ser usados para estimar test–retest variability, poderíamos estabelecer um noise floor para cada família:

$$ \sigma_{\mathrm{retest}} $$

e então avaliar

$$ \frac{|\Delta x_{6m}|}{\sigma_{\mathrm{retest}}}. $$

Isso transformaria uma potencial fraqueza da escolha de seis meses em uma análise metodológica bastante sofisticada.

Em síntese: a literatura médica apoia claramente mudanças morfométricas detectáveis em seis meses, especialmente em LMCI/pMCI. Portanto, eu não abandonaria a coorte 48m_6m strict. Pelo contrário: com 73 sMCI × 74 pMCI, três visitas pré-conversão e ~12 meses de extensão total da trajetória, considero agora que ela tem uma fundamentação biológica e metodológica convincente. O cuidado deve estar em não extrapolar essa sustentação igualmente para todas as famílias de atributos, sobretudo textura e primeira ordem.

Se desejar, posso fazer em seguida uma busca sistemática mais profunda especificamente por estudos T1-w MRI de 6 meses em sMCI/pMCI, separando evidência de volume, shape, TBM/deformação e radiômica, para decidirmos exatamente quais trabalhos devem entrar no estado da arte do manuscrito.

##############################

# Testes realizados

## Definição melhor abordagem longitudinal

- deltas absolutos (D21=T2−T1, D31=T3−T1, D32=T3−T2)
- deltas relativos (D21=(T2−T1)/T1, D31=(T3−T1)/T1, D32=(T3−T2)/T2)
- absolutos atributos (T1, T2, T3)
- deltas absolutos + deltas dos deltas (D21, D32, A=D32−D21)
- i1 absoluto + 2 deltas sequenciais (T1, D21, D32) (Vencedor)
- i1 absoluto + 3 deltas sequenciais (T1, D21, D31, D32)
- i1 absoluto + ritmo + aceleração (T1, M=(D21+D32)/2, A=D32−D21)

# Notas — pipeline do paper (pós Patch A)

Allowlist 4–5 nomes **morta**. Números em `cohort_results.csv` / tex antigos = experimento errado. Não citar.

Mapa do repo `graphs/` por etapa. Ordem = fluxo do paper (Patch A / late).

---

## 0. Pré-proc ADNI (fora + patch)

| Script | Etapa |
|---|---|
| Pipeline ADNI em `/mnt/databases/...` (strip→bias→MNI→seg→DKT) | Gera raw/preproc; **não** está neste repo |
| `preproc/dkt_labelling.py` + `run_dkt_labelling.py` (pasta `preproc/`) | Parcellation DKT (antspynet) |
| `fix_missing_dkt.py` | **Patch:** 4D→3D + DKT nos IDs sem `regions` → `insert_to_databases_regions/` |

---

## 1. Coorte / clínica

| Script | Etapa |
|---|---|
| `1_dataset.ipynb` | Define coortes (`36m_*`, `48m_*`, `all_population`), bandas ±2 m, CSVs longitudinais |
| `1_dataset_old.ipynb` | Legado (protocolo antigo) |
| `analysis_adni.ipynb` | Análise exploratória ADNI (não é etapa do claim) |

---

## 2. Espaço comum MNI

| Script | Etapa |
|---|---|
| `2_resample.py` | Rigid T1→MNI 1 mm; warpa `regions`/`seg`/`brain_mask` → `images/{resampled_1.0mm,regions,seg,brain_mask}/` |

---

## 3. Extração de features (store `all_population`)

| Script | Etapa | Saída |
|---|---|---|
| `3_feat_vol.py` | Volumes CSF/GM/WM por ROI | `features_volumetric.csv` |
| `3_feat_rad.py` | Radiomics (shape/GLCM/firstorder) | `features_radiomic.csv` |
| `3_feat_gen_dvf_v2.py` | Warps sujeito→template CN | `images/displacement_field_v2/` |
| `3_feat_dvf_v2.py` | Stats DVF por ROI (espaço template) | `features_displacement_v2.csv` |
| `3_feat_gen_dvf.py` + `3_feat_dvf.py` | **Legado v1** (fixed=clínica) | `features_displacement.csv` — preferir v2 |

---

## 4. Merge / long para ablação

| Script | Etapa |
|---|---|
| `4_run_post_extract.py` | Junta vol+rad+disp_v3 → longs em `csvs/cohorts/{COHORT}/ablation/` (homotetia ICV); scanner batch |

Módulos usados aqui: `ablation_prep` (export long, batch).

---

## 5. Modelos / ablação

| Script | Etapa | Paper? |
|---|---|---|
| `5_ablation.py` | Unimodal nested CV (vol/shape/texture/disp/firstorder) | **Sim** (mono) |
| `5_ablation_late_fusion.py` | Late fusion (média de scores SVM) | **Sim** |
| `5_clinic_img.py` | Clínica e clinic+img | Extra / suplemento |
| `5_ablation_leaky.py` | Controlo leaky | Extra |
| `5_ablation_early_fusion.py` | Early concat | **Fora** do paper |
| `run_ablation_full.sh` | Orquestra mono + late (+ extra) nas 4 coortes | **Sim** (driver) |
| `scripts_compare_fusion_vs_shape.py` | Comparação pontual fusion vs shape | Auxiliar |

### Módulos (`modules/`) — onde entram

| Módulo | Papel | Usado em |
|---|---|---|
| `ablation_prep.py` | ROI filter, denylist/keep, load long, scanner | `4_`, `5_*` |
| `ablation_deltas.py` | Deltas T1/D21/D32; colunas por família | prep / runner / Q4 |
| `ablation_representation.py` | `t1_only`, `t1_d21_d32`, fusion specs, paths | `5_*` |
| `ablation_stable.py` | Seletor `l1_stable` (corr → L1 → π) | runner |
| `ablation_harmonize.py` | Longitudinal ComBat Beer REML (se `--combat true`) | runner |
| `longitudinal_combat.py` | Fit/transform LME+EB (Beer 2020) | `ablation_harmonize` |
| `ablation_optuna.py` | Tune SVM/etc. | runner, clinic |
| `ablation_runner.py` | Nested CV unimodal + early fusion suite | `5_ablation`, early, clinic |
| `ablation_late_fusion.py` | Inner-join por `ID_PT`, mean/weighted scores | `5_ablation_late_fusion` |
| `ablation_runner_leaky.py` | Pipeline leaky | `5_ablation_leaky` |
| `ablation_analysis.py` | AUC patient-level, freq. features, summaries | todos `5_*`, notebooks |
| `cohort_compare.py` | Multi-coorte → `cohort_results` / `cohort_features_long` | pós-run / rebuild planilha |
| `stats_compare.py` | Contrastes estatísticos entre configs | `7_stats` |

---

## 6–7. Resultados, stats, artigo

| Script | Etapa |
|---|---|
| `6_results.ipynb` | Figs / tabelas a partir dos CSVs de ablação |
| `7_stats.ipynb` | Stats oficiais (Q4 vs t1_only, FDR, etc.) |
| `artigo/` (`artigo.tex`) | Draft paper (números só após CSVs novos) |

Notas: `0_notes.md`, `0_todo.md` — protocolo e checklist, não código.

---

## Fluxo resumido (paper atual)

```
1_dataset.ipynb
    → 2_resample.py
    → 3_feat_vol + 3_feat_rad + 3_feat_gen_dvf_v2 + 3_feat_dvf_v2
    → 4_run_post_extract.py
    → run_ablation_full.sh  (5_ablation + 5_ablation_late_fusion [+ extra])
    → cohort_compare / 6_results / 7_stats
    → artigo.tex
```

**Estado agora:** etapa **3** a meio (vol/rad incompletos; disp_v2 ainda não); `4`/`5`/`6`/`7` do claim novo ainda à frente.

## Perguntas

1. sMCI vs pMCI: Q4 (`t1_d21_d32` = T1+D21+D32) agrega sinal vs `t1_only`?
2. Quais atributos o seletor escolhe em cada família (frequência across folds×repeats)?
3. União **late** (3 specs) bate teto unimodal?

## Porquê late, não early

Late = 5 SVMs, um por família, média de scores. Forma não vê 144 GLCM. Volume não vê firstorder. `p≫n` da união **não** aplica: cresce nº de modelos, não a dimensão de um só.

Early / `--modality all` = concat. Classe cheia × Q4 → centenas de colunas, n=120. Fora do paper. `run_ablation_full.sh` não corre early.

## Espaço (antes do seletor)

Classe completa; denylist só definição (`keep_*_feat` em `ablation_prep.py` + `ablation_deltas.py`). Mesmos sufixos em `t1_only` e Q4. Cada sufixo × L e R. `t1_only` ×1 (T1). Q4 ×3 (T1, D21, D32).

| Família | Entra | Denylist | t1 L+R | Q4 L+R |
|---|---|---|---|---|
| vol | 4 sufixos | mm³ crus (`mask_mm3`, `gm_mm3`, …) | 8 | 24 |
| shape | 12 IBSI (`original_shape_`) | MeshVolume, VoxelVolume | 24 | 72 |
| texture | 24 GLCM (`original_glcm_`) | GLRLM/GLSZM/GLDM/NGTDM | 48 | 144 |
| firstorder | 16 (`original_firstorder_`) | Energy, TotalEnergy | 32 | 96 |
| disp | 15 momentos (`mag_`, `logjac_`, `strain_fro_`) | `ux/uy/uz`, `_n`, percentis | 30 | 90 |

Texture = GLCM Original. Sem wavelet/LoG.

### vol (4)
- `gm_norm`
- `wm_norm`
- `csf_norm`
- `original_shape_MeshVolume`

### shape (12)
- `original_shape_SurfaceArea`
- `original_shape_SurfaceVolumeRatio`
- `original_shape_Sphericity`
- `original_shape_Elongation`
- `original_shape_Flatness`
- `original_shape_LeastAxisLength`
- `original_shape_MajorAxisLength`
- `original_shape_MinorAxisLength`
- `original_shape_Maximum2DDiameterColumn`
- `original_shape_Maximum2DDiameterRow`
- `original_shape_Maximum2DDiameterSlice`
- `original_shape_Maximum3DDiameter`

### texture — GLCM (24)
- `original_glcm_Autocorrelation`
- `original_glcm_ClusterProminence`
- `original_glcm_ClusterShade`
- `original_glcm_ClusterTendency`
- `original_glcm_Contrast`
- `original_glcm_Correlation`
- `original_glcm_DifferenceAverage`
- `original_glcm_DifferenceEntropy`
- `original_glcm_DifferenceVariance`
- `original_glcm_Id`
- `original_glcm_Idm`
- `original_glcm_Idmn`
- `original_glcm_Idn`
- `original_glcm_Imc1`
- `original_glcm_Imc2`
- `original_glcm_InverseVariance`
- `original_glcm_JointAverage`
- `original_glcm_JointEnergy`
- `original_glcm_JointEntropy`
- `original_glcm_MCC`
- `original_glcm_MaximumProbability`
- `original_glcm_SumAverage`
- `original_glcm_SumEntropy`
- `original_glcm_SumSquares`

### firstorder (16; deny Energy, TotalEnergy)
- `original_firstorder_10Percentile`
- `original_firstorder_90Percentile`
- `original_firstorder_Entropy`
- `original_firstorder_InterquartileRange`
- `original_firstorder_Kurtosis`
- `original_firstorder_Maximum`
- `original_firstorder_Mean`
- `original_firstorder_MeanAbsoluteDeviation`
- `original_firstorder_Median`
- `original_firstorder_Minimum`
- `original_firstorder_Range`
- `original_firstorder_RobustMeanAbsoluteDeviation`
- `original_firstorder_RootMeanSquared`
- `original_firstorder_Skewness`
- `original_firstorder_Uniformity`
- `original_firstorder_Variance`

### disp (15; prefixos `mag_`, `logjac_`, `strain_fro_`)
- `mag_mean`, `mag_std`, `mag_variance`, `mag_skewness`, `mag_kurtosis`
- `logjac_mean`, `logjac_std`, `logjac_variance`, `logjac_skewness`, `logjac_kurtosis`
- `strain_fro_mean`, `strain_fro_std`, `strain_fro_variance`, `strain_fro_skewness`, `strain_fro_kurtosis`

Fora: `ux_*`, `uy_*`, `uz_*`, `curlmag_*`, `*_n`, `*_p05`, `*_p50`, `*_p95`.

## Seletor (`l1_stable`, só outer train)

1. Variance threshold (se zerar → `var>0`, não restaurar classe)
2. Corr `|ρ|>0.85` **uma vez**; fica maior `|ρ|` com `y`; nunca T1 vs D21/D32 do mesmo `anatomical_key`
3. 50× L1 `C=0.1`; coef=0 → `[]` (não keep-all)
4. π≥70% nos boots. Pool vazio → vazio, **nunca** a classe inteira
5. SVM nesse pool. `min_timepoints=0` em t1/Q4

Entre colineares: representante, não “melhor Haralick”. Reportar frequência across folds.

## Contrastes oficiais

- Uniclasse: 5 famílias × `{t1_only, t1_d21_d32}` × 4 coortes
- Late, 3 specs: tudo t1 · tudo Q4 · âncora `shape:t1_only ∪ resto Q4`
- Não crownear `wide`/`abs`/`t1_deltas`/`t1_deltas_rel`/`t1_ma` como claim longitudinal
- Extra (clínica, leaky, cn_ad, RF/EN): suplemento, não pergunta 1–3

## Paper (quando CSVs novos existirem)

Unimodal = Δ(Q4−t1) por família. Late = teto por juntar papéis, não “onde o tempo mais ganha”. Nomes = tabela de frequência, não assinatura única.

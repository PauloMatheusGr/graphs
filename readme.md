Last updated: 04/05/2006 - By Paulo Girardi

Descrição Pré-processamento

[14:00, 5/4/2026] PM: Todos os volumes de RM estrutural ponderados em T1 foram primeiramente submetidos à extração de crânio (skull-stripping) utilizando a função brain_extraction() do pacote ANTsPyNet, dentro do ecossistema ANTsX \cite{tustison-2021}. As entradas foram carregadas via ants.image_read() e as imagens 4D foram convertidas em 3D através da extração do primeiro volume com ants.slice_image(axis=3, idx=0). A extração cerebral foi realizada com modality="t1" e o diretório de cache do ANTsXNet fixado em /workspace/cache para garantir a reprodutibilidade, resultando em volumes sem crânio e máscaras cerebrais binárias.

Posteriormente, aplicou-se a redução de ruído utilizando o filtro de médias não-locais adaptativo (adaptive non-local means filter) implementado em ants.denoise_image() \cite{manjon-2010}, com os parâmetros: mask=None, shrink_factor=1, p=1, r=2, noise_model="Rician" e v=0. As inomogeneidades do campo de viés (bias field) foram corrigidas utilizando o algoritmo N4 implementado em ants.n4_bias_field_correction() \cite{tustison-2010}, empregando a configuração padrão do ANTsPy (mask=None, shrink_factor=1, rescale_intensities=True, spline_param=200 e convergência {'iters': [50, 50, 50, 50], 'tol': 1e-7}).

Finalmente, a segmentação de tecidos foi realizada por meio do framework Atropos baseado em aprendizado profundo (antspynet.deep_atropos()), com o pré-processamento interno ativado. A parcelação anatômica foi obtida através do método de rotulagem Desikan Killiany Tourville (antspynet.desikan_killiany_tourville_labeling()) com agrupamento lobar, ambos pertencentes ao ecossistema ANTsX \cite{tustison-2021}.

 @article{tustison-2021,

 title         = {{The ANTsX ecosystem for quantitative biological and medical imaging}},

 author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},

 journal       = {Scientific Reports},

 publisher     = {Nature},

 volume        = {11},

 number        = {9068},

 pages         = {1--13},

 note          = {},

 doi           = {10.1038/s41598-021-87564-6},

 issn          = {},

 year          = {2021}

 }


@article{manjon-2010,

 title         = {{Adaptive non-local means denoising of MR images with spatially varying noise levels}},

 author        = {Manjón, José V. and Coupé, Pierrick and Martí-Bonmatí, Luis and Collins, D. Louis and Robles, Montserrat},

 journal       = {journal of Magnetic Resonance imaging},

 publisher     = {Wiley},

 volume        = {31},

 number        = {1},

 pages         = {192--203},

 note          = {},

 doi           = {10.1002/jmri.22003},

 issn          = {},

 year          = {2010}

}


@article{tustison-2010,

 title     = {{N4ITK: Improved N3 Bias Correction}},

 author    = {Tustison, Nicholas J. and Avants, Brian B. and Cook, Philip A. and Zheng, Yuanjie and Egan, Alexander and Yushkevich, Paul A. and Gee, James C.},

 journal   = {IEEE Transactions on Medical Imaging},

 publisher = {IEEE},

 volume    = {29},

 number    = {6},

 pages     = {1310--1320},

 note      = {},

 doi       = {10.1109/TMI.2010.2046908},

 issn      = {},

 year      = {2010}

}
[14:01, 5/4/2026] PM: https://github.com/viswanath-lab/RadQy

@article{mrqy-2020,
  title     = {{Technical Note: MRQy — An open‐source tool for quality control of MR imaging data}},
  author    = {Sadri, Amir Reza and Janowczyk, Andrew and Ren, Zhou and Verma, Ruchika and Beig, Niha and Antunes, Jacob and Madabhushi, Anant and Tiwari, Pallavi and Viswanath, Satish E.},
  journal   = {Medical Physics},
  publisher = {wiley},
  volume    = {47},
  number    = {12},
  pages     = {6029--6038},
  note      = {},
  doi       = {10.1002/mp.14593 },
  issn      = {2473-4209},
  year      = {2020}
}

A verificação de outliers nas imagens foi realizada por meio do pacote MRQy \cite{mrqy-2020}, aplicado diretamente sobre as imagens em formato RAW para a extração automática de métricas quantitativas de qualidade de imagem (Image Quality Metrics -- IQMs), sem a necessidade de imagens de referência. O MRQy gerou uma planilha contendo métricas estatísticas básicas (MEAN, RNG, VAR, CV), métricas de contraste (CPP, CNR), múltiplas estimativas da relação sinal-ruído (PSNR, SNR1 a SNR9), medidas de não uniformidade de intensidade (CVP, CJV) e indicadores sensíveis a artefatos e borramento (EFC, FBER), além de informações geométricas do volume (VRX, VRY, VRZ, ROW, COL, NUM). Em seguida, cada métrica foi avaliada estatisticamente pelo método do intervalo interquartil (Interquartile Range -- IQR), no qual os limites inferior e superior foram definidos como $Q_1 - \alpha \cdot IQR$ e $Q_3 + \alpha \cdot IQR$, respectivamente, com fator $\alpha = 1.0$, considerando direções específicas de degradação (two-sided, low-bad e high-bad) de acordo com a natureza de cada métrica. Para cada imagem, foi computado um escore de outlier correspondente ao número de métricas que ultrapassaram os limites estabelecidos, sendo classificadas como suspeitas aquelas com escore maior ou igual a $3$. Esse procedimento possibilitou a identificação automática de volumes com potenciais artefatos, ruído excessivo, baixa relação sinal-ruído ou inconsistências de intensidade, constituindo um mecanismo objetivo de controle de qualidade antes das etapas subsequentes de pré-processamento e análise quantitativa


Documentação Técnica: Pipeline de Deformação Longitudinal (ANTsPy)

As deformações cerebrais observadas em exames de ressonância magnética (MRI) são o resultado da sobreposição de dois processos: o envelhecimento saudável (atrofia senescente) e a neurodegeneração patológica (demências). O objetivo deste pipeline é isolar as alterações estruturais inerentes exclusivamente à demência através da análise de campos de deformação (DF).

## 1. Análise e Fundamentação Científica

### 1.1. Estratificação por Referência Saudável (CN)

A utilização de atlas/referências estratificados por sexo e faixa etária baseados apenas em indivíduos cognitivamente normais (CN) é considerada o padrão-ouro.

- **Justificativa:** O campo de deformação captura a diferença entre o “envelhecimento esperado” para aquele perfil demográfico e a atrofia observada no paciente.
- **Referência:** Wyman et al. (2012), *Empowering Imaging Biomarkers of Alzheimer's Disease*, sobre o uso de modelos de envelhecimento normal como referência para sensibilidade ao desvio patológico.

### 1.2. Âncora CN e consistência no conjunto longitudinal

Para cada conjunto longitudinal (i1, i2, i3), utiliza-se **a mesma imagem atlas CN** (derivada do baseline i1: sexo e idade na primeira aquisição). Todas as imagens do conjunto são registadas com **fixed** = imagem clínica do tempo correspondente e **moving** = esse template CN.

- **Erro a evitar:** trocar o template CN entre tempos do mesmo conjunto invalidaria a comparação longitudinal no mesmo referencial de atlas.
- **Delta longitudinal (implementação atual):** em vez de subtrair vetorialmente dois inverse warps no atlas, o script `displacement_field.py` calcula a deformação relativa por **composição** no sentido ANTs: \(T_{1\to k} = \mathrm{inv}_k \circ \mathrm{fwd}_1\), isto é \(\phi_{\mathrm{Ref}\to k} \circ \phi_{\mathrm{Ref}\to 1}^{-1}\) com *fixed* = clínica e *moving* = atlas. O campo \(u(x) = T(x) - x\) é definido no **domínio da imagem MNI da baseline (i1)** (mesma grelha que o T1 pré-processado em `resampled_1.0mm`), o que alinha Jacobian e magnitude com o restante do pipeline CNN / features.

## 2. Estratégia de Processamento e Armazenamento

### 2.1. Otimização de I/O

As imagens são processadas de forma unitária (lista em `image_data.txt`).

- **Armazenamento de DF:** Os campos completos não são guardados em CSV (volume elevado). Guardam-se **NIfTI** com os warps por registo e, na fase longitudinal, mapas escalares de log-Jacobian e magnitude em `csvs/` (saída com prefixo `ID_PT_comb_...`).

- **Ficheiros por registo SyN** (em `images/displacement_field/`), para cada `ID_IMG` e etiqueta `CN_SEX-..._AGE-...`:

  - `*_0GenericAffine.mat`
  - `*_1Warp.nii.gz` (campo no domínio *moving* / atlas, cadeia *fwd*)
  - `*_1InverseWarp.nii.gz` (cadeia *inv*)

## 3. Radiomics (PyRadiomics) por ROI

O script `features_radiomic.py` extrai features radiômicas por ROI (labels) com PyRadiomics e salva em `csvs/features_radiomic.csv`.

### 3.1. Classes de features e lista de atributos

Listas obtidas via `Radiomics<Classe>.getFeatureNames()` (PyRadiomics). Referência das classes na documentação: [features.rst](https://github.com/AIM-Harvard/pyradiomics/blob/master/docs/features.rst).

- **firstorder (19)**: `10Percentile`, `90Percentile`, `Energy`, `Entropy`, `InterquartileRange`, `Kurtosis`, `Maximum`, `Mean`, `MeanAbsoluteDeviation`, `Median`, `Minimum`, `Range`, `RobustMeanAbsoluteDeviation`, `RootMeanSquared`, `Skewness`, `StandardDeviation`, `TotalEnergy`, `Uniformity`, `Variance`
- **shape (17)**: `Compactness1`, `Compactness2`, `Elongation`, `Flatness`, `LeastAxisLength`, `MajorAxisLength`, `Maximum2DDiameterColumn`, `Maximum2DDiameterRow`, `Maximum2DDiameterSlice`, `Maximum3DDiameter`, `MeshVolume`, `MinorAxisLength`, `SphericalDisproportion`, `Sphericity`, `SurfaceArea`, `SurfaceVolumeRatio`, `VoxelVolume`
- **glcm (28)**: `Autocorrelation`, `ClusterProminence`, `ClusterShade`, `ClusterTendency`, `Contrast`, `Correlation`, `DifferenceAverage`, `DifferenceEntropy`, `DifferenceVariance`, `Dissimilarity`, `Homogeneity1`, `Homogeneity2`, `Id`, `Idm`, `Idmn`, `Idn`, `Imc1`, `Imc2`, `InverseVariance`, `JointAverage`, `JointEnergy`, `JointEntropy`, `MCC`, `MaximumProbability`, `SumAverage`, `SumEntropy`, `SumSquares`, `SumVariance`
- **glrlm (16)**: `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelRunEmphasis`, `LongRunEmphasis`, `LongRunHighGrayLevelEmphasis`, `LongRunLowGrayLevelEmphasis`, `LowGrayLevelRunEmphasis`, `RunEntropy`, `RunLengthNonUniformity`, `RunLengthNonUniformityNormalized`, `RunPercentage`, `RunVariance`, `ShortRunEmphasis`, `ShortRunHighGrayLevelEmphasis`, `ShortRunLowGrayLevelEmphasis`
- **glszm (16)**: `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelZoneEmphasis`, `LargeAreaEmphasis`, `LargeAreaHighGrayLevelEmphasis`, `LargeAreaLowGrayLevelEmphasis`, `LowGrayLevelZoneEmphasis`, `SizeZoneNonUniformity`, `SizeZoneNonUniformityNormalized`, `SmallAreaEmphasis`, `SmallAreaHighGrayLevelEmphasis`, `SmallAreaLowGrayLevelEmphasis`, `ZoneEntropy`, `ZonePercentage`, `ZoneVariance`
- **ngtdm (5)**: `Busyness`, `Coarseness`, `Complexity`, `Contrast`, `Strength`
- **gldm (16)**: `DependenceEntropy`, `DependenceNonUniformity`, `DependenceNonUniformityNormalized`, `DependencePercentage`, `DependenceVariance`, `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelEmphasis`, `LargeDependenceEmphasis`, `LargeDependenceHighGrayLevelEmphasis`, `LargeDependenceLowGrayLevelEmphasis`, `LowGrayLevelEmphasis`, `SmallDependenceEmphasis`, `SmallDependenceHighGrayLevelEmphasis`, `SmallDependenceLowGrayLevelEmphasis`

## 3. Guia de Implementação (Pipeline)

### I. Espaço de referência e inputs

- **Input:** T1 processada por tempo (`ID_IMG`), em MNI (`*_stripped_nlm_denoised_biascorrected_mni_template.nii.gz` em `images/resampled_1.0mm/`).
- **Atlas CN:** template estratificado por sexo e década etária (baseline do paciente), em `images/groupwise/`.
- **Regra:** Todas as imagens do mesmo `COMBINATION_NUMBER` usam o **mesmo** template CN (âncora definida na primeira aquisição do paciente em `image_data.txt`).

### II. Geração do campo de deformação (registo SyN)

- **Afim:** escala, rotação, translação global.
- **SyN:** deformação não linear (voxel a voxel).
- **Saídas críticas:** ficheiros listados na secção 2.1 (ordem das listas *fwd* / *inv* igual à devolvida por `ants.registration`).

### III. Atributos longitudinais (`displacement_field.py`)

Para cada conjunto (i1, i2, i3), a fase 2 constrói **dois** deltas em relação ao tempo 1: \(\Delta_{1\to 2}\), \(\Delta_{1\to 3}\) e \(\Delta_{2\to 3}\).

1. **Campo de deslocamento relativo**  
   Composição ponto a ponto (via `apply_transforms_to_points`): primeiro `fwd` de i1 (Clin₁ → Ref), depois `inv` do tempo alvo (Ref → Clin_k). O campo \(u\) no domínio da **baseline clínica** é a diferença em coordenadas físicas (LPS) entre o ponto transformado e o original.

2. **Log-Jacobian**  
   Calculado com `create_jacobian_determinant_image` sobre esse campo vetorial no domínio da baseline.

3. **Magnitude**  
   Norma euclidiana de \(u\) por voxel.

4. **Máscara cerebral (recomendado)**  
   Por defeito, tenta-se `images/brain_mask/{ID_IMG_i1}_brain_mask.nii.gz` (p.ex. após `resample.py`). Alternativa: variável de ambiente `DISPLACEMENT_BRAIN_MASK` apontando para uma máscara binária em MNI (reamostrada para o domínio da baseline). Fora da máscara, log-Jacobian e magnitude são zerados. Se não existir máscara, o script avisa e grava sem mascaramento.

**Variáveis de ambiente opcionais**

| Variável | Efeito |
|----------|--------|
| `DISPLACEMENT_BRAIN_MASK` | Caminho para máscara cerebral quando não há ficheiro por `ID_IMG` em `brain_mask/`. |
| `DISPLACEMENT_POINT_CHUNK` | Tamanho do lote de pontos na composição (predefinido: 400000) para controlar memória. |

**Ficheiros de saída (exemplo):** `csvs/{ID_PT}_comb_{COMBINATION_NUMBER}_delta12_logjac.nii.gz`, `_delta13_logjac.nii.gz`, `_delta23_logjac.nii.gz`, `_delta12_mag.nii.gz`, `_delta13_mag.nii.gz`, `_delta23_mag.nii.gz`.

---

## Nota final

Ao concluir este passo, obtém-se, por conjunto de três imagens, mapas que descrevem a evolução estrutural entre a baseline e os tempos seguintes, **no espaço da imagem MNI do tempo i1**, com opcional restrição ao cérebro para reduzir ruído extra-craniano.

---

## 4. Modelagem (CNN / LSTM / PyCaret) + SHAP + Anti-vazamento por paciente

Esta secção documenta os scripts em `colab/` usados para classificação (pMCI vs sMCI) a partir do CSV:

- `csvs/abordagem_teste/all_delta_features_neurocombat.csv`

O CSV contém colunas de metadados (ex.: `ID_PT`, `GROUP`, `SEX`, `roi`, `label`, `pair`, ...) e colunas numéricas (features) que entram nos modelos.

---

## 4.0. Objetivo experimental e “abordagens”

Você vai executar dois blocos principais:

### Abordagem 1 — Detecção de outliers (one-class / clustering)

**Objetivo**: detectar **amostras anômalas** sem usar (ou usando minimamente) o rótulo `GROUP`. Há dois usos típicos e complementares:

- **QC / limpeza de dataset**: encontrar conjuntos/imagens que “fogem” do padrão por artefato, erro de segmentação, erro de ROI, merge errado, ou scanner/site. (Aqui o “outlier” é problema de qualidade, não necessariamente pMCI.)
- **Anomalia como sinal clínico** (semi-supervisionado): treinar o modelo **somente nos sMCI** como “normal” e usar o *anomaly score* para rankear quão “pMCI-like” é um conjunto. A avaliação vira uma *classificação via score* (AUC/PR-AUC) sem treinar um classificador discriminativo.

**O que entra como entrada** (pré-modelagem):

- **Entrada tabular wide**: 1 linha por conjunto `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)` com features “roi|side|pair|atributo” (ver `colab/datasets.py::build_wide_tabular_from_long_pairs`).
- **Entrada por sequência** (opcional): i1→i2→i3 com features radiômicas por imagem (ver `colab/datasets.py::build_triplet_sequence_from_radiomics`).

**Nota (grafos)**: nesta fase, **nenhum script usa grafos**. A geração/modelagem por grafos fica para uma etapa posterior, depois que o pipeline “sem grafos” estiver fechado (normalização por fold, balanceamento por paciente, SHAP e baseline de resultados).

**Modelos a inserir (convencionais)**:

- **IsolationForest**: robusto em alta dimensão, rápido, bom baseline.  
  - Por quê: não assume distribuição gaussiana e lida melhor com múltiplas fontes de variação.
- **One-Class SVM (RBF)**: forte em fronteira não-linear, mas sensível a escala.  
  - Por quê: boa alternativa quando a “normalidade” é um manifold.
- **EllipticEnvelope** (covariância robusta): útil quando as features ficam “quase gaussianas” após z-score e redução de dimensão.  
  - Por quê: baseline interpretável quando o comportamento é elipsoidal.
- **LocalOutlierFactor / KNN distance**: captura anomalia “local” (densidade).  
  - Por quê: útil se o dataset tem subgrupos normais (ex.: por idade/sexo).
- **PCA reconstruction error** (ou Autoencoder MLP como extensão deep):  
  - Por quê: detecta amostras que não podem ser reconstruídas bem por um subespaço “normal”.

**Clustering** (para exploração e pseudo-rótulos):

- **KMeans / MiniBatchKMeans** (com distância ao centróide como score de anomalia)
- **Gaussian Mixture** (score = -loglik)
- **DBSCAN/HDBSCAN** (quando houver clusters densos + ruído)

**Como avaliar**:

- Se for **QC**: inspeção manual das top-N amostras por score + checagem de metadados (scanner/site) para evitar que o “outlier” seja só batch.  
- Se for **anômalo vs normal (semi-supervisionado)**:  
  - treine só em `sMCI`, aplique em todos, use `GROUP` apenas para computar métricas (AUC, balanced accuracy no threshold escolhido, etc.).

### Abordagem 4 — Classificação binária (pMCI vs sMCI)

**Objetivo**: treinar modelos supervisionados com `GROUP` como ground truth.

**Regras**:

- **Split por paciente** (sempre): `StratifiedGroupKFold` com `groups=ID_PT`.
- **Estratificação**: `GROUP+SEX` como `strat_col` para reduzir viés.
- **Normalização sem vazamento**: ver a secção “Regras importantes” abaixo (varia por script).

---

## 4.1. Variáveis de tempo (t12/t13/t23) e TIME_PROG (sem vazamento)

### t12/t13/t23 (intervalo entre imagens do conjunto)

- **Uso recomendado**: como feature **por par** (isto é, `dt` dependente de `pair`) quando você está no modo por pares (12/13/23).  
  - Exemplo: em cada “passo” do LSTM por pares, adicionar a feature `dt`. (O `lstm_example.py` já faz isso quando `t12/t13/t23` existem.)
  - No tabular wide, você pode armazenar `roi|side|pair|dt` como mais uma coluna.
- **Por quê**: separa “mudança por mês” de “mudança total”. Sem isso, o modelo pode confundir deltas grandes com intervalos longos.

### TIME_PROG (tempo até progressão para demência)

Como você notou, **TIME_PROG é estruturalmente ligado ao rótulo** (em geral pMCI terá TIME_PROG>0 e sMCI=0), então:

- **Para classificação binária (abordagem 4)**: **NÃO usar TIME_PROG como feature**.  
  - Motivo: vazamento de target (o modelo aprende um atalho).
- **Usos válidos**:
  - **análise pós-hoc**: correlacionar score do modelo com TIME_PROG apenas nos pMCI (quanto menor TIME_PROG, mais “agressivo”).
  - **modelagem alternativa**: como alvo de regressão/survival (Cox/DeepSurv) ou classificação em múltiplos horizontes (ex.: converter em “progride em <=24 meses?”), mas isso vira outro experimento.

---

## 4.2. Padronização de entrada: “uma função por script” vs vários scripts

**Recomendação**: manter **um construtor de dataset** (funções utilitárias) e scripts separados por modelo.

- **Por quê**:
  - reduz duplicação e inconsistências (ex.: scaler, pivots, colunas ignoradas)
  - mantém cada experimento reprodutível e simples de rodar (um comando por modelo)
  - facilita log/outputs por abordagem (pasta por script/execução)

Implementação sugerida (já iniciada):

- `colab/datasets.py`
  - `build_wide_tabular_from_long_pairs`: gera wide+flatten para sklearn/MLP
  - `build_triplet_sequence_from_radiomics`: monta i1→i2→i3 a partir de `radiomics_merge.csv` + `cj_data_*.txt`

---

## 4.3. Deep tabular (MLP)

Script novo:

- `colab/mlp_example.py`

**Entrada**:

- `--input-mode wide_from_long`: faz pivot e usa 1 linha por conjunto (recomendado)
- `--input-mode as_is_rows`: usa o CSV como está (1 linha = 1 exemplo)

**O que testar (hiperparâmetros)**:

- `--kbest`: \{200, 400, 600, 1000, 2000\} (depende do nº total de features)
- `--hidden`: \{`256,64`, `512,128`, `1024,256`\}
- `--dropout`: \{0.0, 0.2, 0.3, 0.5\}
- `--lr`: \{1e-4, 3e-4, 1e-3\}
- `--batch-size`: \{32, 64, 128\}
- `--balance`: `none` vs `downsample`

---

## 4.4. Hiperparâmetros a testar por script (resumo prático)

### `colab/sklearn_models_teste.py`

Esse script hoje roda “várias famílias” com hiperparâmetros default. Para uma bateria de testes controlada, os knobs mais importantes são:

- **Pré-processamento/seleção**
  - `--remove-constant` (on/off)
  - `--corr-threshold`: \{0.0, 0.90, 0.95\}
  - `--feature-selection`: `none` / `kbest` / `two_stage`
  - `--fs-k-pre`: \{50, 100, 200, 400\}
  - `--fs-k-final`: \{20, 30, 50\}
- **Modelos (rodar um subconjunto)**
  - `--models`: começar com `"Logistic Regression,Extra Trees Classifier,SVM - RBF Kernel,Gradient Boosting Classifier"`

Motivo: você controla a capacidade efetiva via seleção de features (sem explodir o espaço de busca) e compara famílias com vieses diferentes (linear / kernel / árvores / boosting).

### `colab/cnn_example.py`

Hoje a CNN é “tabular conv1D” com `kbest` + z-score e early stopping.

Parâmetros a testar:

- `--kbest`: \{50, 100, 200, 400\}
- `--epochs`: \{50, 80, 120\}
- `--batch-size`: \{32, 64, 128\}
- `--balance`: `none` vs `downsample`

(Se você evoluir a arquitetura para tratar `pair` como canais e conv sobre `ROI`, aí os hiperparâmetros passam a incluir `filters`, `kernel_size`, `dropout`, `lr`.)

### `colab/lstm_example.py`

Parâmetros a testar:

- `--sequence-source`: `pairs` (deltas por par) vs `columns` (se você exportar *_base/_follow/_delta)
- `--pair-order`: fixo `12,13,23` (ou experimentar `12,23,13` só se houver hipótese forte de ordem)
- `--epochs`: \{40, 60, 80\}
- `--batch-size`: \{32, 64\}
- `--balance`: `none` vs `downsample`

Para a sua nova proposta (reconstrução i1→i2→i3 via radiomics_merge), o ideal é criar um modo extra no LSTM que consuma a sequência real (não implementado ainda neste script; está implementado no construtor em `colab/datasets.py`).

### 4.1. Regras importantes (sem vazamento)

- **Separação por paciente**: os scripts consideram `ID_PT` como grupo; um paciente não aparece em treino/validação/teste ao mesmo tempo.
- **Nested split**:
  - **split externo**: (folds) para avaliação
  - **split interno**: validação (early stopping / seleção) dentro do treino externo
- **Z-score sem vazamento** (por script):
  - `cnn_example.py`, `lstm_example.py`, `mlp_example.py`: scaler fitado **somente** no `fit` (treino interno) e aplicado em `val/test`.
  - `sklearn_models_teste.py`: o `StandardScaler` faz parte do `ColumnTransformer` e é fitado no **treino do fold externo** (antes do CV interno de ranking), e aplicado no teste externo.
  - `models_teste.py` (PyCaret): z-score manual nas features numéricas selecionadas, fitado no **treino do fold externo** e aplicado no teste externo.

### 4.2. Filtro por ROI/label (todas as abordagens)

Os filtros atuam em **linhas** do CSV (cada linha representa um exemplo associado a uma ROI/label/side/pair):

- `--roi "hippocampus,amygdala"`: filtra pela coluna `roi`
- `--label "17,53"`: filtra pela coluna `label`
- `--roi-label "hippocampus:17,hippocampus:53"`: define combinações explícitas `roi:label`

Se você **omitir** `--roi`, `--label` e `--roi-label`, o script usa **todas** as ROIs do CSV.

### 4.3. Balanceamento (2 modos)

Todos os scripts suportam a flag `--balance` (quando existe no script), mas **o ponto exato onde ela atua varia**:

- `--balance none`: sem balanceamento (default)
- `--balance downsample`: **downsampling por paciente (`ID_PT`)** balanceando simultaneamente os estratos `GROUP+SEX` (ex.: `pMCI_F`, `pMCI_M`, `sMCI_F`, `sMCI_M`).
  - `cnn_example.py`, `lstm_example.py`, `mlp_example.py`: aplica no `fit` (treino interno) e nunca altera `val/test`.
  - `sklearn_models_teste.py`: aplica no **treino do fold externo** (antes do preprocessing), e nunca altera o teste externo.
  - `models_teste.py` (PyCaret): aplica no **treino do fold externo**.

### 4.4. SHAP (importância de features e de ROIs)

Em `cnn_example.py` e `lstm_example.py`:

- `--shap`: gera
  - `shap_feature_importance_*.csv` (rank de atributos numéricos)
  - `shap_roi_importance_*.csv` (rank por `roi,label`)
  - plots: `shap_bar_*.png`, `shap_beeswarm_*.png`, `roi_bar_*.png`
- `--shap-outdir`: diretório de saída (default `colab/outputs/`)
- `--shap-samples` / `--shap-background`: amostragem para acelerar

Os plots mostram **nomes reais** das features.

---

## 5. Scripts em `colab/` e comandos

### 5.1. `colab/cnn_example.py` (CNN 1D tabular)

**O que faz**

- CNN 1D sobre features tabulares
- Split por paciente (`ID_PT`) sem vazamento
- Split interno (validação) por paciente
- Z-score fitado no `fit` e aplicado em `val`/`test`
- Seleção de atributos: `SelectKBest(f_classif, k=--kbest)` **fitado apenas no treino externo**
- (Opcional) SHAP + plots

**Comandos**

- Holdout (por paciente) em CPU:

```bash
python colab/cnn_example.py --device cpu --test-size 0.2
```

- Holdout + downsample:

```bash
python colab/cnn_example.py --device cpu --balance downsample --test-size 0.2
```

- Cross-validation externa (ex.: 5 folds) + validação interna:

```bash
python colab/cnn_example.py --device cpu --n-splits 5 --inner-fold 5 --kbest 100
```

- Rodar SHAP (todas as ROIs):

```bash
python colab/cnn_example.py --device cpu --shap
```

- Rodar SHAP filtrando ROI/label:

```bash
python colab/cnn_example.py --device cpu --shap --roi hippocampus --label 17,53
```

### 5.2. `colab/lstm_example.py` (LSTM com sequência por pares)

**O que faz**

- Monta sequência com 3 passos temporais por amostra (linhas `pair=12,13,23`)
- Split externo por paciente (`ID_PT`) com `StratifiedGroupKFold`
- Split interno por paciente para validação (`--inner-fold`)
- Z-score fitado no `fit` e aplicado em `val`/`test`
- (Opcional) SHAP + plots (por padrão roda SHAP no fold 1 para reduzir custo)

**Comandos**

- Cross-validation em CPU:

```bash
python colab/lstm_example.py --device cpu --n-splits 5 --inner-fold 5 --sequence-source pairs --pair-order 12,13,23
```

- Com downsample:

```bash
python colab/lstm_example.py --device cpu --n-splits 5 --inner-fold 5 --balance downsample --sequence-source pairs
```

- SHAP + plots:

```bash
python colab/lstm_example.py --device cpu --n-splits 5 --inner-fold 5 --shap --sequence-source pairs
```

### 5.3. `colab/models_teste.py` (PyCaret + nested CV externo)

**O que faz**

- Loop externo: `StratifiedGroupKFold(n_splits=--n-splits)` por paciente (`ID_PT`)
- Dentro de cada fold externo:
  - Seleção opcional de features numéricas (kbest/sfs/two_stage) **fitada só no treino**
  - Z-score manual nas features numéricas selecionadas (fit no treino externo, aplica no teste externo)
  - CV interno do PyCaret **group-aware** (splits gerados com `StratifiedGroupKFold` no treino externo)
  - Balanceamento opcional por downsample (por paciente) no treino externo
- SHAP:
  - `--shap`: baseline SHAP (LogReg numérico) para ranking rápido
  - `--shap-pycaret-topk K`: SHAP + plots para os top-K modelos do PyCaret em cada fold (usa features já pré-processadas pelo PyCaret)

**Comandos**

- Rodar PyCaret com 10 folds externos (default):

```bash
python colab/models_teste.py
```

- Sem balanceamento (explícito):

```bash
python colab/models_teste.py --balance none
```

- Downsample por `GROUP+SEX` (por paciente):

```bash
python colab/models_teste.py --balance downsample
```

- Seleção de features em 2 estágios (recomendado quando há muitas features):

```bash
python colab/models_teste.py --feature-selection two_stage --fs-k-pre 200 --fs-k-final 30
```

- SHAP baseline (rápido):

```bash
python colab/models_teste.py --shap
```

- SHAP dos top-2 modelos PyCaret por fold (gera CSV + plots por fold/modelo):

```bash
python colab/models_teste.py --shap-pycaret-topk 2 --shap-pycaret-samples 150 --shap-pycaret-background 120
```

## TODO

O script `displacement_field.py` produz os atributos escalares de DF (log-Jacobian e magnitude). Existem ainda `csvs/features_volumetric.csv` e `csvs/features_radiomic.csv` por ROI.

1. Como calcular também as diferenças volumétricas e radiómicas entre as imagens i1 e i2, e i1 e i3?
2. Como calcular ou extrair o DF por ROI?


TODO- 

Aqui vai um enquadramento direto para fechares a etapa “nível imagem” e alinhares com o grafo mais tarde.

1. O que features_displacement.py deve fazer (conceito)
O displacement_field.py já deixa quatro volumes por conjunto (por ID_PT + COMBINATION_NUMBER), todos no espaço da baseline (i1):

delta12_logjac, delta13_logjac, delta12_mag, delta13_mag
O novo script deve:

Ler o mesmo ficheiro de conjuntos que o pipeline longitudinal (ex.: cj_data_abordagem_4.txt) para saber (ID_PT, COMBINATION_NUMBER) e os três ID_IMG ordenados no tempo.
Localizar os .nii.gz em csvs/ (ou pasta que definires) com o padrão de nomes que o displacement_field.py grava.
Para cada conjunto, carregar esses volumes e uma máscara de ROI alinhada à baseline i1 (ex.: images/regions/{ID_IMG_i1}_regions.nii.gz ou o mesmo critério do features_radiomic.py, com ants.resample_image_to_target se necessário).
Por ROI (e opcionalmente __global__ no cérebro), calcular estatísticas escalares: p.ex. média, mediana, desvio-padrão, percentis do logjac e da magnitude para Δ₁→₂ e Δ₁→₃.
Escrever um CSV — formato long compatível com o resto do projeto (ID_PT, COMBINATION_NUMBER, roi, side, label, delta_pair ∈ {12,13}, metric ∈ {logjac,mag}, stat, value) ou wide com colunas explícitas; o importante é uma linha lógica por conjunto (ou por conjunto×ROI) para o modelo.
Assim, cada conjunto de 3 imagens fica descrito pelos campos de deslocamento resumidos (não pelo vetor 3D bruto), que é o que entra no modelo tabular.

2. Representar “conjuntos de 3 imagens” de forma consistente
Hoje o volumétrico e o radiómico estão em formato long por ID_IMG (como em features_volumetric.csv: uma série de linhas por imagem). Isso serve para análise por tempo; para um modelo que consome sequências de 3, precisas de uma camada de agregação:

Abordagem	Ideia	Quando usar
Chave de conjunto
Todas as linhas/features passam a ter ID_PT + COMBINATION_NUMBER (e opcionalmente ID_IMG_i1, i2, i3).
Essencial para juntar DF, volume e radiómica no mesmo exemplo.
Formato largo (wide)
Uma linha por (ID_PT, COMBINATION_NUMBER); colunas feat_roi_t1, feat_roi_t2, feat_roi_t3 ou Δ_t2-t1, Δ_t3-t1.
Muito comum para MLP/tabular e para comparar com grafos.
Manter long + índice temporal
Coluna time_index ∈ {1,2,3} por linha; o modelo ou o DataLoader agrupa por conjunto.
Útil se quiseres redes que tratam explicitamente a sequência.
Para deslocamento, a informação já é intra-conjunto (só tens Δ₁→₂ e Δ₁→₃; não há “DF por imagem” isolada no mesmo sentido que volume por tempo). Para volume/radiómica, podes ou empilhar três medições por ROI ou usar diferenças relativas à baseline, alinhado ao que fazes com o DF.

3. Alinhamento com a fase de grafos
Se no grafo cada nó for tempo/ROI e as arestas ligarem tempos ou regiões, no nível imagem o análogo é uma matriz de features por conjunto (ou por nó, com COMBINATION_NUMBER fixo).
Definir já agora a chave canónica: (ID_PT, COMBINATION_NUMBER) evita ambiguidade porque COMBINATION_NUMBER repete entre pacientes (como no displacement_field.py).
4. O que falta explicitamente no repositório
Um script tipo features_displacement.py (o que propuseste) que faça o passo 1–5.
Um passo opcional mas útil: script ou notebook que leia features_volumetric.csv / features_radiomic.csv e exporte features_*_set.csv com uma linha por (ID_PT, COMBINATION_NUMBER) (ou junte os três tempos em colunas), para o mesmo esquema de chaves que o CSV de DF.
Estou em modo Ask: não posso criar ficheiros. Se quiseres que implemente features_displacement.py e o esquema de CSV no repositório, ativa Agent mode e indica preferência (long vs wide e lista exata de estatísticas por ROI).

Sim: se a unidade de análise é progressão relativamente à baseline, faz sentido que volume e radiómica sigam a mesma lógica que o DF (Δ₁→₂ e Δ₁→₃), em vez de empilhar três “estados absolutos” sem hierarquia.

Como “fechar” volumétrico e radiómico
1. Mesma chave em todo o pipeline
Para cada (ID_PT, COMBINATION_NUMBER) com imagens ordenadas i1, i2, i3 (baseline = i1):

Para cada ROI (e cada feature radiómica já definida no teu CSV longo):
Δ₁₂ = valor em i2 − valor em i1 (ou razão i2/i1, ou log-ratio, conforme a escala da feature).
Δ₁₃ = valor em i3 − valor em i1 (idem).
Assim cada conjunto fica descrito por dois números por feature/ROI, como no DF (dois mapas de delta). O modelo “vê” sempre mudança face à baseline, não três instantâneos independentes.

2. Escolhas que tens de fixar por tipo de atributo

Tipo	Delta em diferença simples	Notas
Volume (mm³, normalizado)
vol_k − vol_1
Natural; para “taxa” podes dividir pelo intervalo de tempo se tiveres TIME_PROG.
Radiómica (first-order, textura)
feat_k − feat_1 ou (feat_k − feat_1) / |feat_1|
Texturas têm escala arbitrária: diferença ou delta relativo evita explosão de escala; para contagens/intensidades positivas, log antes do delta pode estabilizar.
3. O que não obrigas mas podes guardar

Opcionalmente valor absoluto em i1 como covariável separada (“nível inicial”), se o teu desenho experimental precisar de separar nível de mudança. Isso não contradiz “representar o conjunto por deltas”; só enriquece.
4. Formato de dados

Wide: uma linha por (ID_PT, COMBINATION_NUMBER), colunas ..._d12, ..._d13 por feature/ROI — fácil de juntar com DF e com o grafo.
Long: manter delta_pair ∈ {12, 13} como no DF, com as mesmas chaves — consistente com o que já pensaste para features_displacement.
5. Implementação lógica (sem código aqui)

Ler features_volumetric.csv / features_radiomic.csv filtrado aos três ID_IMG do conjunto (via cj_data_abordagem_4.txt).
Para cada chave (feature, ROI, …) presente nas três linhas (ou por i1+i2+i3), calcular Δ₁₂ e Δ₁₃.
Tratar missing (ROI ausente num tempo): NaN ou exclusão, com regra explícita.
Com isto ficas com uma definição única: conjunto = par de deltas à baseline em DF, volume e radiómica — e podes fechar a etapa “nível imagem” e alinhar depois o grafo (nós/arestas que também codifiquem Δ em relação a i1 ou trajetória i1→i2→i3 no mesmo espírito).

Se quiseres que isso fique implementado em scripts/CSV no repositório, ativa Agent mode e indica se preferes formato long ou wide e se usas diferença simples ou relativa para radiómica.
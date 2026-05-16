Last updated: 04/05/2006 - By Paulo Girardi

**Nota (modelagem):** o desenho inicial abaixo (itens 1–5) foi o ponto de partida. O pipeline **implementado** está na [secção 4](#4-modelagem-supervisionada-exp1--exp2--pipeline-implementado) e em [`exp1.md`](exp1.md) / [`exp2.md`](exp2.md) — tensor `(n, 60, atributos)` via `load_tensor`, sem CSV wide intermédio; modelos XGBoost, ROCKET, SVM, LSTM; CV aninhada por `ID_PT`.

---

Desenho inicial (referência histórica) — CSV de exemplo: `csvs/abordagem_teste/all_delta_features_neurocombat.csv`:

1. Cada 3 linhas = um conjunto (20 ROIs × 3 pares). Ideia original: flatten para uma linha por conjunto e gravar CSV wide.

2. Balanceamento em nível de paciente (`GROUP` × `SEX`) com downsampling.

3. Seleção de atributos: correlação, variância, opcionalmente SelectKBest / SFS.

4. Split treino/val/teste sem leakage por `ID_PT`; z-score só no treino.

5. Classificação supervisionada sMCI vs pMCI. 

##############################################

Descrição Pré-processamento

[14:00, 5/4/2026] PM: Todos os volumes de RM estrutural ponderados em T1 foram primeiramente submetidos à extração de crânio (skull-stripping) utilizando a função brain_extraction() do pacote ANTsPyNet, dentro do ecossistema ANTsX \cite{tustison-2021}. As entradas foram carregadas via ants.image_read() e as imagens 4D foram convertidas em 3D através da extração do primeiro volume com ants.slice_image(axis=3, idx=0). A extração cerebral foi realizada com modality="t1" e o diretório de cache do ANTsXNet fixado em /workspace/cache para garantir a reprodutibilidade, resultando em volumes sem crânio e máscaras cerebrais binárias.

Posteriormente, aplicou-se a redução de ruído utilizando o filtro de médias não-locais adaptativo (adaptive non-local means filter) implementado em ants.denoise_image() \cite{manjon-2010}, com os parâmetros: mask=None, shrink_factor=1, p=1, r=2, noise_model="Rician" e v=0. As inomogeneidades do campo de viés (bias field) foram corrigidas utilizando o algoritmo N4 implementado em ants.n4_bias_field_correction() \cite{tustison-2010}, empregando a configuração padrão do ANTsPy (mask=None, shrink_factor=3, rescale_intensities=True, spline_param=200 e convergência {'iters': [50, 50, 50, 50], 'tol': 1e-7}).

Parâmetros de pré-processamento (pipeline de testes)
====================================================
Documentação dos valores escolhidos após experimentação no notebook
preproc.ipynb. Pastas de saída em ./testes/.

Fluxo: skull stripping → denoise → bias field → histogram matching (MNI)


1. Skull stripping (ANTsXNet brain_extraction)
--------------------------------------------

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
 issn          = {2045-2322},
 year          = {2021}
 }

Ferramenta : antspynet.brain_extraction
Modalidade : t1  (recomendado; outras modalidades são mais lentas e o
                  resultado é semelhante)

Parâmetros principais
  modality = "t1"
  verbose  = True

Saídas (em ./testes/skullstrip/)
  {ID}_brain_mask.nii.gz
  {ID}_stripped.nii.gz

Notas
  - Modalidades alternativas no ANTsXNet: t1v0, t1nobrainer, t1combined,
    flair, t2, bold, fa, etc. Não usadas aqui por custo/tempo vs. ganho.
  - Cache dos pesos: ~/antspynet_cache (configurável no notebook).


2. Denoise (ANTs — Non-Local Means adaptativo)
----------------------------------------------

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
 issn          = {1522-2586},
 year          = {2010}
}

Ferramenta : ants.denoise_image
Referência : Manjón et al., JMRI 2010 (NLM adaptativo)

Parâmetros escolhidos
  shrink_factor = 1    # fator de downsample interno (s)
  p             = 2    # raio do patch local (ver abaixo)
  r             = 3    # raio de busca (radius; ver abaixo)
  noise_model   = "Rician"
  mask          = None

O que significa "p"?
  No ANTs, p é o raio do patch local usado para estimar o ruído e aplicar
  o filtro NLM (não é "potência" nem número de bins). Valores maiores
  aumentam o tamanho da vizinhança local e tendem a suavizar mais a imagem.

O que significa "r" (radius no notebook)?
  r é o raio de busca: distância até onde o algoritmo procura patches
  semelhantes para a média não-local. Valores maiores denoisam mais, com
  maior custo computacional.

Sufixo no nome do ficheiro : sf_{shrink_factor}_p_{p}_r_{r}
Exemplo de saída : I100004_stripped_sf_1_p_2_r_3.nii.gz

Observações da experimentação
  - shrink_factor > 1 remove o ruído de forma insatisfatória.
  - r > 3 e/ou p > 2 suaviza demais a imagem (perda de detalhe).
  - r < 3 e/ou p < 2 remove pouco ruído.


3. Bias field correction (N4 / ANTs)
------------------------------------

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

Ferramenta : ants.n4_bias_field_correction

Parâmetros escolhidos
  shrink_factor     = 3
  convergence       = {"iters": [50, 50, 50, 50], "tol": 1e-7}
  return_bias_field = False

Sufixo no nome do ficheiro : sf_{shrink_factor}
Exemplo de saída : I100004_stripped_sf_1_p_2_r_3_sf_3.nii.gz
  (quando encadeado após denoise; no notebook de testes o bias lê
   *_stripped.nii.gz e grava {base}_sf_3.nii.gz)


4. Histogram matching (ANTs → template MNI)
-----------------------------------------

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
 issn          = {2045-2322},
 year          = {2021}
 }

Ferramenta : ants.histogram_match_image2
Referência : atlases/templates/mni152_2009c_template.nii.gz

Parâmetros escolhidos
  bins   = 128   # transform_domain_size no código (b no nome do ficheiro)
  points = 16    # match_points no código (p no nome do ficheiro)

Outros (fixos no notebook)
  match_points e transform_domain_size passados a histogram_match_image2
  Normalização min-max da fonte e referência; reescala pelo p99.9 da MNI

Sufixo no nome do ficheiro : _b_{bins}_p_{points}
Exemplo de saída : I100004_stripped_b_128_p_16.nii.gz

Mapeamento nome ↔ parâmetro ANTs
  b (bins)   → transform_domain_size
  p (points) → match_points


Resumo rápido
-------------
  Etapa              | Parâmetro-chave              | Valor
  -------------------|------------------------------|-------
  Skull stripping    | modality                     | t1
  Denoise            | shrink_factor / p / r        | 1 / 2 / 3
  Bias field (N4)    | shrink_factor                | 3
  Histogram matching | bins / points (MNI)          | 128 / 16

5. Detecção de outliers

https://github.com/viswanath-lab/RadQy

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

A verificação de outliers nas imagens foi realizada por meio do pacote MRQy \cite{mrqy-2020}, aplicado diretamente sobre as imagens em formato RAW para a extração automática de métricas quantitativas de qualidade de imagem (Image Quality Metrics -- IQMs), sem a necessidade de imagens de referência. O MRQy gerou uma planilha contendo métricas estatísticas básicas (MEAN, RNG, VAR, CV), métricas de contraste (CPP, CNR), múltiplas estimativas da relação sinal-ruído (PSNR, SNR1 a SNR9), medidas de não uniformidade de intensidade (CVP, CJV) e indicadores sensíveis a artefatos e borramento (EFC, FBER), além de informações geométricas do volume (VRX, VRY, VRZ, ROW, COL, NUM). Em seguida, cada métrica foi avaliada estatisticamente pelo método do intervalo interquartil (Interquartile Range -- IQR), no qual os limites inferior e superior foram definidos como $Q_1 - \alpha \cdot IQR$ e $Q_3 + \alpha \cdot IQR$, respectivamente, com fator $\alpha = 1.0$, considerando direções específicas de degradação (two-sided, low-bad e high-bad) de acordo com a natureza de cada métrica. Para cada imagem, foi computado um escore de outlier correspondente ao número de métricas que ultrapassaram os limites estabelecidos, sendo classificadas como suspeitas aquelas com escore maior ou igual a $3$. Esse procedimento possibilitou a identificação automática de volumes com potenciais artefatos, ruído excessivo, baixa relação sinal-ruído ou inconsistências de intensidade, constituindo um mecanismo objetivo de controle de qualidade antes das etapas subsequentes de pré-processamento e análise quantitativa.

6. Parcelamento das regiões do cérebro

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
 issn          = {2045-2322},
 year          = {2021}
 }

 função desikan_killiany_tourville_labeling

7. Segmentação dos tecidos do cérebro

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
 issn          = {2045-2322},
 year          = {2021}
 }

 função deep_atropos

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

==============================
= Pipeline de prodcessamento =
==============================

Harmonização neuro combat

@article{beer-2020,
  title     = {{Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data}},
  author    = {Beer, Joanne C. and Tustison, Nicholas J. and Cook, Philip A. and Davatzikos, Christos and Sheline, Yvette I. and Shinohara, Russell T. and Linn, Kristin A.},
  journal   = {NeuroImage},
  publisher = {},
  volume    = {220},
  number    = {},
  pages     = {117129},
  note      = {},
  doi       = {10.1016/j.neuroimage.2020.117129},
  issn      = {},
  year      = {2020}
}

================================

Shap analisys

@inproceedings{lundberg-2017,
  title     = {{A Unified Approach to Interpreting Model Predictions}},
  author    = {Lundberg, Scott M and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems},
  publisher = {Curran Associates, Inc.},
  volume    = {30},
  number    = {},
  address   = {},
  pages     = {1--10},
  note      = {},
  doi       = {10.48550/arXiv.1705.07874},
  issn      = {},
  isbn      = {},
  year      = {2017}
}

===================================

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

## 4. Modelagem supervisionada (exp1 / exp2) — pipeline implementado

Esta secção descreve o pipeline **efetivamente implementado** em `colab/` para classificação binária **sMCI vs pMCI** (`GROUP`). Detalhes por experimento: [`exp1.md`](exp1.md) (deltas) e [`exp2.md`](exp2.md) (absolutos + taxa desde baseline).

**Relação com o desenho inicial (itens 1–5 no topo deste README):** não é necessário gerar um CSV “wide” (uma linha por conjunto com flatten manual). O tensor `(n, 60, n_atributos)` é montado em memória por `colab/exp1_utils.py::load_tensor`. A seleção de atributos (correlação + variância) e o z-score são feitas **por fold**, sem vazamento. O balanceamento é opcional via flag `DOWNSAMPLE_GROUP_SEX` (downsample por **paciente** nos estratos `GROUP×SEX`).

### 4.0. Visão geral dos dois experimentos

| | **Experimento 1 (exp1)** | **Experimento 2 (exp2)** |
|---|--------------------------|---------------------------|
| **CSV** | `csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv` | `csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv` |
| **Conteúdo** | Deltas entre imagens (pares `12`, `13`, `23`) | Absolutos na img1; img2/3 como taxa desde baseline |
| **Ponderação temporal** | `delta_rate`: \(x' = x / \max(dt, \varepsilon)\) por par | `baseline_rate`: img1 absoluto; img2/3 \((x-x_{\mathrm{baseline}})/\max(t_{12\|13},\varepsilon)\) |
| **`PAIR_ORDER`** | `["12", "13", "23"]` | `["1", "2", "3"]` |
| **Scripts** | `exp1_xgboost.py`, `exp1_rocket.py`, `exp1_svm.py`, `exp1_lstm.py` | `exp2_xgboost.py`, `exp2_rocket.py`, `exp2_svm.py`, `exp2_lstm.py` |
| **Regenerar figuras** | `colab/exp1_plots.py` | `colab/exp2_plots.py` |

Em ambos: cada amostra = `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)` com tensor **`X_3d` de forma `(n, 60, n_feat)`** — 60 linhas = **20 ROIs × 3 passos temporais**; alvo `y`: sMCI=0, pMCI=1; split por **`ID_PT`**.

### 4.0.1. Utilitários partilhados

- **`colab/exp1_utils.py`:** `load_tensor`, `downsample_train_indices`, `inner_cv_splits`, `inner_train_val`, filtros correlação/variância, plots, CSVs de métricas OOF, agregação SHAP/coef por ROI e atributo (`slot_labels`: `pair|roi|side|label`).
- **`colab/exp_lstm_common.py`:** pipeline LSTM (Optuna, Keras, SHAP Kernel) reutilizado por `exp1_lstm.py` e `exp2_lstm.py`.

### 4.0.2. Cenários balanced / unbalanced

Em cada script de treino, a constante **`DOWNSAMPLE_GROUP_SEX`** controla o cenário:

| Valor | Pasta de saída | Comportamento |
|-------|----------------|---------------|
| `True` | `colab/exp{1,2}/balanced/<modelo>/` | Antes do split interno, equipara o nº de **pacientes** por estrato `GROUP×SEX` no treino externo de cada fold |
| `False` | `colab/exp{1,2}/unbalanced/<modelo>/` | Usa todos os pacientes disponíveis no treino externo |

O conjunto de **teste externo (~20 %)** mantém a distribuição natural; o downsample **não** altera o teste.

Para comparar as duas estratégias: correr cada script duas vezes (ou duplicar a flag) — **4 modelos × 2 cenários × 2 experimentos = até 16 runs** completos.

### 4.0.3. Validação cruzada (sem vazamento por paciente)

**Não há** `train_test_split` fixo 70/15/15. Métricas reportadas = **OOF** em 5 folds externos.

| Nível | Método | Folds | Papel |
|-------|--------|-------|--------|
| **Externo** | `StratifiedGroupKFold` (`groups=ID_PT`) | 5 | Teste OOF (~20 % por fold) |
| **Interno (Optuna)** | `StratifiedGroupKFold` no treino externo | 5 (`INNER_NCV_SPLITS`) | Média da AUC → hiperparâmetros |
| **Holdout interno** | `inner_train_val()` | ~64 % `tr_fit` / ~16 % `val` do total | Early stopping (XGBoost/LSTM), curvas no fold 1 |

Constantes comuns: `random_state=42`, `CORR_THR=0.9`, `VAR_THR=0.0`, `DT_EPSILON=0.5`.

**Pré-processamento por fold externo** (fit só em `tr_fit`, salvo nota):

1. Ponderação temporal na carga (`load_tensor`).
2. Correlação entre colunas > 0,9 no flatten de `X_3d[tr_fit]`.
3. `VarianceThreshold(0.0)` (remove colunas constantes no treino).
4. `StandardScaler` (z-score).

No **NCV interno do Optuna** (XGBoost e LSTM), correlação + variância + scaler são **recalculados em cada split interno** (sem vazamento).

### 4.0.4. Modelos e saídas

| Modelo | Script(s) | Entrada | Optuna (trials) | Importância |
|--------|-----------|---------|-----------------|-------------|
| **XGBoost** | `exp*_xgboost.py` | Tabular achatado `(n, 60·p)` | 30 | SHAP (`TreeExplainer`) → ROI / atributo |
| **ROCKET** | `exp*_rocket.py` | `(n, p, 60)` + L1 logística | 30 (`C`) | \|coef.\| L1 (sem SHAP no espaço kernel) |
| **SVM linear** | `exp*_svm.py` | Tabular achatado | 30 (`C`) | \|coef.\| agregado por ROI / atributo |
| **LSTM** | `exp*_lstm.py` | `(n, 3, 20·p)` após `panels_to_seq` | 20 | SHAP Kernel no vetor `60·p` achatado |

**Artefactos por run** (`colab/exp{1,2}/{balanced|unbalanced}/{xgboost|rocket|svm|lstm}/`):

- `tables/`: `metrics_per_fold.csv`, `oof_predictions.csv`, `fold_test_scores.csv`, `importance_shap_*` ou `importance_coef_*`, `feature_counts_fold0.csv`, `run_meta.json` (+ `training_curves_fold0.csv` no LSTM).
- `figures/`: confusão OOF, ROC/PR, boxplot de métricas, contagens de atributos, barras SHAP/coef; XGBoost/LSTM: curvas de treino no fold 1.

**LSTM e GPU:** definir **antes** de importar TensorFlow (já nos scripts `exp*_lstm.py`):

```python
os.environ.setdefault("LSTM_DEVICE", "gpu")      # ou "cpu"
os.environ.setdefault("LSTM_GPU_INDEX", "0")   # "1" para a segunda RTX 4090
```

Treino com `use_cudnn=False` e XLA auto-jit desligado (`TF_XLA_FLAGS=--tf_xla_auto_jit=0`). Correr exp1 e exp2 em paralelo: um processo por GPU (`LSTM_GPU_INDEX=0` vs `1`).

### 4.0.5. Execução (raiz do repositório)

```bash
# Experimento 1 — deltas
python colab/exp1_xgboost.py
python colab/exp1_rocket.py
python colab/exp1_svm.py
python colab/exp1_lstm.py

# Experimento 2 — absolutos + baseline_rate
python colab/exp2_xgboost.py
python colab/exp2_rocket.py
python colab/exp2_svm.py
python colab/exp2_lstm.py

# Regenerar PDFs a partir dos CSV (editar RUN_DIR em CONFIG)
python colab/exp1_plots.py
python colab/exp2_plots.py
```

**Dependências:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `xgboost`, `shap`, `sktime` (ROCKET), `tensorflow` (LSTM).

Alterar `DOWNSAMPLE_GROUP_SEX` no topo de cada script antes de correr (balanced vs unbalanced).

---

## 4.1. Variáveis de tempo e vazamento de rótulo

### t12 / t13 / t23

Integrados na ponderação temporal (`delta_rate` no exp1; `baseline_rate` no exp2). Ver fórmulas em `exp1.md` e `exp2.md`.

### TIME_PROG

**Não usar como feature** na classificação binária sMCI/pMCI (vazamento de target). Usos válidos: análise pós-hoc ou modelos de survival (experimento futuro).

---

## 4.2. Outros objetivos e scripts exploratórios (legado)

### Detecção de outliers (futuro / paralelo)

Planeado para etapa posterior (QC ou one-class em sMCI): IsolationForest, One-Class SVM, LOF, etc. Entrada wide/sequência ainda em `colab/datasets.py`. **Não** faz parte dos runs `exp*_*.py`.

**Nota (grafos):** modelagem por grafos fica para depois do baseline tabular/sequencial (exp1/exp2) estar fechado.

### Scripts exploratórios anteriores (CLI, CSV `abordagem_teste`)

Protótipos com interface `--balance`, `--shap`, `--roi`, etc. O pipeline **principal** de produção é `exp1_*` / `exp2_*` acima.

| Script | Papel |
|--------|--------|
| `colab/cnn_example.py` | CNN 1D tabular + `SelectKBest` |
| `colab/lstm_example.py` | LSTM antigo (substituído por `exp*_lstm.py` + `exp_lstm_common.py`) |
| `colab/mlp_example.py` | MLP tabular |
| `colab/sklearn_models_teste.py` | Várias famílias sklearn |
| `colab/models_teste.py` | PyCaret + nested CV |
| `colab/datasets.py` | Construtores wide / sequência i1→i2→i3 |

Exemplo (legado):

```bash
python colab/cnn_example.py --device cpu --n-splits 5 --inner-fold 5 --kbest 100
python colab/lstm_example.py --device cpu --n-splits 5 --shap --sequence-source pairs
```

---

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
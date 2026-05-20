# Pipeline de neuroimagem e modelagem supervisionada (sMCI vs pMCI)

**Última atualização:** maio/2026 — Paulo Girardi

Este documento descreve o fluxo **do volume T1 ao classificador**, com ênfase no que está **implementado em código**. Detalhes por experimento: [`exp1.md`](exp1.md) (deltas) e [`exp2.md`](exp2.md) (absolutos + taxa desde baseline). Comandos rápidos: [`colab/readme.txt`](colab/readme.txt).

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Pré-processamento de imagens](#2-pré-processamento-de-imagens)
3. [Extração de atributos e harmonização](#3-extração-de-atributos-e-harmonização)
4. [Modelagem supervisionada (exp1 / exp2)](#4-modelagem-supervisionada-exp1--exp2--pipeline-implementado)
5. [Trabalho futuro e legado](#5-trabalho-futuro-e-legado)

---

## 1. Visão geral

```text
RM T1  →  ANTs (skull, denoise, N4, MNI)  →  QC (MRQy)  →  parcelação/segmentação
      →  registo longitudinal + DF  →  radiomics/volume  →  NeuroComBat (§3.1)  →  CSV de features
      →  load_tensor (exp1 ou exp2)  →  CV por ID_PT  →  XGB / SVM / ROCKET / LSTM  →  OOF + artefactos
```

| Fase | Onde está documentado / implementado |
|------|--------------------------------------|
| Imagem → CSV | Secções 2–3 deste README; scripts em `images/`, `csvs/`, notebooks |
| CSV → modelo | Secção 4; `colab/exp{1,2}_*.py`, `colab/exp1_utils.py` |
| **Segmento ativo** | **Experimento 2** (`exp2.md`): melhorias (AP, checkpoints, demografia, ablação ROI) |
| Referência histórica | Experimento 1 (`exp1.md`): deltas; não re-treinado no ciclo atual |

**Chave canónica de amostra:** `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)` — três aquisições por conjunto; split de validação sempre por **`ID_PT`** (paciente).

---

## 2. Pré-processamento de imagens

Volumes de RM estrutural T1: extração de crânio (ANTsPyNet / ANTsX), denoise NLM, correção N4, histogram matching para MNI. Documentação dos valores em `preproc.ipynb`; saídas de teste em `./testes/`; produção em `images/`.

**Fluxo:** skull stripping → denoise → bias field → histogram matching (MNI)

### 2.1. Skull stripping (ANTsXNet `brain_extraction`)

```bibtex
@article{tustison-2021,
  title         = {{The ANTsX ecosystem for quantitative biological and medical imaging}},
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
  journal       = {Scientific Reports},
  publisher     = {},
  volume        = {11},
  number        = {9068},
  pages         = {1--13},
  note          = {},
  doi           = {10.1038/s41598-021-87564-6},
  issn          = {},
  year          = {2021}
  }
```

- **Ferramenta:** `antspynet.brain_extraction` dentro do ecossistema ANTsX.
- **Entrada:** `ants.image_read()`; volumes 4D → 3D com `ants.slice_image(axis=3, idx=0)`.
- **Parâmetros:** `modality = "t1"` (recomendado; outras modalidades são mais lentas com ganho semelhante), `verbose = True`.
- **Cache:** diretório ANTsXNet (ex.: `/workspace/cache` ou `~/antspynet_cache`).
- **Saídas** (`./testes/skullstrip/`): `{ID}_brain_mask.nii.gz`, `{ID}_stripped.nii.gz`.
- **Notas:** modalidades alternativas no ANTsXNet (t1v0, t1nobrainer, flair, t2, …) não usadas aqui por custo/tempo.

### 2.2. Denoise (ANTs — Non-Local Means adaptativo)

```bibtex
@article{manjon-2010,
  title         = {{Adaptive non-local means denoising of MR images with spatially varying noise levels}},
  author        = {Manjón, José V. and Coupé, Pierrick and Martí-Bonmatí, Luis and Collins, D. Louis and Robles, Montserrat},
  journal       = {Journal of Magnetic Resonance imaging},
  publisher     = {},
  volume        = {31},
  number        = {1},
  pages         = {192--203},
  note          = {},
  doi           = {10.1002/jmri.22003},
  issn          = {},
  year          = {2010}
}
```

- **Ferramenta:** `ants.denoise_image` (Manjón et al., JMRI 2010).
- **Parâmetros escolhidos:**

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `shrink_factor` | `1` | Downsample interno do filtro |
| `p` | `2` | Raio do patch local NLM |
| `r` | `3` | Raio de busca de patches semelhantes |
| `noise_model` | `"Rician"` | Ruído RM |
| `mask` | `None` | Sem máscara explícita |

- **Sufixo no ficheiro:** `sf_{shrink_factor}_p_{p}_r_{r}` — ex.: `I100004_stripped_sf_1_p_2_r_3.nii.gz`.
- **Observações da experimentação:** `shrink_factor > 1` remove pouco ruído; `r > 3` e/ou `p > 2` suaviza demais; `r < 3` e/ou `p < 2` remove pouco ruído.

### 2.3. Bias field correction (N4 / ANTs)

```bibtex
@article{tustison-2010,
  title     = {{N4ITK: Improved N3 Bias Correction}},
  author    = {Tustison, Nicholas J. and Avants, Brian B. and Cook, Philip A. and Zheng, Yuanjie and Egan, Alexander and Yushkevich, Paul A. and Gee, James C.},
  journal   = {IEEE Transactions on Medical Imaging},
  publisher = {},
  volume    = {29},
  number    = {6},
  pages     = {1310--1320},
  note      = {},
  doi       = {10.1109/TMI.2010.2046908},
  issn      = {},
  year      = {2010}
}
```

- **Ferramenta:** `ants.n4_bias_field_correction`.
- **Parâmetros escolhidos:** `shrink_factor = 3`; `convergence = {"iters": [50, 50, 50, 50], "tol": 1e-7}`; `return_bias_field = False`. No notebook de produção também: `rescale_intensities=True`, `spline_param=200`, `mask=None`.
- **Sufixo:** `sf_{shrink_factor}` — ex.: `{base}_sf_3.nii.gz` após denoise.

### 2.4. Histogram matching (ANTs → template MNI)

```bibtex
@article{tustison-2021,
  title         = {{The ANTsX ecosystem for quantitative biological and medical imaging}},
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
  journal       = {Scientific Reports},
  publisher     = {},
  volume        = {11},
  number        = {9068},
  pages         = {1--13},
  note          = {},
  doi           = {10.1038/s41598-021-87564-6},
  issn          = {},
  year          = {2021}
  }
```

- **Ferramenta:** `ants.histogram_match_image2`.
- **Referência:** `atlases/templates/mni152_2009c_template.nii.gz`.
- **Parâmetros escolhidos:**

| Parâmetro (nome ficheiro) | Valor | Parâmetro ANTs |
|---------------------------|-------|----------------|
| `bins` (`b`) | `128` | `transform_domain_size` |
| `points` (`p`) | `16` | `match_points` |

- **Outros (notebook):** normalização min-max da fonte e referência; reescala pelo p99.9 da MNI.
- **Sufixo:** `_b_{bins}_p_{points}` — ex.: `I100004_stripped_b_128_p_16.nii.gz`.

### 2.5. Resumo dos parâmetros (pré-processamento T1)

| Etapa | Parâmetro-chave | Valor |
|-------|-----------------|-------|
| Skull stripping | `modality` | `t1` |
| Denoise | `shrink_factor` / `p` / `r` | `1` / `2` / `3` |
| Bias field (N4) | `shrink_factor` | `3` |
| Histogram matching | `bins` / `points` | `128` / `16` |

### 2.6. Detecção de outliers (MRQy)

Repositório: [RadQy / MRQy](https://github.com/viswanath-lab/RadQy)

```bibtex
@article{sadri-2020,
  title     = {{Technical Note: MRQy — An open‐source tool for quality control of MR imaging data}},
  author    = {Sadri, Amir Reza and Janowczyk, Andrew and Ren, Zhou and Verma, Ruchika and Beig, Niha and Antunes, Jacob and Madabhushi, Anant and Tiwari, Pallavi and Viswanath, Satish E.},
  journal   = {Medical Physics},
  publisher = {},
  volume    = {47},
  number    = {12},
  pages     = {6029--6038},
  note      = {},
  doi       = {10.1002/mp.14593 },
  issn      = {},
  year      = {2020}
}
```

A verificação de outliers foi realizada com **MRQy** sobre imagens RAW (IQMs sem referência). Métricas: estatísticas (MEAN, RNG, VAR, CV), contraste (CPP, CNR), SNR (PSNR, SNR1–SNR9), não uniformidade (CVP, CJV), artefatos (EFC, FBER), geometria (VRX, VRY, VRZ, ROW, COL, NUM).

- **Regra IQR:** limites \(Q_1 - \alpha \cdot IQR\) e \(Q_3 + \alpha \cdot IQR\), com **\(\alpha = 1.0\)**.
- **Direções:** two-sided, low-bad ou high-bad conforme a métrica.
- **Classificação suspeita:** escore de outlier = nº de métricas fora dos limites; **escore ≥ 3** → volume suspeito.

### 2.7. Parcelação e segmentação (ANTsPyNet)

```bibtex
@article{tustison-2021,
  title         = {{The ANTsX ecosystem for quantitative biological and medical imaging}},
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
  journal       = {Scientific Reports},
  publisher     = {},
  volume        = {11},
  number        = {9068},
  pages         = {1--13},
  note          = {},
  doi           = {10.1038/s41598-021-87564-6},
  issn          = {},
  year          = {2021}
  }
```

- **Parcelação:** `antspynet.desikan_killiany_tourville_labeling()` com agrupamento lobar.
- **Segmentação de tecidos:** `antspynet.deep_atropos()` com pré-processamento interno ativado.

---

## 3. Extração de atributos e harmonização

### 3.1. Harmonização entre scanners (NeuroComBat)

Implementação em **`features_selection.ipynb`** (pacote [`neuroCombat`](https://pypi.org/project/neuroCombat/) ≥ 0.2.10, ver `requirements-neurocombat.txt`). Ajuste **cross-sectional** (Fortin et al.), **não** Longitudinal ComBat (Beer et al.) — ver nota no fim desta secção.

```bibtex
@article{fortin-2018,
  title     = {{Harmonization of cortical thickness measurements across scanners and sites}},
  author    = {Fortin, Jean‐Philippe and Cullen, Nicholas and Sheline, Yvette I. and Taylor, Warren D. and Aselcioglu, Irem and Cook, Philip A. and Adams, Phil and Cooper, Crystal and Fava, Maurizio and McGrath, Patrick J. and McInnis, Melvin G. and Phillips, Mary L. and Trivedi, Madhukar H. and Weissman, Myrna M. and Shinohara, Russell T.},
  journal   = {NeuroImage},
  publisher = {},
  volume    = {167},
  number    = {},
  pages     = {104--120},
  note      = {},
  doi       = {10.1016/j.neuroimage.2017.11.024},
  issn      = {},
  year      = {2018}
}
```

#### Entradas e saídas

| Experimento | Entrada | Saída (usada em `colab/exp{1,2}_*.py`) |
|-------------|---------|----------------------------------------|
| **Exp1** (deltas) | `csvs/{ab}/all_delta_features.csv` | `all_delta_features_neurocombat.csv` |
| **Exp2** (unitários, long) | `csvs/{ab}/all_unitary_features.csv` | `all_unitary_features_neurocombat.csv` |

`{ab}` habitual: `abordagem_4_sMCI_pMCI`. No unitário harmonizado, a coluna **`MRI_DATE`** é removida da saída.

#### Coluna `batch` (efeito de scanner)

Definida no notebook a partir de metadados de aquisição (quando presentes):

`batch = MANUFACTURER + "_" + MFG_MODEL + "_" + FIELD_STRENGTH + "_" + SLICE_THICKNESS`

(como string concatenada por linha). NeuroComBat exige **≥ 2** níveis de `batch`; batches com **&lt; 3** amostras geram aviso (ComBat pode ser instável).

#### Design matrix do ComBat (implementação atual)

Covariáveis passadas a `neuroCombat()` — preservadas no ajuste além do batch:

| Coluna | No ComBat? | Papel |
|--------|------------|--------|
| `batch` | Sim (`batch_col`) | Efeito de scanner a remover |
| `AGE` | Sim (contínua) | Covariável biológica |
| `SEX` | Sim (`categorical_cols`) | Covariável biológica |
| `DIAG` | **Não** por defeito | Flag `INCLUDE_DIAG_IN_COMBAT = False` nas células do notebook |
| `GROUP` (sMCI/pMCI) | **Não** | Alvo do exp2 — incluir no ComBat **vazaria** o rótulo |
| `TIME_PROG` | **Não** | Progressão longitudinal — **vazamento** para conversão pMCI |
| `t12`, `t13`, `t23`, IDs, `roi`, `side`, `label`, … | Metadados no CSV | **Não** entram no ajuste; só radiomics/volume em `feature_cols` |

Parâmetros fixos nas células: `eb=True`, `parametric=True`, `mean_only=False`. Matriz de entrada: `dat` com shape `(n_features, n_amostras)`.

#### Pré-filtro e ficheiro de saída

Linhas são excluídas se faltar `batch`, `AGE`, `SEX` ou se alguma feature for NaN. O CSV harmonizado contém **apenas** linhas que passaram no filtro (metadados originais + features ajustadas; ordem `meta_cols + feature_cols`).

#### Baseline sem harmonização (sensibilidade)

Para comparar com o pipeline sem ComBat, apontar `CSV_PATH` nos scripts `colab/exp2_*.py` (e exp1) para `all_unitary_features.csv` / `all_delta_features.csv` em vez dos ficheiros `*_neurocombat.csv`.

#### Limitações metodológicas (documentadas no notebook)

1. **Coorte inteira:** o ComBat é ajustado em **todas** as linhas do CSV antes do CV. O z-score e a seleção de features em `colab/` são por fold (§4.3); o ComBat **não** é refitado por fold externo (melhoria futura).
2. **Formato long:** cada linha ROI×`side`×`label`×`pair` conta como uma amostra independente no ComBat (correlação intra-paciente não modelada).
3. **Longitudinal ComBat** (Beer et al., 2020) — referência para harmonização temporal explícita; **não implementado** neste repositório:

```bibtex
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
```

### 3.2. Campo de deslocamento longitudinal (`displacement_field.py`)

As deformações em RM refletem envelhecimento saudável e neurodegeneração. O pipeline isola alterações estruturais via **campos de deformação (DF)**.

#### Fundamentação e registo

- **Referência CN:** atlas estratificado por sexo e década etária (indivíduos cognitivamente normais) — alinhado à ideia de modelos de envelhecimento normal (Wyman et al., 2012).
- **Âncora longitudinal:** para cada trio (i1, i2, i3), **o mesmo** template CN (sexo e idade na i1). Registo: **fixed** = imagem clínica no tempo; **moving** = template CN.
- **Delta (implementação):** composição ANTs \(T_{1\to k} = \mathrm{inv}_k \circ \mathrm{fwd}_1\); campo \(u(x) = T(x) - x\) no domínio MNI da **baseline clínica i1** (`resampled_1.0mm`).

#### I/O e ficheiros SyN

- Lista de processamento: `image_data.txt`.
- Por registo (`images/displacement_field/`), etiqueta `CN_SEX-..._AGE-...`:
  - `*_0GenericAffine.mat`
  - `*_1Warp.nii.gz` (fwd)
  - `*_1InverseWarp.nii.gz` (inv)
- Mapas escalares em `csvs/` (prefixo `ID_PT_comb_...`), não CSV de campo 3D completo.

#### Registo e atributos longitudinais

| Etapa | Método |
|-------|--------|
| Registo global | Afim + **SyN** (não linear) via `ants.registration` |
| \(\Delta_{1\to2}\), \(\Delta_{1\to3}\), \(\Delta_{2\to3}\) | Composição de warps; log-Jacobian (`create_jacobian_determinant_image`); magnitude = \(\|u\|\) |
| Máscara | `images/brain_mask/{ID_IMG_i1}_brain_mask.nii.gz` ou env `DISPLACEMENT_BRAIN_MASK` |

| Variável de ambiente | Efeito |
|---------------------|--------|
| `DISPLACEMENT_BRAIN_MASK` | Máscara binária MNI reamostrada para a baseline |
| `DISPLACEMENT_POINT_CHUNK` | Lote na composição (default **400000**) |

**Exemplos de saída:** `csvs/{ID_PT}_comb_{COMBINATION_NUMBER}_delta12_logjac.nii.gz`, `_delta13_logjac.nii.gz`, `_delta23_logjac.nii.gz`, `_delta12_mag.nii.gz`, etc.

**Input T1:** `images/resampled_1.0mm/*_stripped_nlm_denoised_biascorrected_mni_template.nii.gz`. **Atlas CN:** `images/groupwise/`.

### 3.3. Radiomics (PyRadiomics)

```bibtex
@article{zwanenburg-2020,
  title     = {{The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping}},
  author    = {Zwanenburg, Alex and Vallières, Martin and Abdalah, Mahmoud A. and Aerts, Hugo J. W. L. and Andrearczyk, Vincent and Apte, Aditya and Ashrafinia, Saeed and Bakas, Spyridon and Beukinga, Roelof J. and Boellaard, Ronald and Bogowicz, Marta and Boldrini, Luca and Buvat, Irène and Cook, Gary J. R. and Davatzikos, Christos and Depeursinge, Adrien and Desseroit, Marie-Charlotte and Dinapoli, Nicola and Dinh, Cuong Viet and Echegaray, Sebastian and El Naqa, Issam and Fedorov, Andriy Y. and Gatta, Roberto and Gillies, Robert J. and Goh, Vicky and Götz, Michael and Guckenberger, Matthias and Ha, Sung Min and Hatt, Mathieu and Isensee, Fabian and Lambin, Philippe and Leger, Stefan and Leijenaar, Ralph T.H. and Lenkowicz, Jacopo and Lippert, Fiona and Losnegård, Are and Maier-Hein, Klaus H. and Morin, Olivier and Müller, Henning and Napel, Sandy and Nioche, Christophe and Orlhac, Fanny and Pati, Sarthak and Pfaehler, Elisabeth A.G. and Rahmim, Arman and Rao, Arvind U.K. and Scherer, Jonas and Siddique, Muhammad Musib and Sijtsema, Nanna M. and Socarras Fernandez, Jairo and Spezi, Emiliano and Steenbakkers, Roel J.H.M. and Tanadini-Lang, Stephanie and Thorwarth, Daniela and Troost, Esther G.C. and Upadhaya, Taman and Valentini, Vincenzo and van Dijk, Lisanne V. and van Griethuysen, Joost and van Velden, Floris H.P. and Whybra, Philip and Richter, Christian and Löck, Steffen},
  journal   = {Radiology},
  publisher = {},
  volume    = {295},
  number    = {2},
  pages     = {328--338},
  note      = {},
  doi       = {10.1148/radiol.2020191145},
  issn      = {},
  year      = {2020}
}
@article{lambin-2017,
  title     = {{Radiomics: the bridge between medical imaging and personalized medicine}},
  author    = {Sobrenome-autor1, Nome-autor1 and Sobrenome-autor2, Nome-autor2 and Sobrenome-autor3, Nome-autor3},
  journal   = {Nature Reviews Clinical Oncology},
  publisher = {},
  volume    = {14},
  number    = {},
  pages     = {749--762},
  note      = {},
  doi       = {10.1038/nrclinonc.2017.141},
  issn      = {},
  year      = {2017}
}
@article{griethuysen-2017,
  title     = {{Computational Radiomics System to Decode the Radiographic Phenotype}},
  author    = {van Griethuysen, JJM and Fedorov, A and Parmar,  C and Hosny, A and Aucoin, N and Narayan, V and Beets-Tan, RGH and Fillion-Robin, JC and Pieper, S and Aerts, HJWL},
  journal   = {Cancer Research},
  publisher = {},
  volume    = {77},
  number    = {21},
  pages     = {e104--e107},
  note      = {},
  doi       = {10.1158/0008-5472.CAN-17-0339},
  issn      = {},
  year      = {2017}
}
```

Script `features_radiomic.py` → `csvs/features_radiomic.csv`. Listas via `Radiomics<Classe>.getFeatureNames()` — [documentação PyRadiomics](https://github.com/AIM-Harvard/pyradiomics/blob/master/docs/features.rst).

- **firstorder (19):** `10Percentile`, `90Percentile`, `Energy`, `Entropy`, `InterquartileRange`, `Kurtosis`, `Maximum`, `Mean`, `MeanAbsoluteDeviation`, `Median`, `Minimum`, `Range`, `RobustMeanAbsoluteDeviation`, `RootMeanSquared`, `Skewness`, `StandardDeviation`, `TotalEnergy`, `Uniformity`, `Variance`
- **shape (17):** `Compactness1`, `Compactness2`, `Elongation`, `Flatness`, `LeastAxisLength`, `MajorAxisLength`, `Maximum2DDiameterColumn`, `Maximum2DDiameterRow`, `Maximum2DDiameterSlice`, `Maximum3DDiameter`, `MeshVolume`, `MinorAxisLength`, `SphericalDisproportion`, `Sphericity`, `SurfaceArea`, `SurfaceVolumeRatio`, `VoxelVolume`
- **glcm (28):** `Autocorrelation`, `ClusterProminence`, `ClusterShade`, `ClusterTendency`, `Contrast`, `Correlation`, `DifferenceAverage`, `DifferenceEntropy`, `DifferenceVariance`, `Dissimilarity`, `Homogeneity1`, `Homogeneity2`, `Id`, `Idm`, `Idmn`, `Idn`, `Imc1`, `Imc2`, `InverseVariance`, `JointAverage`, `JointEnergy`, `JointEntropy`, `MCC`, `MaximumProbability`, `SumAverage`, `SumEntropy`, `SumSquares`, `SumVariance`
- **glrlm (16):** `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelRunEmphasis`, `LongRunEmphasis`, `LongRunHighGrayLevelEmphasis`, `LongRunLowGrayLevelEmphasis`, `LowGrayLevelRunEmphasis`, `RunEntropy`, `RunLengthNonUniformity`, `RunLengthNonUniformityNormalized`, `RunPercentage`, `RunVariance`, `ShortRunEmphasis`, `ShortRunHighGrayLevelEmphasis`, `ShortRunLowGrayLevelEmphasis`
- **glszm (16):** `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelZoneEmphasis`, `LargeAreaEmphasis`, `LargeAreaHighGrayLevelEmphasis`, `LargeAreaLowGrayLevelEmphasis`, `LowGrayLevelZoneEmphasis`, `SizeZoneNonUniformity`, `SizeZoneNonUniformityNormalized`, `SmallAreaEmphasis`, `SmallAreaHighGrayLevelEmphasis`, `SmallAreaLowGrayLevelEmphasis`, `ZoneEntropy`, `ZonePercentage`, `ZoneVariance`
- **ngtdm (5):** `Busyness`, `Coarseness`, `Complexity`, `Contrast`, `Strength`
- **gldm (16):** `DependenceEntropy`, `DependenceNonUniformity`, `DependenceNonUniformityNormalized`, `DependencePercentage`, `DependenceVariance`, `GrayLevelNonUniformity`, `GrayLevelNonUniformityNormalized`, `GrayLevelVariance`, `HighGrayLevelEmphasis`, `LargeDependenceEmphasis`, `LargeDependenceHighGrayLevelEmphasis`, `LargeDependenceLowGrayLevelEmphasis`, `LowGrayLevelEmphasis`, `SmallDependenceEmphasis`, `SmallDependenceHighGrayLevelEmphasis`, `SmallDependenceLowGrayLevelEmphasis`

### 3.4. Ponderação temporal na modelagem (exp1 vs exp2)

| | **Exp1** (`delta_rate`) | **Exp2** (`baseline_rate`) |
|---|-------------------------|----------------------------|
| CSV | `all_delta_features_neurocombat.csv` | `all_unitary_features_neurocombat.csv` |
| `PAIR_ORDER` | `["12","13","23"]` | `["1","2","3"]` |
| Transformação | \(x' = x / \max(dt, \varepsilon)\) por par | `pair=1`: absoluto; `pair=2,3`: \((x-x_{\mathrm{baseline}})/\max(t_{12\|13},\varepsilon)\) |
| `SEX` | Não dividido pelo tempo | Idem |
| \(\varepsilon\) | `DT_EPSILON = 0.5` meses | Idem |

**Racional:** taxa de mudança mensal, não delta bruto multiplicado pelo intervalo. **Exp2** ancora sempre na imagem 1 (evita acúmulo de erro de registo em cadeia i1→i2→i3).

**`TIME_PROG`:** não usar como feature na classificação sMCI/pMCI (vazamento de rótulo). Também **não** entra no design do NeuroComBat (§3.1).

Implementação: `colab/exp1_utils.py` — `apply_temporal_rate_norm`, `apply_temporal_baseline_rate`, chamados em `load_tensor()`.

---

## 4. Modelagem supervisionada (exp1 / exp2) — pipeline implementado

### 4.1. Entrada comum (`load_tensor`)

- **Sem CSV wide intermédio:** o tensor `(n, 60, n_feat)` é montado em memória a partir do CSV long.
- **60 linhas** = 3 passos temporais × 20 slots (`roi`, `side`, `label`) por datapoint.
- **`y`:** GROUP → sMCI=0, pMCI=1.
- **`groups`:** `ID_PT`.
- **`slot_labels`:** `pair|roi|side|label` (agregação SHAP / coeficientes).
- Grupos com ≠ 60 linhas são ignorados.

Lista de colunas de atributos: primeira linha `As colunas de atributos são` em `exp1.md` / `exp2.md`.

### 4.2. Cenários balanced / unbalanced

Controlado por **`DOWNSAMPLE_GROUP_SEX`** (env ou constante no script; default **`True`** nos `exp2_*.py`):

| Valor | Pasta | Comportamento |
|-------|-------|----------------|
| `True` | `colab/exp{1,2}/balanced/<modelo>/` | Downsample de **pacientes** no treino externo: igualar contagem por estrato **GROUP × SEX** |
| `False` | `.../unbalanced/<modelo>/` | Todos os pacientes disponíveis no treino externo |

- Semente downsample: `RANDOM_STATE + 31 * fold_id`.
- **Teste externo (~20%)** não é alterado pelo downsample.

`run_exp2_all.py` define `DOWNSAMPLE_GROUP_SEX=1` (balanced) e `=0` (unbalanced) automaticamente.

### 4.3. Validação cruzada e anti-vazamento

**Não há** `train_test_split` fixo. Métricas principais = **OOF** (5 folds externos).

```text
                    ┌─────────────────────────────────────┐
  train_idx (80%)   │  NCV interna (5 folds) → Optuna   │
  (opc. downsample) │  tr_fit (~64%) │ val (~16%)       │
                    └─────────────────────────────────────┘
  test_idx (20%)  →  predição OOF (nunca usada no fit)
```

| Nível | Método | Folds | Papel |
|-------|--------|-------|--------|
| Externo | `StratifiedGroupKFold(groups=ID_PT)` | 5 | Teste OOF |
| Interno (Optuna) | `StratifiedGroupKFold` no `train_idx` | 5 (`INNER_NCV_SPLITS`) | Média da AUC → hiperparâmetros |
| Holdout | `inner_train_val()` (1.º split SGK) | — | Refit final + early stopping (XGB/LSTM) |

Constantes globais: `RANDOM_STATE=42`, `CORR_THR=0.9`, `VAR_THR=0.0`, `DT_EPSILON=0.5`.

#### Pré-processamento por fold (fit só em treino)

| Ordem | Passo | Anti-leakage |
|-------|--------|----------------|
| (pré) | NeuroComBat no CSV (§3.1) | Ajuste na **coorte inteira** antes do CV — não por fold |
| 0 | Ponderação temporal em `load_tensor` | Determinística por linha do CSV |
| 1 | Correlação \|ρ\| > 0,9 (greedy) | `fit` em linhas de **`tr_fit`** achatadas |
| 2 | `VarianceThreshold(0.0)` | Colunas constantes no treino |
| 3 | `StandardScaler` (z-score) | `fit` em `tr_fit`; `transform` em val/test |

No **Optuna (NCV interna)**, correlação + variância + scaler são **recalculados em cada split interno** (`flat_scaled_tabular_train_val`, `seq_scaled_train_val`, `prepare_scaled_rocket_inputs`).

**Classificação:** limiar **0,5** em probabilidade (XGB, ROCKET, LSTM). SVM: `predict()` para rótulo; **sigmoid(decision_function)** para score/AUC/OOF.

### 4.4. Modelos, Optuna e interpretabilidade

Hiperparâmetros são escolhidos **por fold externo** (5 estudos Optuna independentes); valores finais em `checkpoints/fold_k/meta.json` → `best_params`.

#### XGBoost (`colab/exp2_xgboost.py`)

| Item | Valor |
|------|--------|
| Entrada | `(n, 60·p')` achatado |
| Optuna | 30 trials, TPE, maximizar média AUC NCV interna |
| Espaço | `spw_mul` [0.25,4] log; `max_depth` [2,8]; `learning_rate` [0.01,0.3] log; `subsample` [0.5,1]; `colsample_bytree` [0.2,1]; `reg_lambda` [1e-3,50] log; `min_child_weight` [1,20]; `gamma` [0,5] |
| Treino | `binary:logistic`, early stopping 50 rondas, máx 200 árvores, `scale_pos_weight = (n_neg/n_pos)·spw_mul` |
| Interpretação | SHAP `TreeExplainer` no teste; agregação \|SHAP\| por ROI e atributo |

#### SVM linear (`colab/exp2_svm.py`)

| Item | Valor |
|------|--------|
| Modelo | `LinearSVC`, `dual="auto"`, `max_iter=50000` |
| Optuna | 30 trials; **`C`** ∈ [1e-4, 1e4] log |
| Score | AUC com `decision_function` (NCV e refit) |
| Interpretação | Média de \|coef.\| por fold → ROI / atributo |

#### ROCKET + L1 (`colab/exp2_rocket.py`)

| Item | Valor |
|------|--------|
| Entrada | `(n, p', 60)` após transpose para sktime |
| ROCKET | `num_kernels=2000`; **novo fit por fold interno** no Optuna (sem vazamento) |
| Classificador | `LogisticRegression` L1, `solver="saga"`, `max_iter=10000` |
| Optuna | 30 trials; **`C`** ∈ [1e-4, 1e4] log |

#### LSTM (`colab/exp2_lstm.py` → `exp_lstm_common.py`)

| Item | Valor |
|------|--------|
| Entrada | `(n, 3, 20·p')` via `panels_to_seq` |
| Arquitetura | `LSTM(units, dropout, use_cudnn=False)` → `Dropout` → `Dense(1, sigmoid)` |
| Optuna | 20 trials: `units` 16–96 step 16; `dropout` [0.1,0.5]; `lr` [1e-4,3e-3] log; `batch_size` ∈ {16,32,64} |
| Treino | até 100 épocas; early stopping `val_auc`, patience 10; `class_weight` balanceado |
| GPU | `LSTM_DEVICE`, `LSTM_GPU_INDEX`; XLA/jit desligados; `use_cudnn=False` |
| Interpretação | SHAP Kernel no vetor achatado (exp2) |

#### Optuna (otimização de hiperparâmetros)

```bibtex
@inproceedings{akiba-2019,
  title     = {{Optuna: A Next-generation Hyperparameter Optimization Framework}},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '19)},
  publisher = {},
  volume    = {},
  number    = {},
  address   = {New York, NY, USA},
  pages     = {2623--2631},
  note      = {},
  doi       = {10.1145/3292500.3330701},
  issn      = {},
  isbn      = {},
  year      = {2019}
}
```

- **Sampler:** `TPESampler` com semente derivada de `fold_id`.
- **Direção:** maximizar média da **AUC** na NCV interna (5 folds).

#### SHAP (XGBoost e LSTM)

```bibtex
@inproceedings{lundberg-2017,
  title     = {{A Unified Approach to Interpreting Model Predictions}},
  author    = {Lundberg, Scott M and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems},
  publisher = {},
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
```

- **XGBoost:** `shap.TreeExplainer` no conjunto de teste externo; agregação de \|SHAP\| médio por ROI (`roi_from_slot_label`) e por nome de atributo.
- **LSTM:** `shap.Explainer` com masker independente; background `SHAP_BACKGROUND=40`, amostras `SHAP_SAMPLES=60`.
- **SVM / ROCKET:** interpretação por **\|coeficientes\|** (L1 ou `LinearSVC`), mesma agregação ROI/atributo — sem SHAP no espaço kernel ROCKET.

### 4.5. Métricas e artefactos

| Métrica | Uso |
|---------|-----|
| Acc | Limiar 0,5 (pode ser alta com classe majoritária ~79% sMCI) |
| AUC | Ranqueamento |
| F1 | Classe pMCI |
| AP | Desbalanceamento (classe 1) |

Por run (`colab/exp2/{balanced|unbalanced}/{xgboost|svm|rocket|lstm}/`):

| Pasta | Conteúdo |
|-------|----------|
| `tables/` | `metrics_per_fold.csv`, `oof_predictions.csv`, `fold_test_scores.csv`, importâncias, `feature_counts_fold0.csv`, `run_meta.json` |
| `figures/` | Confusão OOF, ROC/PR, boxplot, SHAP/coef, curvas (fold 0) |
| `checkpoints/fold_{0..4}/` | Modelo, `preprocess.joblib`, `meta.json` (exp2) |
| `tables/demographics/`, `figures/demographics/` | Via `analyze_oof_demographics.py` (sexo F/M) |

### 4.6. Execução (orquestradores)

Raiz do repositório; Python: `.venv/bin/python` ou `python`.

```bash
# Experimento 2 — 8 runs (4 modelos × balanced/unbalanced)
.venv/bin/python colab/run_exp2_all.py

# Regenerar CSV harmonizado (após alterar features_selection.ipynb §3.1)
#   → executar células NeuroComBat (delta + unitário) no notebook

# Regenerar figuras + demografia (8 runs com OOF)
.venv/bin/python colab/postprocess_exp2_runs.py

# Ablação LOO por ROI — XGB balanced (~10 ROIs no CSV atual)
.venv/bin/python colab/run_roi_ablation_exp2.py

# Opcional: verificar checkpoint XGB fold 0 vs OOF
.venv/bin/python colab/verify_xgb_checkpoint.py
```

**Experimento 1 (manual, referência):**

```bash
.venv/bin/python colab/exp1_xgboost.py   # idem rocket, svm, lstm
.venv/bin/python colab/exp1_plots.py
```

**Dependências:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `xgboost`, `shap`, `sktime`, `tensorflow` (LSTM).

### 4.7. Ablação por ROI (exp2, XGBoost)

- Script: `colab/run_roi_ablation_exp2.py`.
- Remove uma **`roi`** de cada vez (máscara a zero nos 3 pares temporais).
- Saída: `colab/exp2/balanced/xgboost_ablation/drop_<roi>/`.
- Resumo: `ablation_summary.csv`, `ablation_delta_auc_oof.pdf`.
- Por defeito reutiliza `best_params` do baseline (`ABLATION_SKIP_OPTUNA=1`); `ABLATION_FORCE_OPTUNA=1` para Optuna em cada ROI.

### 4.8. Comparação exp1 vs exp2

| | Exp1 | Exp2 (ativo) |
|---|------|----------------|
| Features | Deltas entre pares | Absolutos + taxa desde baseline |
| Scripts | `exp1_*.py` | `exp2_*.py` |
| Pasta resultados | `colab/exp1/...` | `colab/exp2/...` |

Mesma arquitetura de CV, filtros, Optuna e utilitários (`exp1_utils.py`, `exp_lstm_common.py`).

---

## 5. Trabalho futuro e legado

### 5.1. Em aberto (nível imagem / features)

- `features_displacement.py`: estatísticas de DF por ROI a partir dos NIfTI de delta.
- Agregar volume/radiomics por conjunto com deltas \(\Delta_{12}\), \(\Delta_{13}\) face à baseline (chave `ID_PT` + `COMBINATION_NUMBER`).
- **Harmonização:** NeuroComBat por fold externo (evitar estatísticas globais na validação); avaliar **Longitudinal ComBat** (Beer et al., §3.1) para tripletas temporais.
- Modelagem por **grafos** após fechar baseline tabular/sequencial.

### 5.2. Scripts exploratórios (não são o pipeline de produção)

| Script | Papel |
|--------|--------|
| `colab/cnn_example.py` | CNN 1D + SelectKBest |
| `colab/lstm_example.py` | LSTM antigo |
| `colab/sklearn_teste.py`, `colab/models_teste.py` | Protótipos sklearn/PyCaret |
| `colab/datasets.py` | Construtores wide / sequência |

Detecção de outliers em features (IsolationForest, etc.) planeada para etapa posterior; **não** integrada em `exp*_*.py`.

### 5.3. Desenho inicial (referência histórica)

Ideia original (substituída pelo código atual):

1. Flatten manual 20 ROIs × 3 tempos → CSV wide.
2. Balanceamento GROUP×SEX.
3. Seleção de atributos (correlação, variância, SFS).
4. Split sem leakage por `ID_PT`.
5. Classificação sMCI vs pMCI.

**Implementação atual:** tensor em memória, seleção e z-score **por fold**, Optuna com NCV interna, OOF agregado — ver secção 4.

---

## Referências bibliográficas

As entradas BibTeX completas estão **inline** nas secções correspondentes:

| Tema | Secção | Chave sugerida |
|------|--------|----------------|
| ANTs / ANTsPyNet | §2.1, §2.4, §2.7 | `tustison-2021` |
| Denoise NLM | §2.2 | `manjon-2010` |
| N4 bias | §2.3 | `tustison-2010` |
| MRQy | §2.6 | `mrqy-2020` |
| NeuroComBat (implementado) | §3.1 | `fortin-2018` |
| Longitudinal ComBat (referência) | §3.1 | `beer-2020` |
| SHAP | §4.4 | `lundberg-2017` |
| Optuna | §4.4 | `akiba-2019` |

Literatura adicional citada no texto: Reuter et al. (2011–2012) — registo longitudinal; Wyman et al. (2012) — biomarcadores e referência CN.

---

*Para edição do pipeline de modelagem exp2, alterar os scripts em `colab/` e sincronizar [`exp2.md`](exp2.md); este README é o mapa do repositório.*

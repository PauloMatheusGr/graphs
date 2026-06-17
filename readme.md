# Pipeline de neuroimagem e modelagem supervisionada (sMCI vs pMCI)

**Última atualização:** maio/2026 — Paulo Girardi

Este documento descreve o fluxo **do volume T1 ao classificador**, com ênfase no que está **implementado em código**. Detalhes por experimento: [`exp1.md`](exp1.md) (deltas) e [`exp2.md`](exp2.md) (absolutos + taxa desde baseline). Comandos rápidos: [`colab/readme.txt`](colab/readme.txt).

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Pré-processamento de imagens](#2-pré-processamento-de-imagens)
3. [Extração de atributos e harmonização](#3-extração-de-atributos-e-harmonização)
4. [Modelagem supervisionada (exp1 / exp2)](#4-modelagem-supervisionada-exp1--exp2--pipeline-implementado) (§4.3: divisão dos dados e CV aninhada)
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
| CSV → modelo | Secção 4; `colab/exp{1,2}_*.py`, `colab/exp_utils.py` |
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
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
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
  journal       = {Journal of Magnetic Resonance Imaging},
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
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
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
  doi       = {10.1002/mp.14593},
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
  author        = {Tustison, Nicholas J. and Cook, Philip A. and Holbrook, Andrew J. and Johnson, Hans J. and Muschelli, John and Devenyi, Gabriel A. and Duda, Jeffrey T. and Das, Sandhitsu R. and Cullen, Nicholas C. and Gillen, Daniel L. and Yassa, Michael A. and Stone, James R. and Gee, James C. and Avants, Brian B.},
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

Duas vias no repositório (ajuste **cross-sectional** NeuroComBat, Fortin et al. 2018; base ComBat: Johnson et al. 2007). **Longitudinal ComBat** (Beer et al. 2020) — referência, não implementado (ver nota no fim):

| Via | Onde | Pacote | Uso |
|-----|------|--------|-----|
| **Coorte inteira** | `1_features_selection.ipynb` (legado) | [`neuroCombat`](https://pypi.org/project/neuroCombat/) ≥ 0.2.10 | Exploração / CSV `*_neurocombat.csv` |
| **Por fold (colab exp1/exp2)** | `colab/exp_harmonize.py` + `exp{1,2}_*.py` | [`neuroCombat`](https://pypi.org/project/neuroCombat/) | CV sem vazamento (`RUN_NEUROCOMBAT=1`) |
| **Por fold (ablação)** | `ablation_harmonize.py` + `ablation_runner.py` / `3_ablation.ipynb` | [`neuroCombat`](https://pypi.org/project/neuroCombat/) | Flag `WITH_COMBAT`; CSV long por `ID_IMG` |

Dependências: `requirements-neurocombat.txt` (notebook legado + ablação + exp2 com `RUN_NEUROCOMBAT=1`).

```bibtex
@article{johnson-2007,
  title   = {{Adjusting batch effects in microarray expression data using empirical Bayes methods}},
  author  = {Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel},
  journal = {Biostatistics},
  volume  = {8},
  number  = {1},
  pages   = {118--127},
  doi     = {10.1093/biostatistics/kxj037},
  year    = {2007}
}
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
| **Exp2** (unitários, long) | `csvs/{ab}/all_unitary_features.csv` | `all_unitary_features_neurocombat.csv` (só notebook) |

`{ab}` habitual: `abordagem_4_sMCI_pMCI`. Nos scripts **exp2**, `CSV_PATH` aponta sempre para **`all_unitary_features.csv`** (sem harmonização prévia). No unitário harmonizado do notebook, a coluna **`MRI_DATE`** é removida da saída.

#### Coluna `batch` (efeito de scanner)

Definida a partir de metadados de aquisição em `run_nb1_post_extract.py` / notebook 1:

`batch = MANUFACTURER + "_" + MFG_MODEL + "_" + FIELD_STRENGTH`

(No notebook legado de harmonização na coorte inteira pode incluir também `SLICE_THICKNESS`.)

NeuroComBat exige **≥ 2** níveis de `batch`; batches com **&lt; 3** amostras geram aviso (ComBat pode ser instável).

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

Nos scripts exp2, manter `RUN_NEUROCOMBAT=False` (predefinição). No exp1, usar `all_delta_features.csv` em vez de `*_neurocombat.csv` se quiser o mesmo tipo de comparação.

#### Exp2: ComBat por fold (`RUN_NEUROCOMBAT`)

Implementação em **`colab/exp_harmonize.py`** (`neuroCombat` + `neuroCombatFromTraining`, espelhado em `ablation_harmonize.py`).

| Parâmetro | Valor |
|-----------|--------|
| Flag no script | `RUN_NEUROCOMBAT = False` ou env `RUN_NEUROCOMBAT=1` |
| Entrada | `csvs/abordagem_4_sMCI_pMCI/all_unitary_features.csv` |
| Unidade estatística | **Uma linha wide por `ID_IMG_ref`** (colunas = `roi\|side\|label` × feature radiomics) |
| Fit ComBat | Imagens dos triplets do **treino externo** do fold (após downsample, se ativo) |
| Transform | Imagens dos triplets **treino ∪ teste** desse fold |
| Covariáveis | `batch`, `AGE`, `SEX` (sem `GROUP`, `TIME_PROG`, `DIAG`) |
| Ordem no pipeline | ComBat → `baseline_rate` (`load_tensor`) → correlação/variância → `StandardScaler` (§4.3) |
| Saídas | `colab/exp2/{balanced\|unbalanced}/{modelo}_neurocombat/` quando a flag está ativa |

**Requisitos:** ≥ 2 níveis de `batch` no treino do fold. Batches com &lt; 3 imagens no treino: aviso. Batches que aparecem **só no teste** do fold: aviso e imagens mantêm features **não harmonizadas** (ComBat não pode aplicar parâmetros de scanner não vistos no fit).

**Modelos:** `exp2_xgboost.py`, `exp2_svm.py`, `exp2_rocket.py`, `exp2_lstm.py` (via `exp_lstm_common.py`).

```bash
# Exemplo (venv do projeto)
cd colab
RUN_NEUROCOMBAT=1 ../.venv/bin/python exp2_svm.py
```

#### Limitações metodológicas

**Notebook (coorte inteira):**

1. ComBat em **todas** as linhas do CSV antes de qualquer CV.
2. Formato **long** no notebook: cada linha ROI×`pair` conta como amostra independente (correlação intra-imagem não modelada; semântica de coluna misturada por ROI).

**Exp2 com `RUN_NEUROCOMBAT=1`:** formato **wide por imagem** no ComBat; ainda assim `baseline_rate` é calculado na coorte inteira antes do split (ver §3.4). ComBat no **treino externo** inclui imagens do holdout interno (Optuna) — vazamento leve dentro do bloco de treino; não inclui triplets do teste externo.

**Longitudinal ComBat** (Beer et al., 2020) — referência; **não implementado**:

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

### Formação dos conjuntos com 3 imagens

```bibtex
@article{smith-2002,
  title     = {{Accurate, Robust, and Automated Longitudinal and Cross-Sectional Brain Change Analysis}},
  author    = {Smith, Stephen M. and Zhang, Yongyue and Jenkinson, Mark and Chen, Jue and Matthews, Paul M. and Federico, Antonio and De Stefano, Nicola},
  journal   = {NeuroImage},
  publisher = {},
  volume    = {17},
  number    = {},
  pages     = {479--489},
  note      = {},
  doi       = {10.1006/nimg.2002.1040},
  issn      = {},
  year      = {2002}
}
@article{freeborough-1997,
  title     = {{The Boundary Shift Integral: An Accurate and Robust Measure of Cerebral Volume Changes from Registered Repeat MRI}},
  author    = {Freeborough, Peter A. and Fox, Nick C.},
  journal   = {IEEE Transactions on Medical Imaging},
  publisher = {},
  volume    = {16},
  number    = {5},
  pages     = {623--629},
  note      = {},
  doi       = {10.1109/42.640753},
  issn      = {},
  year      = {1997}
}
@article{reuter-2012,
  title     = {{Within-subject template estimation for unbiased longitudinal image analysis}},
  author    = {Reuter, Martin and Schmansky, Nicholas J. and Rosas, H. Diana and Fischl, Bruce},
  journal   = {NeuroImage},
  publisher = {},
  volume    = {61},
  number    = {},
  pages     = {1402--1418},
  note      = {},
  doi       = {10.1016/j.neuroimage.2012.02.084},
  issn      = {},
  year      = {2012}
}
```

Definição do porquê utilizar a abordagem baseline em vez da abordagem sequencial nos conjuntos com 3 imagens.

1. Mitigação do Acúmulo de Erros (Estabilidade Estatística): Na análise longitudinal com três ou mais pontos no tempo (ex: t0​,t1​,t2​), a comparação sequencial (t0​→t1​ somado a t1​→t2​) está frequentemente sujeita à propagação de erros de registro e interpolação. Smith et al. (2002) apontam que, na verdade, a comparação entre a soma das medidas sequenciais e a medida direta (t0​→t2​) é um método sensível justamente para evidenciar fontes de erro nos procedimentos de estimativa de atrofia. Ao adotar a comparação direta sempre com o baseline (i1​→i2​ e i1​→i3​), você elimina o acúmulo de variância e o ruído que ocorreriam ao usar a imagem i2​ (que já sofreu transformações ou representa um estado intermediário) como referência para i3​, garantindo maior estabilidade estatística à sua medida direta.

2. Consistência do Referencial Anatômico (Precisão Biológica): Para garantir a precisão biológica, as alterações estruturais sutis (como atrofia tecidual, mudanças de volume e deformações locais) devem ser mapeadas contra a anatomia original do paciente. A literatura tradicional de processamento longitudinal frequentemente emprega métodos de registro que alinham os exames de acompanhamento (follow-up) diretamente ao exame baseline para computar os campos de deformação e analisar as alterações. Freeborough & Fox (1997) também validam essa abordagem ao descrever que a imagem repetida deve ser registrada com a imagem baseline determinando rotações e translações que minimizem o desvio padrão entre os voxels correspondentes. Usar o baseline garante que a métrica de progressão reflita a verdadeira alteração desde o início do estudo, ancorando os biomarcadores extraídos a um estado biológico inicial constante.

A escolha da abordagem baseada em baseline (i1​→i2​,i1​→i3​), em detrimento de uma abordagem puramente sequencial (i1​→i2​,i2​→i3​), fundamenta-se na necessidade de garantir a estabilidade estatística e a precisão biológica na extração de biomarcadores longitudinais. Do ponto de vista estatístico, o uso da imagem i1​ como referencial único mitiga a propagação e o acúmulo de erros de registro e de interpolação geométrica que frequentemente afetam cadeias de comparações sucessivas. A literatura demonstra que a soma de medidas sequenciais pode introduzir variâncias adicionais que são evitadas ao se realizar a medição direta entre o ponto inicial e os tempos subsequentes \cite{smith-2002}. Sob a ótica da precisão biológica, o processamento longitudinal classicamente estabelece o baseline como o sistema de coordenadas absoluto para computar os campos de deformação e as perdas volumétricas ao longo do tempo \cite{reuter-2012,freeborough-1997}. Dessa forma, garante-se que todas as alterações estruturais extraídas reflitam o desvio morfológico real em relação ao estado anatômico nativo do indivíduo no momento de inclusão no estudo, otimizando a confiabilidade das trajetórias dos biomarcadores.


### 3.2. Campo de deslocamento longitudinal (`displacement_field.py`)

As deformações em RM refletem envelhecimento saudável e neurodegeneração. O pipeline isola alterações estruturais via **campos de deformação (DF)**.

#### Fundamentação e registo

- **Referência CN:** atlas estratificado por sexo e década etária (indivíduos cognitivamente normais) — alinhado à padronização ADNI (Wyman et al., 2013).

```bibtex
@article{wyman-2013,
  title     = {{Standardization of analysis sets for reporting results from ADNI MRI data}},
  author    = {Wyman, Bradley T. and Harvey, Danielle J. and Crawford, Karen and Bernstein, Matt A. and Carmichael, Owen and Cole, Patricia E. and Crane, Paul K. and DeCarli, Charles and Fox, Nick C. and Gunter, Jeffrey L. and Hill, Derek L. and Killiany, Ronald J. and Pachachi, Chahin and Schwarz, Adam J. and Schuff, Norbert and Senjem, Matthew L. and Suhy, Joyce and Thompson, Paul M. and Weiner, Michael and Jack, Clifford R., Jr. and {Alzheimer's Disease Neuroimaging Initiative}},
  journal   = {Alzheimer's \& Dementia},
  publisher = {},
  volume    = {9},
  number    = {},
  pages     = {332--337},
  note      = {},
  doi       = {10.1016/j.jalz.2012.06.004},
  issn      = {},
  year      = {2013}
}
```

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
  author    = {Lambin, Philippe and Rios-Velazquez, Emmanuel and Leijenaar, Ralph and Carvalho, Sara and van Stiphout, Ruud G. P. M. and Granton, Bram and Zegers, Catharina M. L. and Gillies, Richard and Boellard, Ronald and Dekker, Andre and Shinohara, Russell T. and Kerkmeijer, Ferda and Lambrecht, Matthias and Menyhart, Orsolya and Wientjes, Celine and Natussen, Ursula and Dekker, Floris and Hoebers, Frank and Aerts, Hugo J. W. L.},
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
  author    = {van Griethuysen, Joost J. M. and Fedorov, Andriy and Parmar, Chintan and Hosny, Ahmed and Aucoin, Nicole and Narayan, Vivek and Beets-Tan, Regina G. H. and Fillion-Robin, Jean-Christophe and Pieper, Steve and Aerts, Hugo J. W. L.},
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
| CSV | `all_delta_features.csv` ou `*_neurocombat.csv` (exp1) | `all_unitary_features.csv` (`RUN_NEUROCOMBAT` no pipeline) |
| `PAIR_ORDER` | `["12","13","23"]` | `["1","2","3"]` |
| Transformação | \(x' = x / \max(dt, \varepsilon)\) por par | `pair=1`: absoluto; `pair=2,3`: \((x-x_{\mathrm{baseline}})/\max(t_{12\|13},\varepsilon)\) |
| `SEX` | Não dividido pelo tempo | Idem |
| \(\varepsilon\) | `DT_EPSILON = 0.5` meses | Idem |

**Racional:** taxa de mudança mensal, não delta bruto multiplicado pelo intervalo. **Exp2** ancora sempre na imagem 1 (evita acúmulo de erro de registo em cadeia i1→i2→i3).

**`TIME_PROG`:** não usar como feature na classificação sMCI/pMCI (vazamento de rótulo). Também **não** entra no design do NeuroComBat (§3.1).

Implementação: `colab/exp_utils.py` — `apply_temporal_rate_norm`, `apply_temporal_baseline_rate`, chamados em `load_tensor()`.

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

### 4.3. Etapas de divisão dos dados (validação cruzada aninhada)

**Não há** `train_test_split` fixo único. As métricas reportadas no artigo vêm das previsões **out-of-fold (OOF)**: em cada um dos 5 folds externos, o modelo prevê o conjunto de **teste** desse fold (dados que não entraram no treino daquela repetição). A junção das cinco partições de teste cobre os **1276** conjuntos uma vez.

Implementação: `StratifiedGroupKFold` com `groups=ID_PT` em `colab/exp_utils.py` (`inner_train_val`, `inner_cv_splits`, `downsample_train_indices`).

Constantes globais: `RANDOM_STATE=42`, `CORR_THR=0.9`, `VAR_THR=0.0`, `DT_EPSILON=0.5`, `INNER_NCV_SPLITS=5`.

#### Coorte, conjuntos e pacientes

| Conceito | Significado | Exp2 (exemplo auditado) |
|----------|-------------|-------------------------|
| **Linha do CSV** | 1 ROI × 1 `pair` (i1, i2 ou i3) | 76 560 linhas no CSV harmonizado |
| **Conjunto / amostra do modelo** | Chave `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)` com bloco **60×atributos** (20 ROIs × 3 pares) | **1276** conjuntos válidos |
| **Paciente** | `ID_PT` — unidade do split de CV | **525** pacientes |
| **Rótulo `y`** | `GROUP`: sMCI=0, pMCI=1 | 1011 conjuntos sMCI, 265 pMCI (~79% / 21%) |

O classificador recebe tensores `(n, 60, n_feat)` (ou variantes sequenciais / ROCKET), **não** as 76 560 linhas do CSV. Vários conjuntos podem pertencer ao mesmo paciente.

**Auditoria reprodutível:** script em `experimentos.ipynb` (ou funções em `exp_utils`) com o mesmo CSV e `exp2.md`; export opcional para `colab/paper_split_audit/`.

#### Cinco folds externos: uma coorte, cinco repetições

Não são cinco coortes independentes. É o **mesmo** conjunto de 525 pacientes / 1276 amostras, com **papéis** diferentes em cada fold:

- Em **um** fold, cada paciente está no **teste** ou no **pool de treino** (~80%).
- Ao longo dos **5** folds, cada paciente é testado **exatamente uma vez** → 5 × 105 = **525** atribuições de teste (não 525×5 pacientes distintos).
- Cada paciente entra no pool de treino em **quatro** folds → 5 × 420 = **2100** atribuições de treino (não 2100 pacientes).

```text
COORTE: 1276 conjuntos · 525 pacientes · 1011 sMCI / 265 pMCI (conjuntos)

     FOLD 0      FOLD 1      FOLD 2      FOLD 3      FOLD 4
        │           │           │           │           │
 TESTE  241/105    261/105    256/105    258/105    260/105   Σ conjuntos = 1276 (OOF)
        │           │           │           │           │
 POOL   1035/420   1015/420   1020/420   1018/420   1016/420   420 pac. = 525−105 (fixo)
        │           │           │           │           │
        ▼ downsample GROUP×SEX (só no pool; ver §4.2)
        │
 TREINO 435/184    374/164    362/164    380/168    427/184   (média ~396 conj. / 173 pac.)
 bal.
        │
   ┌────┴────┐
   ▼         ▼
 NCV×5    tr_fit | val  (~80% / ~20% do treino bal.; holdout fixo)
 Optuna
   │
   └──► previsão no TESTE deste fold → OOF
```

Por fold externo, **teste + treino balanceado** usam tipicamente **676–687** conjuntos; o restante do pool de treino (~600 conjuntos) fica **fora** do treino **desse** fold (pacientes não sorteados no downsample), sem passar ao teste desse fold.

#### Passo a passo (um fold externo)

1. **Split externo** (`StratifiedGroupKFold`, 5 folds, `random_state=42`): ~**20%** dos conjuntos → **teste externo**; ~**80%** → **pool de treino**. Por paciente: **105** no teste, **420** no pool (valores exatos no exp2 auditado; constantes em todos os folds).
2. **Downsample** (se `DOWNSAMPLE_GROUP_SEX=True`, §4.2): aplicado **só** ao pool de treino. Equilibra **pacientes** nos quatro estratos **GROUP × SEX** (`min_n` pacientes por estrato); mantém **todos os conjuntos** dos pacientes selecionados. **Não** iguala o número de conjuntos sMCI vs pMCI; **não** altera o teste externo.
3. **NCV interna** (5 folds no treino balanceado): em cada trial do Optuna, treina nos splits internos de treino e avalia AUC nos splits internos de **validação**; o score do trial é a **média das 5 AUC**. **Sem** novo downsample. Pré-processamento (correlação, variância, scaler) **recalculado por split interno**.
4. **Holdout `tr_fit` | `val`** (primeiro split do mesmo tipo de SGK no treino balanceado): divisão **fixa** ~80% / ~20% por paciente para o **modelo final** desse fold externo (não confundir com os 5 splits da NCV).
5. **Treino final** com os melhores hiperparâmetros do passo 3, usando `tr_fit` (e `val` conforme o modelo — ver tabela abaixo).
6. **Avaliação**: previsão apenas no **teste externo** do fold → agregação OOF.

#### Tabelas de contingência — exp2 balanced (auditoria)

Valores **conjuntos / pacientes** por fold (treino antes do downsample: **420** pacientes em todos os folds).

| Fold | Teste | Pool treino (antes DS) | Treino bal. (após DS) | tr_fit | val |
|:----:|:-----:|:----------------------:|:---------------------:|:------:|:---:|
| 0 | 241 / 105 | 1035 / 420 | 435 / 184 | 351 / 148 | 84 / 36 |
| 1 | 261 / 105 | 1015 / 420 | 374 / 164 | 302 / 132 | 72 / 32 |
| 2 | 256 / 105 | 1020 / 420 | 362 / 164 | 285 / 130 | 77 / 34 |
| 3 | 258 / 105 | 1018 / 420 | 380 / 168 | 306 / 135 | 74 / 33 |
| 4 | 260 / 105 | 1016 / 420 | 427 / 184 | 341 / 147 | 86 / 37 |
| **Média** | **255** / **105** | **~1021** / **420** | **~396** / **~173** | **~317** / **~138** | **~79** / **~34** |

Exemplo **fold 0** — rótulo em conjuntos: teste 184 sMCI / 57 pMCI; treino bal. 240 / 195; `val` 38 / 46 (prevalência em `val` pode desviar da coorte por ser holdout pequeno e haver vários conjuntos por paciente).

NCV interna (ordem de grandeza no fold 0, treino bal. 435 conj.): ~**348** conjuntos no treino interno e ~**87** na validação interna **por split** (~147 / ~37 pacientes).

#### O que cada nível faz (resumo)

| Nível | Método | Folds | Treino | Avaliação | Objetivo |
|-------|--------|:-----:|--------|-----------|----------|
| **Externo** | `StratifiedGroupKFold(ID_PT)` | 5 | Pool → (opc.) downsample → bal. | **Teste externo** | Métricas **OOF** (artigo) |
| **Interno (Optuna)** | SGK no treino bal. | 5 | `in_tr` (~80% do bal.) | `in_va` (~20% do bal.) | Média AUC → **hiperparâmetros** |
| **Holdout** | `inner_train_val()` (1.º split SGK) | 1 | `tr_fit` | `val` | **Modelo final** + curvas / early stop |

A NCV interna e o holdout `tr_fit`|`val` atuam sobre o **mesmo** treino balanceado, em **paralelo** (não em série): o Optuna **não** usa `val` holdout; o refit final **não** substitui a média dos cinco modelos internos por um único modelo da NCV.

#### Papel do holdout `val` (por modelo)

| Modelo | Optuna (NCV interna) | Holdout `tr_fit` \| `val` |
|--------|----------------------|---------------------------|
| **XGBoost** | Média AUC; early stopping em cada `in_va` | Refit em `tr_fit`; **early stopping** em `val`; curvas logloss/acc |
| **LSTM** | Idem (épocas / `val_auc` nos splits internos) | Refit em `tr_fit`; **early stopping** em `val`; curvas |
| **SVM** | Média AUC em `in_va` | `LinearSVC.fit(tr_fit)`; `val` para **curvas** (SGD), sem early stop no SVC |
| **ROCKET+L1** | Média AUC; ROCKET refit por split interno no Optuna | ROCKET+LR em `tr_fit`; `val` para **curvas** e diagnóstico (fold 0) |

O `best_val_auc` registado após Optuna no log refere-se à **média da NCV interna**, não à AUC do `val` holdout (salvo caminhos de ablação com parâmetros fixos).

#### Diagrama — fold externo 0 (com NCV interna)

```text
                    COORTE 1276 conj. · 525 pac.
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
     TESTE EXTERNO                      POOL TREINO
     241 conj. · 105 pac.               1035 conj. · 420 pac.
     (sem downsample)                         │
                                             ▼ DOWNsample GROUP×SEX
                                        TREINO BALANCEADO
                                        435 conj. · 184 pac.
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
                 NCV INTERNA ×5                           HOLDOUT tr_fit | val
                 (Optuna: média AUC)                        351/148  |  84/36
                 treino int. ~348 | val int. ~87              │
                 por split; sem novo DS                       ▼
                        │                              MODELO FINAL
                        └──────── hiperparâmetros ──────────┘
                                             │
                                             ▼
                                   PREVISÃO OOF (241 teste)
```

#### Pré-processamento por fold (fit só em treino relevante)

| Ordem | Passo | Onde `fit` | Anti-vazamento |
|-------|--------|------------|----------------|
| (pré) | NeuroComBat no CSV (§3.1) | Coorte inteira (antes do CV) | Não por fold — ver limitação em §3.1 |
| 0 | Ponderação temporal em `load_tensor` | Determinística por linha | Sem vazamento por fold |
| 1 | Correlação \|ρ\| > 0,9 (greedy) | Treino do split em uso (`tr_fit` ou `in_tr`) | Sem `fit` em val/teste do mesmo passo |
| 2 | `VarianceThreshold(0.0)` | Idem | Remove só colunas constantes no treino |
| 3 | `StandardScaler` (z-score) | Idem | `transform` em val/teste |

No **Optuna**, os passos 1–3 são **recalculados em cada split interno** (`flat_scaled_tabular_train_val`, `seq_scaled_train_val`, `prepare_scaled_rocket_inputs`). No **refit final**, o pipeline de atributos é ajustado em `tr_fit` e aplicado a `val` e teste externo.

**Classificação:** limiar **0,5** em probabilidade (XGB, ROCKET, LSTM). SVM: `predict()` para rótulo; **sigmoid(decision_function)** para score/AUC/OOF.

### 4.4. Modelos, Optuna e interpretabilidade

Hiperparâmetros são escolhidos **por fold externo** (5 estudos Optuna independentes); valores finais em `checkpoints/fold_k/meta.json` → `best_params`.

#### XGBoost (`colab/exp2_xgboost.py`)
```bibtex
@inproceedings{chen-2016,
  title     = {{XGBoost: A Scalable Tree Boosting System}},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  publisher = {},
  volume    = {},
  number    = {},
  address   = {New York, NY, USA},
  pages     = {785--794},
  note      = {},
  doi       = {10.1145/2939672.2939785},
  issn      = {},
  isbn      = {},
  year      = {2016}
}
```

| Item | Valor |
|------|--------|
| Entrada | `(n, 60·p')` achatado |
| Optuna | 30 trials, TPE, maximizar média AUC NCV interna |
| Espaço | `spw_mul` [0.25,4] log; `max_depth` [2,8]; `learning_rate` [0.01,0.3] log; `subsample` [0.5,1]; `colsample_bytree` [0.2,1]; `reg_lambda` [1e-3,50] log; `min_child_weight` [1,20]; `gamma` [0,5] |
| Treino | `binary:logistic`, early stopping 50 rondas, máx 200 árvores, `scale_pos_weight = (n_neg/n_pos)·spw_mul` |
| Interpretação | SHAP `TreeExplainer` no teste; agregação \|SHAP\| por ROI e atributo |

#### SVM linear (`colab/exp2_svm.py`)

```bibtex
@article{cortes-1995,
  title     = {{Support-vector networks}},
  author    = {Cortes, Corinna and Vapnik, Vladimir},
  journal   = {Machine learning},
  publisher = {},
  volume    = {20},
  number    = {3},
  pages     = {273--297},
  note      = {},
  doi       = {10.1007/BF00994018},
  issn      = {},
  year      = {1995}
}
```

| Item | Valor |
|------|--------|
| Modelo | `LinearSVC`, `dual="auto"`, `max_iter=50000` |
| Optuna | 30 trials; **`C`** ∈ [1e-4, 1e4] log |
| Score | AUC com `decision_function` (NCV e refit) |
| Interpretação | Média de \|coef.\| por fold → ROI / atributo |

#### ROCKET + L1 (`colab/exp2_rocket.py`)

```bibtex
@article{dempster-2020,
  title     = {{ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels}},
  author    = {Dempster, A. and Petitjean, F. and Webb, G.I.},
  journal   = {Alzheimer's \& Dementia},
  publisher = {},
  volume    = {34},
  number    = {},
  pages     = {1454--1495},
  note      = {},
  doi       = {10.1007/s10618-020-00701-z},
  issn      = {},
  year      = {2020}
}
```

| Item | Valor |
|------|--------|
| Entrada | `(n, p', 60)` após transpose para sktime |
| ROCKET | `num_kernels=2000`; **novo fit por fold interno** no Optuna (sem vazamento) |
| Classificador | `LogisticRegression` L1, `solver="saga"`, `max_iter=10000` |
| Optuna | 30 trials; **`C`** ∈ [1e-4, 1e4] log |

#### LSTM (`colab/exp2_lstm.py` → `exp_lstm_common.py`)

```bibtex
@article{hochreiter-1997,
  title     = {{Long short-term memory}},
  author    = {Hochreiter, Sepp and Schmidhuber, Jürgen},
  journal   = {Neural Computation},
  publisher = {},
  volume    = {34},
  number    = {},
  pages     = {1735--1780},
  note      = {},
  doi       = {10.1162/neco.1997.9.8.1735},
  issn      = {},
  year      = {1997}
}
```

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
| `tables/` | `metrics_per_fold.csv`, `oof_predictions.csv`, `fold_test_scores.csv`, importâncias, `feature_counts_fold0.csv`, `training_curves_fold{0..4}.csv`, `training_curves_mean.csv`, `run_meta.json` |
| `figures/` | Confusão OOF, ROC/PR, boxplot, SHAP/coef; **XGB / LSTM / ROCKET / SVM:** `training_curves_fold{0..4}.pdf`, `training_curves_mean.pdf` (logloss e acurácia treino/val no holdout tr_fit\|val) |
| `checkpoints/fold_{0..4}/` | Modelo, `preprocess.joblib`, `meta.json` (exp2) |
| `tables/demographics/`, `figures/demographics/` | Via `analyze_oof_demographics.py` (sexo F/M) |

### 4.6. Execução (orquestradores)

Raiz do repositório; Python: `.venv/bin/python` ou `python`.

```bash
# Experimento 2 — 4 runs balanced (4 modelos)
.venv/bin/python colab/run_exp_all.py

# Regenerar CSV harmonizado (após alterar features_selection.ipynb §3.1)
#   → executar células NeuroComBat (delta + unitário) no notebook

# Opcional: verificar checkpoint XGB fold 0 vs OOF
.venv/bin/python colab/verify_xgb_checkpoint.py
```

**Experimento 1 (manual, referência):**

```bash
.venv/bin/python colab/exp1_xgboost.py   # idem rocket, svm, lstm
.venv/bin/python colab/exp_plots.py
```

**Dependências:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `xgboost`, `shap`, `sktime`, `tensorflow` (LSTM).

### 4.7. Comparação exp1 vs exp2

| | Exp1 | Exp2 (ativo) |
|---|------|----------------|
| Features | Deltas entre pares | Absolutos + taxa desde baseline |
| Scripts | `exp1_*.py` | `exp2_*.py` |
| Pasta resultados | `colab/exp1/...` | `colab/exp2/...` |

Mesma arquitetura de CV, filtros, Optuna e utilitários (`exp_utils.py`, `exp_lstm_common.py`).

---

## 5. Trabalho futuro e legado

### 5.1. Em aberto (nível imagem / features)

- `features_displacement.py`: estatísticas de DF por ROI a partir dos NIfTI de delta.
- Agregar volume/radiomics por conjunto com deltas \(\Delta_{12}\), \(\Delta_{13}\) face à baseline (chave `ID_PT` + `COMBINATION_NUMBER`).
- **Harmonização:** NeuroComBat por fold externo (evitar estatísticas globais na validação); avaliar **Longitudinal ComBat** (Beer et al., §3.1) para tripletas temporais.
- Modelagem por **grafos** após fechar baseline tabular/sequencial.

### 5.1.1. TODO pós-treino (publicação / estatística / ablação)

Este bloco é um *checklist* para executar **após** finalizar os treinos `exp1_*` / `exp2_*`. A maioria das análises abaixo é possível **sem retreinar modelos**, usando apenas os artefactos por run em `tables/` (principalmente `oof_predictions.csv`, `metrics_per_fold.csv`, `fold_test_scores.csv`) e os `checkpoints/`.

#### A) Definir hipótese primária vs. sensibilidade (evitar *data dredging*)

- Escolher e documentar **uma configuração primária** para o artigo:
  - eixo: `exp1` (deltas) vs `exp2` (unitários)
  - `balanced` vs `unbalanced`
  - `harmonized` (NeuroComBat) vs `no_harmon`
  - modelo: XGBoost / SVM / ROCKET / LSTM
- Tratar as restantes combinações como **análises secundárias/sensibilidade** (suplemento/heatmap), com correção para comparações múltiplas quando aplicável.

#### B) Estatística com as previsões OOF (sem retreino)

- Consolidar uma tabela da grelha completa (até \(2\times 2\times 2\times 4 = 32\) configurações) com:
  - métricas OOF (AUC, AP, acc, F1) e dispersão por fold (`metrics_per_fold.csv`).
  - heatmap/figura de comparação (principal + suplemento).
- Estimar **IC 95% por bootstrap ao nível do paciente** (cluster bootstrap por `group_id` do `oof_predictions.csv`), reportando:
  - AUC/AP globais por configuração.
  - diferenças pareadas \(\Delta\)AUC/\(\Delta\)AP entre pares de interesse (ex.: `exp2` vs `exp1`, harmonizado vs não, balanced vs unbalanced) usando as mesmas amostras (`row_idx`) em OOF.
- Comparações múltiplas:
  - aplicar Bonferroni/FDR quando houver muitas comparações paralelas (ex.: grelha completa).
- Calibração e decisão (baseadas em OOF):
  - curvas de calibração / Brier score para a configuração primária.
  - (opcional) *decision curve analysis* (benefício líquido vs limiar).
- Nota metodológica: há **múltiplos conjuntos por paciente** (cluster). Para ICs/p-valores, preferir bootstrap por paciente em vez de assumir IID por amostra.

#### C) Ablação por ROI na configuração vencedora (exige novo treino, mas pode evitar Optuna)

Objetivo: quantificar impacto de cada ROI na performance, focando **apenas** na melhor abordagem (reduz custo e evita interpretabilidade difusa).

- **Leave-One-ROI-Out (LORO)** (recomendado para o corpo do artigo):
  - para cada ROI \(r\) (20 ROIs), remover os **3 slots** correspondentes (3 tempos/pares) do tensor e repetir o treino/avaliação no mesmo 5-fold SGK.
  - preferir ablação com hiperparâmetros **fixos** (usar `best_params` guardados nos `checkpoints/fold_k/meta.json`) para evitar re-Optuna 20×.
  - reportar \(\Delta\)AUC OOF = AUC(full) − AUC(sem ROI \(r\)), com IC 95% por bootstrap por paciente e distribuição por fold (boxplot).
- **LORO + re-Optuna**:
  - opcional só para 2–3 ROIs top (suplemento), devido ao custo.
- **Permutation ablation** (sanity check, sem retreino completo):
  - com o modelo já treinado (checkpoint), permutar/blindar features da ROI \(r\) apenas no teste do fold e medir queda de AUC.
  - mede dependência do modelo treinado (não substitui LORO), mas é rápido e útil como validação.
- Confrontar ablação (LORO/permutação) com interpretabilidade já exportada:
  - XGBoost/LSTM: `importance_shap_roi_mean.csv`
  - SVM/ROCKET: `importance_coef_roi_mean.csv`
  - verificar concordância entre ROIs com alto \|SHAP\|/\|coef.\| e ROIs com maior \(\Delta\)AUC.

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

| Tema | Secção | Chave BibTeX |
|------|--------|----------------|
| ANTs / ANTsPyNet | §2.1, §2.4, §2.7 | `tustison-2021` |
| Denoise NLM | §2.2 | `manjon-2010` |
| N4 bias | §2.3 | `tustison-2010` |
| MRQy | §2.6 | `sadri-2020` |
| ComBat (base) | §3.1 | `johnson-2007` |
| NeuroComBat (implementado) | §3.1 | `fortin-2018` |
| Longitudinal ComBat (referência) | §3.1 | `beer-2020` |
| IBSI / radiomics | §3.3 | `zwanenburg-2020` |
| Radiomics review | §3.3 | `lambin-2017` |
| PyRadiomics | §3.3 | `griethuysen-2017` |
| ADNI MRI standardization | §3.2 | `wyman-2013` |
| Longitudinal SIENA | §3.2 | `smith-2002` |
| Boundary Shift Integral | §3.2 | `freeborough-1997` |
| Longitudinal template (FreeSurfer) | §3.2 | `reuter-2012` |
| SHAP | §4.4 | `lundberg-2017` |
| Optuna | §4.4 | `akiba-2019` |

Literatura adicional citada no texto: Johnson et al. (2007) — ComBat original; Fortin et al. (2018) — NeuroComBat; Beer et al. (2020) — Longitudinal ComBat (não implementado).

---

*Para edição do pipeline de modelagem exp2, alterar os scripts em `colab/` e sincronizar [`exp2.md`](exp2.md); este README é o mapa do repositório.*

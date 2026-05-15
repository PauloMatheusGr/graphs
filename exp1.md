Tenho uma planilha de delta atributos de um conjunto com 3 imagens, cuja alocação está em csvs/abordagem_teste/all_delta_features_neurocombat.csv.

Cada 60 linhas dessa planilha eu tenho um datapoint, que é 20 regiões de interesse na coluna label vezes 3 diferenças entre as aquisições na coluna pair, que são 12=i2-i1, 13=i3-i1, 23=i3-i2.

Cada linha representa um vetor de atributos de uma região de interesse especifica em um temnpo de aquisição especifico. Para você saber qual imagem é essa linha olhe a coluna pair, para saber qual paciente olhe a coluna ID_PT, para saber qual região de interesse olhe as colunas roi+side+label. 

As colunas **t12**, **t13** e **t23** são o intervalo em meses entre as imagens 1, 2 e 3 (`t12=t2−t1`, `t13=t3−t1`, `t23=t3−t2`). Nos scripts `colab/exp1_*.py`, elas **ponderam os deltas** antes do treino: cada atributo numérico da linha é dividido pelo intervalo do par correspondente, obtendo uma **taxa de mudança por mês** (ver secção *Ponderação temporal* abaixo). A coluna **SEX** não é dividida.

A coluna SEX é o sexo do paciente, que deverá ser convertida para F=0 e M=1. 

O target é a coluna GROUP que precisará ser convertida também para sMCI=0 e pMCI=1.

As colunas COMBINATION_NUMBER	TRIPLET_IDX
são as quantidades de conjuntos (datapoint) de cada paciente.

O split é por paciente na coluna ID_PT.

Drope as colunas ID_IMG_i1	ID_IMG_i2	ID_IMG_i3	roi	side    DIAG	AGE	TIME_PROG	ID_IMG_ref	FIELD_STRENGTH	SLICE_THICKNESS	MANUFACTURER	MFG_MODEL	batch pois não são atributos.

As colunas de atributos são SEX	centroid_x	centroid_y	centroid_z	logjac_n	logjac_mean	logjac_std	logjac_p05	logjac_p50	logjac_p95	mag_n	mag_mean	mag_std	mag_p05	mag_p50	mag_p95	div_n	div_mean	div_std	div_p05	div_p50	div_p95	ux_n	ux_mean	ux_std	ux_p05	ux_p50	ux_p95	uy_n	uy_mean	uy_std	uy_p05	uy_p50	uy_p95	uz_n	uz_mean	uz_std	uz_p05	uz_p50	uz_p95	curlmag_n	curlmag_mean	curlmag_std	curlmag_p05	curlmag_p50	curlmag_p95	original_firstorder_10Percentile	original_firstorder_90Percentile	original_firstorder_Energy	original_firstorder_Entropy	original_firstorder_InterquartileRange	original_firstorder_Kurtosis	original_firstorder_Maximum	original_firstorder_Mean	original_firstorder_MeanAbsoluteDeviation	original_firstorder_Median	original_firstorder_Minimum	original_firstorder_Range	original_firstorder_RobustMeanAbsoluteDeviation	original_firstorder_RootMeanSquared	original_firstorder_Skewness	original_firstorder_TotalEnergy	original_firstorder_Uniformity	original_firstorder_Variance	original_glcm_Autocorrelation	original_glcm_ClusterProminence	original_glcm_ClusterShade	original_glcm_ClusterTendency	original_glcm_Contrast	original_glcm_Correlation	original_glcm_DifferenceAverage	original_glcm_DifferenceEntropy	original_glcm_DifferenceVariance	original_glcm_Id	original_glcm_Idm	original_glcm_Idmn	original_glcm_Idn	original_glcm_Imc1	original_glcm_Imc2	original_glcm_InverseVariance	original_glcm_JointAverage	original_glcm_JointEnergy	original_glcm_JointEntropy	original_glcm_MCC	original_glcm_MaximumProbability	original_glcm_SumAverage	original_glcm_SumEntropy	original_glcm_SumSquares	original_gldm_DependenceEntropy	original_gldm_DependenceNonUniformity	original_gldm_DependenceNonUniformityNormalized	original_gldm_DependenceVariance	original_gldm_GrayLevelNonUniformity	original_gldm_GrayLevelVariance	original_gldm_HighGrayLevelEmphasis	original_gldm_LargeDependenceEmphasis	original_gldm_LargeDependenceHighGrayLevelEmphasis	original_gldm_LargeDependenceLowGrayLevelEmphasis	original_gldm_LowGrayLevelEmphasis	original_gldm_SmallDependenceEmphasis	original_gldm_SmallDependenceHighGrayLevelEmphasis	original_gldm_SmallDependenceLowGrayLevelEmphasis	original_glrlm_GrayLevelNonUniformity	original_glrlm_GrayLevelNonUniformityNormalized	original_glrlm_GrayLevelVariance	original_glrlm_HighGrayLevelRunEmphasis	original_glrlm_LongRunEmphasis	original_glrlm_LongRunHighGrayLevelEmphasis	original_glrlm_LongRunLowGrayLevelEmphasis	original_glrlm_LowGrayLevelRunEmphasis	original_glrlm_RunEntropy	original_glrlm_RunLengthNonUniformity	original_glrlm_RunLengthNonUniformityNormalized	original_glrlm_RunPercentage	original_glrlm_RunVariance	original_glrlm_ShortRunEmphasis	original_glrlm_ShortRunHighGrayLevelEmphasis	original_glrlm_ShortRunLowGrayLevelEmphasis	original_glszm_GrayLevelNonUniformity	original_glszm_GrayLevelNonUniformityNormalized	original_glszm_GrayLevelVariance	original_glszm_HighGrayLevelZoneEmphasis	original_glszm_LargeAreaEmphasis	original_glszm_LargeAreaHighGrayLevelEmphasis	original_glszm_LargeAreaLowGrayLevelEmphasis	original_glszm_LowGrayLevelZoneEmphasis	original_glszm_SizeZoneNonUniformity	original_glszm_SizeZoneNonUniformityNormalized	original_glszm_SmallAreaEmphasis	original_glszm_SmallAreaHighGrayLevelEmphasis	original_glszm_SmallAreaLowGrayLevelEmphasis	original_glszm_ZoneEntropy	original_glszm_ZonePercentage	original_glszm_ZoneVariance	original_ngtdm_Busyness	original_ngtdm_Coarseness	original_ngtdm_Complexity	original_ngtdm_Contrast	original_ngtdm_Strength	original_shape_Elongation	original_shape_Flatness	original_shape_LeastAxisLength	original_shape_MajorAxisLength	original_shape_Maximum2DDiameterColumn	original_shape_Maximum2DDiameterRow	original_shape_Maximum2DDiameterSlice	original_shape_Maximum3DDiameter	original_shape_MeshVolume	original_shape_MinorAxisLength	original_shape_Sphericity	original_shape_SurfaceArea	original_shape_SurfaceVolumeRatio	original_shape_VoxelVolume

A seguir trago uma ideia ludica acerca da quantidade de dados, que não condiz com a realidade, i.e., é apenas para definir a modelagem.

Baseado nessa ideia monte dois scripts python, um para o modelo rocket e outro para o modelo xgboost.

Para a seleção de atributos faça dois filtros, um filtro de atributos altamente correlacionados (>0.9) e outro filtro para atributos de baixa variância.

Plote a quantidade de atributos Raw,  a quantidade de atributos após a filtragem da correlação, a quantidade de atributos após a filtragem da variancia.

Avalie a acurácia durante as epocas de treinamento e plote os gráficos da loss e da acurácia conforme for iterando.

Rode os treinamentos para testar.

Com 525 amostras para 9.600 dimensões ($60 \times 160$), você está em um terreno perigoso de *overfitting* (quando o modelo decora o treino e falha no teste). Para o XGBoost, isso é um desafio; para o ROCKET, é onde ele costuma brilhar.

Aqui está o roteiro técnico para as duas abordagens:

---

### Passo 0: Preparação do "Cubo" de Dados

Independentemente do modelo, primeiro transforme seu CSV flat em um array 3D no NumPy.

python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Carregar e Normalizar
# É vital escalar os dados ANTES do reshape
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features) 

# 2. Reshape para (Amostras, Timesteps, Features)
# X_3d shape: (525, 60, 160)
X_3d = X_scaled.reshape(525, 60, 160)
y = labels # Seu array de 0 e 1



---

### Opção 1: XGBoost (Abordagem Tabular)

Para o XGBoost, precisamos "achatar" o tempo. Cada ponto no tempo vira uma coluna nova.

1. *Flattening:* Transforme $(525, 60, 160)$ em $(525, 9600)$.
2. *Redução de Dimensionalidade (Opcional mas Recomendada):* Com 9.600 colunas para apenas 525 linhas, o XGBoost vai sofrer. Considere usar um PCA para reduzir para umas 100-200 componentes antes.
3. *Treinamento:* Use parâmetros conservadores para evitar que a árvore cresça demais.

python
from xgboost import XGBClassifier

# Achata os dados: 60 * 160 = 9600 colunas
X_flat = X_3d.reshape(525, -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, stratify=y)

model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,      # Profundidade baixa para evitar overfitting
    learning_rate=0.05,
    subsample=0.8,    # Treina com 80% das linhas por árvore
    colsample_bytree=0.3 # Usa apenas 30% das colunas por árvore (importante!)
)
model_xgb.fit(X_train, y_train)



---

### Opção 2: ROCKET (O Estado da Arte para Séries Temporais)

O ROCKET não "achata" o tempo de forma burra. Ele aplica milhares de convoluções aleatórias para extrair padrões de formato e frequência nas janelas temporais.

1. *Instalação:* Você precisará da biblioteca sktime.
2. *Transformação:* Ele criará novas features baseadas na resposta das convoluções.
3. *Classificador Simples:* O ROCKET funciona melhor com um RidgeClassifierCV (uma regressão linear com regularização forte).

python
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

# O sktime espera o formato (n_samples, n_columns, n_timepoints)
# Então vamos transpor nosso 3D: (525, 160, 60)
X_rocket_input = X_3d.transpose(0, 2, 1)

# 1. Aplicar o Rocket
rocket = Rocket(num_kernels=10000) # Gera 10k kernels aleatórios
rocket.fit(X_rocket_input)
X_train_transform = rocket.transform(X_rocket_input)

# 2. Classificar (RidgeClassifier é extremamente rápido e lida com muitas features)
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y)



---

### Qual escolher para os seus 525 dados?

| Critério | XGBoost | ROCKET |
| --- | --- | --- |
| *Risco de Overfitting* | Altíssimo (muitas colunas/poucas linhas) | Baixo (convoluções são aleatórias e robustas) |
| *Captura de Tempo* | Fraca (trata t=1 e t=60 como colunas independentes) | Excelente (foca em padrões morfológicos) |
| *Complexidade* | Fácil de implementar, difícil de tunar aqui | Requer sktime, mas geralmente "funciona de primeira" |

*Veredito:* Comece pelo *ROCKET*. Para dados neurocientíficos, onde a relação entre as regiões (colunas) e o tempo (linhas) é complexa e não-linear, o ROCKET costuma ter uma acurácia 15-20% superior a modelos tabulares como o XGBoost.

---

## Implementação e pipeline (`colab/exp1_xgboost.py`, `colab/exp1_rocket.py`, `colab/exp1_svm.py`)

Esta secção descreve o que está **efetivamente implementado** nos scripts Colab (pode divergir do roteiro ilustrativo acima, que é pedagógico).

### Dados e ficheiros

| Item | Valor nos scripts |
| --- | --- |
| Raiz do projeto | `ROOT = parents[1]` relativamente a `colab/` |
| CSV lido | `csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv` |
| Lista de atributos | Lida de `exp1.md`, linha que começa por `As colunas de atributos são` |
| Figuras e tabelas | `colab/exp1/{balanced|unbalanced}/{xgboost|rocket|svm}/` — subpastas `figures/`, `tables/` (CSV + `run_meta.json`); regenerar PDFs a partir dos CSV: editar `RUN_DIR` e títulos no topo de `colab/exp1_plots.py` e correr `python colab/exp1_plots.py` |

*(A primeira linha deste ficheiro pode referir outro caminho de CSV; o treino reproduzível usa o caminho da tabela acima.)*

### Tensor e alvo

- Cada amostra é um tensor **`X_3d` de forma `(n_amostras, 60, n_atributos)`**: 60 linhas = 20 ROIs × 3 pares (`12`, `13`, `23` em `PAIR_ORDER`), por `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)`.
- **`y`**: `GROUP` → sMCI=0, pMCI=1.
- **`groups`**: `ID_PT` (split por paciente).
- **`sex`**: codificado para downsample (F=0, M=1); coluna `SEX` entra na matriz de atributos como float conforme `exp1.md`.
- **`slot_labels`**: um rótulo por linha do bloco 60, formato `pair|roi|side|label`, usado para agregar SHAP por ROI / nome de atributo (XGBoost).

### Ponderação temporal (t12 / t13 / t23)

**Objetivo:** separar a magnitude do **delta** do **tempo entre aquisições**. Um mesmo delta absoluto em 6 meses implica mudança mais rápida do que em 24 meses.

**Mapeamento `pair` → coluna de tempo** (por linha do CSV / do tensor):

| `pair` | Coluna `dt` |
| --- | --- |
| `12` | `t12` |
| `13` | `t13` |
| `23` | `t23` |

**Fórmula** (aplicada em `colab/exp1_utils.py`, função `apply_temporal_rate_norm`, chamada a partir de `load_tensor` quando `temporal_rate_norm=True`):

Para cada linha do bloco de 60 (uma ROI × um par), e para cada coluna de atributo exceto `SEX`:

\[
x'_{i,j} = \frac{x_{i,j}}{\max(\mathrm{dt}_i,\,\varepsilon)}
\]

em que \(\mathrm{dt}_i\) é o valor de `t12`, `t13` ou `t23` conforme o `pair` da linha \(i\), e \(\varepsilon =\) `DT_EPSILON` (por defeito **0,5** meses nos scripts, constante `DT_EPSILON`).

**Ordem no pipeline:** leitura do CSV → montagem do tensor 60×atributos → **ponderação temporal** → validação cruzada → por fold: correlação → variância → z-score (`StandardScaler`). A ponderação **não** depende do fold (usa apenas `pair` e `t*` do CSV; é determinística por amostra).

**Flags nos scripts** (`exp1_xgboost.py`, `exp1_svm.py`, `exp1_rocket.py`):

- `TEMPORAL_RATE_NORM = True` — ativa a ponderação (passado a `load_tensor`).
- `DT_EPSILON = 0.5` — piso do denominador em meses.

Metadados do run (`tables/run_meta.json`): `temporal_rate_norm`, `dt_epsilon`.

### Validação cruzada (externa e interna)

**Não há** um split fixo único (ex.: 70 % / 15 % / 15 %) nem `train_test_split` nos scripts de treino. O desenho é **CV aninhada por paciente** (`ID_PT`); as métricas reportadas são **out-of-fold (OOF)** — cada amostra entra no teste externo exactamente uma vez.

| Nível | Método | `n_splits` | Papel |
|--------|--------|------------|--------|
| **Externo** | `StratifiedGroupKFold` | **5** | Métricas de teste (OOF) |
| **Interno (Optuna)** | `StratifiedGroupKFold` no treino externo | **5** (`INNER_NCV_SPLITS`) | Hiperparâmetros (média da AUC) |
| **Holdout interno** | `inner_train_val()` — 1.º fold de um SGK (até 5) | ~4 treino / 1 val | Early stopping (XGBoost), curvas, refit final |

Constantes nos scripts (`exp1_xgboost.py`, `exp1_svm.py`, `exp1_rocket.py`): `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` no externo; `INNER_NCV_SPLITS = 5`.

#### Percentagens aproximadas (por fold externo)

Com 5 folds externos e pacientes inteiros em cada parte:

| Conjunto | % do total | Notas |
|----------|------------|--------|
| **Teste (externo)** | **~20 %** | 1 de 5 folds |
| **Treino externo** | **~80 %** | Restante |

Dentro do treino externo, `inner_train_val` (em `colab/exp1_utils.py`) aplica outro SGK e usa o **primeiro** split:

| Conjunto | % do total | % do treino externo |
|----------|------------|---------------------|
| **`tr_fit`** (ajuste final / early stopping) | **~64 %** | ~80 % |
| **`val`** (holdout interno) | **~16 %** | ~20 % |

Resumo por fold: **~64 % treino · ~16 % validação · ~20 % teste** (em amostras; grupos = pacientes).

O **NCV interno do Optuna** (5 folds dentro dos ~80 % de treino externo) **não reserva um terceiro bloco fixo**: roda vários pares treino/val só para a média da AUC; o modelo final treina em `tr_fit` e avalia-se no teste externo (~20 %).

**Não existe** conjunto de teste holdout separado dos 5 folds — só OOF agregado.

#### Regras e flags

- **Externo:** `StratifiedGroupKFold`, **5 folds**, `shuffle=True`, `random_state=42`, agrupamento por `ID_PT`.
- **Opcional — downsample no treino externo:** flag `DOWNSAMPLE_GROUP_SEX` (por defeito **False** em XGBoost, SVM e ROCKET): equipara o número de **pacientes** por estrato **rótulo × sexo** dentro do treino de cada fold; **não altera** as frações 80/20 do fold.
- **Nested CV interno (Optuna):** `INNER_NCV_SPLITS = 5` folds `StratifiedGroupKFold` **dentro do treino externo** de cada fold. O Optuna maximiza a **média da AUC** nesses validadores internos (sem usar o teste externo).
- **Holdout para refit / curvas:** `inner_train_val` devolve `tr_fit` / `val` sobre o treino externo; o modelo final do fold usa esse par para **early stopping** (XGBoost) e para PDFs de curvas no fold 1.
- Com poucos pacientes únicos, `inner_train_val` / `inner_cv_splits` usam `min(5, n_pacientes)` folds; se o SGK falhar, fallback **80 % / 20 %** por ordem de grupo.

### Pré-processamento (por fold externo, alinhado ao `tr_fit` do holdout)

Ordem global e por fold:

0. **Ponderação temporal** na carga dos dados (`load_tensor`, ver acima) — antes de qualquer split.
1. **Correlação entre colunas** no flatten de `X_3d[tr_fit]`: limiar **0,9** (`CORR_THR`), seleção greedy.
2. **`VarianceThreshold(VAR_THR)`** com `VAR_THR = 0.0` (remove só colunas constantes no treino de ajuste).
3. **`StandardScaler`** ajustado nas linhas achatadas do tensor de treino de ajuste, aplicado a val e teste.

**Nota (NCV interno no XGBoost):** em cada trial Optuna e cada split interno, correlação + variância + scaler são **recalculados só no treino interno** daquele split (sem vazamento para a validação interna).

### `colab/exp1_xgboost.py` (tabular / XGBoost)

- **Achatar:** `(n, 60, p) → (n, 60·p)` para o classificador.
- **Hiperparâmetros (Optuna, 25 trials, TPE):** `spw_mul` (multiplicador sobre `scale_pos_weight = n_neg/n_pos` do treino relevante), `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `min_child_weight`, `gamma`.
- **Treino:** API nativa `xgboost.train` com `eval_metric=logloss`, até `N_ESTIMATORS_MAX = 500` árvores, **`EARLY_STOPPING_ROUNDS = 50`**, validação no conjunto interno apropriado; o booster final é carregado em `XGBClassifier` via `load_model` (compatível com SHAP).
- **Métricas de teste por fold:** acurácia, AUC, F1 (limiar 0,5 nas probabilidades).
- **Figuras:** em `figures/`: contagens de atributos; curvas **logloss** e **acurácia na validação** vs número de árvores (só fold 1); matriz de confusão **OOF** agregada; **ROC** e **PR** (média ±1 dp entre folds); boxplot Acc/AUC/F1; barras horizontais **|SHAP|** agregadas por ROI e por nome de atributo; **SHAP summary** no maior fold de teste. **Não** se gera gain/importância nativa XGB.

### `colab/exp1_rocket.py` (ROCKET + regressão logística L1)

- **Entrada ROCKET:** tensor transposto `(n, n_atributos, 60)` (eixo de “tempo” = 60 posições).
- **Pré-processamento:** o mesmo pipeline correlação + variância + scaler que no XGBoost, no `tr_fit` do fold, depois transposição.
- **NCV interno:** para cada um dos `INNER_NCV_SPLITS` splits internos, **novo** `Rocket(num_kernels=10_000)` ajustado só no treino interno; guardam-se `Z_tr` e `Z_val` transformados. O Optuna **não** refaz o ROCKET a cada trial — só a regressão.
- **Classificador:** `LogisticRegression(penalty="l1", solver="saga", max_iter=10000)`, hiperparâmetro **`C`** em escala log \([10^{-4}, 10^4]\), **25 trials** Optuna, objetivo = **média da AUC** nos folds internos.
- **Refit final:** melhor `C`, ajuste em `Z_tr` do `tr_fit` do holdout (ROCKET desse fold).
- **Figuras:** contagens de atributos; no fold 1, acurácia na validação vs **`log10(C)`** (`figures/l1_C_validation_curve.pdf`); matriz de confusão OOF; ROC/PR; boxplot de métricas. **Sem SHAP** neste script (espaço ROCKET de alta dimensão + modelo linear).

### `colab/exp1_svm.py` (tabular / SVM linear)

- **Mesmo pré-processamento** que o XGBoost (achatado, correlação, variância, `StandardScaler` no `tr_fit` do fold).
- **Hiperparâmetros (Optuna, 25 trials):** `C` em escala log \([10^{-4}, 10^4]\); objetivo = média da AUC nos folds internos (`decision_function` na validação interna).
- **Modelo final:** `LinearSVC` ajustado em `tr_fit`; no teste usam-se `predict` e **sigmóide** da `decision_function` como score para AUC/PR e limiar 0,5 para F1/acurácia alinhados ao XGBoost.
- **Importância:** agregação por ROI / atributo com **média dos \|coef.\|** por fold (análogo ao agregado de \|SHAP\| no XGBoost).
- **Figuras:** contagens de atributos; confusão OOF; ROC/PR; boxplot; barras por ROI e por atributo (`coef_top_*.pdf`).

### `colab/exp1_plots.py`

Lê `tables/*.csv` de um run e grava PDFs em `figures/`. **Sem argumentos de linha de comando:** defina `RUN_DIR` e os textos dos gráficos na secção `CONFIG` no início do ficheiro, depois execute o script. Inclui o gráfico de contagens de atributos a partir de `tables/feature_counts_fold0.csv` (gravado no fold 1 dos scripts de treino; volte a correr o treino se o CSV não existir).

### Dependências relevantes

Python: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `xgboost`, `shap` (XGBoost), `sktime` (ROCKET).

### Execução

Na raiz do repositório (com ambiente que tenha as dependências):

```bash
python colab/exp1_xgboost.py
python colab/exp1_rocket.py
python colab/exp1_svm.py
python colab/exp1_plots.py
```
Tenho uma planilha de atributos absolutos por imagens individuais de um conjunto com 3 imagens, cuja alocação está em /mnt/study-data/pgirardi/graphs/csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv.

Cada 60 linhas dessa planilha eu tenho um datapoint, que é 20 regiões de interesse na coluna label vezes 3 aquisições na coluna **pair**: `1` = imagem 1 (baseline), `2` = imagem 2, `3` = imagem 3.

Cada linha representa um vetor de **atributos absolutos** de uma região de interesse numa aquisição específica. Para saber qual imagem é essa linha olhe a coluna `pair`, para saber qual paciente olhe a coluna `ID_PT`, para saber qual região de interesse olhe as colunas `roi`+`side`+`label`.

As colunas **t12**, **t13** e **t23** são o intervalo em meses entre as imagens (`t12=t2−t1`, `t13=t3−t1`, `t23=t3−t2`). Nos scripts `colab/exp2_*.py`, elas entram na **ponderação temporal** (taxa de mudança desde a imagem 1); ver secção *Ponderação temporal* abaixo. A coluna **SEX** não é dividida pelo tempo.

A coluna SEX é o sexo do paciente, que deverá ser convertida para F=0 e M=1. 

O target é a coluna GROUP que precisará ser convertida também para sMCI=0 e pMCI=1.

As colunas COMBINATION_NUMBER	TRIPLET_IDX
são as quantidades de conjuntos (datapoint) de cada paciente.

O split é por paciente na coluna ID_PT.

Drope as colunas ID_IMG_i1	ID_IMG_i2	ID_IMG_i3	ref_tag	roi	side    DIAG	AGE	TIME_PROG	ID_IMG_ref	FIELD_STRENGTH	SLICE_THICKNESS	MANUFACTURER	MFG_MODEL	batch pois não são atributos.

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

## Segmento ativo (desenvolvimento)

O **experimento 2** (atributos absolutos + `baseline_rate`) é o segmento em que se aplicam melhorias de pipeline (métrica AP, checkpoints, demografia OOF, plots). O experimento 1 (`colab/exp1/`, deltas) permanece como referência histórica; não é re-treinado neste ciclo.

## Métricas de avaliação (teste por fold e OOF)

| Métrica | Definição | Leitura com ~79% sMCI nos datapoints |
| --- | --- | --- |
| **Acc** | Acurácia com limiar 0,5 | Pode ser alta prevendo só sMCI (classe 0) |
| **AUC** | Área sob a curva ROC | Capacidade de ranquear pMCI acima de sMCI; insensível ao limiar |
| **F1** | F1 da classe positiva (pMCI = 1) | Foco na minoria |
| **AP** | Average precision (classe 1) | Muito informativa com poucos positivos |

Todas entram em `tables/metrics_per_fold.csv` após re-treino. OOF agregado em `tables/oof_predictions.csv` e `figures/confusion_oof.pdf`.

## Artefactos por run

| Pasta / ficheiro | Conteúdo |
| --- | --- |
| `figures/` | Confusão OOF, ROC/PR, boxplot métricas, SHAP/coef, curvas (fold 0) |
| `tables/` | `metrics_per_fold.csv`, `oof_predictions.csv`, `fold_test_scores.csv`, importâncias |
| `checkpoints/fold_{0..4}/` | Modelo final do fold (`model.json` / `.joblib` / `.keras`), `preprocess.joblib`, `meta.json` |
| `tables/demographics/` | Métricas OOF por sexo (via `colab/analyze_oof_demographics.py`) |
| `figures/demographics/` | Matrizes de confusão por F/M |
| `run_meta.json` | Metadados do run (`metrics_schema`, `checkpoints`, etc.) |

### Ablação LOO (XGBoost balanced)

Após o baseline em `colab/exp2/balanced/xgboost/`:

```bash
python colab/run_roi_ablation_exp2.py
```

- Uma pasta por ROI removida (máscara a zero): `colab/exp2/balanced/xgboost_ablation/drop_<roi>/` com o mesmo `figures/` e `tables/` do treino.
- Resumo: `xgboost_ablation/ablation_summary.csv` e `ablation_delta_auc_oof.pdf`.
- Hiperparâmetros reutilizados do baseline (`ABLATION_SKIP_OPTUNA=1`); variável `ABLATION_FORCE_OPTUNA=1` para Optuna em cada ROI.
- Uma execução manual: `ABLATION_DROP_ROIS=hippocampus RUN_DIR=colab/exp2/balanced/xgboost_ablation/drop_hippocampus DOWNSAMPLE_GROUP_SEX=1 python colab/exp2_xgboost.py` (com `ABLATION_BASELINE_RUN_DIR` e `ABLATION_SKIP_OPTUNA` se aplicável).

## Implementação e pipeline (`colab/exp2_xgboost.py`, `colab/exp2_rocket.py`, `colab/exp2_svm.py`, `colab/exp2_lstm.py`)

Esta secção descreve o que está **efetivamente implementado** nos scripts Colab (espelho do experimento 1, com CSV unitário e ponderação `baseline_rate`).

### Dados e ficheiros

| Item | Valor nos scripts |
| --- | --- |
| Raiz do projeto | `ROOT = parents[1]` relativamente a `colab/` |
| CSV lido | `csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv` |
| Lista de atributos | Lida de `exp2.md`, linha que começa por `As colunas de atributos são` |
| Utilitários | `colab/exp_utils.py` (`load_tensor`, CV, plots) |
| Figuras e tabelas | `colab/exp2/{balanced\|unbalanced}/{xgboost\|rocket\|svm\|lstm}/` — `figures/`, `tables/`, `checkpoints/`, `run_meta.json`; PDFs: `colab/exp_plots.py`; demografia: `colab/analyze_oof_demographics.py` |

### Tensor e alvo

- Cada amostra é um tensor **`X_3d` de forma `(n_amostras, 60, n_atributos)`**: 60 linhas = 20 ROIs × 3 imagens (`1`, `2`, `3` em `PAIR_ORDER`), por `(ID_PT, COMBINATION_NUMBER, TRIPLET_IDX)`.
- **`y`**: `GROUP` → sMCI=0, pMCI=1.
- **`groups`**: `ID_PT` (split por paciente).
- **`sex`**: F=0, M=1; `SEX` entra na matriz de atributos.
- **`slot_labels`**: `pair|roi|side|label` por linha (SHAP / coef no XGBoost e SVM).

### Ponderação temporal (t12 / t13 / t23) — atributos absolutos

**Objetivo:** usar o tempo entre aquisições sem dividir volumes absolutos por meses (o que não é interpretável). Em vez disso, para as imagens 2 e 3 calcula-se a **taxa de mudança desde a baseline** (imagem 1, `pair=1`), alinhada à lógica do exp1 (taxa por mês).

**Baseline:** `pair = 1` (primeira imagem da tripla) — valores **absolutos** mantidos.

**Mapeamento temporal:**

| `pair` | Papel | Transformação (por ROI e atributo, exceto `SEX`) |
| --- | --- | --- |
| `1` | baseline | \(x' = x\) (absoluto) |
| `2` | imagem 2 | \(x' = (x - x_{\mathrm{baseline}}) / \max(t_{12}, \varepsilon)\) |
| `3` | imagem 3 | \(x' = (x - x_{\mathrm{baseline}}) / \max(t_{13}, \varepsilon)\) |

\(x_{\mathrm{baseline}}\) é o valor na **mesma ROI** (`roi`, `side`, `label`) na linha com `pair=1` do mesmo datapoint. \(\varepsilon =\) `DT_EPSILON` (por defeito **0,5** meses).

**Implementação:** `colab/exp_utils.py`, função `apply_temporal_baseline_rate`, activada por `load_tensor(..., temporal_mode="baseline_rate")`.

**Nota:** não se usa `t23` nesta variante (mudança img2→img3); usa-se sempre o referencial desde a **imagem 1** (`t13` para `pair=3`).

**Ordem no pipeline:** CSV → tensor 60×atributos → **baseline_rate** → CV → por fold: correlação (0,9) → variância → z-score.

**Flags nos scripts** (`exp2_xgboost.py`, `exp2_svm.py`, `exp2_rocket.py`, `exp2_lstm.py`):

- `TEMPORAL_MODE = "baseline_rate"`
- `DT_EPSILON = 0.5`
- `PAIR_ORDER = ["1", "2", "3"]`

`TIME_PROG` **não** entra como feature (vazamento de rótulo); ver `readme.md` §4.1.

Metadados: `tables/run_meta.json` inclui `temporal_mode`, `dt_epsilon`.

### Validação cruzada (externa e interna)

**Igual ao experimento 1** nos scripts `exp2_xgboost.py`, `exp2_svm.py`, `exp2_rocket.py`, `exp2_lstm.py` (mesma lógica em `colab/exp_utils.py` e `colab/exp_lstm_common.py`). **Não há** split fixo 70/15/15 nem `train_test_split`; métricas = **OOF** em 5 folds por `ID_PT`.

| Nível | Método | `n_splits` | Papel |
|--------|--------|------------|--------|
| **Externo** | `StratifiedGroupKFold` | **5** | Teste (OOF) |
| **Interno (Optuna)** | `StratifiedGroupKFold` no treino externo | **5** (`INNER_NCV_SPLITS`) | Hiperparâmetros (média da AUC) |
| **Holdout interno** | `inner_train_val()` — 1.º fold SGK (até 5) | ~4 treino / 1 val | Early stopping, curvas, refit |

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; `INNER_NCV_SPLITS = 5`.

#### Percentagens aproximadas (por fold externo)

| Conjunto | % do total |
|----------|------------|
| **Teste (externo)** | **~20 %** |
| **`tr_fit`** | **~64 %** |
| **`val`** (holdout interno) | **~16 %** |

Por fold: **~64 % treino · ~16 % validação · ~20 % teste** (pacientes inteiros em cada parte). O NCV interno do Optuna roda **dentro** dos ~80 % de treino externo e não define um terceiro holdout fixo além de `tr_fit`/`val`.

#### Regras e flags

- **Externo:** 5 folds, `random_state=42`, agrupamento por `ID_PT`.
- **Downsample opcional:** `DOWNSAMPLE_GROUP_SEX` (por defeito **True** nos scripts `exp2_*.py`; `False` em `unbalanced` via `run_exp2_all.py`); só reduz amostras no treino externo, sem mudar ~20% de teste do fold.
- **Nested CV (Optuna):** `INNER_NCV_SPLITS = 5`; objetivo = média da AUC nos folds internos.
- **Holdout `tr_fit` / `val`:** early stopping (XGBoost) e curvas no fold 1.
- Poucos pacientes: até `min(5, n_pacientes)` folds internos; fallback 80/20 se o SGK falhar.

### Pré-processamento (por fold)

0. **Ponderação temporal** na carga (`baseline_rate`).
1. Correlação > **0,9** (`CORR_THR`).
2. `VarianceThreshold(0.0)` (`VAR_THR`).
3. `StandardScaler` (z-score, fit só em `tr_fit`).

### Modelos

- **`exp2_xgboost.py`:** tabular achatado, Optuna, SHAP, mesmas figuras que exp1.
- **`exp2_rocket.py`:** ROCKET + L1 logística, Optuna em `C`.
- **`exp2_svm.py`:** `LinearSVC`, Optuna em `C`, |coef.| por ROI/atributo.
- **`exp2_lstm.py`:** LSTM Keras em `(n, 3, 20·p)` após `baseline_rate`; Optuna; NCV interno; SHAP; `colab/exp_lstm_common.py`. **GPU:** `LSTM_DEVICE=gpu`, `LSTM_GPU_INDEX=0|1`; `use_cudnn=False` + XLA auto-jit off (evita erro cuDNN no driver 570 / TF 2.21).

### Execução

```bash
python colab/exp2_xgboost.py
python colab/exp2_rocket.py
python colab/exp2_svm.py
python colab/exp2_lstm.py
python colab/exp_plots.py
```

Com `DOWNSAMPLE_GROUP_SEX = False` em `exp2_lstm.py` (e nos outros scripts), os resultados vão para `colab/exp2/unbalanced/lstm/`. Requer `tensorflow` além das dependências do exp1.

Comparar com o experimento 1: mesmo pipeline de CV e filtros; difere o CSV (deltas vs absolutos), `PAIR_ORDER` e modo temporal (`delta_rate` no exp1 vs `baseline_rate` no exp2).
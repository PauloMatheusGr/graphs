Tenho uma planilha de atributos absolutos por imagens individuais de um conjunto com 3 imagens, cuja alocação está em /mnt/study-data/pgirardi/graphs/csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv.

Cada 60 linhas dessa planilha eu tenho um datapoint, que é 20 regiões de interesse na coluna label vezes 3 diferenças entre as aquisições na coluna pair, que são 1=imagem 1, 2=imagem 2, 3=imagem 3.

Cada linha representa um vetor de atributos de uma região de interesse especifica em um temnpo de aquisição especifico. Para você saber qual imagem é essa linha olhe a coluna pair, para saber qual paciente olhe a coluna ID_PT, para saber qual região de interesse olhe as colunas roi+side+label. 

As colunas t12 t13 t23 é a diferença de tempo em meses entre as imagens 1 2 3, i.e., t12=t2-t1,t13=t3-t1,t23=t3-t2, que são importantes para ponderar os atributos, pois quanto mais próximo (menos valor pro delta tempo) mais similares as imagens serão, e quanto mais distante (maior valor pro delta tempo) mais alterações estruturais as imagens terão entre si, e consequentemente, mais diferenças entre os atributos. Será uma maneira de reduzir as diferenças dos atributos em imagens próximas e aumentar as diferenças dos atributos em imagens distantes. 

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
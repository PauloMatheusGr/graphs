O comando “mínimo” que você quer (resultados finais: melhores modelos + melhores atributos/ROIs agregados)
Se sua intenção é: avaliar modelos, escolher os melhores, e ter SHAP agregado (média dos folds) dos melhores modelos, eu usaria assim:

python colab/sklearn_models_teste.py \
  --csv "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv" \
  --n-splits 10 \
  --inner-fold 5 \
  --top-k 3 \
  --seed 123 \
  --feature-selection two_stage \
  --fs-k-pre 50 \
  --fs-k-final 30 \
  --balance downsample \
  --remove-constant \
  --corr-threshold 0.90 \
  --shap \
  --shap-folds all \
  --shap-samples 100 \
  --shap-background 50 \
  --outdir "/mnt/study-data/pgirardi/graphs/colab/outputs"
  
O que você usa como “resultado final”:
Melhores modelos (métricas): leaderboard_cv_agg_by_model_ranked_by_accuracy.csv
Melhores atributos por modelo (SHAP médio entre folds): shap/shap_feature_importance_agg_by_model.csv
Melhores ROIs por modelo (SHAP médio entre folds): shap/shap_roi_importance_agg_by_model.csv
Comentando seu comando linha a linha (e o que cada parâmetro aceita)
python colab/sklearn_models_teste.py \
Executa o script.
  --csv "..." \
CSV de entrada. Precisa ter pelo menos: ID_PT, GROUP, SEX + colunas de features.
Também pode ter roi e label se você quiser ranking de ROI no SHAP.
  --n-splits 10 \
Folds externos (avaliação final). Usa StratifiedGroupKFold com grupo = ID_PT (não mistura paciente entre treino/teste).
Valores típicos: 5, 10.
  --inner-fold 5 \
Folds internos (ranking de modelos dentro do treino do fold externo).
Valores típicos: 3, 5.
  --top-k 3 \
Em cada fold externo, só os top 3 modelos do ranking interno são avaliados no teste externo (e têm SHAP, se ligado).
Se você quer menos coisa/mais rápido: --top-k 1.
  --seed 123 \
Semente para splits e alguns modelos.
  --feature-selection two_stage \
Seleção de features por fold externo (sem vazamento).
Opções:
none
kbest
sfs
two_stage
  --fs-k-pre 50 \
Só usado no two_stage: tamanho do “pool” do estágio 1 (KBest).
Vai de 1 até o nº de features disponíveis.
  --fs-k-final 30 \
Só usado no two_stage: nº final de features (SFS em cima do pool).
Vai de 1 até fs-k-pre.
  --balance downsample \
Balanceia o treino por paciente para igualar #pacientes em estratos GROUP+SEX.
Opções: none, downsample.
  --remove-constant \
Remove features constantes (variância zero) no treino do fold e aplica no teste do fold.
É flag (liga/desliga).
  --corr-threshold 0.90 \
Remove features muito correlacionadas no treino do fold (ex.: 0.90, 0.95).
0.0 desliga.
  --shap \
Liga SHAP (features e, se existir roi/label, também ROI).
  --shap-folds all \
Novo: quais folds terão SHAP.
Valores:
all (recomendado se você quer média robusta)
ou lista: 1,2,3 (se quiser acelerar e ainda ter uma “média parcial”).
  --shap-samples 100 \
Quantas amostras do teste do fold usar no SHAP (quanto menor, mais rápido).
Típico: 50–300.
  --shap-background 50 \
Quantas amostras do treino do fold usar como background.
Típico: 50–200.
  --outdir "..." 
Pasta base onde ele cria uma subpasta com timestamp e salva tudo.
Como testar modelos individualmente (sem se perder)
Use --models com os nomes exatos dos modelos.

Exemplos:

Rodar só um modelo (e garantir que ele seja avaliado):
python colab/sklearn_models_teste.py --models "SVM - RBF Kernel" --top-k 1 --shap --shap-folds all
Rodar um conjunto pequeno:
python colab/sklearn_models_teste.py --models "Logistic Regression,SVM - RBF Kernel,Random Forest Classifier" --top-k 3 --shap --shap-folds all
Nomes que existem hoje no script (principais):

Logistic Regression
Extra Trees Classifier
Naive Bayes
Quadratic Discriminant Analysis
K Neighbors Classifier
Ridge Classifier
Linear Discriminant Analysis
Random Forest Classifier
Ada Boost Classifier
Decision Tree Classifier (no script está como GradientBoosting com nome errado; se quiser eu arrumo o rótulo)
Gradient Boosting Classifier
SVM - Linear Kernel
SVM - RBF Kernel
Dummy Classifier
Se você quiser, eu implemento uma flag --list-models para o script imprimir essa lista automaticamente (aí você não precisa copiar/colar nomes).
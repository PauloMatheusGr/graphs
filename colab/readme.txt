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

TODO 

Vou olhar rapidamente o cabeçalho e algumas linhas do seu CSV all_delta_features_neurocombat.csv para sugerir a melhor padronização (quais colunas são metadados vs features) e como reformatar para o formato de entrada dos modelos (sklearn / CNN / LSTM) sem vazar informação.

Sua planilha não está “raw errada” por ter muitos números — ela está “errada para modelagem” porque o formato atual é long/tidy e mistura:

estrutura do experimento (ex.: COMBINATION_NUMBER, pair, roi, side, IDs das imagens…)
metadados (scanner/site/batch…)
features de fato (ex.: logjac_*, mag_*, original_*, strain_* etc.)
E, do jeito que está, qualquer pipeline que “pega tudo que é numérico” vai mesmo tratar COMBINATION_NUMBER, t12/t13/t23, coordenadas etc. como atributos.

A solução é padronizar uma única entrada por amostra (um “conjunto/triplet”) e separar explicitamente quais colunas são features.

Como transformar seus dados em entrada de modelo (sem perder o sentido do triplet)
1) Defina a “amostra” (1 linha = 1 conjunto)
Pelo seu cabeçalho, faz sentido que 1 amostra = (ID_PT, COMBINATION_NUMBER) (e possivelmente TRIPLET_IDX se existir mais de um triplet por combinação).

Então você quer transformar o CSV long em wide, agregando:

20 ROIs (roi)
3 pares (pair = 12/13/23)
muitas features radiômicas/deformação (logjac_*, mag_*, original_*, …)
Ou seja, cada amostra vira um vetor grande com colunas do tipo:

roi=hippocampus|pair=12|logjac_mean
roi=hippocampus|pair=13|logjac_mean
…
repetido para cada ROI e cada feature.
Isso vira um 
X
X pronto para o sklearn_models_teste.py (modelos tabulares).

2) Separe colunas estruturais/metadados vs features
Com o seu cabeçalho (linha 1), eu sugeriria:

Chaves/estrutura (NÃO features):
ID_PT, COMBINATION_NUMBER, TRIPLET_IDX
pair, roi, side
ID_IMG_i1, ID_IMG_i2, ID_IMG_i3, ID_IMG_ref
t12, t13, t23 (isso é estrutura do par/tempo; se entrar, pode vazar “qual par é qual”)
Alvo e estratificação (NÃO features):
GROUP (parece ser sua classe sMCI/pMCI), label (parece numérica), DIAG
SEX (você decide se entra como feature; no seu script ela entra na estratificação)
Metadados de aquisição (em geral eu não colocaria como feature no começo):
FIELD_STRENGTH, SLICE_THICKNESS, MANUFACTURER, MFG_MODEL, batch
(esses podem virar “atalho” para scanner/site em vez de doença)
Features de fato (candidatas a entrar no modelo):
tudo que começa com logjac_, mag_, div_, ux_, uy_, uz_, curlmag_, strain_, original_
Você pode começar com um whitelist por prefixos (mais seguro) do que “drop algumas”.

Como padronizar para usar nos 3 scripts (sklearn / CNN / LSTM)
A) Para @colab/sklearn_models_teste.py (tabular)
Faça wide + flatten: 
X
X com dimensão 
(
n
_
a
m
o
s
t
r
a
s
,
20
×
3
×
n
_
f
e
a
t
u
r
e
s
)
(n_amostras,20×3×n_features).

CV: sempre GroupKFold/StratifiedGroupKFold por ID_PT (porque o mesmo paciente tem vários conjuntos).
Balanceamento: também por paciente (como você já vinha fazendo).
SHAP: fica perfeito nesse formato; você consegue SHAP por feature “roi|pair|atributo”.
B) Para CNN (@colab/CNN_Example.ipynb)
Em vez de flatten, mantenha um tensor:

Opção mais natural aqui: (pair, roi, feat) = (3, 20, F)
Trate pair como canais (3 canais) e aplique conv ao longo da dimensão ROI (20).
Ou trate como “imagem” 2D: altura=20 (roi), largura=F (features), canais=3 (pairs).
Isso faz sentido se você quer que a rede aprenda padrões “espaciais” entre ROIs.

C) Para LSTM (@colab/lstm_example.py)
Aqui é a parte crítica: LSTM faz sentido quando você tem sequência temporal com passos ordenados.

No seu dataset, você tem deltas (12, 13, 23) — isso é uma representação dinâmica, mas não é uma sequência temporal linear (porque 23 depende de 1 e 2; e 12/13/23 não são “t1,t2,t3” como observações independentes).

Se você não tem os valores originais por tempo (i1,i2,i3) e só tem deltas, eu acho que:

LSTM não é a melhor primeira escolha.
Um MLP/tabular ou um CNN com canais=pair tende a ser mais coerente.
Se você tiver (ou puder construir) features em cada tempo i1,i2,i3, aí sim LSTM/Transformer fica bem mais justificável.

Modelos que eu sugiro testar (ordem prática)
Comece simples e forte para tabular, depois parta para deep se necessário:

1) Baselines tabulares (fortes e interpretáveis)
Regressão logística (L1/L2/ElasticNet) — ótima para high-dim, dá estabilidade
Linear SVM
Gradient Boosting / XGBoost / LightGBM / CatBoost (se disponível) — geralmente muito fortes em tabular
2) Modelos de árvore clássicos
RandomForest / ExtraTrees (podem funcionar, mas em high-dim às vezes ficam instáveis)
3) Deep (se você tiver amostras suficientes)
MLP em cima do flatten (muito competitivo e simples)
CNN 1D/2D com (3,20,F) como descrito acima
Eu deixaria LSTM por último (ou só se você reconstruir série i1→i2→i3).

O que você precisaria me dizer para eu cravar a padronização “oficial”
Só 4 decisões objetivas:

A amostra é (ID_PT, COMBINATION_NUMBER) ou precisa incluir TRIPLET_IDX também?
Label final: usar GROUP (pMCI/sMCI) ou label (numérico)? Qual é o ground truth correto?
side: entra como parte da chave (ou ROI já inclui lado)? Hoje você tem roi=hippocampus e side=L/R, então o “ROI efetivo” provavelmente é roi+side.
Quais metadados você quer excluir com certeza (ex.: batch, MANUFACTURER, etc.)?
Com isso, dá para padronizar um “construtor de dataset” único que alimenta: sklearn (flatten), CNN (tensor), e (se fizer sentido) LSTM.
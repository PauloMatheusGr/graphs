Preciso criar meus grafos em que cada grafo unificado seja a representação de um conjunto com 3 imagens conforme descrito pelo arquivo @plano_grafos.md .

Preciso que você forneça omo sugestão de implementação todas as ferramentas computacionais necessárias e os meios  necessários para eu gerar todos os grafos descritos pelo arquivo @cj_data_abordagem_1_sMCI_pMCI.txt . 

Padronize as descrições, pois o arquivo @cj_data_abordagem_1_sMCI_pMCI.txt piossui todos os conjuntos possiveis com 3 imagens, ou seja, os grafos gerados serão aproveitados para executar os experimentos em que os conjuntos serão obtidos pelo arquivo @cj_data_abordagem_4_sMCI_pMCI.txt .

Os grafos deverão ser gerados em "/mnt/study-data/pgirardi/graphs/graphs"

crie um script python que gere esses grafos, e nele, separe as funções necessárias para criar cada grafo (intra e inter-tempos) e o grafo unificado. 

os atributos 

########################

Esse plano é parar criar um classificador binário utilizando o modelo LSTM, em que o arquivo "/mnt/study-data/pgirardi/graphs/csvs/features_all_abordagem_4 copy_zscore.csv" contém os atributos, em que cada linha é uma região de interesse, as colunas ID_PT	COMBINATION_NUMBER	TRIPLET_IDX	GROUP	TIME_PROG	pair	ID_IMG_i1	ID_IMG_i2	ID_IMG_i3	ID_IMG_base	ID_IMG_follow	roi	side	label   disp_ID_IMG_i1	disp_ID_IMG_i2	disp_ID_IMG_i3.

são do header e não devem ser utilizadas para treinamento, as demais colunas são os atributos. 

cada datapoint é constituido por varias linhas e são agrupados formando o mesmo conjunto com 3 imagens sequenciais caso os valores das 3 colunas ID_PT	COMBINATION_NUMBER	TRIPLET_IDX sejam iguais.

O rótulo de cada atributo está na coluna GROUP e considere sMCI a classe negativa e pMCI a classe postiva para a classificação, nesse caso desconsidere os CN que não devem ser utilizados como datapoints.

O conjunto de dados está desbalanceado tanto na relação sMCI e pMCI (coluna GROUP) quanto por sexo (coluna SEX) masculino e feminino.

Deve utilizar um amostrador aleatório ponderado (wheighed random sampler) para balancear a epoca com os dados fornecidos. 

O split deverá ser feito considerando 80% dos pacientes válidos (coluna ID_PT) para treino e 20% para validação.

Os pacientes válidos são somente sMCI ou pMCI (coluna GROUP).

A coluna pair indica quais diferenças entre duas imagens estão sendo comparadas, e.g., 12 é da imagem ID_IMG_i1 para ID_IMG_i2 e 13 é da imagem ID_IMG_i1 para ID_IMG_i3. 

Os demais atributos é referente a essa comparação. 

O modelo deverá treinar por 100 épocas ou até saturar. 

Você deverá testar os valores de learning rate para encontrar o espaço ideal. 

Utilize o learning rate com função linear de decaimento.

Use estratégias de regularização wheigth decay e dropout.

O resultado da validação deverá ser impresso no terminal em cada fim de época. 

As métricas utilizadas para avaliação serão f1-score, acurácia, recall, e precisão. E deverão ser avaliadas em conjunto de treino e conjunto de validação em cada epoca.

Deverá ser utililzado o wandb para registro de metricas. 


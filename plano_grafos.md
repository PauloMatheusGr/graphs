Prompt de Geração: Construção de Grafos Espaço-Temporais para Neuroimagem
Objetivo

Gerar um script Python que transforme dados tabulares (extraídos de 3 imagens de RM longitudinais: t1​,t2​,t3​) em objetos de grafo compatíveis com PyTorch Geometric. O objetivo final é alimentar uma rede Spatio-Temporal Graph Convolutional Network (ST-GCN).
Definição do Conjunto de Dados

Cada amostra é um conjunto CJ={i1​,i2​,i3​}, onde it​ representa a aquisição no tempo t.

    Nós (V): 20 Regiões de Interesse (ROIs), divididas entre Hemisfério Esquerdo (L) e Direito (R).

    Atributos dos Nós (X): Vetor de características resultante da seleção de atributos (Radiômica + Volumetria).

Estrutura dos Grafos a serem Gerados
1. Grafos Intra-tempo (Conectividade Espacial)

Para cada conjunto CJ, gerar 3 grafos individuais (Gt1​,Gt2​,Gt3​).

    Topologia: Cada grafo intra-tempo deve ser totalmente conectado (FC - Fully Connected).

    Arestas (Eintra​): O peso das arestas deve ser definido pela Correlação de Pearson entre os vetores de atributos das ROIs dentro do mesmo frame temporal.

    Propósito: Capturar a relação estrutural/morfométrica entre diferentes áreas cerebrais em um ponto estático no tempo.

2. Grafos Inter-tempo (Conectividade Longitudinal/Delta)

Representar a dinâmica de alteração entre as aquisições através de grafos de "Diferença" (Deltas).

    Pares Temporais: Gerar representações para os deltas Δ12​(t2​−t1​), Δ13​(t3​−t1​) e Δ23​(t3​−t2​).

    Atributos de Diferença: Os nós desses grafos contêm os atributos de variação radiômica e campos de deformação (Jacobianos/Deslocamento).

    Conectividade: As arestas conectam a mesma ROI ao longo do tempo de forma direcionada e progressiva (t1​→t2​, t1​→t3​, t2​→t3​).

3. Grafo Unificado (Estrutura Espaço-Temporal Completa)

Criar um objeto de grafo único que integre toda a informação do conjunto CJ.

    Arquitetura Multi-camada: * O grafo unificado contém todos os nós de t1​,t2​,t3​ (Total de 60 nós: 20 por tempo).

        Arestas Espaciais: Manter as conexões intra-tempo (FC) dentro de cada camada temporal.

        Arestas Temporais: Conectar cada ROI n no tempo t à mesma ROI n no tempo t+1 (e t+2).

    Atributos de Aresta: As arestas temporais devem carregar os valores dos Deltas calculados anteriormente como pesos ou atributos de aresta.

Requisitos do Script Python

    Entrada: Dois DataFrames (ou CSVs):

        df_estatico: Atributos das ROIs por tempo.

        df_dinamico: Atributos de deltas e deformação.

    Processamento: * Implementar uma função build_unified_graph(patient_id) que retorne um objeto torch_geometric.data.Data.

        Garantir a normalização dos atributos antes da criação das arestas.

    Saída: Uma lista de grafos processados pronta para o DataLoader do PyTorch.

Dicas para o Cursor:

    Interpretabilidade: Peça para o Cursor comentar como ele está indexando as ROIs (0-19 para t1​, 20-39 para t2​, etc.).

    ST-GCN: Reforce que a matriz de adjacência unificada deve ser esparsa para eficiência, mas que as sub-matrizes intra-tempo são densas (totalmente conectadas).

    Batching: Solicite que o script utilize o formato Edge Index (COO) do PyG para facilitar o treinamento em GPU.

Com esse guia, o script gerado terá uma estrutura muito mais próxima do que a literatura científica de Graph Deep Learning utiliza para dados longitudinais.
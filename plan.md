# Plano da versão 1 do artigo — fechar análises e figuras

**Idioma:** português do Brasil.  
**Público:** pessoas e assistentes de linguagem (por exemplo Gemini). Evitar jargão; quando um nome técnico for inevitável, explicar na primeira vez.  
**Estado:** plano confirmado para implementação. Em seguida editar os cadernos Jupyter `6_results.ipynb` (figuras) e `7_stats.ipynb` (estatísticas).

**Atenção:** ignore versões antigas deste arquivo que pediam “fazer tudo só na coorte 48m_6m” ou trocar a terceira imagem pela segunda de forma confusa. Esse desenho foi **substituído** pelo plano abaixo.

---

## Glossário (ler antes)

| Nome no plano | Significado |
|---------------|-------------|
| Coorte | Conjunto de pacientes escolhido com regras de tempo (janela clínica e intervalo entre exames). |
| Coorte principal | A coorte cujos números entram no resumo e nas figuras extras. Pasta no disco: `48m_6m`. |
| Soft verdadeiro | Regra que permite, em parte dos pacientes progressores, que a **última** das três ressonâncias seja a primeira imagem já com diagnóstico de Alzheimer. Aumenta o número de progressores (120). |
| Soft falso (estrito) | Todas as três ressonâncias usadas na previsão ainda em comprometimento cognitivo leve. Menos progressores (74). Análise de sensibilidade “antes da conversão”. |
| Visita 1, 2 e 3 | Três exames de ressonância magnética ponderada em T1 do mesmo paciente, separados cerca de 6 ou 12 meses. |
| Só a primeira visita | O classificador usa apenas atributos da primeira imagem. Nome do protocolo no código: `t1_only`. |
| Duas visitas | Primeira imagem mais a mudança entre a visita 1 e a visita 2. Nome no código: `t1_d21`. |
| Três visitas | Primeira imagem mais a mudança 1→2 e a mudança 2→3. Nome no código: `t1_d21_d32`. |
| Família de atributos | Um dos cinco blocos: volume, forma, textura, estatísticas de primeira ordem, deslocamento do registro entre imagens. |
| Análise uniclasse | Um classificador por família de atributos (não misturar famílias no mesmo modelo). |
| Fusão tardia (análise multiclasse) | No fim, combinar as **probabilidades** de vários classificadores uniclasse (em geral pela média). |
| Máquina de vetores de suporte | Classificador principal do protocolo. Nome no código: `svm`. |
| Quatro algoritmos | Máquina de vetores de suporte, floresta aleatória, rede elástica e XGBoost — só na coorte principal. |
| Sem harmonização ComBat | Protocolo padrão do artigo: não aplicar o método ComBat entre aparelhos de ressonância. |
| Com ComBat | Análise de sensibilidade: harmonizar atributos entre aparelhos ou lotes. Resultados em pastas separadas cujo nome contém `combat`. |
| Experimento com vazamento | Normalização ou seleção de atributos usa o conjunto inteiro antes da validação (resultado otimista; só para comparar com a literatura). |
| Área sob a curva ROC | Medida principal de discriminação (quanto mais perto de 1, melhor). |
| Correção de Benjamini–Hochberg | Ajuste quando se fazem vários testes estatísticos ao mesmo tempo, para controlar falsos positivos. |
| Volume intracraniano | Medida do tamanho da cabeça. Atributos de forma foram corrigidos por **homotetia** (comprimento dividido pela raiz cúbica do volume; área pela raiz na potência dois terços; volume pelo próprio volume intracraniano). |

Os nomes de pastas e de protocolos no código (`t1_only`, `48m_6m`, e similares) permanecem iguais aos arquivos no computador — são identificadores dos programas, não abreviações soltas no texto.

---

## 1. Pergunta científica e desenho

### Coorte principal e regra soft

- **Principal (soft verdadeiro):** pasta `csvs/cohorts/48m_6m/` — 73 pacientes estáveis e 120 progressores (cerca de 46 progressores usam a última imagem já com Alzheimer).
- **Sensibilidade (soft falso):** pasta `csvs/cohorts/48m_6m_soft_False/` — 73 estáveis e 74 progressores; as três imagens ainda em comprometimento leve.

No texto do artigo: os números principais vêm do soft verdadeiro; o soft falso responde “e se exigirmos imagens só antes da conversão?”. O resumo **não** deve chamar tudo de previsão estritamente anterior à conversão se o número principal for soft verdadeiro.

### Camada principal (quatro coortes, mesmo protocolo)

Comparar, para cada uma das **cinco famílias** e com a **máquina de vetores de suporte**, as três formas de usar o tempo:

1. só a primeira visita;  
2. duas visitas;  
3. três visitas.

As quatro coortes (pastas): `36m_6m`, `36m_12m`, `48m_6m`, `48m_12m`  
(janela clínica de 36 ou 48 meses × intervalo nominal de 6 ou 12 meses entre imagens).

**Importante:** as quatro coortes compartilham pacientes; não são quatro estudos independentes. Isso deve estar escrito no artigo.

### Camada complementar (somente coorte principal)

- Fusão tardia: **três combinações escolhidas de antemão** (todas as famílias na primeira visita; todas nas três visitas; âncora = forma na primeira visita + o restante nas três visitas). As 237 combinações gravadas no disco ficam no material suplementar e são **exploratórias** (não concluir que “a melhor combinação prova superioridade”).
- Quatro algoritmos na primeira visita e nas três visitas.
- Experimento com vazamento (família volume nas três visitas; também já existe para duas visitas).
- Só variáveis clínicas; clínicas mais imagem de **volume** na primeira visita.
- ComBat ligado versus desligado (sensibilidade; ver seção 8).
- Soft verdadeiro versus soft falso (máquina de vetores de suporte; primeira visita e três visitas; cinco famílias).

### O que não entra no corpo da versão 1

Fusão precoce; experimento que remove o pareamento de histograma; três mapas de calor (um por forma de usar o tempo); fusão tardia com duas visitas nas quatro coortes; quatro algoritmos na representação de duas visitas; calibração ou curva de decisão clínica; contagem de pacientes por dobra da validação; estimativa de ruído com exames repetidos muito próximos no tempo.

### Como justificar o intervalo de cerca de seis meses (Métodos)

Não escrever que “em seis meses a atrofia do hipocampo é grande”. Escrever que:

- mudanças de forma e volume em seis meses já foram medidas em **grupos** (especialmente comprometimento leve mais avançado e quem progride);
- detectar no **grupo** não é o mesmo que medir com precisão em cada **indivíduo**;
- o desenho de **três visitas** cobre cerca de **doze meses** no total, com duas mudanças consecutivas de cerca de seis meses;
- volume, forma e campo de deslocamento têm melhor apoio biológico; textura e primeira ordem (brilho da imagem T1) são mais exploratórias e sensíveis à normalização de intensidade.

Referências úteis: Mubeen e outros (2017), Schuff e outros (2009), Hua e outros (2016), Leung e outros (2013).

---

## 2. O que já está no disco e o que ainda falta correr

### Já pronto (não voltar a lançar esses experimentos)

| Conteúdo | Onde |
|----------|------|
| Coorte principal: só visita 1, duas visitas e três visitas; cinco famílias (quatro algoritmos, exceto forma nas duas visitas = só máquina de vetores de suporte) | pastas cujo nome termina em `t1_only`, `d21` e `d21d32` dentro de `48m_6m` |
| Controles cognitivamente normais versus Alzheimer nas três visitas | mesmos resultados das três visitas |
| Fusão tardia (muitas combinações) | pasta `ablation_results_late_fusion` |
| Só clínico e clínico mais volume | pastas `ablation_results_clinic` e `ablation_results_clinic_img_t1_only` |
| Experimento com vazamento (volume) | pastas cujo nome contém `leaky` |
| Soft falso: só visita 1 e três visitas, cinco famílias, máquina de vetores de suporte | `48m_6m_soft_False` (**sem** pasta de duas visitas) |
| Quatro coortes: só visita 1 e três visitas, máquina de vetores de suporte | pastas das quatro coortes |
| Correção de forma pelo volume intracraniano (homotetia) | script `4_run_post_extract.py` (já aplicado) |

### Duas visitas nas três coortes — **já feito** (10 de setembro de 2026)

O script na sessão tmux (arquivo de log `logs/v1_d21_3cohorts_20260910_143820.log`) correu a representação de **duas visitas**, máquina de vetores de suporte, cinco famílias, em:

- `36m_6m`  
- `36m_12m`  
- `48m_12m`  

Começo cerca de 14:38 (UTC) · fim com mensagem de conclusão às 14:50 (UTC) · verificação automática OK (quinze arquivos de resumo).  
A coorte `48m_6m` **não** foi relançada (já tinha duas visitas).

A planilha que junta as coortes também já foi reconstruída:

- `csvs/cohort_comparison/cohort_results.csv`  
- `csvs/cohort_comparison/cohort_features_long.csv`  

**Não** voltar a lançar esse bloco.

### Opcional neste fecho

- Duas visitas no soft falso: **não** precisa para comparar soft verdadeiro versus soft falso (basta só a primeira visita e as três visitas).
- ComBat: ver decisão na seção 8 (sensibilidade na coorte principal, não novo padrão).

### Cuidados ao lançar scripts (se no futuro relançar outra coisa)

O programa `5_ablation.py` **substitui** o arquivo de resultados da pasta (não junta linhas antigas).  
Não pedir ComBat verdadeiro e falso ao mesmo tempo nas pastas que hoje guardam só o protocolo sem ComBat.  
Não pedir só a tarefa controles normais versus Alzheimer se isso apagar a tarefa estável versus progressor.  
Uma família de atributos por pasta de saída.

---

## 3. Figuras e tabelas mínimas da versão 1

| Código | O que é | Onde gerar |
|--------|---------|------------|
| Figura A | Quatro painéis (uma coorte em cada): cinco famílias no eixo; três barras (só visita 1 / duas visitas / três visitas); máquina de vetores de suporte | caderno `6_results.ipynb` — hoje a lista de protocolos nas quatro coortes só tem visita 1 e três visitas; **incluir duas visitas** |
| Tabela A | Áreas sob a curva das três representações e diferenças em relação à só visita 1; quatro coortes × cinco famílias | a partir do arquivo `cohort_results.csv` |
| Tabela B | Coorte principal: cinco famílias × quatro algoritmos; duas metades (só visita 1 \| três visitas) | caderno de figuras (tabela; mapa de calor opcional com a mesma escala de cores) |
| Figura B | Uma curva ROC da família forma, três representações temporais, só coorte principal | já existe arquivo semelhante |
| Tabela C | Testes: três visitas versus só visita 1 **e** duas visitas versus só visita 1; cinco famílias; correção por vários testes; coorte principal e quatro coortes | caderno `7_stats.ipynb` |
| Tabela D | Uma linha por análise complementar na coorte principal (fusão âncora, vazamento, clínico, clínico mais volume, ComBat se houver, soft) | caderno `7_stats.ipynb` |
| Figura C / Tabela E | Soft verdadeiro versus soft falso: barras ou tabela; máquina de vetores de suporte; só visita 1 e três visitas; cinco famílias | **nova seção** em `6_results.ipynb` (dados já no disco) |

**Não** colocar no corpo do artigo: três mapas de calor (um por representação temporal); curvas ROC separadas só da visita 1 e só das três visitas se a Figura B já mostrar as três; figuras grandes só de fusão, vazamento ou ComBat (esses números vão na Tabela D).

---

## 4. Alterações no caderno de estatísticas (`7_stats.ipynb`)

1. Textos e títulos: coorte principal `48m_6m` (o código já aponta; textos e saídas antigas ainda falam em `48m_12m`).  
2. Família usada na fusão clínico mais imagem: de forma para **volume** — o arquivo no disco é de volume.  
3. Voltar a executar demografia, variáveis que confundem, três visitas versus só visita 1, as quatro coortes, fusão tardia, clínico e vazamento com a coorte principal correta.  
4. **Novo:** contraste duas visitas versus só visita 1 (mesmo tipo de teste que três visitas versus só visita 1) → Tabela C.  
5. Fusão tardia: reportar só as três combinações escolhidas de antemão.  
6. Clínico: diferença (clínico mais imagem) menos (só clínico) com intervalo de confiança; imagem = volume na primeira visita.  
7. **Novo:** soft verdadeiro versus soft falso (descritivo; tamanhos de amostra diferentes → **não** usar reamostragem pareada paciente a paciente).  
8. **Novo (se os resultados ComBat já existirem):** ComBat ligado versus desligado, lendo as pastas cujo nome contém `combat` (não misturar dentro dos arquivos sem ComBat).  
9. Montar a Tabela D e gravar em `artigo/tables/`.

Depois, se quiser: também o valor-p bilateral; renomear arquivos de tabela que ainda dizem `48m12` no nome para refletir `48m_6m`.

---

## 5. Alterações no caderno de figuras (`6_results.ipynb`)

1. Atualizar o índice: pergunta principal = quatro coortes × três representações uniclasse; complementares = só coorte principal.  
2. As duas visitas nas três coortes e a planilha consolidada já existem: na lista de protocolos das quatro coortes, **incluir a representação de duas visitas** e regenerar a Figura A.  
3. Exportar Tabela A.  
4. Exportar Tabela B (quatro algoritmos).  
5. Manter Figura B (curva ROC da forma, três representações); a curva ROC do volume pode ir ao material suplementar.  
6. **Nova seção soft:** caminhos explícitos para `48m_6m` e `48m_6m_soft_False`; tabela e barras; diferença = soft verdadeiro menos soft falso; nota: 120 versus 74 progressores.  
7. Não mudar a variável global de coorte do caderno para a pasta do soft falso.  
8. Não fazer figura grande só de ComBat na versão mínima (só uma linha na Tabela D).

---

## 6. Ordem de trabalho (após este plano)

0. Confirmar duas visitas e planilha consolidada → **já feito** (10 de setembro de 2026, cerca de 14:50 UTC, mais a reconstrução da planilha).  
1. Em `7_stats.ipynb`: família clínica mais imagem = volume; textos da coorte principal.  
2. Em `7_stats.ipynb`: contraste duas visitas versus só visita 1; atualizar três visitas versus só visita 1.  
3. Em `6_results.ipynb`: Figura A (já pode incluir as três representações nas quatro coortes) e Tabelas A e B.  
4. Em `6_results.ipynb`: seção soft verdadeiro versus soft falso.  
5. Em `7_stats.ipynb`: Tabela D (fusão, vazamento, clínico mais volume, soft, ComBat se estiver pronto).  
6. Executar os cadernos do início ao fim; gravar tabelas e figuras em `artigo/`.  
7. Reescrever o texto LaTeX do artigo com esses números (passo separado; meta cerca de 18 a 22 páginas, não 59).

---

## 7. Depois: texto do artigo (não bloqueia os passos 1 a 5)

- Coorte principal `48m_6m`; enxugar o manuscrito.  
- Métodos: normalização de forma pelo volume intracraniano (homotetia).  
- Soft verdadeiro = números principais; soft falso = sensibilidade antes da conversão.  
- Fusão = três combinações / caráter exploratório; não repetir área sob a curva antiga inconsistente (por exemplo 0,802).  
- Reamostragem e permutação: dizer que os escores já obtidos fora da amostra ficam fixos (não é “incerteza de treino completo”).  
- Trabalhos relacionados: Mubeen, Schuff, Hua; morfometria versus intensidade.

---

## 8. Decisões já fechadas (lista de verificação)

- [x] Pergunta principal = análise uniclasse, quatro coortes, três representações temporais, máquina de vetores de suporte.  
- [x] Complementares (fusão, quatro algoritmos, vazamento, clínico, ComBat, soft) = só coorte principal.  
- [x] Soft: principal = verdadeiro; sensibilidade = falso; figura no caderno de resultados; **não** exige duas visitas no soft falso.  
- [x] Figura A = quatro painéis com três barras; Tabela B = quatro algoritmos (não três mapas de calor).  
- [x] Fusão clínico mais imagem usa **volume**, não forma.  
- [x] Duas visitas nas quatro coortes = pré-requisito da Figura A → **cumprido** (três coortes novas mais a principal que já existia).  
- [x] **ComBat:** o padrão do artigo continua **sem** ComBat. Harmonização entre aparelhos entra como **sensibilidade obrigatória na coorte principal** (ligado versus desligado), não como novo protocolo padrão de todo o estudo. Duas visitas no soft falso continuam opcionais.

**Motivo do ComBat:** o disco e os cadernos estão calibrados sem ComBat; o ComBat clássico pode enfraquecer o sinal longitudinal; estudo multi-centro exige discussão e uma comparação na coorte principal, não relançar as quatro coortes × três representações só para mudar o padrão.

---

## 9. Próximo passo para o assistente de código

Os resultados de duas visitas e a planilha que junta as coortes já estão prontos. Implementar nesta ordem:

1. Ajustes e novas células no caderno `7_stats.ipynb`: usar volume na fusão clínico mais imagem; corrigir textos da coorte principal; contraste duas visitas versus só a primeira visita; soft verdadeiro versus soft falso; ComBat se as pastas existirem; montar a Tabela D.  
2. Ajustes e nova seção soft no caderno `6_results.ipynb`; regenerar a Figura A incluindo as **três** formas de usar o tempo (só a primeira visita, duas visitas e três visitas) nas quatro coortes — na lista de protocolos do caderno, passar a incluir também a representação de duas visitas.  
3. Não relançar os experimentos já listados como prontos na seção 2.

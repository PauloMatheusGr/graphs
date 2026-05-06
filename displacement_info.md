# displacement_info.md

## Visão geral

Este documento descreve o pipeline de **registro deformável** e a construção de **atributos derivados de campos de deslocamento** utilizados em `features_displacement.py`. A ideia central é usar um **template groupwise estratificado** (por sexo e faixa etária) como **referencial comum**, e então caracterizar (i) o deslocamento do template até cada imagem clínica de um indivíduo (por tempo) e (ii) formas opcionais de representar mudanças entre tempos via esse mesmo template.

---

## 0) Construção do template groupwise estratificado (`groupwise_ants.py`)

O template groupwise estratificado \(T\) é construído previamente por estrato (diagnóstico, sexo e faixa etária) usando ANTsPy, produzindo uma imagem “média” (template) que representa um subconjunto de indivíduos cognitivamente normais (CN). No repositório, esse processo está implementado em `groupwise_ants.py`.

### 0.1 Seleção do estrato (CN, sexo, idade)

Seja um *dataset* tabular \(D\) (CSV) contendo, para cada exame, pelo menos:
\(\{\texttt{ID\_IMG}, \texttt{SEX}, \texttt{AGE}, \texttt{DIAG}\}\).

O script filtra o estrato fixando:

- Diagnóstico: \(\texttt{DIAG} = \text{"CN"}\)
- Sexo: \(\texttt{SEX} \in \{\text{"F"}, \text{"M"}\}\)
- Idade: \(\texttt{AGE} \in [a_{\min}, a_{\max}]\)

Se o conjunto filtrado tiver mais de \(N_{\max}\) imagens, o script aplica uma amostragem aleatória reprodutível (semente fixa) para limitar o número de casos:

\[
S = \mathrm{sample}(D_{\text{filtrado}}, N_{\max}; \text{seed})
\]

Por padrão em `groupwise_ants.py`:

- \(N_{\max}=20\)
- `RANDOM_SEED = 7`
- `DIAG_FILTER = "CN"`

O script também salva um CSV com os casos efetivamente usados (`selected_*.csv`), para auditoria e reprodutibilidade.

### 0.2 Resolução de caminhos e entradas de imagem

Para cada `ID_IMG` selecionado, o caminho do volume T1 pré-processado é resolvido como:

\[
p(\texttt{ID\_IMG}) = \texttt{IMAGES\_DIR} \; / \; (\texttt{ID\_IMG} + \texttt{SUFFIX})
\]

IDs sem arquivo correspondente são descartados. O groupwise exige pelo menos 2 imagens válidas.

### 0.3 Padronização de orientação, intensidades e grade

Antes do groupwise, cada imagem é padronizada para reduzir variações não-biológicas.

1) **Reorientação**: todas as imagens são reorientadas para uma convenção fixa (por padrão, RAS).

2) **Casamento de histograma (Histogram Matching)**: as intensidades são harmonizadas usando um template MNI como referência (\(\mathrm{MNI}\)), de forma mascarada no cérebro. Em notação conceitual:

\[
\tilde{C}_i = \mathrm{HM}\big(C_i; \mathrm{MNI}\big)
\]

3) **Reamostragem isotrópica**: cada volume é reamostrado para espaçamento isotrópico desejado (por padrão, \(1\text{ mm}\)) com interpolação adequada para T1 contínua (Bspline).

\[
\hat{C}_i = \mathrm{Resample}\big(\tilde{C}_i; 1\text{ mm}\big)
\]

4) **Escolha de um “target” comum com padding**: dentre as imagens, escolhe-se a de maior extensão física como target e aplica-se padding para garantir FOV suficiente. Em seguida, todas as imagens são reamostradas para o grid desse target:

\[
\bar{C}_i = \mathrm{ResampleToTarget}(\hat{C}_i; \text{target})
\]

5) **Máscaras e normalização robusta**: para cada imagem e para o target, obtém-se uma máscara cerebral automática e faz-se uma reescala robusta baseada em percentis dentro do cérebro para reduzir outliers:

\[
C_i^{(0)} = \mathrm{RobustRescale}(\bar{C}_i; \text{mask}, p_{0.5}, p_{99.5})
\]

### 0.4 Pré-alinhamento (Rigid → Affine)

Para estabilizar o groupwise, cada imagem \(C_i^{(0)}\) é pré-alinhada ao target por uma cadeia de registros globais:

\[
C_i^{(1)} = \mathrm{Affine}\big(\mathrm{Rigid}(C_i^{(0)} \rightarrow \text{target}) \rightarrow \text{target}\big)
\]

onde as máscaras são usadas para restringir o cálculo ao cérebro.

### 0.5 Inicialização do template e iterações groupwise (SyN + média)

O template inicial é a média das imagens pré-alinhadas:

\[
T^{(0)} = \frac{1}{n}\sum_{i=1}^{n} C_i^{(1)}
\]

Em seguida, o template é refinado iterativamente. Na iteração \(t\):

1) para cada imagem \(C_i^{(1)}\), estima-se um registro deformável para o template corrente:

\[
G_{i}^{(t)} = \mathrm{Reg}\big(C_i^{(1)} \rightarrow T^{(t-1)}\big)
\]

2) aplica-se o warp para obter a imagem deformada no espaço do template:

\[
W_i^{(t)} = C_i^{(1)} \circ \big(G_{i}^{(t)}\big)^{-1}
\]

3) atualiza-se o template como a média das imagens deformadas:

\[
T^{(t)} = \frac{1}{n}\sum_{i=1}^{n} W_i^{(t)}
\]

No `groupwise_ants.py`, o tipo de transformação deformável padrão é `SyN` e o número de iterações do template é `N_ITER_TEMPLATE = 5`.

Ao final, o template é novamente normalizado (percentis no cérebro) para facilitar visualização/consistência de intensidades.

### 0.6 Saídas e convenção de nomes

O script grava o template final como NIfTI comprimido:

`groupwise_DIAG-CN_SEX-<F|M>_AGE-<min>-<max>_N-<n>_template.nii.gz`

e o CSV correspondente:

`selected_DIAG-CN_SEX-<F|M>_AGE-<min>-<max>_N-<n>.csv`

Esses arquivos são usados posteriormente como \(T\) (template estratificado) no pipeline de deslocamento.

**Observação importante (faixas etárias):** `features_displacement.py` constrói o nome do template usando faixas do tipo `50-59`, `60-69`, etc. Portanto, ao gerar templates com `groupwise_ants.py`, recomenda-se chamar o script com limites coerentes (por exemplo, `50 59` em vez de `50 60`) para que o nome do arquivo resultante seja compatível com a regra de nomenclatura usada na etapa longitudinal.

---

## 1) Espaços (referenciais) e notação

Sejam:

- \(T\): template groupwise estratificado (CN) do paciente.
- \(C_k\): imagem clínica no tempo \(k\) (por exemplo \(k \in \{1,2,3\}\) para \(i_1,i_2,i_3\)).
- \(\Omega_T \subset \mathbb{R}^3\): domínio físico (mm) do template.
- \(\Omega_k \subset \mathbb{R}^3\): domínio físico (mm) da clínica no tempo \(k\).
- \(x \in \Omega_T\): um ponto (em coordenadas físicas, mm) no espaço do template.

Transformações:

- \(F_k: \Omega_T \rightarrow \Omega_k\): transformação **Template → Clínica\(_k\)**.
- \(I_k: \Omega_k \rightarrow \Omega_T\): transformação **Clínica\(_k\) → Template** (inversa).

### Relação com ANTs/ANTsPy

No script, o registro é executado como:

- `fixed = C_k` (imagem clínica do tempo \(k\))
- `moving = T` (template estratificado)

No padrão ANTs/ANTsPy:

- `fwdtransforms` mapeia **moving → fixed**  \(\Rightarrow\) **Template → Clínica\(_k\)** \(\equiv F_k\)
- `invtransforms` mapeia **fixed → moving**  \(\Rightarrow\) **Clínica\(_k\) → Template** \(\equiv I_k\)

---

## 2) Registro deformável template↔clínica (formulação abstrata)

Um registro deformável (ex.: SyN) estima uma transformação que alinha o template à clínica. De forma abstrata, pode-se escrever:

\[
F_k \;=\; \arg\min_{F} \Big( \mathcal{D}\big(C_k,\, T \circ F^{-1}\big) \;+\; \lambda \,\mathcal{R}(F)\Big)
\]

onde:

- \(\mathcal{D}(\cdot,\cdot)\): termo de dissimilaridade (ex.: correlação cruzada, MI),
- \(\mathcal{R}(F)\): termo de regularização (suavidade / difeomorfismo),
- \(\lambda\): peso da regularização.

A notação \(T \circ F^{-1}\) indica que o template é reamostrado no espaço da clínica por meio do inverso do mapeamento (convenção comum em registro baseado em intensidades).

---

## 3) Símbolo de composição “\(\circ\)”

O símbolo **\(\circ\)** representa **composição de funções/transformações**:

\[
(A \circ B)(x) = A(B(x))
\]

Isto é: **aplica-se primeiro \(B\), depois \(A\)**.

Em registro, a composição é essencial para “conectar” espaços via um terceiro referencial (por exemplo, via template).

---

## 4) Campo de deslocamento no referencial do template (por tempo)

Se o **referencial de análise** é sempre o template groupwise, o objeto natural é o campo de deslocamento por tempo definido em \(\Omega_T\):

\[
u_k(x) = F_k(x) - x, \quad x \in \Omega_T
\]

Interpretação:

- \(x\) é uma posição no template (mm),
- \(F_k(x)\) é a posição correspondente no espaço da clínica \(C_k\),
- \(u_k(x)\) é um vetor 3D (mm) medindo “quanto o template precisa se deslocar” para alinhar-se à clínica do tempo \(k\), naquele ponto.

### Componentes

\[
u_k(x) = \big(u_{x,k}(x),\; u_{y,k}(x),\; u_{z,k}(x)\big)
\]

Cada componente preserva direção e sinal do deslocamento ao longo de cada eixo.

---

## 5) Atributos derivados do campo \(u_k\)

O campo vetorial \(u_k\) permite derivar mapas escalares e componentes, que são resumidos por ROIs.

### 5.1 Magnitude do deslocamento

\[
\|u_k(x)\| = \sqrt{u_{x,k}(x)^2 + u_{y,k}(x)^2 + u_{z,k}(x)^2}
\]

Interpretação: intensidade do deslocamento (mm) independentemente da direção.

### 5.2 Divergência (expansão/contração)

\[
\nabla \cdot u_k(x) =
\frac{\partial u_{x,k}}{\partial x} +
\frac{\partial u_{y,k}}{\partial y} +
\frac{\partial u_{z,k}}{\partial z}
\]

- valores positivos: comportamento local expansivo,
- valores negativos: comportamento local contrativo.

### 5.3 Rotacional (curl) e sua magnitude

\[
\nabla \times u_k(x) =
\begin{pmatrix}
\frac{\partial u_{z,k}}{\partial y} - \frac{\partial u_{y,k}}{\partial z} \\
\frac{\partial u_{x,k}}{\partial z} - \frac{\partial u_{z,k}}{\partial x} \\
\frac{\partial u_{y,k}}{\partial x} - \frac{\partial u_{x,k}}{\partial y}
\end{pmatrix}
\]

e

\[
\|\nabla \times u_k(x)\|
\]

Interpretação: intensidade da componente “giratória” local do campo.

### 5.4 Jacobiano e log-Jacobiano (variação volumétrica local)

Define-se o mapeamento total:

\[
\varphi_k(x) = x + u_k(x)
\]

Seu Jacobiano é:

\[
J_{\varphi_k}(x) = \frac{\partial \varphi_k(x)}{\partial x}
= I + \frac{\partial u_k(x)}{\partial x}
\]

O determinante \(\det(J_{\varphi_k}(x))\) mede variação volumétrica local:

- \(> 1\): expansão,
- \(< 1\): contração,
- \(= 1\): preservação local de volume.

Frequentemente usa-se o log-Jacobiano:

\[
\log\det(J_{\varphi_k}(x))
\]

---

## 6) Representações opcionais “via template” para mudança entre tempos

Mesmo sem registrar clínica↔clínica diretamente, é possível representar relações entre tempos usando o template como ponte.

### 6.1 Diferença vetorial direta entre campos no template

Se \(u_1\) e \(u_2\) são ambos definidos em \(\Omega_T\), então:

\[
\Delta u_{1\to2}(x) = u_2(x) - u_1(x)
\]

Interpretação: quanto o deslocamento (template→clínica) “mudou” do tempo 1 para o tempo 2, ponto a ponto no template.

### 6.2 Composição Template→Template entre dois alinhamentos

Outra forma é construir um mapeamento no próprio template que “compara” dois alinhamentos:

\[
S_{1\to2} = I_2 \circ F_1
\quad \Rightarrow \quad
S_{1\to2}(x) = I_2(F_1(x))
\]

O campo correspondente no template pode ser escrito como:

\[
w_{1\to2}(x) = S_{1\to2}(x) - x
\]

Esse \(w_{1\to2}\) também pode ter magnitude, Jacobiano, divergência, curl, etc.

---

## 7) Agregação por ROIs (features por região)

Seja \(R \subset \Omega_T\) uma região de interesse (ROI) no referencial do template (ou reamostrada para ele). Para um mapa escalar \(s(x)\) (por exemplo, \(\|u_k(x)\|\) ou \(\log\det J_{\varphi_k}(x)\)), extrai-se um conjunto de estatísticas:

- média: \(\mu_R = \frac{1}{|R|}\sum_{x\in R} s(x)\)
- desvio padrão: \(\sigma_R\)
- quantis: \(q_{0.05}, q_{0.50}, q_{0.95}\)
- tamanho amostral: \(|R|\)

O resultado final é um vetor de atributos por ROI, por tempo (e/ou por relação entre tempos, se adotada).

---

## Mini-glossário

- **\(T\)**: template groupwise estratificado (CN).
- **\(C_k\)**: imagem clínica no tempo \(k\).
- **\(\Omega_T, \Omega_k\)**: domínios físicos (mm) de template e clínica.
- **\(F_k\)**: transformação Template→Clínica\(_k\) (*forward*).
- **\(I_k\)**: transformação Clínica\(_k\)→Template (*inverse*).
- **\(\circ\)**: composição; \((A\circ B)(x)=A(B(x))\).
- **\(u_k(x)=F_k(x)-x\)**: campo de deslocamento no template.
- **\(\varphi_k(x)=x+u_k(x)\)**: mapeamento total.
- **\(J_{\varphi_k}(x)\)**: Jacobiano do mapeamento; **\(\det\)** determinante.
- **\(\log\det(J)\)**: log-Jacobiano (expansão/contração em escala log).
- **\(\nabla\cdot u\)**: divergência; **\(\nabla\times u\)**: rotacional (curl).


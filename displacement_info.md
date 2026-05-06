# displacement_info.md

## Visão geral

Este documento descreve o pipeline de **registro deformável** e a construção de **atributos derivados de campos de deslocamento** utilizados em `features_displacement.py`. A ideia central é usar um **template groupwise estratificado** (por sexo e faixa etária) como **referencial comum**, e então caracterizar (i) o deslocamento do template até cada imagem clínica de um indivíduo (por tempo) e (ii) formas opcionais de representar mudanças entre tempos via esse mesmo template.

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


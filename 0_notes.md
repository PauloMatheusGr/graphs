# Atributos dos classificadores — guia de estudo

Objetivo: entender o que cada atributo mede e por que ele muda com a atrofia do hipocampo.
Sugestão de estudo: copiar à mão o **mapa geral** (passo 1), depois uma família por vez
(passos 2–6), e no fim responder às **perguntas de fixação** (passo 8) sem olhar.

---

## Passo 0 — A ideia em uma frase

> Na doença de Alzheimer o hipocampo **encolhe**, **perde neurônios** e o **líquor (CSF) ocupa
> o espaço**. Cada família de atributos olha esse mesmo fenômeno por um ângulo diferente.

Na T1: substância cinzenta = **cinza claro**, líquor = **escuro**. Atrofia ⇒ mais escuro
na borda, forma mais irregular, deformação maior em relação ao "normal".

---

## Passo 1 — Mapa geral (copiar primeiro)

| # | Família (eixo x) | Pergunta que responde | Olha para | Nº por lado |
|---|---|---|---|---|
| 1 | **Volume** | *Quanto* tecido existe? | máscara + segmentação de tecidos | 4 |
| 2 | **Shape** | *Que forma* tem? | só a máscara (geometria) | 12 |
| 3 | **1st order** | *Que intensidades* existem? | histograma da T1 (sem posição) | 16 |
| 4 | **Texture** | *Como as intensidades se organizam* no espaço? | pares de vizinhos (GLCM) | 24 |
| 5 | **DVF** | *Quanto e onde* difere do cérebro de referência? | campo de deformação (registro) | 12 |

Tudo é medido **separado** no hipocampo esquerdo (L) e direito (R).

Agrupamento para memorizar:

- **Geometria da máscara** → Volume, Shape ("quanto" e "que forma").
- **Sinal dentro da máscara** → 1st order, Texture ("que tons" e "como se arrumam").
- **Geometria relativa a uma referência** → DVF ("como deformar o sujeito até o normal").

Mnemônico: **V-S-P-T-D** = **V**olume, **S**hape, **P**rimeira ordem, **T**extura, **D**VF
("**V**ocê **S**ó **P**ensa **T**udo **D**epois").

Representações temporais (como o atributo entra no classificador):

| Abordagem | Código | O que entra |
|---|---|---|
| 1 visita | `t1_only` | valor no baseline (T1) |
| 2 visitas (S0,R10) | `t1_r10` | T1 + taxa de mudança entre visita 0 e 1 (por mês) |
| 3 visitas (D) | `t1_ols` | T1 + inclinação da reta (mínimos quadrados) pelas 3 visitas |

Pré-processamento comum (lembrar): T1 sem crânio, sem ruído, corrigida de não-uniformidade
(N4), intensidades igualadas ao MNI (histogram matching) e alinhada ao MNI por registro rígido.
Medidas de tamanho divididas pelo tamanho da cabeça (ICV): comprimento ÷ ICV^(1/3),
área ÷ ICV^(2/3), volume ÷ ICV — "cabeça grande não é hipocampo grande".

---

## Passo 2 — Volume (4 atributos): "quanto tecido?"

Em uma frase: **tamanho do hipocampo e do que ele é feito** (cinzenta, branca, líquor).

| Atributo | O que é | Com atrofia |
|---|---|---|
| `MeshVolume` | volume do hipocampo (malha de superfície) ÷ ICV | ↓ |
| `gm_norm` | fração de substância cinzenta dentro da ROI | ↓ |
| `wm_norm` | fração de substância branca dentro da ROI | varia (borda, fímbria) |
| `csf_norm` | fração de líquor dentro da ROI | ↑ |

Lembrar: as frações (`*_norm`) já são proporções (0–1), por isso não dividem pelo ICV.
É o marcador clássico: volume hipocampal é o "padrão-ouro" da atrofia.

---

## Passo 3 — Shape (12 atributos): "que forma?"

Em uma frase: **geometria da máscara**, sem olhar intensidade.

Truque: imagine o hipocampo como um **charuto curvado**. A PCA das coordenadas dos voxels
dá 3 eixos: maior (comprimento) ≥ menor (largura) ≥ mínimo (espessura).

| Grupo | Atributos | O que medem |
|---|---|---|
| Eixos (tamanho) | `MajorAxisLength`, `MinorAxisLength`, `LeastAxisLength` | comprimento, largura, espessura (÷ ICV^(1/3)) |
| Proporções (sem unidade) | `Elongation` = √(menor/maior), `Flatness` = √(mínimo/maior) | quão alongado / achatado (0–1) |
| Diâmetros máximos | `Maximum3DDiameter`; `Maximum2DDiameterSlice` (axial), `...Column` (coronal), `...Row` (sagital) | maior distância entre 2 pontos (no espaço / em cada plano) |
| Superfície | `SurfaceArea` (÷ ICV^(2/3)), `SurfaceVolumeRatio` (área/volume) | tamanho da "casca"; razão ↑ quando o volume cai mais rápido que a área |
| Compacidade | `Sphericity` (1 = esfera) | contorno irregular ou atrofia focal ⇒ ↓ |

Fora do classificador: `MeshVolume` (já está em Volume) e `VoxelVolume` (redundante).

---

## Passo 4 — 1st order (16 atributos): "que intensidades?"

Em uma frase: **estatísticas do histograma** das intensidades T1 dentro do hipocampo.
Não importa **onde** está cada voxel — só **quanto** ele brilha.

Imagem mental: despejar todos os voxels num saco e fazer um histograma.

| Grupo | Atributos | Com atrofia (mais líquor escuro) |
|---|---|---|
| Centro | `Mean`, `Median`, `RootMeanSquared` | ↓ |
| Extremos | `Minimum`, `Maximum`, `10Percentile`, `90Percentile`, `Range` | P10 e mínimo ↓ (cauda escura) |
| Dispersão | `Variance`, `InterquartileRange`, `MeanAbsoluteDeviation`, `RobustMeanAbsoluteDeviation` (só P10–P90) | ↑ |
| Forma do histograma | `Skewness` (assimetria), `Kurtosis` (caudas) | assimetria fica negativa (cauda escura) |
| Organização do histograma | `Entropy` (desordem), `Uniformity` (Σp², concentração) | entropia ↑, uniformidade ↓ |

Os **4 momentos** estão aqui (média, variância, assimetria, curtose), mas a família tem mais
que isso — por isso o rótulo é "1st order", não "moments".
Fora: `Energy`, `TotalEnergy` (crescem com o tamanho da ROI → redundantes com volume).

---

## Passo 5 — Texture / GLCM (24 atributos): "como as intensidades se organizam?"

Em uma frase: **padrão espacial** — vizinhos parecidos (homogêneo) ou diferentes (heterogêneo)?

Como funciona a GLCM (matriz de coocorrência):
1. Reduz a T1 a 64 tons de cinza.
2. Para cada voxel de tom *i*, olha o vizinho imediato (13 direções em 3D) e anota o tom *j*.
3. A matriz conta quantas vezes cada par (*i*, *j*) aparece.
4. Diagonal cheia (i ≈ j) ⇒ tecido homogêneo. Fora da diagonal ⇒ transições bruscas.

Diferença-chave para o 1st order: **mesmo histograma pode ter texturas diferentes**
(tabuleiro de xadrez × metade preta/metade branca: mesmo histograma, GLCM oposta).

| Grupo | Atributos | Ideia |
|---|---|---|
| Contraste / diferença | `Contrast`, `DifferenceAverage`, `DifferenceVariance`, `DifferenceEntropy` | quão diferentes são os vizinhos |
| Homogeneidade | `Id`, `Idm`, `Idn`, `Idmn`, `InverseVariance` | o oposto: quão parecidos são os vizinhos |
| Uniformidade / desordem | `JointEnergy`, `JointEntropy`, `MaximumProbability` | poucos pares dominam (regular) × muitos pares (complexo) |
| Nível e dispersão dos pares | `JointAverage`, `Autocorrelation`, `SumAverage`, `SumEntropy`, `SumSquares` | brilho médio dos pares e espalhamento |
| Agrupamento | `ClusterTendency`, `ClusterShade`, `ClusterProminence` | vizinhos de tons parecidos formam "manchas"? assimetria/caudas da GLCM |
| Dependência | `Correlation`, `Imc1`, `Imc2`, `MCC` | saber o tom de um voxel ajuda a prever o do vizinho? |

Com atrofia (bolsões de líquor, bordas irregulares): `Contrast`, `JointEntropy`,
`ClusterProminence` ↑; `Idm`, `JointEnergy` ↓.
Outras matrizes (GLRLM, GLSZM, GLDM, NGTDM) são extraídas mas **não** entram.

---

## Passo 6 — DVF (12 atributos): "quanto e onde difere do normal?"

Em uma frase: **quanto é preciso deformar o cérebro do sujeito para ele ficar igual ao
template** (cérebro médio de referência, pareado por sexo e faixa etária).

Como funciona:
1. Registro não linear (SyN, ANTs) entre a T1 do sujeito e o template.
2. Resultado: campo de deslocamento **u(x)** — uma setinha (mm + direção) em cada voxel.
3. Desse campo saem **3 mapas**; de cada mapa, **4 estatísticas** no hipocampo ⇒ 3 × 4 = 12.

Os 3 mapas (mnemônico **D-V-S** do artigo MBEC):

| Mapa | Letra | O que é | Analogia |
|---|---|---|---|
| `mag` | **D** (displacement) | tamanho da setinha \|u\| em mm | "quanto andou" |
| `jac_det` | **V** (volume) | determinante do Jacobiano: 1 = igual, ≠ 1 = encolheu/expandiu | "balão esvaziou ou encheu" |
| `strain_fro` | **S** (strain) | norma do tensor de deformação ε = ½(∇u + ∇uᵀ) | "quanto esticou/torceu" (ignora girar/transladar) |

As 4 estatísticas (os **4 momentos**):

| Sufixo | Significado |
|---|---|
| `_mean` | deformação média |
| `_variance` | heterogênea? (focal × difusa) |
| `_skewness` | há cauda de voxels com deformação extrema? |
| `_kurtosis` | deformação concentrada em poucos pontos? |

Variantes: `disp` (template CN), `disp_ad` (template AD), `disp_cnad` (os dois juntos);
`disp_oasis*` = mesmos 12 atributos com templates OASIS-3 e ROI `hippocampus_d2`
(hipocampo + 2 mm) ou `hippocampus` (núcleo).
Atenção ao sinal: ADNI fixed = template; OASIS fixed = sujeito ⇒ no OASIS `jac_det > 1`
significa sujeito **menor** que o template (atrofia).
Fora do classificador: desvio-padrão, percentis e `logjac` (ficam só no CSV).

---

## Passo 7 — Resumo comparativo (refazer de memória)

| | Volume | Shape | 1st order | Texture | DVF |
|---|---|---|---|---|---|
| Usa intensidade? | não* | não | sim | sim | indiretamente (registro) |
| Usa posição dos voxels? | não | sim | não | sim (vizinhos) | sim |
| Precisa de referência? | não | não | não | não | **sim** (template) |
| Normalizado por ICV? | sim | tamanhos sim | não | não | já normalizado pelo afim do registro |
| Sinal típico da atrofia | volume ↓, CSF ↑ | esfericidade ↓ | média ↓, dispersão ↑ | contraste ↑, homogeneidade ↓ | deformação ↑ |

\* `gm/wm/csf_norm` vêm da segmentação de tecidos, que usa a intensidade para classificar.

---

## Passo 8 — Perguntas de fixação (responder sem olhar)

1. Quais são as 5 famílias e a pergunta que cada uma responde?
2. Por que dividir comprimento por ICV^(1/3) e não por ICV?
3. Qual a diferença entre 1st order e Texture? Dê o exemplo do tabuleiro de xadrez.
4. Por que "1st order moments" é um rótulo errado?
5. O que o `csf_norm` mede e para que lado vai com a atrofia?
6. O que é `Sphericity` e por que cai com atrofia focal?
7. O que a GLCM conta? O que significa uma diagonal "cheia"?
8. Quais são os 3 mapas do DVF (D-V-S) e o que cada um representa?
9. Quais são as 4 estatísticas aplicadas a cada mapa do DVF?
10. Por que `Energy`/`TotalEnergy` e `VoxelVolume` saem do classificador?
11. Qual a diferença entre `t1_only`, `t1_r10` e `t1_ols`?
12. No OASIS, `jac_det > 1` significa atrofia ou expansão? Por quê?

---

# Métodos estatísticos — fórmulas, termos e exemplos

Referência: `7_stats.ipynb`, `modules/stats_compare.py`. Classe positiva = pMCI.

| Análise | Pergunta | Método |
|---|---|---|
| §1 Idade sMCI vs pMCI | idades diferem? | Mann-Whitney U |
| §1 Sexo sMCI vs pMCI | proporção M/F difere? | Qui-quadrado |
| §2 Confundimento | idade/sexo sozinhos preveem? imagem supera demografia? | Permutação da AUC + bootstrap pareado ΔAUC |
| §3/§4 Baseline vs 2v/B/D | longitudinal supera baseline? | Bootstrap pareado ΔAUC + FDR BH (por coorte) |
| §5 Fusões (tetos) | contrastes pré-especificados diferem? | Bootstrap pareado + BH (7 contrastes) |
| §6 Clínico vs imagem | imagem acrescenta ao clínico? | Bootstrap pareado ΔAUC |
| §7 Soft True vs False | definição de coorte muda AUC? | descritivo (pacientes diferentes → sem pareamento) |
| §8 ComBat vs nocombat (só T1) | harmonização muda AUC? | Bootstrap pareado ΔAUC + BH (5 famílias) |
| §10 Sensibilidade | achados resistem a α = 1%? | mesmos p + IC 99% + BH por coorte e global |

---

## 1. Score do paciente

$$s_i = \frac{1}{R}\sum_{r=1}^{R} s_{i,r}$$

| Termo | Significado |
|---|---|
| $s_{i,r}$ | probabilidade de pMCI dada ao paciente $i$ na repetição $r$, quando ele estava no fold de teste (out-of-fold) |
| $R$ | nº de repetições da validação cruzada (10) |
| $s_i$ | score final do paciente, usado em todas as AUCs |

Exemplo: paciente recebeu 0,62; 0,70; 0,58; … (10 valores) → $s_i$ = média ≈ 0,64.

---

## 2. AUC (área sob a curva ROC)

$$\text{AUC} = \frac{1}{n_1 n_0}\sum_{i \in \text{pMCI}}\sum_{j \in \text{sMCI}} \Big[\mathbb{1}(s_i > s_j) + 0{,}5\cdot\mathbb{1}(s_i = s_j)\Big]$$

| Termo | Significado |
|---|---|
| $n_1$, $n_0$ | nº de pMCI e de sMCI |
| $\mathbb{1}(\cdot)$ | vale 1 se a condição é verdadeira, 0 caso contrário |
| AUC | probabilidade de um pMCI sorteado receber score maior que um sMCI sorteado. 0,5 = acaso; 1 = separação perfeita |

Exemplo: pMCI = {0,8; 0,4}, sMCI = {0,5; 0,3}. Pares: 0,8>0,5 ✓; 0,8>0,3 ✓; 0,4>0,5 ✗; 0,4>0,3 ✓ → AUC = 3/4 = **0,75**.

Por que AUC: não depende de limiar e não é distorcida pelo desbalanceamento (120 pMCI vs 73 sMCI), ao contrário da acurácia.

### ΔAUC

$$\Delta\text{AUC} = \text{AUC}_{\text{método}} - \text{AUC}_{\text{baseline}}$$

Exemplo (ilustrativo): D = 0,733, T1 = 0,600 → ΔAUC = **+0,133** (D melhor). ΔAUC não é "taxa de acerto".

---

## 3. Bootstrap pareado da ΔAUC (§2–§6, §10)

Ideia: simular "e se eu repetisse o estudo com outros n pacientes parecidos?".

**Passo 1 — reamostrar** (repetir $b = 1,\dots,B$):

$$I^{*}_b = (i_1, \dots, i_n), \quad i_k \sim \text{Uniforme}\{1,\dots,n\} \text{ com reposição}$$

$$\Delta^{*}_b = \text{AUC}_{\text{método}}(I^{*}_b) - \text{AUC}_{\text{baseline}}(I^{*}_b)$$

| Termo | Significado |
|---|---|
| $n$ | nº de pacientes da coorte (ex.: 120 em 48m_12m). **Cada reamostra tem tamanho n** |
| $B$ | nº de reamostras = 5000 (5000 conjuntos, não amostras menores) |
| com reposição | paciente pode sair repetido ou não sair (~37% ficam fora de cada reamostra) |
| pareado | as duas AUCs são calculadas nos **mesmos** pacientes sorteados → cancela a dificuldade individual de cada paciente |
| $\Delta^{*}_b$ | ΔAUC na reamostra $b$ |

Reamostras sorteadas só com uma classe são descartadas (AUC indefinida); $B'$ = nº de reamostras válidas.

**Passo 2 — IC por percentis** (nível $1-\alpha$):

$$\text{IC}_{1-\alpha} = \Big[Q_{\alpha/2}(\Delta^{*}),\; Q_{1-\alpha/2}(\Delta^{*})\Big]$$

| Termo | Significado |
|---|---|
| $Q_x$ | percentil $x$ das 5000 ΔAUC ordenadas |
| IC 95% | percentis 2,5 e 97,5 → 125º e 4875º valor |
| IC 99% | percentis 0,5 e 99,5 → 25º e 4975º valor (mais largo) |

**Passo 3 — p-valores:**

$$p_{\text{uni}} = \frac{1 + \sum_{b=1}^{B'} \mathbb{1}(\Delta^{*}_b \le 0)}{B' + 1} \qquad p_{\text{bi}} = 2\cdot\min(p_{\text{uni}},\, 1 - p_{\text{uni}})$$

| Termo | Significado |
|---|---|
| $p_{\text{uni}}$ | fração de reamostras em que o método **não** superou o baseline. H1: método > baseline |
| +1 | evita p = 0 (resultado nunca é "impossível") |
| $p_{\text{bi}}$ | versão bilateral (H1: método ≠ baseline) |

**Exemplo real (48m_12m, vol, D vs T1):** ΔAUC = 0,133.
- 5000 reamostras; ~19 deram $\Delta^{*} \le 0$ → $p_{\text{uni}} = (19+1)/5001 = $ **0,004**; $p_{\text{bi}} = 0{,}008$.
- 125º valor = 0,040; 4875º = 0,227 → IC 95% **[0,040; 0,227]**.
- 25º valor = 0,007; 4975º = 0,256 → IC 99% **[0,007; 0,256]**.

Leitura: IC exclui 0 → dados compatíveis com ganho positivo. p ≠ "probabilidade de ser sorte".

Relação IC ↔ p: IC $1-\alpha$ exclui 0 ⇔ $p_{\text{uni}} < \alpha/2$ (aprox.). Exigir IC 95% > 0 = teste bilateral a 5%.

---

## 4. Teste de permutação da AUC (§2)

Pergunta: "o modelo é melhor que o acaso (AUC > 0,5)?".

$$p_{\text{perm}} = \frac{1 + \sum_{b=1}^{B} \mathbb{1}\big(\text{AUC}(\pi_b(y), s) \ge \text{AUC}_{\text{obs}}\big)}{B + 1}$$

| Termo | Significado |
|---|---|
| $\pi_b(y)$ | rótulos sMCI/pMCI embaralhados aleatoriamente (quebra qualquer relação real com o score) |
| $B$ | 5000 permutações |
| $\text{AUC}_{\text{obs}}$ | AUC real com rótulos verdadeiros |

Exemplo (ilustrativo): AUC do modelo só-idade = 0,70; em 9 de 5000 embaralhamentos AUC ≥ 0,70 → $p = 10/5001 = $ **0,002** → idade sozinha prevê acima do acaso.

Diferença do bootstrap: permutação compara **um** modelo com o acaso; bootstrap pareado compara **dois** modelos entre si.

---

## 5. Mann-Whitney U (§1, idade)

Compara posições (ranks) dos dois grupos; não exige distribuição normal.

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2} \qquad U_0 = n_1 n_0 - U_1$$

| Termo | Significado |
|---|---|
| $R_1$ | soma dos ranks do grupo pMCI após ordenar todas as idades juntas |
| $n_1$, $n_0$ | tamanhos dos grupos |
| $U_1$ | nº de pares (pMCI, sMCI) em que o pMCI é mais velho |
| p | calculado a partir da distribuição de U sob H0 (grupos iguais), bilateral |

Exemplo (ilustrativo): pMCI = {72, 75, 80}, sMCI = {70, 73}. Ranks: 70→1, 72→2, 73→3, 75→4, 80→5. $R_1 = 2+4+5 = 11$; $U_1 = 11 - 6 = 5$ de $n_1 n_0 = 6$ pares → pMCI tendem a ser mais velhos.

Curiosidade: $U_1 / (n_1 n_0)$ = AUC. AUC é a estatística de Mann-Whitney aplicada a scores.

---

## 6. Qui-quadrado de independência (§1, sexo)

$$\chi^2 = \sum_{\text{células}} \frac{(O - E)^2}{E} \qquad E = \frac{\text{total da linha}\times\text{total da coluna}}{N} \qquad gl = (r-1)(c-1)$$

| Termo | Significado |
|---|---|
| $O$ | contagem observada na célula (ex.: pMCI homens) |
| $E$ | contagem esperada se sexo e grupo fossem independentes |
| $N$ | total de pacientes |
| $gl$ | graus de liberdade; tabela 2×2 → 1 |

Exemplo (ilustrativo, N = 193):

| | M | F | total |
|---|---|---|---|
| pMCI | 70 | 50 | 120 |
| sMCI | 40 | 33 | 73 |
| total | 110 | 83 | 193 |

$E_{\text{pMCI,M}} = 120\cdot110/193 = 68{,}4$ … $\chi^2 \approx 0{,}23$, gl = 1 → p ≈ 0,63 → sem diferença de sexo.
Obs.: `scipy.stats.chi2_contingency` aplica correção de Yates em 2×2 (p um pouco maior).

---

## 7. Correção de múltiplas comparações — FDR Benjamini–Hochberg (§3–§5, §10)

Problema: m testes a α = 5% → ~$m \cdot 0{,}05$ falsos positivos esperados mesmo sem efeito real (20 testes → ~1; 60 testes → ~3).

**FDR** (False Discovery Rate) = proporção esperada de falsos positivos **entre os resultados declarados significativos**. BH garante FDR ≤ α.

$$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)} \qquad q_{(k)} = \min_{j \ge k} \; \min\!\Big(1,\; \frac{m\, p_{(j)}}{j}\Big)$$

| Termo | Significado |
|---|---|
| $m$ | nº de testes da família (5 por coorte; 20 global; 7 nos tetos) |
| $p_{(k)}$ | k-ésimo menor p-valor |
| $k$ | posição (rank) do p na ordem crescente |
| $m\,p/k$ | "pedágio": p multiplicado por m/k (o menor p paga mais) |
| $\min_{j \ge k}$ | garante que q nunca diminui ao descer no ranking (mantém a ordem) |
| $q$ | menor α em que o teste seria declarado descoberta |

Passos: (1) ordenar p; (2) $m\,p/k$; (3) de baixo para cima, cada q = mínimo entre ele e o q abaixo.

**Exemplo por coorte (m = 5; vol real, demais ilustrativos):**

| Família | p | k | $5p/k$ | q | q < 0,05? |
|---|---|---|---|---|---|
| vol | 0,004 | 1 | 0,020 | **0,020** | sim |
| texture | 0,030 | 2 | 0,075 | 0,075 | não |
| shape | 0,200 | 3 | 0,333 | 0,333 | não |
| disp | 0,450 | 4 | 0,563 | 0,563 | não |
| firstorder | 0,800 | 5 | 0,800 | 0,800 | não |

texture: p = 0,03 parecia significativo; após BH q = 0,075 → "raw sig." mas não "FDR sig.".

**Global (m = 20):** vol continua menor p → $q = 0{,}004 \cdot 20/1 = $ **0,080** > 0,05 → não passa.

Leitura de q = 0,020: se declaro significativos todos os testes com q ≤ 0,02, espera-se no máximo 2% de falsos entre eles.

Por que BH e não Bonferroni ($p \cdot m$ para todos): Bonferroni controla a chance de **qualquer** falso positivo; com testes correlacionados (mesmos pacientes em todas as famílias) é conservador demais. BH é o padrão para várias famílias de atributos.

Família por coorte vs global: por coorte = cada coorte é uma pergunta separada (critério atual); global = uma pergunta única (mais rigoroso). Reportar o outro como sensibilidade.

---

## 8. Rótulos de significância

| Rótulo | Regra | Significado |
|---|---|---|
| FDR sig. | $q < \alpha$ **e** $\text{IC}_{\text{lo}} > 0$ | ganho resiste à correção de múltiplos testes |
| raw sig. | $p_{\text{uni}} < \alpha$ **e** $\text{IC}_{\text{lo}} > 0$ | ganho no teste isolado, não resiste à correção |
| sem evidência | caso contrário | não mostrou superioridade (≠ provar que não há efeito) |

Por que exigir p/q **e** IC: p diz se há evidência; IC mostra tamanho e incerteza. Exigir IC > 0 evita declarar ganho cujo intervalo inclui perda.

---

## 9. Sensibilidade α = 1% (§10)

Mesmos p (bootstrap refeito com as mesmas seeds); muda só o limiar e o IC.

$$\text{sig}_{1\%} = (p \text{ ou } q < 0{,}01) \;\wedge\; Q_{0{,}005}(\Delta^{*}) > 0$$

Critérios: sem correção, FDR por coorte, FDR global (20).

**Resultados atuais (unimodal, 60 testes):**

| Contraste | ΔAUC | IC 95% | IC 99% | p | q coorte | q global | 5% | 1% |
|---|---|---|---|---|---|---|---|---|
| D vol 48m_12m | 0,133 | [0,040; 0,227] | [0,007; 0,256] | 0,004 | 0,020 | 0,080 | FDR sig. | só sem correção |
| 2v texture 36m_6m | 0,076 | [0,012; 0,143] | [−0,004; 0,162] | 0,009 | 0,045 | 0,180 | FDR sig. | não |
| D texture 36m_6m | 0,055 | [0,005; 0,104] | [−0,010; 0,119] | 0,016 | 0,080 | 0,160 | raw sig. | não |
| B vol 48m_12m | 0,085 | [0,004; 0,167] | [−0,024; 0,191] | 0,021 | 0,104 | 0,382 | raw sig. | não |

Racional: 60 testes a 5% → ~3 falsos positivos esperados; 2 FDR sig. e 4 raw sig. são compatíveis com acaso. α = 1% testa robustez do achado principal.

**Frase para o artigo:** "Na coorte 48m_12m, o método D superou o baseline no volume (ΔAUC = 0,133; IC 95% 0,040–0,227; p = 0,004), resistindo à correção FDR dentro da coorte (q = 0,020), mas não à correção global sobre 20 contrastes (q = 0,080). Em α = 1%, o efeito se manteve apenas sem correção (IC 99% 0,007–0,256). Trata-se de achado sugestivo e localizado, que requer confirmação independente."

---

## 10. Escolha do método D entre B, C, D — pendente (não implementado)

Hoje: escolha por valores absolutos → sem teste e com viés de seleção (mesmos dados escolhem e testam).

Opções: (a) justificar a priori (D usa as 3 visitas numa reta: nível + inclinação, menos sensível a ruído de uma visita); (b) teste de Friedman; (c) bootstrap pareado D vs B e D vs C + BH.

**Friedman** (Demšar, 2006 — comparar métodos em vários cenários):

$$\chi^2_F = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k} \bar{R}_j^{\,2} - \frac{k(k+1)^2}{4}\right]$$

| Termo | Significado |
|---|---|
| $N$ | nº de blocos (cenários) = 5 famílias × 4 coortes = 20 |
| $k$ | nº de métodos comparados (ex.: B, C, D → 3) |
| $\bar{R}_j$ | rank médio do método $j$ (1º = maior AUC em cada bloco) |
| gl | $k-1$ |

Exemplo (ilustrativo): ranks médios D = 1,6; B = 2,0; C = 2,4 (N = 20, k = 3) → $\chi^2_F = \frac{240}{12}[1{,}6^2 + 2^2 + 2{,}4^2 - 12] = 20 \cdot 0{,}32 = 6{,}4$, gl = 2 → p ≈ 0,04 → algum método se destaca; seguir com Wilcoxon pareado/Nemenyi + correção.
Ressalva: blocos não totalmente independentes (mesmos pacientes em várias famílias).

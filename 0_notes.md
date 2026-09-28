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

# TODO — handoff lab → laptop

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Branch:** `main` (merge de `exp/disp-mag-paper-attrs` feito; push `main` pode estar pendente)  
**Docs:** `artigo.md` = briefing reunião; este ficheiro = pendências técnicas

---

## Achados já consolidados (`48m_12m`, svm, smci, nocombat)

| Setup | AUC patient | Nota |
|-------|------------:|------|
| Mono teto `t1_only/shape` | **0.786** | baseline |
| Early concat shape T1 ∪ tex Δ | 0.794 | gate FAIL |
| Late multi-T1 | ≈0.76 | abaixo do mono |
| Late âncora shape T1 ∪ tex Δ | **0.823** | claim oficial; gate FAIL |
| Late tripla (+vol Δ) | **0.829** | ponto; gate FAIL |

Scripts: `5_ablation_late_fusion.py`, `5_ablation_early_fusion.py`, `scripts_compare_fusion_vs_shape.py`.

---

## NOVO — gap 6m/12m: só mínimo hoje (vulnerabilidade do orientador)

**Código actual** (`1_dataset.ipynb`, `_pick_evenly_spaced`): só `gap ≥ 6` ou `≥ 12` meses. **Sem máximo.**  
Selecção = extremos do pool + meio equidistante → coorte “6 m” estica gaps.

**Gaps consecutivos reais (sMCI/pMCI, selecção actual):**

| Coorte | min | mediana | máx |
|--------|----:|--------:|----:|
| `36m_6m` | 6.0 | **12.1** | **24.1** |
| `48m_6m` | 6.0 | **12.2** | **34.5** |
| `36m_12m` | 12.0 | 12.5 | 23.4 |
| `48m_12m` | 12.0 | **16.8** | **34.5** |

~70% dos gaps “6 m” estão **mais perto de 12 do que de 6**.  
`12m` ⊂ `6m` na mesma janela (100% dos IDs 12m estão no 6m).

**Como falar:** coortes = espaçamento **mínimo**, não protocolo ± banda. Fix: tabela/histograma de gaps + texto honesto; ou redesign com min+máx (rerun).

---

## NOVO — proposta orientador: banda ±3 / ±2 (min e máx)

### Regra de elegibilidade (estimativa “justa”)
Não filtrar o CSV actual (mata o 6 m). Em vez disso, no histórico ADNI (`csvs/adnimerged.csv`):

1. Mesma `classify_patient` (janela 36/48, soft pMCI).  
2. Existe triplo $(i_1,i_2,i_3)$ ordenado com **ambos** gaps consecutivos na banda.  
3. Conta 1 conjunto se ≥1 triplo válido (escolhe o mais perto do centro 6 ou 12).

### 1) Problema de sobreposição na fronteira (±3)

Se 6±3 = **[3, 9]** e 12±3 = **[9, 15]** (ambos fechados):  
**gap = 9 meses** pertence às duas bandas → ambiguidade. **Tem problema** se 9 entra nos dois.

**Fix (obrigatório se adoptarem bandas):** intervalos **disjuntos**, ex.:
- 6 m → **[3, 9)**  (inclui 3, exclui 9)  
- 12 m → **[9, 15]** (inclui 9)

Assim gap=9 vai **só** para 12 m. Alternativa: 9 em nenhuma → [3,9) + (9,15].

#### Notação de intervalos (lembrar)

| Escrita | Significado | Extremos |
|---------|-------------|---------|
| **[a, b]** | fechado | inclui **a** e **b** |
| **[a, b)** | semiaberto à direita | inclui **a**, **não** inclui **b** |
| **(a, b]** | semiaberto à esquerda | **não** inclui **a**, inclui **b** |
| **(a, b)** | aberto | não inclui a nem b |

Ex.: **[3, 9]** inclui o 9; **[3, 9)** vai até 8.999… e **exclui** o 9.  
Por isso **[3,9]∪[9,15]** partilha o 9; **[3,9)∪[9,15]** não.

Nota: mesmo com bandas disjuntas, o **mesmo paciente** pode entrar nas duas coortes se tiver *dois triplos diferentes* (um denso ~6 m e um ~12 m). Isso é overlap de ID, não ambiguidade de um gap. Nos números abaixo: ~60–70% dos IDs da banda 12 também têm triplo na banda 6.

### 2) Contagens sMCI / pMCI (reescolha no histórico, soft=True)

Estimativa lab 2026-08-07 (sem reaplicar MRQy/filtros finos; modo só-mínimo deu ~155/92 vs artigo 153/90 — ordem ok).

#### ±3 disjunto — `[3,9)` e `[9,15]`

| Janela | Banda | sMCI | pMCI | Total | Artigo (só mín.) |
|--------|-------|-----:|-----:|------:|------------------|
| 36 m | 6±3 [3,9) | 149 | 119 | **268** | 153/90 |
| 36 m | 12±3 [9,15] | 134 | 39 | **173** | 69/21 |
| 48 m | 6±3 [3,9) | 87 | 137 | **224** | 96/109 |
| 48 m | 12±3 [9,15] | 81 | 56 | **137** | 71/40 |

Overlap IDs 6∩12: 36m ≈169; 48m ≈132.

#### ±2 — `[4,8]` e `[10,14]` (faixa 8–10 “órfã”, sem partilha de bordo)

| Janela | Banda | sMCI | pMCI | Total |
|--------|-------|-----:|-----:|------:|
| 36 m | 6±2 [4,8] | 140 | 112 | **252** |
| 36 m | 12±2 [10,14] | 127 | 35 | **162** |
| 48 m | 6±2 [4,8] | 82 | 130 | **212** |
| 48 m | 12±2 [10,14] | 77 | 52 | **129** |

Overlap IDs 6∩12: 36m ≈157; 48m ≈124.

### ±3 vs ±2 — o que compensa?

| Critério | ±3 disjunto | ±2 |
|----------|-------------|-----|
| n | **maior** (~+6–8% vs ±2) | menor |
| Fronteira 9 m | resolvida com `[3,9)` / `[9,15]` | não toca (gap 8–10 vazio) |
| Separação 6 vs 12 | bandas coladas em 9 | **mais clara** (buraco 2 m) |
| Custo | rerun total igual | rerun total igual |
| Risco | 48m 6±3 muito pMCI-pesado (87/137) | idem, um pouco menos n |

**Recomendação lab:**  
- Se o objectivo é **fechar a ambiguidade do orientador com máximo n** → **±3 com `[3,9)` + `[9,15]`**.  
- Se o objectivo é **contraste limpo 6 vs 12** (menos zona cinzenta) → **±2 `[4,8]` + `[10,14]`** (pequena perda de n).  
- **Não** usar ±3 fechado `[3,9]`∩`[9,15]` (gap=9 indefinido).  
- Qualquer banda = **nova população** → rerun ablações (não só filtrar CSV actual: filtrar actual deixa 6 m em ~3 sMCI / 39 pMCI).

**Alternativa sem rerun:** manter só-mínimo + reportar distribuição/máx dos gaps + linguagem “mínimo nominal”.

---

## Paper — ainda fazer

- [ ] Tabela resumo âncora: mono | multi-T1 | late | early + Δ/IC/gate  
- [ ] Forest Δ âncora vs shape T1 (4 cohorts)  
- [ ] Decidir com orientador: (A) honestidade só-mínimo + figura gaps, ou (B) redesign ±2/±3 + rerun  
- [ ] Se (B): regenerar cohorts em `1_dataset.ipynb` com min+máx; depois mono → late → early  
- [ ] Escrever Results; alinhar `artigo.md`

## Não fazer (enquanto claim Art.1 actual)

- Crownear top late com disp  
- Grade early completa / weighted late  
- Claim “significativamente > shape” sem `gate_pass`

---

## Status

Experimentos `48m_12m` + early/late pares **ok** no desenho actual (só mínimo).  
**Bloqueio metodológico novo:** definição de gap sem máximo — resolver na reunião antes de mais runs grandes.

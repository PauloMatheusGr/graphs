# TODO — fechar paper (handoff lab → laptop)

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Estado:** experimentos **`48m_12m` fechados**. Próximo = expansão mínima da âncora late + escrita.

---

## Achados já consolidados (`48m_12m`, svm, smci, nocombat)

| Setup | AUC patient | Nota |
|-------|------------:|------|
| Mono teto `t1_only/shape` | **0.786** | baseline |
| Mono `t1_deltas/shape` | ≈0.787 | empate com T1 |
| Early concat shape T1 ∪ tex Δ | 0.794 | gate FAIL; early esgotado |
| Late multi-T1 (ex. shape∪tex T1) | ≈0.76 | **abaixo** do mono |
| Late âncora shape T1 ∪ tex Δ | **0.823** | gate FAIL (IC cruza 0) |
| Late tripla shape T1 ∪ vol Δ ∪ tex Δ | **0.829** | melhor ponto; gate FAIL |

**Conclusões:** late mean > early; long late > multi-T1 e > mono no ponto; gate vs shape n.s.  
**Método:** só `--combine mean`. Não weighted / early grid / vol concat.

Scripts: `5_ablation_late_fusion.py`, `scripts_compare_fusion_vs_shape.py --mode late`.

---

## Enredo do paper (3 atos)

1. Mono por cohort × gap 6m vs 12m (vol/texture ganham com 12m; shape empata; 6m redundante).  
2. Fusão justa em **48m_12m**: mono | multi-T1 late | long late | early ref.  
3. Replicação: âncora late nos outros cohorts → claim A (12m consistente) ou B (só 48m_12m).

---

### 4. Fazer
- [ ] Tabela 48m_12m: mono | multi-T1 | long | early + Δ/IC/gate  
- [ ] Forest Δ âncora late vs shape T1 local (4 cohorts)  
- [ ] Escrever Results IMRAD (3 atos)  
- [ ] Commit código se ainda pendente no laptop  

---

## Não fazer

- Reabrir grade pairwise / early / triplas / outros modelos em `48m_12m`  
- Crowdear early ou full ablation nos outros cohorts  
- Claim “significativamente > shape T1” sem `gate_pass`  
- `--combine weighted` como método principal  

---

## Status

`48m_12m` late/early/matriz justa **ok**. Falta: **âncora late × 3 cohorts** → rebuild → figuras/texto.

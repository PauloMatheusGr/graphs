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

## Fazer agora (laptop / tmux)

```bash
cd /mnt/study-data/pgirardi/graphs
```

### 1. Âncora late nos 3 cohorts restantes

```bash
for co in 36m_6m 36m_12m 48m_6m; do
  echo "======== LATE ANCHOR $co ========"
  .venv/bin/python 5_ablation_late_fusion.py \
    --fusion shape:t1_only,texture:t1_deltas --combine mean \
    --cohort "$co" --tasks smci_pmci --selection l1_stable \
    --models svm --combat false \
    --run-missing --repeats 10 --tuner optuna --optuna-trials 30 \
    --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
    --stable-bootstrap 50 --stable-l1-c 0.1

  .venv/bin/python scripts_compare_fusion_vs_shape.py \
    --cohort "$co" --mode late --fingerprint t1_shape__t1_deltas_texture || true
done
```

### 2. Multi-T1 late (só se §1 subir vs shape local)

```bash
for co in 36m_6m 36m_12m 48m_6m; do
  .venv/bin/python 5_ablation_late_fusion.py \
    --fusion shape:t1_only,texture:t1_only --combine mean \
    --cohort "$co" --tasks smci_pmci --selection l1_stable \
    --models svm --combat false \
    --run-missing --repeats 10 --tuner optuna --optuna-trials 30 \
    --stable-pool-min-pct 70 --stable-pool-min-timepoints 0
done
```

Fingerprint tipicamente `t1_shape__t1_texture`.

### 3. Rebuild planilha

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "modules")
from cohort_compare import save_cohort_comparison
save_cohort_comparison(
    ["36m_6m", "36m_12m", "48m_6m", "48m_12m"],
    Path("csvs/cohort_comparison"),
    n_boot=2000,
)
print("ok → csvs/cohort_comparison/cohort_results.csv")
PY
```

### 4. Depois dos runs
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

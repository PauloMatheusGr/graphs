# Handoff — early vs late fusion (`48m_12m`)

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Decisão de método:** se late fusion for **melhor que early** (AUC patient + preferencialmente gate vs shape T1), **late vira o método escolhido** do paper.  
**Late fusion oficial:** só `--combine mean` (média uniforme de probas OOF). Sem weighted no método principal.

---

## 1. Goal & gate

- Tarefa: sMCI vs pMCI · cohort **`48m_12m`** · SVM · `l1_stable` · combat false  
- Teto transversal: `t1_only` / `shape` ≈ **0.786**  
- Gate vs shape: ΔAUC>0 **e** IC95 lo>0 → `scripts_compare_fusion_vs_shape.py --mode early|late`  
- Comparar early vs late no **mesmo** fingerprint de slots (ex. shape T1 ∪ texture t1_deltas)

| Método | Melhor AUC patient (mesmo par texture) | Gate vs shape |
|--------|----------------------------------------:|---------------|
| Early concat | 0.794 | FAIL |
| Late mean | **0.823** | FAIL (IC lo ≈ −0.01) |

Late já **ganha early no ponto**. Falta (ideal) gate vs shape; mesmo sem gate, late > early justifica escolher late como fusão.

Scores branches: ambas proba SVM em [0,1] — mean ok.

---

## 2. Scripts

| Script | Papel |
|--------|--------|
| `5_ablation_late_fusion.py` | **Candidato a método** — média de scores mono-mod |
| `5_ablation_early_fusion.py` | Baseline de fusão (concat) — já explorado |
| `scripts_compare_fusion_vs_shape.py` | Gate (`--mode late` / `early`) |

Saída late: `csvs/cohorts/{cohort}/ablation_results_late_fusion/{fingerprint}/`  
Protocol planilha: `late__{fingerprint}`

---

## 3. Principais CLIs — late fusion (`mean` only)

```bash
cd /mnt/study-data/pgirardi/graphs
```

Flags comuns:

```text
--combine mean \
--cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false
```

Default: `--reuse-disk` (rápido). `--run-missing` só se faltar CSV mono-mod.

### 3.1 Obrigatório — âncora (já rodou; re-gate OK)

```bash
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_only,texture:t1_deltas --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false

.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_shape__t1_deltas_texture
```

Fingerprint: `t1_shape__t1_deltas_texture` · AUC ≈ **0.823**

### 3.2 Controles 2-view (reuse disk)

```bash
# shape T1 ∪ vol t1_deltas
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_only,vol:t1_deltas --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false
.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_shape__t1_deltas_vol

# shape T1 ∪ vol deltas_only
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_only,vol:deltas_only --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false
.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_shape__deltas_vol

# shape t1_deltas ∪ texture t1_deltas
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_deltas,texture:t1_deltas --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false
.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_deltas_shape__t1_deltas_texture
```

### 3.3 Tripla (opcional)

```bash
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_only,vol:t1_deltas,texture:t1_deltas --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false
.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_shape__t1_deltas_vol__t1_deltas_texture
```

### 3.4 Se faltar mono-mod (ex. texture deltas_only)

```bash
.venv/bin/python 5_ablation_late_fusion.py \
  --fusion shape:t1_only,texture:deltas_only --combine mean \
  --cohort 48m_12m --tasks smci_pmci --selection l1_stable --models svm --combat false \
  --run-missing --repeats 10 --tuner optuna --optuna-trials 30
.venv/bin/python scripts_compare_fusion_vs_shape.py \
  --cohort 48m_12m --mode late --fingerprint t1_shape__deltas_texture
```

### 3.5 Rebuild planilha

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

---

## 4. Early fusion (referência — não crowdear)

Melhor early: `t1_shape__t1_deltas_texture` ≈ **0.794**, gate fail. Vol dilui.  
Script: `5_ablation_early_fusion.py`. Comparar com late no mesmo fingerprint via `--mode early` vs `--mode late`.

---

## 5. Critério de escolha late vs early

1. Rodar late `mean` nos CLIs §3 (âncora + controles).  
2. Para cada fingerprint: AUC late vs AUC early (mesmo slots) e gate vs shape.  
3. **Escolher late** se AUC late > early no par principal (shape∪texture) — já verdadeiro (0.823 > 0.794).  
4. Gate vs shape: desejável; se continuar n.s., reportar late como melhor fusão **com** IC / p-value honestos (não claim “significativamente > T1” sem gate).

---

## 6. Não fazer

- `--combine weighted` como método principal  
- Mais early-fusion concat (vol / 3×)  
- Misturar cohorts  
- Claim “> shape T1” sem `gate_pass`  
- Grid de stable-pool / Optuna só na fusion  

---

## 7. Status

Late **mean** âncora: **0.823** vs early **0.794** vs shape **0.786**; gate late ainda FAIL.  
Próximo: controles §3.2–3.3 + rebuild; depois **fixar late como método de fusão** se continuar ≥ early. Weighted fora do plano.

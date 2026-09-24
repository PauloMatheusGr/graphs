# Ablação longitudinal volume (revisor 2)

**Objectivo:** fechar críticas do revisor 2 no mesmo nested CV do artigo:
(1) normalização temporal ÷Δt; (2) encoding consecutivos vs uma velocidade (taxa global / OLS).

**Escopo runs:** `--modality vol` primeiro em `48m_6m`. Código genérico a todas as famílias.

**Não sobrescreve** `plan.md` (fecho figs/stats).

---

## Braços

| ID | Representation | Vetor | Tempo |
|----|----------------|-------|-------|
| A | `t1_only` | `[V0]` | — |
| B₀ | `t1_d21_d32` | `[V0, Δ10, Δ21]` abs | não (encoding manuscrito) |
| B | `t1_r10_r21` | `[V0, r10, r21]` | sim — padrão temporal |
| C | `t1_rate02` | `[V0, (V2−V0)/(t2−t0)]` | sim |
| D | `t1_ols` | `[V0, β̂1]` OLS 3 pts | sim |

Unidade: **meses** desde baseline (`t0=0`); `dias / 30.436875`.

Contrastes: B vs B₀ (÷Δt); B vs C/D (estrutura); C vs D (taxa vs slope); \* vs A (vale longitudinal?).

Claim: `48m_6m`, soft True, combat false, SVM, `l1_stable`, seed 42, repeats 10, Optuna 10.

## CLI

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
for REP in t1_only t1_d21_d32 t1_r10_r21 t1_rate02 t1_ols; do
  python 5_ablation.py \
    --cohort 48m_6m --representation "$REP" \
    --modality vol --tasks smci_pmci --selection l1_stable \
    --models svm --combat false --repeats 10 --seed 42 \
    --tuner optuna --optuna-trials 10
done
```

A posteriori: mesmo loop com `--modality shape|texture|firstorder|disp`.

## Rebuild compare (após todos os runs, incl. soft_False)

**Não** corrige Q4 abs — só **adiciona** `t1_r10_r21` / `t1_rate02` / `t1_ols` a `cohort_results.csv` e escreve `ablation_ABCD_grid.csv`.

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
# esperar soft_False terminar
PYTHONPATH=modules python modules/cohort_compare.py --include-soft --n-boot 2000
```

Saídas em `csvs/cohort_comparison/`:
- `cohort_results.csv` — todos os protocolos (claim + ABCD + late/…)
- `cohort_features_long.csv`
- `ablation_ABCD_grid.csv` — A/B0/B/C/D + deltas

Sanidade sem escrever: `PYTHONPATH=modules python modules/cohort_compare.py --self-check`

## Resultados (48m_6m, vol, SVM, l1_stable, seed 42, R=10)

Fonte: `csvs/cohorts/48m_6m/ablation_vol_ABCD_table.csv` (corrida 2026-09-24).

| Braço | Representation | auc_patient_mean | Δ vs A | n_feat_mean |
|-------|----------------|------------------|--------|-------------|
| A | t1_only | 0.7559 | — | 2.04 |
| B₀ | t1_d21_d32 | 0.7630 | +0.007 | 9.32 |
| B | t1_r10_r21 | 0.7764 | +0.020 | 9.32 |
| C | t1_rate02 | 0.7828 | +0.027 | 5.98 |
| D | t1_ols | 0.7838 | +0.028 | 5.96 |

### Interpretação (Discussão)

1. **B vs B₀ (+0.013):** ÷Δt nas taxas consecutivas melhora vs deltas absolutos do manuscrito → normalização temporal importa mesmo com intervalos ~6 m.
2. **C ≈ D ≫ B₀:** colapsar a uma velocidade (taxa global / OLS) supera encoding `[V0,Δ10,Δ21]` → crítica do revisor 2 confirmada na família volume.
3. **C ≈ D (0.783 vs 0.784):** com 3 visitas quase equiespaçadas, OLS ≈ taxa ponta-a-ponta (como o revisor previa).
4. **C/D vs A (~+0.028):** longitudinal **com** tempo + encoding estável acrescenta ao baseline volume (não é “zero ganho”).
5. **Mensagem para o artigo:** o claim “longitudinal não ajuda” no volume era em parte artefacto de deltas consecutivos absolutos; com taxas/slope o ganho é modesto mas consistente. B₀ fica como controlo do encoding publicado.

### Próximo (se Discussão exigir)

Repetir CLI com outras modalidades; soft False / `48m_12m` só se precisar de SNR / pré-conversão.

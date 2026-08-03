# TODO — fusão shape T1 + deltas (furar teto)

Prompt: *implementa e roda a âncora abaixo; depois exploratórios.*

## 1. Implementar feature set cross-mod
Hoje **não existe** modality/rep pra `shape_T1 ∪ X_deltas`. Fazer:

- Concat cols: `shape` com `t1_only` + `{vol|texture|disp}` com `deltas_only`
- Hook em `modules/ablation_representation.py` (`feature_columns_for_representation`) e/ou CLI novo fino
- Pasta saída: `csvs/cohorts/48m_12m/ablation_results_fusion_shape_t1_{mod}_deltas/`
- Mesmo pipeline: SVM · Optuna · `l1_stable` · combat=False · 10×5 · `auc_patient_mean`

## 2. Rodar âncora (pré-registrada)
Cohort **`48m_12m`** só.

```text
fusion = shape_t1 ∪ vol_deltas_only
```

Comparar bootstrap pareado vs baseline já existente:
`csvs/cohorts/48m_12m/ablation_results_t1_only/shape/`  
(sucesso = ΔAUC>0 com IC/p estilo `7_stats`, não só ponto > 0.78)

## 3. Rodar exploratórios (só depois da âncora)
```text
shape_t1 ∪ texture_deltas_only
shape_t1 ∪ disp_deltas_only
```
FDR nos 3 testes (âncora + 2).

## 4. Controles (não re-rodar se CSV já existe)
- `t1_only/shape` — teto
- `t1_only/all`, `deltas/all` — união densa já ≤ teto
- Opcional: `shape_t1 ∪ shape_deltas_only` (espera Δ≈0)

## 5. Não fazer
- Misturar cohort 36m com 48m
- Crownear exploratório se âncora falhar
- Reivindicar “longitudinal > shape T1” sem ΔAUC+IC vs shape T1
- Mexer em disp/DVF neste TODO

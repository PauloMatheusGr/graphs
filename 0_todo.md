# Handoff Cursor (2026-07-31) — andamento do paper

## Objetivo
Avaliar se encodings **longitudinais (deltas)** melhoram discriminação **sMCI→pMCI** vs **baseline (T1-only)**, por modalidade hipocampal.  
**Não** reivindicar: “longitudinal > melhor baseline global”.  
**Sim** reivindicar: ganho **modalidade- e janela-dependente**.

## Duas coortes primárias (papéis distintos)
| Papel | Cohort | N (sMCI+pMCI) | Uso |
|-------|--------|---------------|-----|
| Âncora metodológica | `36m_6m` | 243 | potência, ranking, redundância gap curto (deltas≈T1 em vol) |
| Âncora longitudinal | `48m_12m` | 111 | claim formal **vol_deltas vs vol_T1**; tabela T1 vs deltas 5 mods |

Outras (`36m_12m`, `48m_6m`) = sensibilidade / figura de Δ — sem fishing de p-values.

## Achados (pré-rerodada; AUC patient-level)
- **Shape-T1** = melhor estático (teto); deltas de shape **não** ganham.
- **Vol:** ganho forte em `48m_12m` (~0.65→0.75); Δ≈0 em gap curto (`36m_6m`).
- **Texture:** ganho consistente nas 4 cohorts.
- **All:** eco da texture; fusão não bate melhores singles.
- **Disp:** ~acaso — **provisório**; DVF sendo re-extraído (correção fixa/móvel). Não fechar discussão de disp até rerun.

## Pipeline seleção (só isto)
`l1_stable` = 50 bootstraps no TRAIN outer → StandardScaler **por boot** → corr → var → LogReg L1 → pool (τ=70%, ≥2 tempos) → corr/var de novo → SVM.  
**Sem mRMR.** Nested CV: 10 repeats × 5 outer folds. TEST nunca entra na seleção.  
Patches recentes: scaler no L1-boot; bootstrap **com reposição** (sem `np.unique`).  
Métrica de ranking: **`auc_patient_mean`** (média OOF por `ID_PT`).

## Rerodada em andamento (Optuna nos dois encodings)
Tmux paralelo; ordem por cohort: `t1_only` → `t1_deltas`.

```bash
# Ex.: 48m_12m (igual para 36m_6m trocando --cohort)
python 5_run_ablation.py --cohort 48m_12m --representation t1_only \
  --modality vol,shape,texture,all --tasks smci_pmci --selection l1_stable \
  --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 30

python 5_run_ablation_deltas.py --cohort 48m_12m --representation t1_deltas \
  --modality vol,shape,texture,all --tasks smci_pmci --selection l1_stable \
  --models svm --combat false --repeats 10 --tuner optuna --optuna-trials 30
```

Saídas: `csvs/cohorts/{cohort}/ablation_results_t1_only|deltas/{mod}/`  
**Não** misturar tuner (T1 e deltas ambos Optuna). SVM só nesta rodada. `disp` depois do DVF.

## Próximos passos (outro Cursor)
1. Quando jobs terminarem: comparar **novo vs antigo** AUC patient (T1 vs deltas) em `36m_6m` e `48m_12m`.
2. Confirmar ranking de ganho: vol/texture sobem; shape ≈0; all ≤ singles.
3. Após DVF: `--modality disp` nas mesmas configs.
4. Opcional ROI paper: fusão **shape_T1 ∪ vol_deltas** em `48m_12m`.
5. Análise: `6_results.ipynb` / `7_stats.ipynb`; teste formal único = vol deltas vs vol T1 em `48m_12m`.
6. Discussão: ganho condicional; teto shape-T1; N menor em 48m_12m = limitação.

## Arquivos-chave
- CLIs: `5_run_ablation.py`, `5_run_ablation_deltas.py`
- Seleção: `modules/ablation_stable.py`, `CorrVarSelector` em `modules/ablation_runner.py`
- Plano experimental legado (parcialmente desatualizado): `readme.md`

---

# O que falta para fechar o artigo

Actualizado 2026-07-31.

## TODO

### Displacement / DVF
Displacement muito ruim (perto do acaso), porém teoricamente deveria ter resultados próximos dos resultados do volume, pois volume mostra atrofia nas parcelas normalizadas de gm,csf,wm enquanto que atributos dvf mostra o quanto se deformou. Solução: dado o conjunto de imagens longitudinais i={i1,i2,i3} realizar:

- O corregistro deformável entre i2 e i1 sendo i1 fixa e i2 móvel para gerar o campo de deformação entre i1 e i2.
- O corregistro deformável entre i3 e i1 sendo i1 fixa e i3 móvel para gerar o campo de deformação entre i1 e i3.
- O corregistro deformável entre i3 e i2 sendo i2 fixa e i3 móvel para gerar o campo de deformação entre i2 e i3.

A partir disso calcular os atributos do dvf.

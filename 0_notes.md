# Notas — pipeline do paper (pós Patch A)

Allowlist 4–5 nomes **morta**. Números em `cohort_results.csv` / tex antigos = experimento errado. Não citar.

## Perguntas

1. sMCI vs pMCI: Q4 (`t1_d21_d32` = T1+D21+D32) agrega sinal vs `t1_only`?
2. Quais atributos o seletor escolhe em cada família (frequência across folds×repeats)?
3. União **late** (3 specs) bate teto unimodal?

## Porquê late, não early

Late = 5 SVMs, um por família, média de scores. Forma não vê 144 GLCM. Volume não vê firstorder. `p≫n` da união **não** aplica: cresce nº de modelos, não a dimensão de um só.

Early / `--modality all` = concat. Classe cheia × Q4 → centenas de colunas, n=120. Fora do paper. `run_ablation_full.sh` não corre early.

## Espaço (antes do seletor)

Classe completa; denylist só definição (`keep_*_feat` em `ablation_prep.py` + `ablation_deltas.py`). Mesmos sufixos em `t1_only` e Q4. Cada sufixo × L e R. `t1_only` ×1 (T1). Q4 ×3 (T1, D21, D32).

| Família | Entra | Denylist | t1 L+R | Q4 L+R |
|---|---|---|---|---|
| vol | `gm_norm` `wm_norm` `csf_norm` `MeshVolume` | mm³ crus | 8 | 24 |
| shape | 12 IBSI (`original_shape_`) | MeshVolume, VoxelVolume | 24 | 72 |
| texture | 24 GLCM (`original_glcm_`) | GLRLM/GLSZM/GLDM/NGTDM | 48 | 144 |
| firstorder | 16 (`original_firstorder_`) | Energy, TotalEnergy | 32 | 96 |
| disp | momentos `mag_` `logjac_` `strain_fro_` (15 no long) | `ux/uy/uz`, `_n`, percentis | 30 | 90 |

Texture = GLCM Original. Sem wavelet/LoG.

## Seletor (`l1_stable`, só outer train)

1. Variance threshold (se zerar → `var>0`, não restaurar classe)
2. Corr `|ρ|>0.85` **uma vez**; fica maior `|ρ|` com `y`; nunca T1 vs D21/D32 do mesmo `anatomical_key`
3. 50× L1 `C=0.1`; coef=0 → `[]` (não keep-all)
4. π≥70% nos boots. Pool vazio → vazio, **nunca** a classe inteira
5. SVM nesse pool. `min_timepoints=0` em t1/Q4

Entre colineares: representante, não “melhor Haralick”. Reportar frequência across folds.

## Contrastes oficiais

- Uniclasse: 5 famílias × `{t1_only, t1_d21_d32}` × 4 coortes
- Late, 3 specs: tudo t1 · tudo Q4 · âncora `shape:t1_only ∪ resto Q4`
- Não crownear `wide`/`abs`/`t1_deltas`/`t1_deltas_rel`/`t1_ma` como claim longitudinal
- Extra (clínica, leaky, cn_ad, RF/EN): suplemento, não pergunta 1–3

## Paper (quando CSVs novos existirem)

Unimodal = Δ(Q4−t1) por família. Late = teto por juntar papéis, não “onde o tempo mais ganha”. Nomes = tabela de frequência, não assinatura única.

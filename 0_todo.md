# v0 100%

**Nunca** `WIPE=1`.

`cn_ad` de 2026-08-30 falhou (`minority=0` + `--results-dir vol_cn_ad`). Ignora. **Não** mistures disp em `vol_cn_ad`.

`3.1` a correr (`all_population.csv` → `displacement_field_v3`). SKIP sMCI/pMCI. `3.2` já aponta para a mesma lista. Late fusion (`WIPE=0 SKIP_MONO=1`) também a correr noutro terminal.

---

## Feito

- [x] SVM uniclasse disp v3 × 4 coortes (`t1_only`, `t1_d21_d32`)
- [x] Stats claim (`7_stats`) + prosa/tabela disp no `artigo.tex`
- [x] **A.** RF + elasticnet disp, `48m_12m`, sMCI/pMCI, T1 e Q4
- [x] Heatmaps sMCI/pMCI T1/Q4 re-exportados
- [x] **B.1** `3.1` CSV = `all_population.csv` (não `smci_pmci`)

## A correr

- [ ] **B.2** `3.1` warps CN/AD (espera `ok`/`skip`/`err` no fim)
- [ ] **D** late fusion 3 specs × 4 coortes (`--reuse-disk`)

## Ainda por fazer

### B. DVF CN+AD — depois do 3.1 acabar

3. `3.2_feat_dvf_v2.py` — `IMAGES_CSV` já é `all_population.csv`, `RESUME=True`. **Não** apaga `features_displacement_v3.csv`.
4. `4_run_post_extract.py`: `COHORT = "48m_12m"` (e as outras 3 se o left-join de CN/AD importar). Regenera `disp_long.csv`.

Sanity: `disp_long` tem `GROUP` CN e AD com feats **não-NaN**. Sem isto ablation `cn_ad` volta a `minority=0`.

### C. Ablation `cn_ad` (depois do B)

**Sem** `--results-dir` (grava em `ablation_results_d21d32/disp/`):

```bash
.venv/bin/python 5_ablation.py --cohort 48m_12m --representation t1_d21_d32 \
  --modality disp --tasks cn_ad --selection l1_stable --models svm \
  --combat false --repeats 10 --seed 42 \
  --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

Heatmap T1 também com disp CN×AD? Repete `--representation t1_only`.

Depois: só a figura `heatmap_cn_ad.pdf` (sMCI/pMCI já feitos).

### D. Late fusion — **já lançado** (`--reuse-disk`)

Se o bash só fez as 3 specs: 100% argmax ainda pede `ablation_results_late_fusion/*disp*` extra. Specs **sem** `disp` não mexem.

### E. Rebuild + paper (depois de C e D)

1. `save_cohort_comparison`
2. Células encoding / fusion / HM / `heatmap_cn_ad` — **não** heatmaps sMCI/pMCI
3. `7_stats` se D mudou late
4. Se argmax mudou: `artigo.tex` (`48m_6m` `$0{,}828$`)
5. Compila PDF

### F. Opcional (fotometria limpa) — **não** bloqueia v0

Warps v3 em T1 **sem** HM + `3.2` + Q4 `48m_12m` disp. Sem isto HM `$0{,}621\to0{,}600$` continua misto (`csvs_old`).

---

**Ordem:** espera 3.1 + late → B.3–4 (`3.2`, `4_run_post_extract`) → C → E.

######################################################
# Lock histórico (não é a fila de amanhã)
######################################################

Estado: Patch A **no código**. Bash **corrigido**. Run `SKIP_EXTRA=0 WIPE=1` **a correr** (2026-08-17).
Primary: SVM + `l1_stable` (corr 0.85 uma vez, L1 `C=0.1`, π=70%, `min_timepoints=0`) + nocombat.
Métrica: AUC patient-level. Long. oficial = **Q4 `t1_d21_d32`**. Baseline = `t1_only`.
Espaço: classe completa + denylist (`ablation_prep.py` / `ablation_deltas.py`). Allowlist morta.
Early / `--modality all` fora. MeshVolume ∈ vol. Energy ∉ firstorder.

CSVs `cohort_results.csv` e AUCs no `artigo.tex` = allowlist → **não usar**.

---

## 1. Racional (lock)

### Perguntas
1. Q4 vs `t1_only` agrega sinal (uniclasse + late)?
2. Principais atributos por família (freq. `selected_features`)?
3. Late (3 specs) bate teto unimodal?

### Contraste
- Long: **`t1_d21_d32`**
- Baseline: **`t1_only`**
- Não crownear `wide`/`abs`/`t1_deltas`/`t1_deltas_rel`/`t1_ma`

### Coortes
| Papel | Coorte |
|-------|--------|
| Claim / multimodal | **`48m_12m`** |
| Âncora n | `36m_6m` |
| Gradiente | 4 coortes |

### Multimodal
Late = 5 SVMs, média. Specs: tudo t1 · tudo Q4 · âncora `shape:t1_only ∪ resto Q4`.
Clinic+img (extra): shape + `t1_only`.

### Script
```bash
# paper + extra (wipe t1_only / d21d32 / late_fusion):
SKIP_EXTRA=0 ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
# RF/EN na claim (pisa Q4 svm):
# SENS_REPS="t1_only t1_d21_d32" SKIP_EXTRA=0 ./run_ablation_full.sh ...
```
40 mono + 12 late + extra só `48m_12m`. `WIPE=0 SKIP_MONO=1` só se mono novo já no disco.

---

## 2. Ainda por fazer

### A. Esperar o run
- Log: `logs/ablation_full_*.log`
- Mono: `ablation_results_t1_only/{vol,shape,texture,disp,firstorder}/` e `ablation_results_d21d32/` nas 4 coortes
- Late: `ablation_results_late_fusion/` × 3 fingerprints × 4
- Extra: `vol_cn_ad/`, clínica, clinic+img, leaky na claim
- Log cn_ad: `tasks: ('cn_ad',)` — não `smci_pmci`

Não crownear AUCs até o bash `DONE`.

### B. Rebuild planilha (depois do DONE)
```bash
cd /mnt/study-data/pgirardi/graphs
.venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules')
from cohort_compare import save_cohort_comparison
root = Path('csvs/cohorts')
cohorts = [c for c in ('36m_6m','36m_12m','48m_6m','48m_12m')
           if (root / c / 'ablation_results_t1_only').is_dir()]
p_res, p_feat, *_ = save_cohort_comparison(
    cohorts, Path('csvs/cohort_comparison'), cohorts_root=root, n_boot=2000)
print(p_res, p_feat)
"
```
Snapshot opcional de `cohort_comparison/` antes. cn_ad não entra.
Q2: `cohort_features_long.csv` (freq. por família, t1 vs Q4 à parte).

### C. Stats
`7_stats.ipynb` top→bottom, **CSV fresco**:
- `PROTOCOL_LONGITUDINAL = "t1_d21_d32"`
- `MODS_COMPARE` = vol, shape, texture, disp, firstorder (sem `all`)
- Claim B: `48m_12m` Q4 vs `t1_only`, FDR 5 mods
- Gradiente 4 coortes
- clinic/leaky em `BASE_CLAIM` se extra acabou

### D. Figs
`6_results.ipynb`: `PROTO_ENC` = `t1_d21_d32`; âncora late spec 3. Heatmaps sMCI/pMCI **já v3** (svm/rf/en). Falta `heatmap_cn_ad` depois do DVF CN/AD. Sem early.

### E. Artigo (`artigo.tex` = draft allowlist)
Reescrever **depois** dos números:
- Classe completa + denylist (não lista de 4 GLCM)
- Seletor: var → corr 0.85 (`|ρ|` com y; protecção temporal) → 50× L1 → π=70% → SVM
- `tab:dims`: contagens **antes** do seletor (texture Q4 = 144, não 24)
- Firstorder = família (16), não “excluído das isoladas”
- Late only; early fora
- Q4 vs `t1_only`; representante colinear ≠ melhor nome IBSI
- Cruzar AUCs com CSV fresco

### F. Armadilhas
- `WIPE=1` apaga t1_only / d21d32 / late — não os long
- Late `--reuse-disk` com CSVs allowlist = experimento errado (bash já wipeia)
- `SENS_REPS=... SKIP_EXTRA=0` reescreve SVM nocombat na claim
- `--tasks smci_pmci` depois de `cn_ad` no mesmo argparse = bug (já evitado)
- Não `4_run_post_extract` (IBSI já no long)
- Grade 72 late / early = fora

---

## 3. Ficheiros-chave

- Denylist / keep: `modules/ablation_prep.py`, `modules/ablation_deltas.py`
- Seletor: `modules/ablation_stable.py`, `corr_keep_mask` em `ablation_runner.py`
- Run: `run_ablation_full.sh` + `5_ablation.py` + `5_ablation_late_fusion.py`
- Resultados: `csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv`
- Stats / figs: `7_stats.ipynb`, `6_results.ipynb`
- Artigo: `artigo/artigo.tex` (desactualizado até E)

## 4. Dimensões (antes L1; L+R)

| Família | Sufixos | t1_only | Q4 |
|---------|---------|---------|-----|
| vol | 4 | 8 | 24 |
| shape | 12 | 24 | 72 |
| texture | 24 GLCM | 48 | 144 |
| firstorder | 16 | 32 | 96 |
| disp | 15 (long) | 30 | 90 |

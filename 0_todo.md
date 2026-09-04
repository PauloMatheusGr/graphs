# Artigo 1 — fila actual (2026-09-03)

**Nunca** `WIPE=1`. `5_ablation.py` reescreve `ablation_results_all.csv` (sem merge).

`cn_ad`: **não** `--results-dir vol_cn_ad`. Extra Q4 grava `cn_ad` + `smci_pmci` nos 5 mods. T1 e `t1_d21` ficam só `smci_pmci` (misturar `cn_ad` no T1 quebrou `48m_12m/disp`).

Não lançar outro `5_ablation.py` nos mesmos paths em paralelo.

---

## Lock

| | |
|---|---|
| Claim | **`48m_6m`**, `soft_pmci=True` (73 sMCI / 120 pMCI) |
| Baseline | `t1_only` |
| 2 visitas | `t1_d21` (T1+D21, sem i3) |
| Long. oficial | Q4 `t1_d21_d32` |
| Extra 4 modelos / clinic / leaky / `cn_ad` | só na claim |
| Gradiente SVM | 4 coortes **já no disco** — não relançar |
| Soft=False | `csvs/cohorts/48m_6m_soft_False/` (T1 vs Q4 SVM) — **sensibilidade**, não claim |
| Early / `--modality all` | fora |
| HM | store à parte; figura 12m = suplemento; **não** no bash |

Primary: SVM + `l1_stable` (corr 0.85, L1 `C=0.1`, π=70%, `min_timepoints=0`) + nocombat. AUC patient-level.

Código **já feito**: `t1_d21`; bash `LATE_GRID=full` na claim = **232** late k≥2 (cada fam. T1 ou Q4). `LATE_GRID=paper` = 3 specs. Âncora nomeada = shape T1 ∪ resto Q4 (e o par shape T1+vol Q4 está na grelha).

`COHORT` / `COHORT_CLAIM` nos notebooks = `48m_6m`. `artigo.tex` ainda cita `48m_12m` (`n=120`).

---

## Disco agora (`48m_6m`)

Há: `ablation_results_t1_only`, `ablation_results_d21d32`, late 3 specs — **só SVM** `smci_pmci`.

Falta: `ablation_results_d21`, 4 modelos (svm/rf/elasticnet/xgb), `cn_ad` no Q4, clinic, clinic+img, leaky vol (Q4 e D21), late specs D21.

---

## Feito (não voltar)

- [x] Patch A / denylist / Q4 = T1+D21+D32
- [x] SVM uniclasse T1 + Q4 × 4 coortes
- [x] Late 3 specs × 4 coortes (SVM, reuse)
- [x] Extra 4 modelos + `cn_ad` + clinic + leaky em **`48m_12m`** (claim antiga)
- [x] Soft=False `48m_6m` uniclasse SVM T1 vs Q4
- [x] Representação `t1_d21` no código + bash
- [x] DVF CN/AD no long (Q4 `cn_ad` nas 5 fam. já correu em 12m)

## A correr / próximo

- [ ] **A.** Bash só claim (longo; SENS reescreve SVM T1/Q4)

```bash
COHORTS="48m_6m" SKIP_MONO=0 SKIP_EXTRA=0 WIPE=0 \
  ./run_ablation_full.sh 2>&1 | tee logs/ablation_d21_48m6m_$(date +%Y%m%d).log
```

`SKIP_EXISTING=1` salta mono SVM já existente; bloco SENS **regrava** 4 modelos. `TWO_VISIT=1` automático (`COHORTS` = claim).

DONE quando: `ablation_results_d21/{vol,shape,texture,disp,firstorder}/`, 4 `model_key` em T1/D21/Q4, Q4 com `task=cn_ad`, late D21, `ablation_results_clinic*`, `ablation_results_leaky_d21d32` + leaky D21.

Não crownear AUCs até `DONE`.

---

## Ainda por fazer (depois do A)

### B. Rebuild planilha

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

Snapshot opcional de `cohort_comparison/` antes. `cn_ad` não entra. Q2: `cohort_features_long.csv`.

### C. Notebooks — patches (funções já existem)

`6_results.ipynb`:

- Heatmap + ROC `t1_d21` (hoje só T1 e Q4)
- `PROTOCOL_COMPARE`: linhas Shape/Vol D21
- `HM_COHORT` ainda `48m_12m` — pular célula ou deixar suplemento 12m
- `FIG_DIR`: figs vão a `artigo/figures`; draft em `Artigo 1 pgirardi/` — copiar ou apontar

`7_stats.ipynb`:

- Contrastes SVM: `t1_d21` vs T1 **e** Q4 vs `t1_d21` (além Q4 vs T1)
- Tabela n/fold a partir de `test_id_pts` (tex ainda diz ~96/24, n=120)
- Markdown / `save_table(..., "*_48m12")` — nomes velhos; dados = `COHORT_CLAIM`

Kernel cwd = raiz. Reexecutar top→bottom (outputs em cache ainda dizem 12m). Stats oficiais = **SVM**. Heatmaps = 4 modelos.

### D. Artigo

Reescrever **depois** dos números (`Artigo 1 pgirardi/artigo.tex` e/ou `artigo/artigo.tex`):

- Claim `48m_6m`, n=193 (73/120); 12m = célula do gradiente
- Justificativa 6 meses (Schuff / Mubeen / Hua LMCI) — ver `0_notes.md`; 3 visitas ≈ 12 meses de trajectória
- T1 vs D21 vs Q4; late; clinic; leaky; heatmaps/ROC; CN×AD
- Soft=False = sensibilidade (não muda o claim)
- Seletor / denylist / dims **antes** L1; late only; early fora
- Cruzar AUCs com CSV fresco — tex actual = números 12m

Compila PDF.

---

## Fora (não construir)

- Early / `--modality all` / grade 72 late
- 4 modelos nas outras 3 coortes
- Relançar gradiente SVM ou soft=False uniclasse
- `|Δ21|` vs `|Δ31|` e noise floor retest (`0_notes`) — TMI nice-to-have
- HM em `48m_6m` — não bloqueia
- ~~`4_run_post_extract` de novo (IBSI já no long)~~ **ICV homotetia** (2026-09-04): 4_ regrava longs; rerun **só shape** (mono + late reuse). Vol/texture/firstorder/disp longs geometricamente iguais (vol já era /ICV).

---

**Ordem:** A bash → B planilha → C patches + notebooks → D tex.

---

## Armadilhas

- `WIPE=1` apaga t1_only / d21 / d21d32 / late — **não** os long
- `COHORTS` default = 4 coortes: **não** usar; só `"48m_6m"` (12m `disp` T1 está podre: só `cn_ad`)
- SENS `SKIP_EXTRA=0` reescreve SVM nocombat na claim — esperado para heatmaps 4 modelos
- `--tasks` duplicado no argparse: último ganha (bash já separa Q4 `cn_ad,smci_pmci`)
- `param_soft_pmci_of` lê `adnimerged_longitudinal.csv`; `4_` agora grava esse ficheiro na pasta da coorte (além dos `*_True.csv` / `*_False.csv`)

---

## Ficheiros-chave

- Denylist / keep / D21: `modules/ablation_prep.py`, `ablation_deltas.py`, `ablation_representation.py`
- Run: `run_ablation_full.sh` + `5_ablation.py` + `5_ablation_late_fusion.py` + `5_clinic_img.py` + `5_ablation_leaky.py`
- Resultados: `csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv`
- Stats / figs: `7_stats.ipynb`, `6_results.ipynb`
- Artigo: `Artigo 1 pgirardi/artigo.tex`

## Dimensões (antes L1; L+R)

| Família | Sufixos | t1_only | t1_d21 | Q4 |
|---------|---------|---------|--------|----|
| vol | 4 | 8 | 16 | 24 |
| shape | 12 | 24 | 48 | 72 |
| texture | 24 GLCM | 48 | 96 | 144 |
| firstorder | 16 | 32 | 64 | 96 |
| disp | 15 | 30 | 60 | 90 |

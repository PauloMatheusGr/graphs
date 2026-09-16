# Todo — DVF v4 (MBEC maps, CN-only)

Atualizado: 2026-09-16. Limpa handoff antigo (combat/notebooks/tex). Foco: extrair → export → ablação isolada → decidir promoção.

---

## Lock

| | |
|---|---|
| Claim | **`48m_6m`**, `PARAM_SOFT_PMCI=True` (sMCI 73 / pMCI 120) |
| Encodings | `t1_only` · Q4 `t1_d21_d32` |
| Knobs | `l1_stable` · combat false · seed 42 · SVM (smoke `-r 1`; full `-r 10`) |
| ROI classificador | hipocampo L+R (`--roi hippocampus`) |
| Extração ROIs | 10×2 DKT em `3.2_feat_dvf_v4.py` (filtro hipocampo no ablation) |
| Keep disp | `mag_`, `jac_det_`, `strain_fro_` × mean/variance/skewness/kurtosis (sem `std`/`logjac`/percentis) |
| Baseline antigo (afim+logjac) | T1 disp ≈ **0.595** · Q4 ≈ **0.577** |

Backup pré-v4: `backups/dvf_pre_nl_*/` (code, features v3, disp/merge long, results disp, compare).

**Não** apontar `--results-dir` a `ablation_results_t1_only|d21|d21d32` até promover.

---

## Feito

- [x] Backup `backups/dvf_pre_nl_*`
- [x] `3.2_feat_dvf_v4.py`: warp só `*_1Warp`; mapas D=`mag`, V=`jac_det`, S=`strain_fro` infinitesimal; ROI_TABLE completa
- [x] `ablation_prep.keep_disp_feat` → `jac_det_` + drop `_std`

---

## Próximos passos (ordem)

### 1. Extrair features v4

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
python 3.2_feat_dvf_v4.py --self-check
python 3.2_feat_dvf_v4.py
```

Saída: `csvs/cohorts/all_population/features_displacement_v4.csv`  
Resume: `images/displacement_field_v3/features_v4/all_population/`

Sanity (afim fora — `mag` v4 ≪ v3):

```bash
python - <<'PY'
import pandas as pd
old = pd.read_csv("csvs/cohorts/all_population/features_displacement_v3.csv")
new = pd.read_csv("csvs/cohorts/all_population/features_displacement_v4.csv")
o = old.query("roi=='hippocampus' and side=='L'")[["ID_IMG","mag_mean"]].rename(columns={"mag_mean":"mag_v3"})
n = new.query("roi=='hippocampus' and side=='L'")[["ID_IMG","mag_mean","jac_det_mean"]]
m = o.merge(n, on="ID_IMG")
print(m[["mag_v3","mag_mean","jac_det_mean"]].describe())
assert "jac_det_mean" in new.columns
print("ok overlap", len(m))
PY
```

Se `mag_mean` ≈ `mag_v3` → parar; A1 falhou.

---

### 2. Reexport long (`4_`)

Em `4_run_post_extract.py`:

```python
DISP_FEATURES = "features_displacement_v4.csv"
```

```bash
python 4_run_post_extract.py
```

Reescreve `disp_long.csv` + `merge_long.csv` nas coortes de `POST_JOBS`. Vol/shape/texture não devem mudar.

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv("csvs/cohorts/48m_6m/ablation/hippocampus/disp_long.csv", nrows=1)
assert any(c.startswith("jac_det_") for c in d.columns), list(d.columns)[:20]
print("ok jac_det in disp_long")
PY
```

---

### 3. Ablação isolada (claim)

```bash
MAG=mag_n,mag_mean,mag_std,mag_p05,mag_p50,mag_p95,mag_variance,mag_skewness,mag_kurtosis
JAC=jac_det_n,jac_det_mean,jac_det_std,jac_det_p05,jac_det_p50,jac_det_p95,jac_det_variance,jac_det_skewness,jac_det_kurtosis
STR=strain_fro_n,strain_fro_mean,strain_fro_std,strain_fro_p05,strain_fro_p50,strain_fro_p95,strain_fro_variance,strain_fro_skewness,strain_fro_kurtosis
ROOT=csvs/cohorts/48m_6m/ablation_results_dvf_v4
COMMON='--cohort 48m_6m --modality disp --tasks smci_pmci --selection l1_stable --combat false --seed 42 --models svm'
```

#### 3a. Smoke `-r 1`

```bash
python 5_ablation.py $COMMON --representation t1_only -r 1 \
  --results-dir $ROOT/t1/all

python 5_ablation.py $COMMON --representation t1_only -r 1 \
  --results-dir $ROOT/t1/jac --exclude-features $MAG,$STR

python 5_ablation.py $COMMON --representation t1_only -r 1 \
  --results-dir $ROOT/t1/mag --exclude-features $JAC,$STR

python 5_ablation.py $COMMON --representation t1_only -r 1 \
  --results-dir $ROOT/t1/strain --exclude-features $MAG,$JAC

python 5_ablation.py $COMMON --representation t1_only -r 1 \
  --results-dir $ROOT/t1/jacstrain --exclude-features $MAG
```

Ler:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("csvs/cohorts/48m_6m/ablation_results_dvf_v4/t1")
for p in sorted(root.glob("*/ablation_summary.csv")):
    s = pd.read_csv(p)
    row = s.query("task=='smci_pmci' and model_key=='svm'").iloc[0]
    print(f"{p.parent.name:12s} auc_patient_mean={row.get('auc_patient_mean')} auc_mean={row.get('auc_mean')}")
PY
```

#### 3b. Se smoke mexer → `-r 10` + Q4 no vencedor

```bash
# exemplo: jac ganhou
python 5_ablation.py $COMMON --representation t1_only -r 10 \
  --results-dir $ROOT/t1/jac --exclude-features $MAG,$STR
python 5_ablation.py $COMMON --representation t1_d21_d32 -r 10 \
  --results-dir $ROOT/q4/jac --exclude-features $MAG,$STR
```

Opcional: `--models svm,rf,elasticnet,xgb`.

---

### 4. Critério de leitura

| Resultado | Decisão |
|---|---|
| jac ≈ vol (~0.75); mag ~0.55 | affine era ruído; V≈volume |
| all < jac | mag prejudica; keep jac (± strain) |
| tudo ~0.55 | CN-only hipocampo saturado; **não** promove |
| Q4 jac > T1 jac | ritmo em det J; senão baseline basta |

Tabela (preencher):

```text
config       | auc_patient_mean | vs old 0.595
t1/all       |                  |
t1/jac       |                  |
t1/mag       |                  |
t1/strain    |                  |
t1/jacstrain |                  |
q4/<win>     |                  |
```

`6_results` / `cohort_results.csv` **não** leem `ablation_results_dvf_v4/`.

---

### 5. Promover (só se critério passar)

1. Congelar keep / `--exclude-features` definitivo.
2. `5_ablation` **sem** `--results-dir` custom → `ablation_results_t1_only/disp` e `ablation_results_d21d32/disp` (pisa antigo; backup já tem cópia).
3. Gradiente 4 coortes T1+Q4 SVM se necessário.
4. Late fusion com ramo `disp` fica stale — relançar ou declarar no tex.
5. Rebuild compare:

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules')
from cohort_compare import save_cohort_comparison
root = Path('csvs/cohorts')
cohorts = [c for c in ('36m_6m','36m_12m','48m_6m','48m_12m')
           if (root / c / 'ablation_results_t1_only').is_dir()]
print(save_cohort_comparison(cohorts, Path('csvs/cohort_comparison'),
                             cohorts_root=root, n_boot=2000)[:2])
"
```

Ou `6_results.ipynb`: `REBUILD_COMPARE = True`.

---

## Não fazer agora

- Relançar vol/shape/texture/official disp sem promoção
- Elastix / template DEM / 15 ROIs NAC (outro desenho)
- `4_` com `DISP_FEATURES=v3` depois de apontar v4
- `--results-dir` compartilhado entre várias mods

---

## Estado rápido

| Passo | Estado |
|-------|--------|
| Backup | OK |
| Código v4 + keep | OK |
| Extract `features_displacement_v4.csv` | **Falta** |
| `DISP_FEATURES=v4` + `4_` | **Falta** |
| Smoke A2 T1 | **Falta** |
| Full `-r 10` / Q4 | após smoke |
| Promover + compare | só se critério OK |

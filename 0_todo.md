# Artigo 1 — handoff v1 (2026-09-06 16:25)

Outro chat: ler isto + `0_notes.md`. **Não** relançar pastas OK. `5_ablation.py` **reescreve** CSV (sem merge).

---

## Amanhã (ordem)

0. **Combat** — `tmux ls` / log `logs/v1_combat_true_*.log`. Esperar `=== DONE`. 10 CSVs em:
   `csvs/cohorts/48m_6m/ablation_results_combat_{t1_only,d21d32}/{vol,shape,texture,disp,firstorder}/ablation_summary.csv`
   Se job morreu: **não** relançar o loop inteiro — só mods em falta, mesmo `--results-dir` (pasta vazia/incompleta). Nunca apontar a `ablation_results_t1_only|d21d32`.
1. Confirmar compare: `ls -l csvs/cohort_comparison/cohort_results.csv` (mtime ≥ 16:18). Se velho: bloco CHECK+compare abaixo.
2. **`7_stats.ipynb`**
   - `CLINIC_MODALITY = "vol"` (obrigatório; fusion no disco é vol).
   - Run §§ 1–7 (claim já `48m_6m`). Header markdown ainda diz `48m_12m` — corrigir.
   - **Célula nova** combat vs nocombat (paths `ablation_results_combat_*`; **não** concat nos CSV False).
3. **`6_results.ipynb`** top→bottom (`COHORT=48m_6m`). Fig 1 leaky já no disco. Depois: `PROTOCOL_ROOTS` + figura combat.
4. Tex `Artigo 1 pgirardi/artigo.tex`: claim `48m_6m`, AUCs, ICV homotetia, ComBat, clinic=vol.

**Não** amanhã (já no disco / aceite): T1/D21/Q4 nocombat, late, leaky, clinic+img vol, 4 coortes, `4_`, late. D21 shape 4 modelos = skip. Soft=False = só se quiseres figura extra.

Detalhe código: **Falta implementar**. Jobs opcionais: **Falta — jobs**.

---

## Lock

| | |
|---|---|
| Claim | **`48m_6m`**, `PARAM_SOFT_PMCI=True` (sMCI 73 / pMCI 120 / CN 100 / AD 149; n=442) |
| Hard | `48m_6m_soft_False` (sMCI 73 / pMCI 74; n=396) |
| Encodings | baseline `t1_only` · 2 vis `t1_d21` · Q4 `t1_d21_d32` |
| Knobs | `l1_stable` · nocombat · repeats 10 · seed 42 · optuna 10 · pool 70% · min-tp 0 · boot 50 · l1-c 0.1 |
| Modelos extra / clinic / leaky / `cn_ad` / D21 | **só claim** |
| Gradiente 4 coortes | SVM · T1+Q4 · 5 famílias · **sem** D21 |
| ICV shape | homotetia (`4_run_post_extract`, 2026-09-04 18:28). **Não** rerodar `4_` nem `3_feat_rad` |

Pastas:

| Protocolo | Disco |
|-----------|--------|
| T1 | `ablation_results_t1_only/{mod}/` |
| D21 | `ablation_results_d21/{mod}/` |
| Q4 | `ablation_results_d21d32/{mod}/` |
| leaky Q4 / D21 | `ablation_results_leaky_d21d32/vol/` · `ablation_results_leaky_d21/vol/` |
| clinic-only | `ablation_results_clinic/` |
| clinic+img T1 | `ablation_results_clinic_img_t1_only/` |
| late | `ablation_results_late_fusion/{spec}/` |
| combat=True | `ablation_results_combat_t1_only/{mod}/` · `ablation_results_combat_d21d32/{mod}/` (pastas **novas**; notebooks ainda **não** leem) |

Mods: `vol` `shape` `texture` `disp` `firstorder`. Modelos paper: `svm,rf,elasticnet,xgb`.

---

## A correr / último estado jobs

- Leaky D21 vol **OK** 16:18 (`ablation_results_leaky_d21/vol/`, svm_smci≈0.747). CHECK vol/clinic OK.
- `cohort_compare` pode ter corrido no mesmo tmux `0` (log `logs/v1_vol_rest_20260906_160735.log`) — confirmar `ok csvs/cohort_comparison/…`.
- Combat=True: CLI abaixo; **não** lançar em paralelo com outro `5_ablation` no mesmo `--results-dir`. Preferir `tmux new -s combat`.

`ConvergenceWarning` `max_iter=2000`: ignorar. `WARNING PROTOCOLO LEAKY`: esperado.

---

## Feito — 4 coortes (mesmo protocolo, SVM, nocombat, smci)

T1 + Q4 · 5 famílias · **sem D21** (D21 só claim). Shape rerodado pós-ICV 04-Sep 18:58–19:01.

| Coorte | n | soft | T1/Q4 5 fam SVM | late specs | D21 |
|--------|---|------|-----------------|------------|-----|
| `36m_6m` | 516 | True | OK | 61 | não |
| `36m_12m` | 371 | True | OK | 21 | não |
| `48m_6m` | 442 | True | OK + extra | **237** | OK |
| `48m_12m` | 305 | True | OK (3 modelos nalguns; shape só svm pós-ICV) | 21 | não |
| `48m_6m_soft_False` | 396 | False | T1+Q4 SVM 5 fam (shape pós-ICV) | não | **não** |

`csvs/cohort_comparison/` — rebuild após leaky (tmux `0`, ~16:18). Confirmar ficheiros frescos.

---

## Feito — claim `48m_6m` (soft=True)

Unimodal T1 / D21 / Q4 · 4 modelos salvo nota. Combat **False** em todos.

| Bloco | Disco | Estado |
|-------|--------|--------|
| Unimodal T1 5 fam 4 modelos smci | `ablation_results_t1_only/*` | OK. Shape 13:22 (ICV). Sem `cn_ad` T1 |
| Unimodal D21 5 fam 4 modelos smci | `ablation_results_d21/*` | OK **excepto shape = só svm** (18:56 ICV) |
| Unimodal Q4 5 fam 4 modelos smci+`cn_ad` | `ablation_results_d21d32/*` | OK. Shape 14:00 · disp `cn_ad` 14:37 |
| Late união multiclasse (SVM) | `ablation_results_late_fusion/` 237 specs | OK 05-Sep 14:07 (shape ICV) |
| Clinical-only | `ablation_results_clinic/` | OK 05-Sep 13:59 |
| Clinic+img T1 **vol** SVM | `…_clinic_img_t1_only/fusion_vol_*` | OK 16:07 |
| Leaky Q4 vol SVM | `…_leaky_d21d32/vol/` | OK 16:14 ≈0.77 |
| Leaky D21 vol SVM | `…_leaky_d21/vol/` | OK 16:18 ≈0.747 |
| Combat=True | `ablation_results_combat_{t1_only,d21d32}/` | **CLI**; pastas novas. Não concat nos CSV False |
| `cn_ad` T1 heatmap | — | skip (notebook só `cn_ad` Q4) |

SVM smci claim (summary, ~auc_patient_mean):

| | vol | shape | texture | disp | FO |
|--|-----|-------|---------|------|-----|
| T1 | 0.756 | 0.712 | 0.628 | 0.595 | 0.687 |
| D21 | 0.740 | 0.716 | 0.671 | 0.548 | 0.691 |
| Q4 | 0.763 | 0.721 | 0.674 | 0.577 | 0.686 |

Late ancora / all-Q4: ver `protocol_compare_primary.csv`.

---

## Feito — `48m_6m_soft_False`

T1+Q4 · 5 fam · **SVM only** · nocombat. Shape 04-Sep 19:01 (ICV). Vol/texture/disp/FO 01-Sep (ICV não aplica). Sem D21, late, clinic, leaky, 4 modelos. Notebooks **não** leem esta pasta (`COHORT=48m_6m`).

---

## Falta — jobs

1. Correr (ou esperar `DONE`) **combat=True** CLI (secção abaixo). 10 jobs: 5 fam × T1+Q4, SVM.
2. Se `cohort_compare` não imprimiu `ok …`: rerodar bloco CHECK+compare.

Jobs opcionais (não bloqueiam v1): D21 shape 4 modelos · `cn_ad` T1 · soft=False D21 · D21 combat.

---

## Falta implementar (código / notebooks / tex)

Disco nocombat já chega p/ figs principais. **Run All sem estas mudanças = combat invisível + clinic fusion aponta shape (CSV não existe).**

### `7_stats.ipynb`

- Header markdown ainda diz claim `48m_12m` — código já `COHORT_CLAIM = "48m_6m"`. Alinhar texto.
- **`CLINIC_MODALITY = "shape"` → `"vol"`.** Fusion no disco: `fusion_vol_l1_stable_nocombat_t1_only_*.csv`. Com `shape`, §6 procura ficheiro em falta.
- Nomes `stats_*_48m12.csv` — opcional rename `48m6m` (ou deixar e documentar).
- **§ nova: combat vs nocombat.** Notebooks **não** conhecem `ablation_results_combat_*`. `CFG.with_combat=False` e `PROTOCOL_ROOTS` só pastas nocombat.
  - Ler T1: `csvs/cohorts/48m_6m/ablation_results_combat_t1_only/{mod}/ablation_results_all.csv`
  - Ler Q4: `…/ablation_results_combat_d21d32/{mod}/…`
  - Emparelhar vs nocombat (`ablation_results_t1_only` / `d21d32`) no mesmo `ID_PT`, SVM, 5 fam.
  - Tabela `artigo/tables/stats_combat_48m6m.csv`: AUC nocombat, AUC combat, Δ, CI, p. Δ = combat − nocombat.
  - **Não** concatenar True para dentro dos CSV False.
- §6 clinic: imagem = **vol T1** (não shape). Texto markdown “Imagem = shape T1” desactualizado.
- §7 leaky: path Q4 já `ablation_results_leaky_d21d32/vol` — deve passar a existir. Opcional 2ª linha D21 leaky (`ablation_results_leaky_d21/vol`) — `7_stats` hoje só Q4.

### `6_results.ipynb`

- `PROTOCOL_ROOTS` **sem** combat. Acrescentar p.ex.:
  - `"t1_only_combat"` → `BASE / "ablation_results_combat_t1_only"`
  - `"t1_d21_d32_combat"` → `BASE / "ablation_results_combat_d21d32"`
- Célula/figura nova: barras ou tabela T1/Q4 × 5 fam × {nocombat, combat}. Heatmaps/ROC **ficam** `CFG.with_combat=False`.
- `build_all_protocols_summary`: incluir clinic+img **vol** (já no glob da pasta) e leaky Q4. Confirmar que `clinica+img_t1` lê `fusion_vol_*` e não exige `fusion_shape_*`.
- Fig 1 leaky: deixou de `MISS` após 16:14 — rerodar célula.
- Encoding T1/D21/Q4 (§2.1): D21 shape só svm — heatmap 4 modelos D21 **não** existe; barras SVM OK.

### Soft=True vs False (opcional artigo)

Disco: `csvs/cohorts/48m_6m_soft_False/` T1+Q4 SVM 5 fam. Nenhum notebook compara. Se entrar no v1: célula ΔAUC True−False, T1 e Q4, SVM, 5 fam (paths paralelos, **não** misturar CSVs). Sem D21 no hard.

### Tex `Artigo 1 pgirardi/artigo.tex`

- Claim **`48m_6m`** (já não `48m_12m`): n, sMCI/pMCI 73/120, justificação 6 meses (`0_notes.md` Mubeen/Schuff).
- AUCs unimodal T1/D21/Q4 + late (tabela daqui / `protocol_compare_primary.csv`).
- ICV **homotetia** (comprimento `/ICV^{1/3}`, área `/ICV^{2/3}`, volume `/ICV`) — não divisão crua.
- Sensibilidade ComBat (após § stats).
- Clinic+img = **vol** T1, não shape.
- Opcional: n por fold externo/interno (`0_notes.md` “quantidade de dados após split”) — não há tabela ainda.

### `modules/` só se preciso

- `stats_compare.fusion_results_path` / `image_ablation_path`: combat roots **não** mapeados. Ou paths explícitos nas células, ou 2 entradas no dict. Não alterar defaults das pastas nocombat.

---

## Não relançar (apaga disco)

```
5_ablation.py --cohort 48m_6m --modality vol|shape|…   # T1/D21/Q4 já OK
5_ablation.py --combat both                           # reescreve CSV; --models svm apaga rf/en/xgb; Q4 smci-only apaga cn_ad
5_clinic_img.py --feature-set clinical                # apaga clinical_summary
5_ablation.py --cohort 48m_6m_soft_False              # apaga SVM T1/Q4 hard
4_run_post_extract / 3_feat_rad
late / clinical-only / EXTRA vol,texture,FO,disp claim
gradiente SVM shape 4 coortes
```

`--results-dir` com **várias** mods no mesmo path = um CSV pisa o outro. 1 mod × 1 `--results-dir`.

`--tasks cn_ad` sozinho no rewrite = perde smci. Dois jobs no mesmo `{mod}/` = CSV podre.

---

## Depois de 4b — CHECK + compare

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate
PY="${PWD}/.venv/bin/python"
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
b = Path("csvs/cohorts/48m_6m")
need = {"svm", "rf", "elasticnet", "xgb"}
ok = True
checks = [
    ("ablation_results_t1_only/shape", {"smci_pmci"}, True),
    ("ablation_results_d21/shape", {"smci_pmci"}, False),  # só svm — aceite
    ("ablation_results_d21d32/shape", {"smci_pmci", "cn_ad"}, True),
    ("ablation_results_d21d32/disp", {"smci_pmci", "cn_ad"}, True),
    ("ablation_results_leaky_d21d32/vol", {"smci_pmci"}, False),
    ("ablation_results_leaky_d21/vol", {"smci_pmci"}, False),
]
for rel, tasks, want4 in checks:
    p = b / rel / "ablation_summary.csv"
    if not p.is_file():
        print("FAIL missing", rel); ok = False; continue
    s = pd.read_csv(p)
    mk, tk = set(s.model_key.astype(str)), set(s.task.astype(str))
    good = (need <= mk if want4 else True) and tasks <= tk
    print(("OK" if good else "FAIL"), rel, "models=", sorted(mk), "tasks=", sorted(tk))
    ok &= good
img = list((b / "ablation_results_clinic_img_t1_only").glob("*vol*summary*.csv"))
print(("OK" if img else "FAIL"), "clinic_img vol", [p.name for p in img])
ok &= bool(img)
raise SystemExit(0 if ok else 1)
PY

"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules')
from cohort_compare import save_cohort_comparison
root = Path('csvs/cohorts')
cohorts = [c for c in ('36m_6m','36m_12m','48m_6m','48m_12m')
           if (root / c / 'ablation_results_t1_only').is_dir()]
print(save_cohort_comparison(cohorts, Path('csvs/cohort_comparison'), cohorts_root=root, n_boot=2000)[:2])
"
```

`compare` só toca `csvs/cohort_comparison/` (derivado).

---

## Combat=True (depois de 4b; pastas novas)

Sensibilidade claim: SVM · `smci_pmci` · 5 fam · T1+Q4. **Não** `both` nas pastas nocombat.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/study-data/pgirardi/graphs
source .venv/bin/activate
mkdir -p logs
PY="${PWD}/.venv/bin/python"
COMMON='--selection l1_stable --combat true --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 --stable-bootstrap 50 --stable-l1-c 0.1 --models svm --tasks smci_pmci'
LOG="logs/v1_combat_true_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date -Is) ==="
ROOT="csvs/cohorts/48m_6m"
for mod in vol shape texture disp firstorder; do
  echo "=== combat T1 $mod ==="
  "$PY" 5_ablation.py --cohort 48m_6m --representation t1_only --modality "$mod" $COMMON \
    --results-dir "$ROOT/ablation_results_combat_t1_only/$mod"
  echo "=== combat Q4 $mod ==="
  "$PY" 5_ablation.py --cohort 48m_6m --representation t1_d21_d32 --modality "$mod" $COMMON \
    --results-dir "$ROOT/ablation_results_combat_d21d32/$mod"
done
echo "=== DONE $(date -Is) | log=$LOG ==="
```

Não concatenar True para dentro dos CSV False até backup + célula `7_stats`. Heatmaps ficam `CFG.with_combat=False`.

Opcional D21 combat: `--representation t1_d21` → `ablation_results_combat_d21/$mod`.

---

## Notebooks / tex — ordem depois de combat `DONE`

1. `6_results.ipynb` top→bottom (`COHORT=48m_6m`) — figs nocombat. Depois célula combat.
2. `7_stats.ipynb` — **primeiro** `CLINIC_MODALITY="vol"`; depois §§; depois § combat.
3. Tex.

Detalhe do que falta no código: secção **Falta implementar** acima.

Gap aceite: D21 **shape** só svm. Fusion clinic = **vol**.

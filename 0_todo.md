# Continuação (Cursor laptop)

Estado: coortes ±2 (`36m_6m`, `36m_12m`, `48m_6m`, `48m_12m`).
Primary: SVM + `l1_stable` (L1 `C=0.1`, π=70%, `min_timepoints=0`) + nocombat.
Métrica: AUC patient-level. Encoding long. oficial = **Q4 `t1_d21_d32`** (T1+D21+D32; paper: `t1_deltas_seq`).
Cinco famílias compactas. `t1_deltas` (T1+D21+D31+D32) = sensibilidade no disco, não claim.
MeshVolume ∈ vol. Firstorder = 4 momentos (Mean, Variance, Skewness, Kurtosis).

---

## 1. Racional

### Pergunta
Há alteração útil no tempo no hipocampo para sMCI→pMCI?

### Contraste primary (claim B)
- Longitudinal: **`t1_d21_d32`** (Q4 / `t1_deltas_seq`)
- Baseline: **`t1_only`**
- Não crownear `wide`/`abs` como longitudinal
- Não crownear `t1_deltas` / `t1_deltas_rel` / `t1_ma` como claim
- `t1_deltas` no disco = identidade D31=D21+D32 (sensibilidade)

### Coortes
| Papel | Coorte | Porquê |
|-------|--------|--------|
| Claim / multimodal | **`48m_12m`** | janela 48 + gap 12 |
| Âncora demografia / maior n | `36m_6m` | §§1–3 em `7_stats` |
| Gradiente | 4 coortes | gap×janela |

### Hierarquia unimodal (claim, svm nocombat)
- Shape = teto estático (`t1_only` ≥ Q4)
- Texture = sinal temporal
- Vol = ganho moderado
- Disp = fraco
- Firstorder = **a correr** (não crownear até números)

### Multimodal
- Late = 5 SVMs, média de scores (não concatena colunas)
- Braços `{t1_only, t1_d21_d32}` (não wide)
- 3 specs pré-especificadas (não grade 72):
  1. tudo `t1_only`
  2. tudo Q4
  3. âncora: `shape:t1_only ∪ {vol,texture,disp,firstorder}:Q4`
- Early = os mesmos 3, só claim
- Clinic+img: **shape + `t1_only`** (teto), não Q4

### Script único
| Script | Conteúdo |
|--------|----------|
| `run_ablation_full.sh` | mono **só firstorder** `{t1_only,Q4}` × 4 coortes → 3 late × 4 → 3 early claim → extra opcional (`SKIP_EXTRA=1` default) |

`cn_ad` → `vol_cn_ad/`; não entra em `cohort_results.csv`.

---

## 2. Discussão (artigo)

1. Alteração ≠ concatenação: claim = Q4 vs `t1_only`; `wide` = âncora metodológica.
2. Shape teto, texture muda; firstorder = intensidade global (família à parte da GLCM).
3. Compacto a priori: um descritor por mecanismo (não dump IBSI).
4. Late > early: selecção conjunta dilui shape; média de scores preserva braços.
5. Clínica como adjunto.
6. Limitações: n claim=120; soft pMCI; só ADNI; cn_ad = sanity.
7. ComBat / RF-EN = sensibilidade; primary SVM nocombat.
8. π=70% = frequência nos **50 boots**, não nas 3 visitas (`min_timepoints=0`).

---

## 3. Pipeline de execução

### A. Runs
```bash
./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
# extra (cn_ad, clínica, leaky) — não default:
# SKIP_EXTRA=0 ./run_ablation_full.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
```
Mono = firstorder only (`t1_only` + Q4). Não toca `vol/` `shape/` `texture/` `disp/`.
Late `--reuse-disk` lê as 4 famílias antigas + firstorder novo.

Verificar firstorder:
`csvs/cohorts/48m_12m/ablation_results_d21d32/firstorder/`
`csvs/cohorts/48m_12m/ablation_results_t1_only/firstorder/`

Verificar cn_ad (se extra): `csvs/cohorts/48m_12m/ablation_results_d21d32/vol_cn_ad/`
Log: `tasks: ('cn_ad',)` — não `smci_pmci`.

### B. Rebuild planilha
Não apaga unimodal das 4 famílias. Reescreve só:
`csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv`

```bash
cd /mnt/study-data/pgirardi/graphs
.venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'modules')
from cohort_compare import save_cohort_comparison
root = Path('csvs/cohorts')
cohorts = [c for c in ('36m_6m','36m_12m','48m_6m','48m_12m')
           if (root / c / 'ablation_results').is_dir()]
p_res, p_feat, *_ = save_cohort_comparison(
    cohorts, Path('csvs/cohort_comparison'), cohorts_root=root, n_boot=2000)
print(p_res, p_feat)
"
```
Cópia opcional de `cohort_comparison/` antes (snapshot). cn_ad não entra.
Leaky Q4 entra como `protocol=t1_d21_d32_global` (`ablation_results_leaky_d21d32/`). `global` continua = leaky **wide**.

### C. Stats
`7_stats.ipynb` top→bottom:
- `PROTOCOL_LONGITUDINAL = "t1_d21_d32"`
- `MODS_COMPARE` = vol, shape, texture, disp, **firstorder** (sem `all`)
- Claim B §4: `48m_12m` Q4 vs `t1_only`, FDR **5** mods
- §4b `36m_6m`; §4c gradiente 4 coortes
- clinic/leaky em `BASE_CLAIM`

### D. Figs
`6_results.ipynb`: `PROTO_ENC` com `t1_d21_d32`; âncora late fingerprint spec 3; cn_ad em `vol_cn_ad`.

### E. Artigo
- Allowlist compacta + porquê (um mecanismo; firstorder = 4 momentos; Energy fora)
- Q4 vs 4 deltas (identidade D31); no texto: `t1_deltas_seq`
- π vs `min_timepoints`
- Late = média de 5 SVMs
- Cruzar AUCs com CSV fresco

### F. Armadilhas
- Mono com `--modality vol,shape,...` **reescreve** as 4 famílias
- `SENS_REPS=... SKIP_EXTRA=0` reescreve SVM nocombat na claim
- `--tasks smci_pmci` depois de `cn_ad` no mesmo argparse = bug (já evitado)
- Não `4_run_post_extract` para firstorder
- Grade 72 late = removida do full.sh

---

## 4. Ficheiros-chave

- Allowlist: `modules/ablation_prep.py`
- Run: `run_ablation_full.sh` + `5_ablation.py` / late / early
- Resultados: `csvs/cohort_comparison/cohort_results.csv`
- Stats: `7_stats.ipynb`
- Figs: `6_results.ipynb`
- Artigo: `artigo/artigo.tex` + `cas-refs.bib`

## 5. Allowlists compactas (L+R; Q4 = ×3 tokens)

| Família | Sufixos | t1_only | Q4 |
|---------|---------|---------|-----|
| vol | 4 | 8 | 24 |
| shape | 5 | 10 | 30 |
| texture | 4 GLCM | 8 | 24 |
| disp | 4 | 8 | 24 |
| firstorder | 4 momentos | 8 | 24 |

**vol:** `gm_norm`, `wm_norm`, `csf_norm`, `original_shape_MeshVolume`  
**shape:** `SurfaceArea`, `SurfaceVolumeRatio`, `Sphericity`, `Elongation`, `Flatness`  
**texture:** `Contrast`, `Correlation`, `Idm`, `JointEntropy`  
**disp:** `mag_mean`, `strain_fro_mean`, `strain_fro_variance`, `logjac_variance`  
**firstorder:** `Mean`, `Variance`, `Skewness`, `Kurtosis`

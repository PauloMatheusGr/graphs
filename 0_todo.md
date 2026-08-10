# Continuação (Cursor laptop)

Estado: coortes ±2 (`36m_6m`, `36m_12m`, `48m_6m`, `48m_12m`); MeshVolume no bloco **vol**; primary = SVM + `l1_stable` + nocombat; métrica AUC patient-level.

---

## 1. Racional dos experimentos

### Pergunta
Há **alteração útil no tempo** no hipocampo para sMCI→pMCI?

### Contraste primary (claim B)
- Longitudinal: **`t1_deltas`** ($t_0$ + deltas absolutos D21/D31/D32)
- Baseline: **`t1_only`**
- **Não** crownear `wide`/`abs` como “longitudinal” (níveis multi-visita ≠ mudança)
- **Não** crownear `t1_deltas_rel` (sensibilidade; texture sobe, vol/disp colapsam)

### Coortes
| Papel | Coorte | Porquê |
|-------|--------|--------|
| Claim / multimodal | **`48m_12m`** | janela 48 + gap 12; maior ganho texture/vol |
| Âncora demografia / maior $n$ | `36m_6m` | §§1–3 em `7_stats` |
| Gradiente | 4 coortes | gap×janela |

### Hierarquia unimodal (claim, svm nocombat)
- Shape = **teto estático** (`t1_only` 0.761 ≥ deltas 0.736)
- Texture = **sinal temporal** (Δ ≈ +0.057)
- Vol = ganho moderado (Δ ≈ +0.029)
- Disp = fraco

### Multimodal
- Late: braços só `{t1_only, t1_deltas}` (não pós-hoc wide)
- Âncora pré-especificada: **`shape:t1_only ∪ texture:t1_deltas`** ≈ **0.801** (bate teto shape)
- Late > early no mesmo spec (+0.03 na âncora)
- Clinic+img: **shape + `t1_only`** (teto), não deltas

### Scripts
| Script | Conteúdo |
|--------|----------|
| `run_ablation_full.sh` | mono 4×{t1_only,t1_deltas,wide} → late → early claim |
| `run_ablation_extra.sh` | sens RF/EN · cn_ad(vol,`t1_deltas`) · clínica · clinic+img(shape,`t1_only`) · leaky(vol,`t1_deltas`) |

`cn_ad` → pasta `vol_cn_ad/`; **não** entra em `cohort_results.csv` (filtro em `cohort_compare.py`). Ver `6_results.ipynb`.

---

## 2. Possíveis discussões (artigo)

1. **Alteração ≠ concatenação:** `wide` pode AUC alta sem responder à pergunta temporal; claim = deltas vs baseline.
2. **Shape teto, texture muda:** geometria já discrimina no $t_0$; textura/GLCM carrega progressão — late mistura os dois.
3. **Gap/janela:** texture ganha deltas nas 4 coortes; máximo em `48m_12m`; vol inconsistente em `36m_12m`.
4. **Late > early:** seleção conjunta dilui shape; média de scores preserva braço texture-Δ.
5. **Clínica como adjunto:** imagem hipocampal sozinha < clínico; fusion clinic+shape útil como contexto, não first-line.
6. **Limitações:** $n$ claim=120; soft pMCI; só ADNI (AIBL/OASIS sem 3 RM); cn_ad = sanity pipeline, não endpoint; selecção pós-gradiente de cohort = pós-hoc no crowning multi-mod.
7. **ComBat / RF-EN:** sensibilidade; primary permanece SVM nocombat.

Números stale no tex antigo (shape 0.785, late 0.823) → já parcialmente actualizados em `artigo/artigo.tex`; rever após extras + rebuild.

---

## 3. Próximos passos

### A. Runs (se ainda não)
```bash
# full já deve existir; se não:
# ./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log

./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
```
Defaults OK: `PRIMARY=48m_12m`, `REP=t1_deltas`, `FUSION_MOD=shape`, `FUSION_REP=t1_only`.

Verificar cn_ad:
`csvs/cohorts/48m_12m/ablation_results_deltas/vol_cn_ad/`  
Log deve mostrar `tasks: ('cn_ad',)` — **não** `smci_pmci`.

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
           if (root / c / 'ablation_results').is_dir()]
p_res, p_feat, *_ = save_cohort_comparison(
    cohorts, Path('csvs/cohort_comparison'), cohorts_root=root, n_boot=2000)
print(p_res, p_feat)
"
```
Actualiza clínica/leaky/mono sens; **não** mete cn_ad no CSV.

### C. Stats formais
Re-run `7_stats.ipynb` top→bottom:
- Claim B §4: `48m_12m` `t1_deltas` vs `t1_only` (FDR 4 mods)
- §4b gap curto `36m_6m`; §4c gradiente 4 coortes
- §§5–6 clinic/leaky em `BASE_CLAIM`
- Outputs limpos; `MODS_COMPARE` sem `all`

### D. Figuras / resultados
- `6_results.ipynb`: ROC/heatmap smci + sanity cn_ad (`vol_cn_ad`)
- Confirmar late âncora e early vs late após rebuild

### E. Artigo
- `artigo/artigo.tex`: cruzar AUCs com CSV fresco; citacoes `cas-refs.bib` (BibTeX cycle)
- Equação 3 visitas: preferir um índice só (`i={i1,i2,i3}` ou `i(t0)..`) — evitar subscrito duplo
- Discussão: pontos da §2 acima
- Tab. clínica 4 coortes já no tex; Tab. ADNI study já no tex

### F. Armadilhas conhecidas
- Extra §0 **reescreve** mono claim (svm+rf+EN nocombat) — pode sobrescrever ComBat do full
- `COMMON` com `--tasks smci_pmci` **depois** de `--tasks cn_ad` → bug (já corrigido no extra)
- Pasta `36m_6m_old/*_cn_ad` = legado; claim actual precisa re-run cn_ad

---

## 4. Ficheiros-chave

- Resultados: `csvs/cohort_comparison/cohort_results.csv`
- Stats: `7_stats.ipynb`
- Figs: `6_results.ipynb`
- Artigo: `artigo/artigo.tex` + `cas-refs.bib`
- Coortes/clínica: `1_dataset1.ipynb`
- Allowlist feats: `modules/ablation_prep.py` (MeshVolume ∈ vol)

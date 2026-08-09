# TODO — handoff (ler isto antes de correr / decidir)

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Docs:** `artigo.md` = briefing científico; **este ficheiro** = estado + comandos + decisões fechadas  
**Python:** `.venv/bin/python` · logs em `logs/`

---

## Estado (2026-08-09)

| Item | Status |
|------|--------|
| Protocolo gap | `forward_band_pm2` (±2: 6→`[4,8]`, 12→`[10,14]`) |
| Coortes | `csvs/cohorts/{36,48}m_{6,12}m/` (`*_old` = legado — **não misturar**) |
| Contagens sMCI/pMCI | 125/106 · 121/33 · 73/120 · **72/48** (`48m_12m` = **claim**) |
| Features + `4_run_post_extract` | OK nas 4 |
| `./run_ablation_full.sh` | **DONE** |
| Planilha `csvs/cohort_comparison/{cohort_results,cohort_features_long}.csv` | **DONE** (pós-full; re-gerar **após** extra terminar) |
| `./run_ablation_extra.sh` | **RUNNING** — `FUSION_MOD=shape REP=wide` · log `logs/ablation_extra_20260809.log` |
| `artigo.md` números | stale / parcialmente old — reescrever pós-extra + stats |

**Próximo após `DONE extra`:** reconsolidar planilha → stats (`7_stats.ipynb`) → escrever.  
**Não** relançar o extra enquanto o log actual estiver a correr.

---

## Decisões fechadas (não reabrir sem dados novos)

### 1) Ranking absoluto cross-cohort engana

Top AUCs globais caem em late fusion em `*6m` (populações diferentes: N e %pMCI).  
**Não** concluir “gap 6 > gap 12” nem “longitudinal não agrega” a partir do top-20 global.  
Comparar **sempre na mesma coorte** (baseline vs long; base-late vs long-late).

### 2) Longitudinal vs baseline (mesma coorte) — SIM agrega

SVM nocombat, best(long) vs `t1_only` por modalidade: maioria long > base; claim `48m_12m` = **4/4** mods.  
Late com deltas > late só `t1_*` nas 4 coortes.

### 3) `wide` vs deltas — não confundir

| Encoding (CLI) | Planilha `protocol` | O que é |
|----------------|---------------------|---------|
| `t1_only` | `t1_only` | baseline (1 visita) |
| `wide` | `abs` | 3 visitas **concatenadas** (long em **nível**) |
| `t1_deltas` | `t1_deltas` | t0 + deltas absolutos (long em **mudança**) |
| `t1_deltas_rel` | `deltas_rel` | t0 + deltas relativos |

- `wide` **é** longitudinal (multi-tempo), **não** é “a hipótese deltas”.  
- Teto uniclasse (melhor mod) nas **4** coortes = **`wide`**.  
- Em **texture/vol** (sobretudo gap 12) `t1_deltas` / `t1_deltas_rel` muitas vezes ≥ `wide` → narrativa de **mudança** vai aí, não no teto global.

### 4) Extra: `FUSION_MOD` ≠ `REP`

- **`FUSION_MOD`** = família de atributos imagem (shape/vol/texture/disp)  
- **`REP`** = encoding temporal do braço imagem (e CN×AD / leaky)

**Teto unimodal claim `48m_12m` (SVM, nocombat, `auc_patient_mean`):**

| REP (CLI) | Melhor mod | AUC | Nota |
|-----------|------------|----:|------|
| **`wide`** | **shape** | **0.783** | teto imagem overall |
| `t1_deltas_rel` | texture | 0.771 | 2º long |
| `t1_only` | **shape** | **0.761** | teto **baseline** |
| `t1_deltas` | texture | 0.753 | &lt; baseline no teto |
| (`t1_deltas`/shape = 0.737 — pior) | | | |

Defaults do script (`FUSION_MOD=shape REP=t1_deltas`) estão **ERRADOS** — **sempre** passar env vars.

**Escolha canónica para fusion clinic+img (melhor braço imagem):**  
`FUSION_MOD=shape REP=wide`

Alternativas (só se a claim da fusion for outra):

| Opção | Quando | Env |
|-------|--------|-----|
| **A (canónica)** | teto imagem = wide/shape | `FUSION_MOD=shape REP=wide` |
| B | fusion em cima do teto **baseline** | `FUSION_MOD=shape REP=t1_only` |
| C | melhor encoding **deltas** | `FUSION_MOD=texture REP=t1_deltas_rel` |
| D | subset passo-0 only (`t1_only`/`t1_deltas`) | `FUSION_MOD=texture REP=t1_deltas` |

Passo 0 do extra **sempre** corre só `t1_only` + `t1_deltas` × svm/rf/elasticnet (hardcoded) — **não** muda com `REP`. `REP` afeta CN×AD, fusion, leaky.

---

## 1) EXTRA — lançado (aguardar fim)

```bash
# já corrido 2026-08-09 — NÃO relançar em paralelo
cd /mnt/study-data/pgirardi/graphs
mkdir -p logs
FUSION_MOD=shape REP=wide ./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
```

**Log activo:** `logs/ablation_extra_20260809.log`  
Ordem: (0) sens mono claim `t1_only`/`t1_deltas` × svm,rf,elasticnet → (1) CN×AD vol `wide` → (2) clínica → (3) fusion clinic+img **shape×wide** → (4) leaky vol `wide`.  
Sucesso = linha `DONE extra` sem `FAIL`. Pasta fusion esperada: `csvs/cohorts/48m_12m/ablation_results_clinic_img/` (wide).

Monitor:
```bash
tail -f logs/ablation_extra_20260809.log
# ou: grep -E 'DONE extra|FAIL|=== ' logs/ablation_extra_20260809.log
```

Só re-correr fusion **depois** se falhar / mudar de ideia (não agora):

```bash
cd /mnt/study-data/pgirardi/graphs
.venv/bin/python 5_clinic_img.py --cohort 48m_12m --feature-set fusion \
  --modality shape --representation wide \
  --tasks smci_pmci --selection l1_stable --models svm --combat false \
  --repeats 10 --tuner optuna --optuna-trials 10 \
  --stable-pool-min-pct 70 --stable-pool-min-timepoints 0 \
  --stable-bootstrap 50 --stable-l1-c 0.1
```

- [x] Lançar extra com `FUSION_MOD=shape REP=wide` (log `ablation_extra_20260809.log`)
- [ ] Log sem `FAIL` / aparecer `DONE extra`
- [ ] Confirmar pasta `csvs/cohorts/48m_12m/ablation_results_clinic_img/`

**Não** expandir clínica / leaky / CN×AD às 4 coortes.

---

## 2) Reconsolidar planilha (só depois de `DONE extra`)

Demora (bootstrap `n_boot=2000`); stdout quiet até ao fim. Sintaxe do `-c` precisa fechar `)` + `print`.

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

- [x] Planilha pós-full
- [ ] Re-gerar após extra (clínica / CN×AD / leaky / fusion entram)

---

## 3) Pós-extra (sem novos experimentos)

### Stats / claims
- [ ] `7_stats.ipynb` — Δ, IC, FDR, **gate_pass** antes de claim “> shape”
- [ ] Tabela âncora: mono | multi-T1 | late | early + Δ/IC/gate  
  (late âncora shape t1 ∪ tex Δ ≈ 0.80 vs shape t1 ≈ 0.76)

### Escrever
- [ ] Actualizar **todos** os números em `artigo.md` aos ±2
- [ ] Narrativa: (i) long vs base na **mesma** coorte; (ii) `wide` = long nível, deltas = long mudança; (iii) vol unimodal claim ≠ ~0.8
- [ ] Atributos estáveis (`cohort_features_long`); figuras

---

## Não fazer

- Crownear top late com disp  
- Grade early completa / weighted late  
- Claim “significativamente > shape” sem `gate_pass`  
- Misturar AUC `*_old` com coortes ±2  
- Usar default `REP=t1_deltas` + `FUSION_MOD=shape` no extra  
- Comparar AUC absoluto entre `*6m` e `*12m` como prova de hipótese temporal  
- Nova ablação além de full+extra  
- `5_ablation_deltas.py` — redundante  
- Promover `t1_deltas_rel` a primary sem análise explícita por modalidade  

---

## Quick ref — ficheiros

| Path | Uso |
|------|-----|
| `run_ablation_full.sh` | grade core (DONE) |
| `run_ablation_extra.sh` | extras claim (**RUNNING** → marcar DONE quando log disser `DONE extra`) |
| `modules/cohort_compare.py` | consolida → `csvs/cohort_comparison/` |
| `5_clinic_img.py` | clínica / fusion clinic+img |
| `5_ablation.py` / `5_ablation_leaky.py` | mono / leaky |
| `artigo.md` | texto científico (números a actualizar) |

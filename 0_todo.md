# TODO — handoff lab → laptop

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Docs:** `artigo.md` = briefing; este ficheiro = pendências técnicas

---

## Estado (2026-08-07)

**Protocolo gap:** `forward_band_pm2` (±2: 6→`[4,8]`, 12→`[10,14]`). Coortes em `csvs/cohorts/{36,48}m_{6,12}m/` (`*_old` = legado).

**Contagens sMCI/pMCI:** 125/106, 121/33, 73/120, **72/48** (`48m_12m` = **claim** multimodal/extras).

**Pré-requisito:** features + `4_run_post_extract.py` nas 4 coortes.

**AUCs em `artigo.md`:** provisórios até rerun ±2 — não misturar com `*_old`.

**Experimentos para o paper:** `full` + `extra` **fecham a grade**. Depois disso **não** falta ablação nova — só consolidar, stats, escrever, tabelas, figuras. Único re-run possível: fusion clinic+img se o teto unimodal da claim ≠ shape.

---

## 1) Runs (última grade experimental)

Ambos executáveis (`-rwx`).

### Core — `./run_ablation_full.sh`

Mono (4 coortes × `t1_only`/`t1_deltas`/`wide`, ComBat **both**) → late (72 specs × 4, **nocombat**) → early (7 specs, claim, nocombat).

```bash
cd /mnt/study-data/pgirardi/graphs
./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
```

- [ ] Correr `run_ablation_full.sh`
- [ ] Logs sem `FAIL`
- [ ] Anotar teto unimodal em `48m_12m` (mod × rep) → decide `FUSION_MOD` do extra

### Extra — `./run_ablation_extra.sh`

Claim `PRIMARY=48m_12m`: sensibilidade (`t1_only`/`t1_deltas` × `svm,rf,elasticnet`) → CN×AD (vol) → clínica → fusion clinic+img (`FUSION_MOD`) → leaky (vol).

Correr **depois** do full; ajustar `FUSION_MOD`/`REP` ao teto SVM nocombat.

```bash
FUSION_MOD=shape REP=t1_deltas ./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
```

- [ ] Correr `run_ablation_extra.sh` (paralelo ao full OK)
- [ ] Logs sem `FAIL`
- [ ] Se mono claim mostrou outro teto: re-correr **só** fusion clinic+img com `FUSION_MOD` certo (não reabrir grade)

**Não** expandir clínica/leaky/CN×AD às 4 coortes.

---

## 2) Próximos passos (pós-runs) — sem novos experimentos

Ordem sugerida:

### A. Consolidar planilhas
Resultados **não** estão num único ficheiro após os bash; cada protocolo/mod/coorte tem pasta. Consolidar:

- [ ] `6_results.ipynb` → `all_protocols_summary.csv` (por coorte) + comparação multi-cohort  
- [ ] ou CLI equivalente via `cohort_compare.save_cohort_comparison` →  
  `csvs/cohort_comparison/cohort_results.csv` + `cohort_features_long.csv`  
  (atributos seleccionados / freqs; união de mods = agregar esse long)

### B. Stats / claims
- [ ] `7_stats.ipynb` — Δ, IC, FDR, **gate_pass** antes de claim “> shape”
- [ ] Tabela âncora: mono | multi-T1 | late | early + Δ/IC/gate

### C. Escrever artigo
- [ ] Actualizar **todos** os números em `artigo.md` / draft (mono, late, early, clínica, CN×AD, leaky)
- [ ] Métodos já OK em grande parte; Results + discussão alinhados aos ±2
- [ ] Tabelas de texto (não só plots)
- [ ] Parágrafo atributos estáveis (de `cohort_features_long`)
- [ ] Figuras: forest Δ âncora vs shape T1 (4 cohorts), barras mono, early vs late, etc.

---

## Não fazer

- Crownear top late com disp  
- Grade early completa / weighted late  
- Claim “significativamente > shape” sem `gate_pass`  
- Misturar AUC `*_old` com coortes ±2  
- Nova ablação além de full+extra (+ opcional `FUSION_MOD`)  
- `5_ablation_deltas.py` — redundante  

---

## Nota

Gap ±2 **fechado**. Claim = `48m_12m`; `36m_6m` = arranque unimodal histórico, não primary dos extras.

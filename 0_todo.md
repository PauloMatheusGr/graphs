# TODO — handoff lab → laptop

**Repo:** `/mnt/study-data/pgirardi/graphs`  
**Docs:** `artigo.md` = briefing; este ficheiro = pendências técnicas

---

## Estado (2026-08-07)

**Protocolo gap:** `forward_band_pm2` adoptado (±2: 6→`[4,8]`, 12→`[10,14]`). Coortes novas em `csvs/cohorts/{36,48}m_{6,12}m/` (pastas `*_old` = legado só-mínimo).

**Contagens sMCI/pMCI (actual):** 125/106, 121/33, 73/120, **72/48** (`48m_12m`).

**Pré-requisito runs:** features + `4_run_post_extract.py` nas 4 coortes (e primary para extra).

**AUCs em `artigo.md`:** provisórios até rerun nestas coortes — não misturar com `*_old`.

---

## Runs — o que executar

Ambos **já executáveis** (`chmod +x` feito: `-rwxrwxr-x`).

### 1. Core — `./run_ablation_full.sh`

Mono (4 coortes × `t1_only`/`t1_deltas`/`wide`) → late (72 specs) → early (7 specs, `48m_12m`).

```bash
cd /mnt/study-data/pgirardi/graphs
./run_ablation_full.sh 2>&1 | tee logs/ablation_full_$(date +%Y%m%d).log
```

- [ ] Correr `run_ablation_full.sh`
- [ ] Verificar logs sem `FAIL`

### 2. Extra — `./run_ablation_extra.sh`

Só primary (`PRIMARY=48m_12m`, `REP=t1_deltas`): CN×AD → clínica → fusion clinic+img (vol) → leaky.

Script: `5_clinic_img.py` (ex-`5_baseline_comparison.py`; `--cohort` OK).

```bash
./run_ablation_extra.sh 2>&1 | tee logs/ablation_extra_$(date +%Y%m%d).log
# opcional: PRIMARY=36m_6m REP=t1_deltas ./run_ablation_extra.sh
```

- [ ] Correr `run_ablation_extra.sh` (pode em paralelo / outro terminal vs full)
- [ ] Verificar logs sem `FAIL`

**Não** meter clínica/leaky/CN×AD nas 4 coortes (YAGNI paper).

---

## Depois dos runs

- [ ] `6_results.ipynb` / `7_stats.ipynb` nas pastas novas
- [ ] Actualizar números em `artigo.md` (mono teto, late, early, clínica, CN×AD, leaky)
- [ ] Tabela resumo âncora: mono | multi-T1 | late | early + Δ/IC/gate
- [ ] Forest Δ âncora vs shape T1 (4 cohorts) se ainda no plano

## Não fazer

- Crownear top late com disp  
- Grade early completa / weighted late  
- Claim “significativamente > shape” sem `gate_pass`  
- Misturar AUC `*_old` com coortes ±2  
- `5_ablation_deltas.py` — redundante (`5_ablation.py --representation t1_deltas`)

---

## Nota histórica (resolvido)

Debate só-mínimo vs banda ±2/±3 → **fechado com ±2**. Secções longas de estimativa lab estão obsoletas; ver git history se precisares dos números pré-adopção.

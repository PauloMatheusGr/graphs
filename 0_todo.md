Pegar informação de como a adni fornece os diagnósticos

"""
Até a etapa 5. é a exclusão padrão do meu grupo de pesquisa, as etapas 6 e 7 são especificas das minhas coortes, com esse arquivo gerado, posso utiliza-lo como base para que meus colegas do grupo de pesquisa também utilizem esse arquivo de imagens excluidas, pelo menos até a etapa 5? 

Por exemplo, o caso do Vinicius, ele utilizou o arquivo (ver em "/home/pgirardi/Desktop/exp_vinicius/raw")
"""

# Todo — fecho artigo v1

Atualizado: 2026-09-17. CSVs experimentais quase fechados; falta remate disp-v4 + rebuild + figuras + stats + tex.

---

## Lock

| | |
|---|---|
| Claim | **`48m_6m`**, `PARAM_SOFT_PMCI=True` (73 sMCI / 120 pMCI) |
| Encodings | `t1_only` · `t1_d21` · Q4 `t1_d21_d32` |
| Knobs | `l1_stable` · **combat false** (primário) · seed 42 · SVM |
| ROI | hipocampo L+R |
| Famílias | vol · shape · texture · disp (v4) · firstorder |
| Métrica | `auc_patient_mean` |
| Coortes gradiente | `36m_6m`, `36m_12m`, `48m_6m`, `48m_12m` (partilham pacientes — dizer no paper) |

---

## Estado experimental

| Bloco | Estado |
|---|---|
| Extract + `4_` disp v4 (`jac_det` no long) | OK |
| `run_dvf_v4.sh` (mono disp + late c/ disp, 4 coortes + soft_False) | **em curso** → esperar `DONE` |
| Unimodal vol/shape/texture/FO (claim + gradiente) | OK (não re-rodar) |
| Soft falso T1+Q4 (5 fam.) | OK; disp actualiza neste run |
| Late paper (all-T1, all-Q4, âncora) | OK; specs c/ disp actualizam neste run |
| LongCombat Q4 vol/shape/texture/FO | OK (`*_longcombat`) |
| LongCombat **disp** + late all-Q4 | **após DONE** (disp ainda v3 na pasta longcombat) |
| `cohort_results.csv` | **stale** → rebuild após DONE |
| `6_results` / `7_stats` / `artigo.tex` | **falta** |

---

## Próximos passos (ordem)

### 0. Esperar `DONE` do `run_dvf_v4.sh`

Log: `logs/ablation_disp_v4_*.log`. Não lançar segundo job nas mesmas pastas `…/disp/`.

### 1. LongCombat remate (claim, só disp + late all-Q4)

```bash
cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate

python 5_ablation.py \
  --cohort 48m_6m --representation t1_d21_d32 --modality disp \
  --tasks smci_pmci --selection l1_stable --models svm \
  --combat true --repeats 10 --seed 42 \
  --tuner optuna --optuna-trials 10 && \
python 5_ablation_late_fusion.py \
  --cohort 48m_6m \
  --fusion vol:t1_d21_d32,shape:t1_d21_d32,texture:t1_d21_d32,disp:t1_d21_d32,firstorder:t1_d21_d32 \
  --tasks smci_pmci --selection l1_stable --models svm \
  --combat true --repeats 10 --seed 42 \
  --tuner optuna --optuna-trials 10 \
  --combine mean --reuse-disk
```

Actualiza Tabela D (linha ComBat). Vol/shape/texture/FO longCombat **não** re-rodes.

### 2. Rebuild compare

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

Ou `6_results.ipynb` com `REBUILD_COMPARE = True`.

### 3. Figuras / tabelas — `6_results.ipynb`

- [ ] Fig A: 4 painéis × 5 famílias × 3 barras (T1 / D21 / Q4)
- [ ] Tabela A: AUCs + Δ vs T1 (4 coortes × 5 fam.)
- [ ] Tabela B: claim × 4 algoritmos (T1 | Q4)
- [ ] Fig B: ROC **volume** (teto unimodal baseline), 3 encodings, só claim
- [ ] Fig / tabela **quatro tetos**: uni T1 · uni Q4 · melhor união baseline · melhor união long (+ âncora)
- [ ] Fig C / Tabela E: soft True vs False (T1 + Q4, 5 fam.)
- [ ] Exportar para `artigo/`

### 4. Stats — `7_stats.ipynb`

- [ ] Coorte principal = `48m_6m` (textos antigos `48m_12m`)
- [ ] Contraste **quatro tetos** + âncora (`vol:t1` ∪ 2.º Q4) vs melhor união grelha
- [ ] Clínica+img = **volume** T1
- [ ] Tabela C: Q4 vs T1 **e** D21 vs T1 (FDR); claim + 4 coortes
- [ ] Late: all-T1, all-Q4, âncora; grelha = ranking (winner’s curse no texto)
- [ ] Clínica: (clinic+vol) − clinic + IC
- [ ] Soft True vs False (descritivo; **não** bootstrap pareado — n diferente)
- [ ] Tabela D: complementar (âncora, leaky, clinic, clinic+vol, **longCombat**, soft)
- [ ] Gravar `artigo/tables/`

### 5. Texto LaTeX

- [ ] Colar números novos; enxugar (~18–22 pp)
- [ ] Soft True = claim; soft False = sensibilidade pré-conversão
- [ ] Coração = quatro tetos + âncora; grelha ≠ prova sem caveat
- [ ] Discussão: checklist abaixo

---

## Checklist — análises a mencionar (Discussão / Conclusões)

Usar como lista de “não esquecer”. **Coração** vs complementares.

### ★ Coração dos experimentos (claim)

Contraste de **quatro tetos** (mesmo pipeline SVM / `l1_stable` / nested CV; `combat=false`):

| Teto | O quê |
|---|---|
| Melhor **unimodal baseline** (T1) | ranking 5 famílias em `t1_only` |
| Melhor **unimodal longitudinal** (Q4) | ranking 5 famílias em `t1_d21_d32` |
| Melhor **união / multiclasse baseline** | grelha late só-T1 (ou all-T1 como spec paper) |
| Melhor **união / multiclasse longitudinal** | grelha late Q4 / mista (ou all-Q4 como spec paper) |

**Âncora (pré-especificada, não o max da grelha):**  
`melhor unimodal T1` ∪ `2.º unimodal Q4` (família **≠** a do 1.º Q4 quando o 1.º Q4 = mesma família do teto T1).

Motivo: Q4 = baseline + Δ21 + Δ32 → unir `vol:t1` ∪ `vol:Q4` **repete** o baseline; o 2.º longitudinal traz família complementar.

Ex. se T1 e Q4 ordenam ambos `vol > shape > FO > texture > disp` → âncora = **`vol:t1_only` ∪ `shape:t1_d21_d32`**.

Grelha de uniões (k≥2) **necessária**: âncora fixada *antes* de saber qual união ganha de facto → comparar âncora vs melhor união empírica (com caveat winner’s curse no suplemento).

Pós-correção ICV (protocolo do paper): **vol = melhor unimodal baseline**. (Bug antigo: eixos/`SurfaceArea` ÷ ICV → unidades erradas; **só** resultados pós-homotetia entram no artigo.)

Encoding unimodal (T1 / D21 / Q4 × 5 fam.) + 4 coortes = camada que alimenta estes tetos. Soft / ComBat / clinic / idade / leaky / CN×AD / 4 algos = **complementares**.

---

### A. Camada principal (detalhe)

1. **Unimodal × encoding** — 5 famílias × {T1, D21, Q4}; SVM; `combat=false`. Pergunta: 1 vs 2 vs 3 imagens agrega?
2. **Quatro coortes** — janela 36/48 m × intervalo 6/12 m; **não** estudos independentes (pacientes partilhados).
3. **Volume = teto unimodal baseline** — protocolo correcto (homotetia ICV). Hierarquia típica T1: vol ≻ shape ≻ … Discutir morfometria (vol/shape/disp) vs intensidade (texture/FO).
4. **Âncora late** — regra acima; CLI/default ainda `--anchor-modality shape` → **re-correr** `vol:t1_only` ∪ 2.º Q4 (provável `shape:t1_d21_d32`) após rankings finais (incl. disp v4).
5. **Três specs paper + grelha** — all-T1; all-Q4; âncora; grelha = achar melhores uniões baseline/long para o contraste dos quatro tetos (suplemento = exploratório).
6. **Disp v4** — mag / jac_det / strain_fro (MBEC); vs baseline antigo afim+logjac.
7. **Intervalo ~6 m** — detectável em **grupo** (Schuff, Mubeen, Hua, Leung); ≠ fiabilidade individual; 3 visitas ≈ 12 m de trajectória.

### B. Complementares (só claim `48m_6m`)

8. **Quatro algoritmos** — SVM, RF, elasticnet, XGB em T1 e Q4 (não em D21 no corpo).
9. **Soft True vs Soft False** — 73/120 vs 73/74; T1+Q4; 5 fam. + late. Leakage MCI–MCI–AD?
10. **Longitudinal ComBat (Beer 2020)** — sensibilidade Q4; batch fabricante×Tesla; **não** melhora teto (pré-v4: late 0.787→0.761; vol/FO caem). Actualizar após remate disp-v4.
11. **Clínico só** vs **clínico + volume T1**.
12. **Controlo demográfico (idade)** — só idade; faixas; idade+faixas. Imagem **supera** demografia.
13. **Leaky** — volume Q4 (e D21 se existir); não misturar no abstract com leak-free.
14. **CN × AD** — sanity nas 3 visitas (mesmo pipeline Q4).

### C. Métodos / limitações a verbalizar

15. Split **por paciente**; nested 5×5 × 10 repeats; Optuna inner.
16. `l1_stable` só no outer-train.
17. ICV shape = **homotetia** (eixos × ICV⁻¹/³, área × ICV⁻²/³); bug antigo ÷ICV **fora** do paper.
18. Batch scanner = MANUFACTURER_FIELD (sem modelo).
19. Soft True **não** é prognóstico estritamente pré-desfecho para os 46 MCI–MCI–AD.
20. Grelha late: melhor união empírica ≠ prova sem caveat (winner’s curse).
21. Fora do corpo v1: early fusion; HM-off; 3 heatmaps; 4 algos em D21; calibração/DCA; n por fold; retest noise floor.

### D. Conclusões — eixos esperados

- **Quatro tetos:** unimodal T1 vs unimodal Q4 vs melhor união baseline vs melhor união long (+ âncora pré-especificada).
- Encoding: onde Q4/D21 ganha vs T1 (e onde não).
- Família dominante baseline: **volume** (protocolo ICV correcto).
- Âncora: vol-T1 ∪ 2.º Q4 (não vol∪vol); grelha diz se outra união sobe mais.
- Demografia: idade/faixas << imagem.
- Soft / longCombat / leaky: sensibilidade; não mudam o protocolo primário se negativos.
- Gradiente de coortes: consistência / poder vs intervalo 6 vs 12 m.
- Limitações: n, IDs partilhados, T1-w não quantitativo, batch grosso, disp SNR, winner’s curse na grelha.

---

## Não fazer agora

- Relançar vol/shape/texture/FO mono
- LongCombat nas 4 coortes ou em `t1_only`
- Early fusion / `--modality all` no corpo
- `4_` com `DISP_FEATURES=v3`
- Segundo `run_dvf_v4.sh` enquanto o actual corre

---

## Refs úteis (Intro / Discussão 6 m)

- Mubeen et al. 2017 (J Neuroradiol) — baseline+6 m vs baseline  
- Schuff et al. 2009 (Brain) — taxa atrofia hipocampal 0–6 m  
- Hua / Leung — morfometria longitudinal ADNI  
- Beer et al. 2020 (NeuroImage) — Longitudinal ComBat  
- Fortin et al. 2018 — ComBat cortical thickness (contexto)

# TODO artigo v2 — estado em 2026-09-28 19:35 (BRT)

> **Escopo congelado.** Experimento novo só se orientador pedir ou revisor exigir.
> Ideias novas → seção "Trabalho futuro" no fim deste arquivo.

## ⚠️ Trabalhar do PC pessoal
- `csvs/`, `logs/`, `Artigo 1 pgirardi/` estão no `.gitignore` → **dados e tabelas só existem no server 2**.
- Usar Cursor **Remote-SSH** no server 2 (não clonar e rodar local).
- Código de hoje: commitar + push antes de trocar de máquina (último commit = `45a29ce backup`, 28/09 01:05).

---

## 🔄 Em execução

### A. tmux 0 — claim_smci (4 modelos, só smci_pmci)
Loop: `for rep in t1_ols t1_only; for mod in shape texture firstorder disp` · logs `logs/claim_smci_<rep>_<mod>.log`
- [x] t1_ols shape (1h40) · [x] t1_ols texture (1h35) · [~] t1_ols firstorder (XGB rep 2/10 às 19:32; lento por contenção de CPU) · [ ] t1_ols disp
- `t1_only` já está completo (4 modelos + cn_ad, feito pelo claim_core4). Rodar de novo **sobrescreve o CSV e apaga cn_ad** → o **vigia** (B) mata esses runs automaticamente.
- Backup feito: `csvs/cohorts/48m_6m/ablation_results_{t1_only,ols}.bak_20260928/`. Se o loop rodou t1_only, restaurar:
  ```bash
  cd /mnt/study-data/pgirardi/graphs/csvs/cohorts/48m_6m
  rm -r ablation_results_t1_only && cp -a ablation_results_t1_only.bak_20260928 ablation_results_t1_only
  ```
- Conferir no fim: `ols/{shape,texture,firstorder,disp}` com svm,rf,elasticnet,xgb (smci). `ols/vol` já tem 4 modelos + cn_ad (não é tocado).

### B. tmux `vigia` — mata t1_only do claim + roda ComBat longitudinal D depois
- ComBat longitudinal D (SVM): [x] vol · [x] shape (em `ablation_results_ols_longcombat/`) · [ ] texture · [ ] disp · [ ] firstorder
- Run antigo do tmux 3 **cancelado** (rodar junto com o claim travava o XGB, que usa todos os núcleos). tmux 3 pode ser fechado: `tmux kill-session -t 3`.
- O vigia (fase 1) mata runs `--representation t1_only ... claim_smci_t1_only` a cada 5 s; quando existir `logs/claim_smci_t1_ols_disp.log` e nenhum run do claim por 30 s → (fase 2) roda texture, disp, firstorder com `--long-combat` (OMP=2, `nice -n 10`, logs `logs/longcombat_t1_ols_<mod>.log`).
- Mensagens esperadas no `tmux a -t vigia`: silêncio → `matou run t1_only` (até 4×) → `claim terminou` → `longcombat <mod>` → `DONE longcombat`. Sair com **Ctrl+B D** (Ctrl+C mata o vigia).
- ⚠️ Nunca pausar run de loop com `kill -STOP`: o bash pula para a próxima iteração.
- Se o vigia morrer: esperar o claim terminar, Ctrl+C no tmux 0 se começar `t1_only`, e rodar à mão:
  ```bash
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  COMMON='--tasks smci_pmci --selection l1_stable --models svm --repeats 10 --seed 42 --tuner optuna --optuna-trials 10 --stable-pool-min-pct 70 --stable-pool-min-timepoints 0'
  for mod in texture disp firstorder; do
    nice -n 10 .venv/bin/python 5_ablation.py --cohort 48m_6m --representation t1_ols --modality "$mod" \
      --long-combat $COMMON --log-file logs/longcombat_t1_ols_${mod}.log
  done
  ```
- Conferir no fim:
  ```bash
  ls csvs/cohorts/48m_6m/ablation_results_ols_longcombat/*/ablation_results_all.csv   # 5 famílias
  diff -rq csvs/cohorts/48m_6m/ablation_results_t1_only csvs/cohorts/48m_6m/ablation_results_t1_only.bak_20260928 && echo "t1_only intacto"
  ```
- Depois de conferido: fechar `tmux kill-session -t vigia`; apagar backups `*.bak_20260928` só após §2 rodar ok.

### C. tmux 1 — DVF groupwise AD (independente do artigo v2)
```
python 3.1_feat_gen_dvf.py --diag AD
python 3.2_feat_dvf.py --diag AD      # rodando
python 4_run_post_extract.py
bash run_dvf_anchor_compare.sh
```

---

## 🧩 Mudanças de código de 28/09 (commitar se ainda não)
- `5_ablation.py`: `--combat true` = **ComBat transversal (NeuroComBat)** → `ablation_results_combat_*`; nova flag `--long-combat` = **ComBat longitudinal (Beer 2020)** → `ablation_results_*_longcombat`. Exclusivas.
  ⚠️ Não usar `--combat both` (grava na pasta padrão, sobrescreve nocombat).
- `modules/ablation_harmonize.py`: NeuroComBat restaurado; `harmonize_long_fold(method="longitudinal"|"transversal")`; longitudinal aceita `t1_only, t1_r10_r21, t1_rate02, t1_ols` (3 visitas).
- `modules/ablation_representation.py`: protocolo `combat` + pastas `longcombat` novas.
- `modules/ablation_runner.py`: `combat_method` propagado; coluna `harmonization_method` = none | neurocombat | longitudinal_combat_reml.
- `modules/stats_compare.py`: `bootstrap_auc_diff_test(..., ci_level=0.95)`; `image_ablation_path` aceita `t1_ols_longcombat` (e t1_only/r10r21/rate02 `_longcombat`).
- `5_ablation_late_fusion.py` **não mudou**: lá `--combat` ainda = longitudinal.
- `7_stats.ipynb`: nova **§10 Sensibilidade α=1%** (`stats_sensitivity_alpha01`). Usa `importlib.reload(stats_compare)`.
- `0_notes.md`: fórmulas + explicação de todos os métodos estatísticos.

---

## ▶️ Próximos passos (em ordem)

### 0. Enquanto os runs rodam (não depende deles)
- [ ] §3 abaixo (Friedman B/C/D) — dados já existem
- [ ] tex v2 (passo 5): itens que não dependem de números novos (renomear figuras, legendas, B₀/Q4 removidos, texto Métodos estatísticos a partir de `0_notes.md`)

### 1. Depois de A terminar (claim em tmux 0)
- [ ] `PYTHONPATH=modules python modules/cohort_compare.py --include-soft --n-boot 2000`
- [ ] `6_results`: heatmap modelos×fam (T1/D, cell `heatmap_models_*_48m6m`), ROC × 4 modelos (T1 + D), cn_ad T1 vs D (cn_ad D só existe em vol)
- [ ] `6_results` V1: fig A, claim, ROC encodings, four_ceilings, soft, heatmaps, ROC modelos

### 2. `7_stats` — reexecutar (Restart kernel: `stats_compare` mudou)
- [ ] §1–§5, §8, §9, §10 → conferir que resultados SVM não mudaram após reescrita dos CSV `ols/*` (seed 42 → deve sair igual)
- Referência atual (unimodal, 60 testes): 48m_12m vol D−T1 ΔAUC=0.133 [0.040, 0.227] p=0.004 q_coorte=0.020 q_global=0.080 → FDR sig. a 5% (por coorte); a 1% só sem correção (IC99 [0.007, 0.256]). 36m_6m texture 2v−T1 q=0.045 (FDR sig. 5%, cai a 1%).

### 3. `7_stats` — nova §11 escolha B/C/D (dados já existem, sem run novo)
- [ ] Friedman: 20 blocos (5 fam × 4 coortes) × {B=t1_r10_r21, C=t1_rate02, D=t1_ols}, AUC paciente SVM; se p<0.05 → Wilcoxon pareado D vs B, D vs C (Holm)
- [ ] Opcional: bootstrap pareado D vs B e D vs C por cenário + BH por coorte (mesmo padrão do §4; C precisa `compare_encoding_vs_t1`-like com path `t1_rate02`)
- Texto: D escolhido a priori por razão conceitual (3 visitas, robusto a ruído de 1 visita, nível+inclinação interpretáveis); Friedman confirma/nega; B e C no suplemento.

### 4. Depois de B terminar (`DONE longcombat`) — estender §8 (ComBat)
- [ ] Contrastes pareados (bootstrap + BH por protocolo, 5 fam):
  - T1 ComBat transversal − T1 nocombat (já existe: `ablation_results_combat_t1_only`)
  - **D longcombat − D nocombat** (`image_ablation_path(BASE, "t1_ols_longcombat", mod)`, cfg `with_combat=True`)
  - **D longcombat − T1 ComBat transversal** (ganho longitudinal com ambos harmonizados)
- [ ] Atualizar Tabela D (§9) com as linhas novas.
- Frase Métodos: "Dados transversais (T1) harmonizados com ComBat; longitudinais (D) com ComBat longitudinal (Beer et al., 2020); ambos ajustados só no treino de cada fold."

### 5. tex v2
- [ ] renomear `*_q4_*` → `*_ols_*` (unimodal, heatmap, roc_models)
- [ ] inserir `trajectories_vol_48m6m.pdf` (cell `v1_fig_traj`) no **suplemento**
- [ ] legendas Fig A / ROC vol / tetos → "3 visitas" = OLS; tetos "vol baseline ∪ forma 3 visitas" = `late__t1_vol__t1_ols_shape`; atualizar tabelas L715/L764
- [ ] `fig_a_encoding_4cohorts.pdf` → grelha 2×2 `fig_a_encoding_{36m6m,36m12m,48m6m,48m12m}.pdf` (`height=` igual; legenda só 36m12m)
- [ ] `soft_true_vs_false_48m6m.pdf` → 2 subfigures `…_baseline.pdf` (sem legenda) + `…_3visits.pdf` (com legenda)
- [ ] conferir se `diagrama.pdf` desenha Δ21/Δ32 → trocar por β̂₁ OLS
- [ ] **B₀/Q4 e D21 abs removidos** (decisão 2026-09-27). Suplemento = grelha A/B/C/D; corpo = A vs 2v R10 vs D
- [ ] Tabela D: tirar linhas leaky (Q4) e ComBat Q4; ComBat = T1 transversal + D longitudinal (passo 4)
- [ ] Métodos estatísticos (base em `0_notes.md`): bootstrap pareado 5000, IC percentil, p unilateral, BH por coorte (+ global como sensibilidade), rótulos, sensibilidade α=1%, Friedman
- [ ] Resultados: frase do achado principal (em `0_notes.md` §9) — "sugestivo e localizado"; não afirmar superioridade geral
- [ ] Soft True vs False: declarar descritivo (pacientes diferentes, sem pareamento)

---

## 💤 Trabalho futuro / limitações (não fazer agora)
- ComBat longitudinal em B/C e nas outras coortes
- Contraste direto D vs 2v
- Soft com teste formal
- Validação externa independente (achado 48m_12m vol, n=120)

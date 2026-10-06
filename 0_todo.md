# TODO — DVF hipocampo com templates OASIS-3 (sMCI vs pMCI)

Objetivo: experimento simples — os atributos `disp` (CN, AD, CN+AD) melhoram sMCI×pMCI ao trocar
os templates ADNI por templates normativos OASIS-3 (DIAG CN|AD × SEX × década 60-69/70-79/80-89,
N=20 cada, MNI152) e o hipocampo por hipocampo dilatado (raio fixo da literatura)? Mesmo pipeline
e mesmos atributos do `disp` ADNI. Rodada completa só se Gate A e Gate B passarem.

Ponto de partida (sMCI×pMCI, t1_only, SVM l1_stable, AUC por paciente, 50 folds):

| Coorte | `vol` | `disp` | `disp_ad` | `disp_cnad` |
|---|---|---|---|---|
| 48m_6m (n=193) | 0,756 | 0,540 | 0,583 | 0,554 |
| 48m_6m_soft_False (n=147) | 0,738 | 0,462 | 0,584 | 0,538 |

## Pré-especificação (fixada em 01/10/2026, antes de qualquer resultado sMCI×pMCI)

Não alterar depois de ver resultados; mudanças aqui invalidam o Gate B. Substitui a versão de
30/09 (núcleo + casca 2/4/8 + TBM `logjac`), descartada antes de qualquer resultado por ser
engenharia demais para a pergunta.

**Templates** (`2.3 build`): `ants.build_template` (atualização de forma + Sharpen), SyN MI
100x70x50x20, 4 iterações, N=20 por estrato, sobre as imagens do `prep` (hist-match → rígido
MNI). Registro sujeito → template (`3.1`): SyNRA CC r4 100x70x50x20, fixed = sujeito.

**ROIs** (`3.2 --src oasis`, espaço do sujeito, sempre dentro da máscara do cérebro):

- `hippocampus_d2` (**principal**) = núcleo ∪ voxels a ≤ 2 mm dele. Raio da literatura
  (Neurocomputing: "morphological dilatation operation, using a 3-D sphere of 2 voxels radius as
  the structuring element"); a 1 mm, EDT ≤ 2 mm = dilatação por esfera r=2 voxels
  (conferido no self-check).
- `hippocampus` (núcleo, 17/53) = **controle**: separa efeito do template do efeito da dilatação.

**Atributos**: os mesmos do `disp` ADNI (`keep_disp_feat`): `mag`, `jac_det`, `strain_fro` ×
mean/variance/skewness/kurtosis, por lado e por âncora (`disp_oasis`, `disp_oasis_ad`,
`disp_oasis_cnad`). Classificador igual (`COMMON_MONO`: SVM, l1_stable, Optuna 10, 10×5 folds,
seed 42, sem ComBat).

**Comparações** (sMCI×pMCI, AUC pareada nos mesmos folds, bootstrap 5000, IC95):

| Comparação | Responde |
|---|---|
| OASIS d2 × `disp` ADNI (núcleo) | ganho total — **decide o Gate B** |
| OASIS núcleo × `disp` ADNI (núcleo) | efeito só do template OASIS |
| OASIS d2 × OASIS núcleo | efeito só da dilatação |
| OASIS d2 / núcleo × `vol` | > / ≈ / < vol |

Folds de `vol` e `disp` ADNI conferidos: idênticos nas duas coortes.

### Adendo 06/10/2026 (antes de qualquer resultado sMCI×pMCI do Gate B)

Registro (`3.1`) intacto; mudanças só no `3.2` e em análises secundárias.

- **Correção de bug** (não muda atributo): `strain_fro` derivava o campo nos eixos de índice,
  sem a matriz de direção da imagem (MNI/LPS ≠ identidade) → cisalhamentos xz/yz misturavam
  rotação. Agora gradiente físico = gradiente de índice · Dᵀ (`3.2`, self-check de rotação rígida
  → strain 0). `jac_det` (ANTs) e `mag` não mudam. Atributos OASIS do Gate A reextraídos.
- **Novos resumos no CSV** (fora de `keep_disp_feat` → família primária inalterada):
  `logjac_rel_mean` = média logJ na ROI − média logJ no cérebro; `jac_nonpos_frac` (QC).
- **Análise secundária** (não altera o veredito do Gate B): famílias só-jacobiano
  `disp_oasis_jac` e `disp_oasis_ad_jac` = {`jac_det_mean`, `logjac_mean`, `logjac_rel_mean`} × L/R
  (6 atributos), d2 e núcleo, 2 coortes, mesmo `COMMON_MONO`. Comparações pareadas (mesmos folds,
  bootstrap 5000, IC95) vs `disp_oasis`/`disp_oasis_ad`, vs `disp`/`disp_ad` ADNI e vs `vol`;
  BH dentro da família secundária; reportar todas.
- **QC** (descritivo, não muda escolha): fração de jac ≤ 0 por imagem e |rho| de `jac_det_mean` /
  `logjac_rel_mean` com volume hipocampal / TIV (TIV = GM+WM+CSF; o `mask_mm3` global é FOV).
- **Fora**: SyN guiado pelo hipocampo (correlação por construção, novo registro), `∫J` (com
  fixed=sujeito ≈ volume do template, constante), mudanças no pipeline ADNI antigo.

**Rodada longitudinal (decidir com orientador ANTES do resultado do Gate B):**
442 pacientes × 3 visitas; t0 já registrado no Gate B. Ritmo atual ≈ 4,4 registros/h
(1 registro = 1 imagem × 1 âncora). Estimativas para t1/t2 restantes:

| opção | registros | tempo |
|---|---|---|
| (a) t1 + t2, âncoras CN e AD (independente, mesmo método do t0) | 1768 | ~17 dias |
| (c1) só t1 (R10, 2 visitas), CN e AD | 884 | ~8 dias |
| (c2) t1 + t2, só âncora CN | 884 | ~8 dias |
| (c3) só t1, só âncora CN | 442 | ~4 dias |

- (a) consistente com o t0 e com o pipeline ADNI; ruído de registro independente por visita.
- (b) template intraindivíduo (SST) + SST→OASIS: menos ruído longitudinal, mas muda o método
  (t0 teria que ser refeito) → não cabe no prazo; fica como trabalho futuro.
- (c) restrição declarada a priori (visitas e/ou âncora); escolha não pode depender do Gate B.

Decisão: ______ (data: __/__/2026)

### Próximos passos (checklist, em ordem)

1. [ ] **Reextração Gate A** (iniciada 06/10 ~12h, fora do tmux, termina ~14h30 do mesmo dia).
       Terminou quando aparecerem 2 linhas `[DONE]` (CN e AD):
       `grep "\[DONE\]" logs/logs_reextract_gateA.txt`
2. [ ] **Conferir Gate A** (segundos; só lê CSVs): `.venv/bin/python pilot_oasis_gate.py gate-a`
       → esperado `GATE A: PASSA`, números idênticos ao backup:
       `diff csvs/pilot/backup_pre_strainfix_20261006/gateA_summary.csv csvs/pilot/gateA_summary.csv && echo idêntico`
3. [ ] **Decidir rodada longitudinal com o orientador** (tabela acima) e preencher "Decisão"
       ANTES do resultado do Gate B.
4. [ ] **Esperar Gate B** (tmux `0`, ~11/10). Progresso:
       `ls images/displacement_field_oasis_{cn,ad}/*1Warp.nii.gz | wc -l` (meta 884; parado 2–5 h é
       normal, ondas). Terminou quando existir `csvs/pilot/gateB_summary.csv` e o tmux mostrar
       `GATE B (...): PASSA | INCONCLUSIVO | FALHA`.
5. [ ] **Análise secundária** (só depois do passo 4), no tmux: `bash run_dvf_oasis_secondary.sh`
       → `csvs/pilot/gateB_qc.csv` e `csvs/pilot/secondary_jac_summary.csv`.
6. [ ] Levar ao orientador: veredito do Gate B + QC + secundária.

Rodar sempre no tmux (`tmux a -t 0`), um bloco por vez. `Ctrl-b d` desanexa sem matar os
processos; `exit` dentro do tmux mata.

## Feito

- [x] `select`: 12 estratos N=20, 240 imagens do pré-processamento próprio
      (`/mnt/databases/mri/oasis/oasis-3/preproc/3-biasfield`). AD 50-59 (1/1) e AD 90-99 (4/4)
      inviáveis → faixas fora de 60-89 usam a década vizinha (`modules/oasis_refs.py`).
- [x] `prep` em 20 shards (hist-match → rígido MNI), logs em `logs/oasis_prep/`: 240 imagens,
      sem Traceback, NCC mínimo 0,201 (> 0,15). Piores: OAS31094_MR_d0103 (0,201),
      OAS30919_MR_d2502 (0,211), OAS30176_MR_d0000 (0,226) — abaixo do normal (0,26–0,43);
      conferir visualmente OAS31094 se algum template sair estranho.
- [x] Código simplificado para núcleo + d2 (01/10; autoverificações ok): núcleo/casca/TBM
      revertidos em `oasis_refs.py`, `ablation_prep.py`, `4_run_post_extract.py`;
      `3.2_feat_dvf.py` só gera `hippocampus` e `hippocampus_d2`; `pilot_oasis_gate.py` sem
      escolha de largura; `run_dvf_oasis_pilot.sh` com `ROIS` = d2 + núcleo.
- [x] `build` dos 12 templates (`ants.build_template`, SyN MI 100x70x50x20, 4 iterações) em
      `images/groupwise/references/oasis_mni/` + QC (`qc.png`, `qc_hippo_zoom.png`).
- [x] Gate A (02/10 12:44 → 04/10 01:00, ~36 h): 80 CN/AD × 2 âncoras, ~300 min por registro
      (1 thread ITK). **PASSA** (`csvs/pilot/gateA_summary.csv`):

      | âncora | ROI | \|rho\| vol/ICV | AUC CN×AD | ADNI antigo (núcleo) \|rho\| / AUC |
      |---|---|---|---|---|
      | cn | d2 | 0,598 | 0,804 | 0,139 / 0,611 |
      | cn | núcleo | 0,717 | 0,883 | |
      | ad | d2 | 0,527 | 0,762 | 0,092 / 0,592 |
      | ad | núcleo | 0,661 | 0,853 | |

      Núcleo > d2 nas duas âncoras: d2 segue principal (pré-especificado); d2 × núcleo é
      reportado no Gate B. \|rho\| alto com volume = risco de redundância com `vol` no Gate B.
- [x] Teste de determinismo descartado: o script ficava em `/tmp/oasis_qc` e foi apagado; não
      muda nenhuma decisão (gates comparam modalidades sobre os mesmos warps; registro já usa
      `--random-seed 42` + 1 thread ITK). Se o artigo afirmar reprodutibilidade exata, recriar
      no repositório (nunca em `/tmp`): registrar I14392 de novo (~5 h) e comparar com o warp
      do Gate A via `np.allclose(atol=1e-4)`.

## 1. Gate B (em andamento desde 04/10 13:26 hora do servidor; ~7–8 dias)

Log: `logs/dvf_oasis_gate-b_20261004_132641.log`; registro em
`logs/dvf_oasis_reg_*_20261004_132641.log`. **Não editar `run_dvf_oasis_pilot.sh` enquanto roda**
(bash lê o script aos poucos).

442 baselines (`csvs/pilot/oasis_gateB_ids.csv`: 149 AD, 100 CN, 120 pMCI, 73 sMCI); as 80 do
Gate A são puladas → 362 novas × 2 âncoras = 724 registros / 24 processos ≈ 30 × 5 h ≈ 6–7 dias,
depois `3.2` (~30 s/imagem), `4_ --oasis-only` e 12 ablações em sequência. Ablação t1_only
cn_ad / smci_pmci em `hippocampus_d2` e `hippocampus` × `disp_oasis`/`_ad`/`_cnad`
(2 coortes × 2 ROIs × 3 = 12 ablações). As ablações t1_only são o resultado final de 1 visita.

Acompanhar:

```bash
grep -h "\[OK\]" logs/dvf_oasis_reg_*_20261004_132641.log | wc -l   # meta: 724
grep -h "\[ERROR\]\|Traceback" logs/dvf_oasis_reg_*_20261004_132641.log
```

### Se cair a energia

Retomada é por item terminado: registro pula warp completo (perde a imagem em curso, ≤ 5 h);
`3.2` pula imagens em `done_keys.txt`; `4_` é barato; **as 12 ablações do gate-b recomeçam do
zero** (não há checagem de "já existe" no piloto).

1. `tmux new -s 0`, `cd /mnt/study-data/pgirardi/graphs && source .venv/bin/activate`.
2. `pgrep -fc 3.1_feat_gen_dvf.py` tem que dar 0.
3. Warp truncado passa pelo teste de tamanho (> 1 KB) do `3.1`; conferir os gravados perto da
   queda e apagar os 3 arquivos (`_1Warp`, `_1InverseWarp`, `_0GenericAffine.mat`) do que falhar:

   ```bash
   for f in $(find images/displacement_field_oasis_cn images/displacement_field_oasis_ad -name "*Warp.nii.gz" -mmin -120); do gzip -t "$f" 2>/dev/null || echo "CORROMPIDO: $f"; done
   ```

4. Se caiu na extração (`[OK] IMG=... rows=4` no log): o `3.2` grava as linhas antes do
   `done_keys`, então pode duplicar ou truncar linha. Conferir (tem que dar 0 e ler sem erro):

   ```bash
   .venv/bin/python -c "
   import pandas as pd
   for a in ('cn','ad'):
       d = pd.read_csv(f'csvs/cohorts/all_population/features_displacement_oasis_{a}.csv')
       print(a, 'duplicadas:', d.duplicated(['ID_IMG','roi','side']).sum())"
   ```

5. Se caiu nas ablações (`=== MONO ...`): antes de relançar, adicionar ao loop do gate-b o
   mesmo `SKIP` por `ablation_results_all.csv` do `run_dvf_oasis_full.sh`.
6. `rm -rf images/displacement_field_oasis_*/_tmp_ants/*` (só com o passo 2 dando 0).
7. Relançar o mesmo comando: `SHARDS=12 bash run_dvf_oasis_pilot.sh gate-b`.

### Critério (impresso no fim do log)

- PASSA: CN×AD de `disp_oasis` d2 (48m_6m) ≥ 0,75 **e** algum par d2 × `disp` ADNI com IC95
  todo > 0.
- INCONCLUSIVO: algum Δ ≥ 0,03 × `disp` ADNI, IC cruza 0.
- FALHA: resto (reportar como resultado negativo).
- Também imprime Δ e IC95 de d2 × núcleo OASIS e de cada ROI × `vol`.
- Comparações múltiplas: 3 âncoras × 2 coortes; reportar todas, não só a melhor.

```bash
SHARDS=12 bash run_dvf_oasis_pilot.sh gate-b
```

Resultado: `csvs/pilot/gateB_summary.csv`.

## 2. Rodada completa (só se Gate B passar) — `run_dvf_oasis_full.sh`

Nada é refeito: registro e `3.2` pulam as 442 baselines (warps e `done_keys`), só acrescentam
t1/t2 ao mesmo CSV de atributos; `t1_r10` / `t1_ols` são calculados na ablação a partir dele.

| Abordagem | Representação | De onde sai |
|---|---|---|
| 1 visita | `t1_only` | Gate B (`gateB_summary.csv`) |
| 2 visitas — S0,R10 (descarta i2; T1 + taxa i0→i1, tempo real) | `t1_r10` | `full` |
| 3 visitas — D (T1 + inclinação OLS em (tₖ, Sₖ)) | `t1_ols` | `full` |

1326 imagens (442 pacientes × t0/t1/t2); as 442 baselines já vêm do Gate B → 884 novas × 2
âncoras ≈ 1768 registros × ~5 h / 24 núcleos ≈ **~15 dias**. t1 é registrado antes de t2.
`refs` nunca junto com `register` (24 núcleos ocupados).

```bash
SHARDS=12 bash run_dvf_oasis_full.sh register   # registro + 3.2 + 4_ --oasis-only
JOBS=4 bash run_dvf_oasis_full.sh refs          # disp_ad/disp_cnad ADNI em t1_r10/t1_ols (faltam)
JOBS=4 bash run_dvf_oasis_full.sh ablate        # 24 ablações OASIS + compare_t1_r10/t1_ols.csv
```

`refs` não depende da OASIS: pode rodar a qualquer momento com CPU livre. Saídas em
`csvs/pilot/compare_t1_r10.csv` e `compare_t1_ols.csv` (mesmas comparações do Gate B).

- Opcional: fusão `vol` + `disp_oasis` (DVF como complemento do volume).

## Pendências menores

- Atualizar o arquivo do plano (ainda descreve faixas de 5 anos e templates antigos do Leandro).
- Opcional: `aff2axcodes` da saída 3-biasfield = T1 bruta (`('R','A','S')`).
- Limpar `images/groupwise/references/`: só `oasis_mni/` é usada (`OASIS_MNI_DIR`).
  `oasis/` = cópia crua dos templates do Leandro (NAC, faixas de 5 anos; original em
  `/mnt/study-data/lprado/groupwise_oasis`, parcialmente sem permissão — pode ser a única cópia
  legível). `oasis_mni_leandro_affine/` = tentativa descartada (Leandro + afim p/ MNI).
  **Apagar é irreversível.** Antes de decidir, rodar `du -sh images/groupwise/references/oasis*`
  para ver quanto espaço cada uma ocupa. Se o espaço não estiver apertado, guardar as duas até a
  decisão do Gate B e só então apagar.

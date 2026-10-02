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

## 1. `build` dos 12 templates (tempo a medir; estimativa 6–15 h)

Protocolo novo (30/09, antes de qualquer resultado): `ants.build_template` com atualização de
forma + Sharpen, SyN MI `reg_iterations=(100,70,50,20)` (com iterações na resolução máxima),
4 iterações (`TEMPLATE_REG` / `TEMPLATE_ITERS` no `2.3`). O primeiro build (SyN padrão 40x20x0,
mesmo do 2.2 ADNI) foi cancelado: deixaria o hipocampo borrado.

Nada pesado em paralelo. 12 builds × `BUILD_THREADS=2` = 24 núcleos. **Rodar uma vez só**: antes,
`pgrep -fc "python 2.3_oasis_templates_mni.py build"` tem que dar 0 (no dia 30 o loop foi colado
duas vezes e ficaram 24 processos disputando 24 núcleos).

```bash
pgrep -fc "python 2.3_oasis_templates_mni.py build"   # tem que ser 0
mkdir -p logs/oasis_build && rm -f logs/oasis_build/*.log
for d in CN AD; do for s in F M; do for a in 60-69 70-79 80-89; do
  BUILD_THREADS=2 .venv/bin/python 2.3_oasis_templates_mni.py build $d $s $a > logs/oasis_build/${d}_${s}_${a}.log 2>&1 &
done; done; done; wait; date
ls images/groupwise/references/oasis_mni/*_template.nii.gz | wc -l   # 12
grep -l -E "Traceback|AssertionError" logs/oasis_build/*.log          # vazio
```

Estimar o tempo total quando a 1ª iteração acabar (total ≈ 4 × iter 1):

```bash
grep -h "\[build\]" logs/oasis_build/*.log      # ex.: "[build] CN F 70-79 iter 1/4 N=20 95.3 min"
pgrep -fc "python 2.3_oasis_templates_mni.py build"   # 12 = rodando, 0 = terminou
```

Se iter 1 passar de ~4 h (total > 16 h), avaliar reduzir para `(100,70,50,10)` ou 3 iterações.
Cancelar de verdade (Ctrl-C só interrompe o `wait`, os builds continuam):
`pkill -f "python 2.3_oasis_templates_mni.py build"` e depois
`rm -rf images/groupwise/references/_tmp_ants/oasis_*`.

## 2. QC dos templates

```bash
.venv/bin/python 2.3_oasis_templates_mni.py qc
```

Inspecionar `images/groupwise/references/oasis_mni/qc.png` (coronal y=-20 mm: hipocampo nítido,
AD com ventrículos maiores que CN) e `qc.csv` (ncc_mni e sharpness parecidos entre estratos).

## 3. Gate A (~21 h)

40 CN + 40 AD; tempo por registro; por âncora, |rho| com `vol/ICV` e AUC univariada CN×AD de
`jac_det_mean` em d2 e no núcleo. Passa se d2 tiver |rho| > o do `jac_det_mean` ADNI antigo
(núcleo) e AUC CN×AD ≥ 0,75 em alguma âncora. Primeiro sinal real: se não separar CN×AD,
dificilmente separa sMCI×pMCI. As 80 imagens fazem parte do Gate B (warps reaproveitados).

```bash
SHARDS=12 bash run_dvf_oasis_pilot.sh gate-a
```

Resultado: `csvs/pilot/gateA_summary.csv`.

## 4. Teste de determinismo (~3 h, depois do Gate A)

Copiar antes `/tmp/oasis_qc/repro_check.py` para o projeto (o `/tmp` pode ser limpo).
Rodar os dois comandos separados — nunca no mesmo bloco com `rm`.

```bash
.venv/bin/python /tmp/oasis_qc/repro_check.py I14392 run
```

```bash
.venv/bin/python /tmp/oasis_qc/repro_check.py I14392 compare     # np.allclose(atol=1e-4)
```

## 5. Gate B (~4–5 dias, só se Gate A passar)

Baselines 48m_6m + soft_False; ablação t1_only cn_ad / smci_pmci em `hippocampus_d2` e
`hippocampus` × `disp_oasis`/`_ad`/`_cnad` (2 coortes × 2 ROIs × 3 = 12 ablações).

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

## 6. Rodada completa (só se Gate B passar)

- Todas as imagens; reps t1_only / t1_d21 / t1_d21_d32; tabela final com d2 (principal) e
  núcleo (controle).
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

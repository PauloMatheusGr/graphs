# TODO — DVF hipocampo com templates OASIS-3 (sMCI vs pMCI)

Objetivo: melhorar `disp` sMCI×pMCI só com hipocampo usando templates OASIS-3 estratificados
(DIAG CN|AD × SEX × década 60-69/70-79/80-89, N=20 cada) em MNI152, SyNRA CC r4 (100x70x50x20),
ROIs hippocampus/d2/d4/d8/shell4 (d4 = principal, fixada antes dos resultados), atributos
jac_det/logjac/mag/strain_fro. Rodada completa só se Gate A e Gate B passarem.

Rodar sempre no tmux (`tmux attach -t oasis`), um bloco por vez.

## Feito

- [x] `select`: 12 estratos N=20, 240 imagens do pré-processamento próprio
      (`/mnt/databases/mri/oasis/oasis-3/preproc/3-biasfield`). AD 50-59 (1/1) e AD 90-99 (4/4)
      inviáveis → faixas fora de 60-89 usam a década vizinha (`modules/oasis_refs.py`).
- [~] `prep` em 20 shards (hist-match → rígido MNI) — rodando, logs em `logs/oasis_prep/`.

## 1. Conferir `prep` (~1 h)

```bash
watch -n 60 'grep -h "\[OK\]" logs/oasis_prep/*.log | wc -l; grep -l Traceback logs/oasis_prep/*.log'
```

Após o `wait` retornar:

```bash
ls images/groupwise/resample_1.0mm_oasis/*.nii.gz | wc -l        # 240
grep -l Traceback logs/oasis_prep/*.log                           # vazio
.venv/bin/python -c "import pandas as pd,glob; q=pd.concat(map(pd.read_csv,glob.glob('images/groupwise/resample_1.0mm_oasis/prep_qc_*.csv'))); print(len(q), q.ncc_mni.min().round(3)); print(q.nsmallest(5,'ncc_mni'))"
```

Critério: 240 imagens, NCC mínimo > 0,15 (normal 0,26–0,43).

## 2. `build` dos 12 templates (~4–6 h)

Nada pesado em paralelo. 12 builds × `BUILD_THREADS=2` = 24 núcleos.

```bash
mkdir -p logs/oasis_build
for d in CN AD; do for s in F M; do for a in 60-69 70-79 80-89; do
  BUILD_THREADS=2 .venv/bin/python 2.3_oasis_templates_mni.py build $d $s $a > logs/oasis_build/${d}_${s}_${a}.log 2>&1 &
done; done; done; wait
ls images/groupwise/references/oasis_mni/*_template.nii.gz | wc -l   # 12
grep -l -E "Traceback|AssertionError" logs/oasis_build/*.log          # vazio
```

## 3. QC dos templates

```bash
.venv/bin/python 2.3_oasis_templates_mni.py qc
```

Inspecionar `images/groupwise/references/oasis_mni/qc.png` (coronal y=-20 mm: hipocampo nítido,
AD com ventrículos maiores que CN) e `qc.csv` (ncc_mni e sharpness parecidos entre estratos).

## 4. Gate A (~21 h)

40 CN + 40 AD; tempo por registro, |rho| entre atributos, AUC univariada CN×AD por ROI.

```bash
SHARDS=12 bash run_dvf_oasis_pilot.sh gate-a
```

Resultado: `csvs/pilot/gateA_summary.csv`.

## 5. Teste de determinismo (~3 h, depois do Gate A)

Copiar antes `/tmp/oasis_qc/repro_check.py` para o projeto (o `/tmp` pode ser limpo).
Rodar os dois comandos separados — nunca no mesmo bloco com `rm`.

```bash
.venv/bin/python /tmp/oasis_qc/repro_check.py I14392 run
```

```bash
.venv/bin/python /tmp/oasis_qc/repro_check.py I14392 compare     # np.allclose(atol=1e-4)
```

## 6. Gate B (~4–5 dias, só se Gate A passar)

Baselines 48m_6m + soft_False; ablação t1_only cn_ad / smci_pmci; `disp_oasis`, `disp_oasis_ad`,
`disp_oasis_cnad` pareados (via `test_id_pts`) contra `disp` ADNI; decisão pelo IC95 do
`bootstrap_auc_diff_test`.

```bash
SHARDS=12 bash run_dvf_oasis_pilot.sh gate-b
```

Resultado: `csvs/pilot/gateB_summary.csv`.

## 7. Rodada completa (só se Gate B passar)

- Todas as imagens; reps t1_only / t1_d21 / t1_d21_d32; tabela final por ROI (d4 principal).

## Pendências menores

- Atualizar o arquivo do plano (ainda descreve faixas de 5 anos e templates antigos do Leandro).
- Opcional: `aff2axcodes` da saída 3-biasfield = T1 bruta (`('R','A','S')`).

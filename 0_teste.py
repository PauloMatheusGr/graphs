#!/usr/bin/env python3
"""Registro MNI -> ADNI do 0_teste.ipynb (células 1 e 3) para rodar no tmux.

Depois de terminar: no notebook, rodar a célula 1 (setup) e pular a 3 (registro); a célula 5 (métricas)
lê o que este script salvou em test_dvf/.
"""
import os
import shutil
import traceback

import ants
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = "/mnt/databases/mri/adni/preproc/4-mni-hist-matching"
REGIONS_DIR = "/mnt/databases/mri/adni/preproc/5-parcellation/regions"
MNI_PATH = "/mnt/study-data/pgirardi/datasets_img/atlases/templates/mni152_2009c_template.nii.gz"
SUFFIX = "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
OUT_DIR = "test_dvf"
SEED = 42

DIAGS = ["CN", "MCI", "AD"]
N_POR_DIAG = 30

SYN_ROBUSTO = {"syn_metric": "CC", "syn_sampling": 4, "reg_iterations": (100, 70, 50, 20)}
VARIANTES = {
    "sem_mask": (False, None),
    "com_mask": (True, None),
    "syn_robusto": (False, SYN_ROBUSTO),
}
RODAR = ["sem_mask", "syn_robusto"]

# ponytail: só fixa seed; determinismo total exige 1 thread (set_ants_deterministic(True)), SyN fica muito lento
ants.config.set_ants_deterministic(False, SEED)
moving = ants.image_read(MNI_PATH)
moving_mask = ants.get_mask(moving)


def t1_path(img_id):
    return os.path.join(INPUT_DIR, img_id + SUFFIX)


def regions_path(img_id):
    return os.path.join(REGIONS_DIR, img_id + "_regions.nii.gz")


def pasta(row):
    return os.path.join(OUT_DIR, f"{row.DIAG}_{row.ID_IMG}")


def registrar(fixed, usar_mask, syn_kw):
    """Rigid -> Affine -> SyN (ordem do prealign_to_target do 2.2). Retorna o resultado do SyN."""
    mk = {"mask": ants.get_mask(fixed)} if usar_mask else {}
    rig = ants.registration(fixed, moving, "Rigid", moving_mask=moving_mask, **mk)
    w = rig["warpedmovout"]
    aff = ants.registration(fixed, w, "Affine", moving_mask=ants.get_mask(w), **mk)
    w = aff["warpedmovout"]
    return ants.registration(fixed, w, "SyNOnly", moving_mask=ants.get_mask(w), **mk, **(syn_kw or {}))


df = pd.read_csv("csvs/adnimerged.csv")
df = df[df["DIAG"].isin(DIAGS)]
df = df[df["ID_IMG"].map(lambda i: os.path.isfile(t1_path(i)) and os.path.isfile(regions_path(i)))]
# 1 imagem sorteada por paciente; paciente não se repete entre diagnósticos (ex.: CN que converteu para MCI)
df = df.sample(frac=1, random_state=SEED).drop_duplicates("ID_PT")
sel = pd.concat([df[df["DIAG"] == d].head(N_POR_DIAG) for d in DIAGS])
os.makedirs(OUT_DIR, exist_ok=True)
sel.to_csv(os.path.join(OUT_DIR, "selected_images.csv"), index=False)
print(sel["DIAG"].value_counts().to_string(), flush=True)

falhas = []
for k, row in enumerate(sel.itertuples(), 1):
    d = pasta(row)
    os.makedirs(d, exist_ok=True)
    fixed = ants.image_read(t1_path(row.ID_IMG))
    if not os.path.isfile(os.path.join(d, "fixed.nii.gz")):
        ants.image_write(fixed, os.path.join(d, "fixed.nii.gz"))
        shutil.copy2(regions_path(row.ID_IMG), os.path.join(d, "regions.nii.gz"))
    for v in RODAR:
        out = os.path.join(d, f"mni_warped_{v}.nii.gz")
        if os.path.isfile(out):
            print(f"[{k}/{len(sel)}] [SKIP] {d} {v}", flush=True)
            continue
        try:
            syn = registrar(fixed, *VARIANTES[v])
            shutil.copy2(syn["fwdtransforms"][0], os.path.join(d, f"warp_{v}_1Warp.nii.gz"))
            ants.image_write(syn["warpedmovout"], out)  # por último: marca de concluído
            print(f"[{k}/{len(sel)}] [OK] {d} {v}", flush=True)
        except Exception:
            falhas.append((row.ID_IMG, v))
            print(f"[{k}/{len(sel)}] [ERROR] {d} {v}\n{traceback.format_exc()}", flush=True)

print(f"[DONE] {len(sel)} imagens, {len(falhas)} falhas: {falhas}", flush=True)
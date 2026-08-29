#!/usr/bin/env python3
"""Groupwise SyN template por estrato CN (sexo × década).

Lê IDs congelados em selected_*.csv. Imagens já rigid-MNI em
images/groupwise/resample_1.0mm (2_resample_groupwise.py). Sem HM,
sem pad, sem reorient, sem rescale [0,1].

Uso:
    python 1_groupwise_ants.py
    python 1_groupwise_ants.py 60 69 F
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import ants

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # noqa: E731

ROOT = Path(__file__).resolve().parent
SELECTED_DIR = ROOT.parent / "groupwise" / "adni" / "CN"
IMAGES_DIR = ROOT / "images" / "groupwise" / "resample_1.0mm"
OUT_DIR = ROOT / "images" / "groupwise" / "references"
SUFFIX = "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
TYPE_OF_TRANSFORM = "SyN"
N_ITER_TEMPLATE = 5
KEEP_TMP_ANTS = False


def usage_and_exit(code: int = 1) -> None:
    print(
        "Uso:\n"
        "  python 1_groupwise_ants.py\n"
        "  python 1_groupwise_ants.py <min_age> <max_age> <sex>\n\n"
        "Exemplo:\n"
        "  python 1_groupwise_ants.py 60 69 F\n"
    )
    sys.exit(code)


def parse_args(argv: list[str]):
    if len(argv) == 1:
        return None
    if len(argv) != 4:
        usage_and_exit(1)
    try:
        age_min = int(float(argv[1]))
        age_max = int(float(argv[2]))
    except ValueError:
        print("Erro: idades numéricas.")
        usage_and_exit(1)
    sex = str(argv[3]).strip().upper()
    if sex not in {"M", "F"}:
        print("Erro: sex = M ou F.")
        usage_and_exit(1)
    if age_min > age_max:
        usage_and_exit(1)
    return age_min, age_max, sex


def selected_csvs(one=None) -> list[Path]:
    if one is not None:
        age_min, age_max, sex = one
        p = SELECTED_DIR / f"selected_DIAG-CN_SEX-{sex}_AGE-{age_min}-{age_max}_N-20.csv"
        if not p.is_file():
            raise FileNotFoundError(f"selected CSV ausente: {p}")
        return [p]
    csvs = sorted(SELECTED_DIR.glob("selected_DIAG-CN_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"nenhum selected_DIAG-CN_*.csv em {SELECTED_DIR}")
    if not any("AGE-50-59" in p.name for p in csvs):
        print("[WARN] selected CSV 50-59 ausente — DVF nessa faixa sem template novo", flush=True)
    return csvs


def stratum_tag(csv_path: Path) -> str:
    # selected_DIAG-CN_SEX-F_AGE-60-69_N-20.csv → DIAG-CN_SEX-F_AGE-60-69_N-20
    name = csv_path.stem
    if not name.startswith("selected_"):
        raise ValueError(csv_path.name)
    return name[len("selected_") :]


def read_id_img(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID_IMG" not in reader.fieldnames:
            raise ValueError(f"{path.name} sem coluna ID_IMG")
        return [row["ID_IMG"] for row in reader]


def resolve_paths(ids: list[str]) -> tuple[list[str], dict[str, Path], list[str]]:
    id_to_path: dict[str, Path] = {}
    missing: list[str] = []
    for img_id in ids:
        p = IMAGES_DIR / f"{img_id}{SUFFIX}"
        if p.is_file():
            id_to_path[img_id] = p
        else:
            missing.append(img_id)
    ok = [i for i in ids if i in id_to_path]
    return ok, id_to_path, missing


def resample_to_target(im: ants.ANTsImage, tgt: ants.ANTsImage) -> ants.ANTsImage:
    try:
        return ants.resample_image_to_target(im, tgt, interp_type="bspline")
    except Exception:
        return ants.resample_image_to_target(im, tgt, interp_type="linear")


def prealign_to_target(
    moving: ants.ANTsImage,
    target: ants.ANTsImage,
    moving_mask: ants.ANTsImage,
    target_mask: ants.ANTsImage,
) -> ants.ANTsImage:
    rigid = ants.registration(
        fixed=target,
        moving=moving,
        type_of_transform="Rigid",
        fixed_mask=target_mask,
        moving_mask=moving_mask,
        verbose=False,
    )
    warped_rigid = rigid["warpedmovout"]
    affine = ants.registration(
        fixed=target,
        moving=warped_rigid,
        type_of_transform="Affine",
        fixed_mask=target_mask,
        moving_mask=ants.get_mask(warped_rigid),
        verbose=False,
    )
    return affine["warpedmovout"]


def build_groupwise_template(image_paths: list[Path], n_iter: int, type_of_transform: str):
    t_global = time.time()
    imgs = [ants.image_read(str(p)) for p in tqdm(image_paths, desc="Lendo", unit="img")]
    if len(imgs) < 2:
        raise RuntimeError("groupwise precisa >=2 imagens")
    for im, p in zip(imgs, image_paths):
        if len(im.shape) != 3:
            raise RuntimeError(f"não-3D: {p.name} shape={im.shape}")

    # ponytail: grid já é MNI 1mm; first image = target. Sem pad/reorient.
    target = imgs[0]
    imgs = [resample_to_target(im, target) for im in imgs]
    masks = [ants.get_mask(im) for im in imgs]
    target_mask = ants.get_mask(target)

    n_imgs = len(imgs)
    pbar = tqdm(
        total=2 * n_imgs + n_iter * n_imgs,
        desc="Rigid+Affine+SyN",
        unit="reg",
        ncols=110,
    )
    pre = []
    for im, mk in zip(imgs, masks):
        pre.append(prealign_to_target(im, target, mk, target_mask))
        pbar.update(2)
    masks_pre = [ants.get_mask(im) for im in pre]

    template = resample_to_target(ants.average_images(pre), target)
    for it in range(1, n_iter + 1):
        print(f"\n[Groupwise] {it}/{n_iter} {type_of_transform}", flush=True)
        t_it = time.time()
        tmpl_mask = ants.get_mask(template)
        warped = []
        for img, mk in zip(pre, masks_pre):
            reg = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform=type_of_transform,
                fixed_mask=tmpl_mask,
                moving_mask=mk,
                verbose=False,
            )
            warped.append(reg["warpedmovout"])
            pbar.update(1)
        template = resample_to_target(ants.average_images(warped), target)
        print(f"[OK] iter {it} {(time.time() - t_it) / 60:.2f} min", flush=True)
    pbar.close()
    print(f"[OK] groupwise {(time.time() - t_global) / 60:.2f} min", flush=True)
    return template


def _set_tmp_env(tmp_dir: Path) -> None:
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


def run_one(csv_path: Path) -> None:
    tag = stratum_tag(csv_path)
    out_template = OUT_DIR / f"groupwise_{tag}_template.nii.gz"
    if out_template.is_file():
        print(f"[SKIP] template existe: {out_template}", flush=True)
        return

    ids = read_id_img(csv_path)
    subset_ok, id_to_path, missing = resolve_paths(ids)
    print(f"[{tag}] csv={len(ids)} achadas={len(subset_ok)} missing={len(missing)}", flush=True)
    if missing:
        print(f"[{tag}] IDs sem resample: {missing[:10]}", flush=True)
    if len(subset_ok) < 2:
        raise RuntimeError(f"{tag}: rode 2_resample_groupwise.py primeiro")

    image_paths = [id_to_path[i] for i in subset_ok]
    template = build_groupwise_template(image_paths, N_ITER_TEMPLATE, TYPE_OF_TRANSFORM)
    ants.image_write(template, str(out_template))
    print(f"[OK] {out_template}", flush=True)


def main(argv: list[str]) -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_base = OUT_DIR / "_tmp_ants"
    tmp_base.mkdir(parents=True, exist_ok=True)
    run_tmp = Path(tempfile.mkdtemp(prefix="ants_", dir=str(tmp_base)))
    _set_tmp_env(run_tmp)
    print(f"[TMP] {run_tmp}", flush=True)
    try:
        for csv_path in selected_csvs(parse_args(argv)):
            run_one(csv_path)
    finally:
        if not KEEP_TMP_ANTS:
            shutil.rmtree(run_tmp, ignore_errors=True)
    print(f"\n[OK] total {(time.time() - t_start) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv)

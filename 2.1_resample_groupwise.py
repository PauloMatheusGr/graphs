#!/usr/bin/env python3
"""Rigid-resample CN selected T1s into images/groupwise/resample_1.0mm.

Same fixed MNI and same Rigid as 2_resample.py. T1 only — no labels.
IDs come from frozen selected_*.csv (no re-sample).

Usage:
    python 2_resample_groupwise.py
    python 2_resample_groupwise.py --ids-only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTED_DIR = ROOT.parent / "groupwise" / "adni" / "CN"
INPUT_DIR = "/mnt/databases/mri/adni/preproc/4-mni-hist-matching"
OUTPUT_DIR = ROOT / "images" / "groupwise" / "resample_1.0mm"
REF_MNI = "/mnt/study-data/pgirardi/preproc/atlases/templates/mni152_2009c_template.nii.gz"
SUFFIX = "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"


def _load_resample2():
    spec = importlib.util.spec_from_file_location("resample2", ROOT / "2_resample.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_id_img(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID_IMG" not in reader.fieldnames:
            raise ValueError(f"{path.name} sem coluna ID_IMG")
        return [row["ID_IMG"] for row in reader]


def collect_ids(selected_dir: Path) -> list[str]:
    csvs = sorted(selected_dir.glob("selected_DIAG-CN_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"nenhum selected_DIAG-CN_*.csv em {selected_dir}")
    ids: list[str] = []
    seen: set[str] = set()
    for p in csvs:
        for img_id in _read_id_img(p):
            if img_id not in seen:
                seen.add(img_id)
                ids.append(img_id)
    decades = {p.name for p in csvs}
    if not any("AGE-50-59" in n for n in decades):
        print("[WARN] selected CSV 50-59 ausente — DVF nessa faixa sem template novo", flush=True)
    return ids


def run_batch(ids: list[str]) -> None:
    import ants

    rs = _load_resample2()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fixed = ants.image_read(REF_MNI)
    n = len(ids)
    t0 = time.perf_counter()
    for k, img_id in enumerate(ids, start=1):
        prog = f"[{k}/{n}]"
        out_path = OUTPUT_DIR / f"{img_id}{SUFFIX}"
        if out_path.is_file():
            print(f"{prog} [SKIP] {out_path.name}", flush=True)
            continue
        moving_path = rs.resolver_caminho_imagem(INPUT_DIR, img_id)
        if moving_path is None:
            print(f"{prog} [WARN] ID_IMG={img_id} não achado em {INPUT_DIR}", flush=True)
            continue
        print(f"{prog} [RUN] {img_id} ← {os.path.basename(moving_path)}", flush=True)
        moving = ants.image_read(moving_path)
        warped = rs.corregistro_rigid_mni(fixed, moving)["warpedmovout"]
        ants.image_write(warped, str(out_path))
        print(f"{prog} [OK] {out_path}", flush=True)
    print(f"[INFO] {n} IDs em {time.perf_counter() - t0:.1f} s → {OUTPUT_DIR}", flush=True)


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids-only", action="store_true", help="lista IDs e sai (check)")
    args = p.parse_args(argv)

    ids = collect_ids(SELECTED_DIR)
    assert ids, "lista ID_IMG vazia"
    print(f"[INFO] {len(ids)} ID_IMG únicos em {SELECTED_DIR}", flush=True)
    if args.ids_only:
        for img_id in ids:
            print(img_id)
        return
    run_batch(ids)


if __name__ == "__main__":
    main(sys.argv[1:])

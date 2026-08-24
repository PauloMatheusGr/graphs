#!/usr/bin/env python3
"""
Corrige raw 4D→3D e corre DKT (antspynet) só nos IDs em falta.
Entrada: /mnt/databases/mri/adni/raw_data/
Saída:   /mnt/study-data/pgirardi/insert_to_databases_regions/
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import ants
import antspynet

INPUT_DIR = Path("/mnt/databases/mri/adni/raw_data")
OUTPUT_DIR = Path("/mnt/study-data/pgirardi/insert_to_databases_regions")
CACHE_DIR = Path("/mnt/study-data/pgirardi/preproc/cache")

# Stems em falta (I+dígitos ou I+dígitos_*)
TARGET_STEMS = (
    "I41449",
    "I58423",
    "I58872",
    "I70013",
    "I150524_Eq_1",
    "I515913_Eq_1",
    "I150935_real",
    "I150936_real",
    "I154499_real",
    "I154500_real",
)

# Aceita I12345 ou I12345_qualquer_coisa no basename
STEM_RE = re.compile(r"^(I\d+(?:_.+)?)$")


def ensure_3d(img: ants.ANTsImage) -> ants.ANTsImage:
    """4D → 1º volume; rejeita ndim ≠ 3/4."""
    if img.dimension == 4:
        print(f"  [4D→3D] shape={img.shape} → slice axis=3 idx=0")
        return ants.slice_image(img, axis=3, idx=0)
    if img.dimension != 3:
        raise ValueError(f"dimensão {img.dimension} inválida (esperado 3D/4D)")
    return img


def resolve_raw(stem: str) -> Path | None:
    """Procura stem.nii.gz / stem.nii em INPUT_DIR."""
    for ext in (".nii.gz", ".nii"):
        p = INPUT_DIR / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def out_paths(stem: str) -> tuple[Path, Path]:
    """Nomes alinhados ao store ADNI: *_regions / *_lobes."""
    return (
        OUTPUT_DIR / f"{stem}_regions.nii.gz",
        OUTPUT_DIR / f"{stem}_lobes.nii.gz",
    )


def run_dkt(stem: str, raw_path: Path) -> None:
    regions_out, lobes_out = out_paths(stem)
    if regions_out.is_file() and lobes_out.is_file():
        print(f"[SKIP] {stem}: já existe em {OUTPUT_DIR}")
        return

    print(f"[RUN] {stem} ← {raw_path}")
    t1 = ants.image_read(str(raw_path))
    print(f"  raw dim={t1.dimension} shape={t1.shape}")
    t1 = ensure_3d(t1)
    print(f"  3D shape={t1.shape}")

    dkt = antspynet.desikan_killiany_tourville_labeling(
        t1,
        do_preprocessing=True,
        do_lobar_parcellation=True,
    )
    ants.image_write(dkt["segmentation_image"], str(regions_out))
    ants.image_write(dkt["lobar_parcellation"], str(lobes_out))
    print(f"  [OK] {regions_out.name}")
    print(f"  [OK] {lobes_out.name}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(antspynet, "set_antsxnet_cache_directory"):
        antspynet.set_antsxnet_cache_directory(str(CACHE_DIR))

    # valida stems
    stems = []
    for s in TARGET_STEMS:
        if not STEM_RE.match(s):
            print(f"[WARN] stem ignorado (padrão I+dígitos[_...]): {s}")
            continue
        stems.append(s)

    n_ok = n_skip = n_err = 0
    for stem in stems:
        raw = resolve_raw(stem)
        if raw is None:
            print(f"[ERR] {stem}: raw não encontrado em {INPUT_DIR}")
            n_err += 1
            continue
        try:
            regions_out, _ = out_paths(stem)
            existed = regions_out.is_file()
            run_dkt(stem, raw)
            if existed:
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:
            print(f"[ERR] {stem}: {e}")
            n_err += 1

    print(f"[DONE] ok={n_ok} skip={n_skip} err={n_err} out={OUTPUT_DIR}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
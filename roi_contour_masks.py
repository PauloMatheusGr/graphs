#!/usr/bin/env python3
"""
Gera máscaras 3D binárias da faixa de contorno (1 = contorno, 0 = resto) por ROI,
a partir de *_regions.nii.gz (24 ROIs do notebook).

Morfologia (padrão **inner**, contorno fino):
  contorno = M & ~erosão(M)  (superfície interna, ~1 voxel de espessura)

Alternativa **shell** (faixa mais grossa):
  faixa = dilatação(M) & ~erosão(M)

Requisitos: nibabel, numpy, scipy.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

ROI_TABLE: tuple[tuple[str, str, int], ...] = (
    ("hippocampus", "L", 17),
    ("hippocampus", "R", 53),
    ("amygdala", "L", 18),
    ("amygdala", "R", 54),
    ("thalamus_proper", "L", 10),
    ("thalamus_proper", "R", 49),
    ("accumbens_area", "L", 26),
    ("accumbens_area", "R", 58),
    ("inf_lateral_ventricle", "L", 5),
    ("inf_lateral_ventricle", "R", 44),
    ("entorhinal", "L", 1006),
    ("entorhinal", "R", 2006),
    ("parahippocampal", "L", 1016),
    ("parahippocampal", "R", 2016),
    ("posterior_cingulate", "L", 1023),
    ("posterior_cingulate", "R", 2023),
    ("isthmus_cingulate", "L", 1010),
    ("isthmus_cingulate", "R", 2010),
    ("rostral_anterior_cingulate", "L", 1026),
    ("rostral_anterior_cingulate", "R", 2026),
    ("medial_orbitofrontal", "L", 1014),
    ("medial_orbitofrontal", "R", 2014),
    ("insula", "L", 1035),
    ("insula", "R", 2035),
)

def contour_inner(mask: np.ndarray, iterations: int, struct: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=iterations)
    return mask & ~eroded

def contour_shell(mask: np.ndarray, iterations: int, struct: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=iterations)
    dilated = ndimage.binary_dilation(mask, structure=struct, iterations=iterations)
    return dilated & ~eroded

def build_contour(mask: np.ndarray, mode: str, iterations: int) -> np.ndarray:
    struct = ndimage.generate_binary_structure(rank=3, connectivity=3)
    if mode == "inner":
        return contour_inner(mask, iterations=iterations, struct=struct)
    if mode == "shell":
        return contour_shell(mask, iterations=iterations, struct=struct)
    raise ValueError(mode)

def main() -> None:
    ap = argparse.ArgumentParser(description="Contornos 3D por ROI (regions.nii.gz)")
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument(
        "--mode",
        choices=("shell", "inner"),
        default="inner",
        help=(
            "inner (padrão): contorno fino — M & ~erosão(M). "
            "shell: faixa mais grossa — dilatação(M) & ~erosão(M)."
        ),
    )
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--combined-labels", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stderr, force=True)

    img = nib.load(str(args.regions))
    data = np.asanyarray(img.dataobj)
    affine = img.affine

    if args.prefix:
        prefix = args.prefix.strip()
    else:
        stem = args.regions.name.replace(".nii.gz", "").replace(".nii", "")
        prefix = stem.replace("_regions", "").strip("_") or stem

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined = np.zeros(data.shape, dtype=np.uint16) if args.combined_labels else None

    n_out = 0
    for roi_name, side, label in ROI_TABLE:
        mask = data == label
        if not np.any(mask):
            logger.warning("Label %s (%s %s) sem voxels", label, roi_name, side)
            continue
        contour = build_contour(mask, args.mode, args.iterations)
        if not np.any(contour):
            logger.warning("Contorno vazio label %s (%s %s)", label, roi_name, side)
            continue

        out_path = args.out_dir / f"{prefix}_contour_{roi_name}_{side}.nii.gz"
        nib.save(nib.Nifti1Image(contour.astype(np.uint8), affine, img.header), str(out_path))
        n_out += 1
        logger.info("Gravado %s (%s voxels)", out_path.name, int(np.sum(contour)))

        if combined is not None:
            overlap = (combined > 0) & contour
            if np.any(overlap):
                logger.warning("Sobreposição em %s voxels (mantém valor anterior)", int(np.sum(overlap)))
            m = contour & (combined == 0)
            combined[m] = np.uint16(label)

    if args.combined_labels is not None and combined is not None:
        args.combined_labels.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(combined, affine, img.header), str(args.combined_labels))
        logger.info("Combinado: %s", args.combined_labels)

    logger.info("Concluído | máscaras=%s | modo=%s | iterations=%s", n_out, args.mode, args.iterations)

if __name__ == "__main__":
    main()
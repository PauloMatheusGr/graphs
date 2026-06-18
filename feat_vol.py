#!/usr/bin/env python3
"""
Calcula atributos volumétricos por ROI (CSF/GM/WM) usando regions, seg e brain_mask.

Saída CSV wide: 1 linha por (ID_IMG, roi, side, label) + linha __global__ por imagem.
Grava incrementalmente após cada ID_IMG; se OUT_CSV já existir, pula imagens completas.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# =========================
# CONFIG (edite aqui)
# =========================

ab = "longitudinal_window_4_groups"
# ab = "longitudinal_4_groups"

IMAGES_CSV = "csvs/adnimerged_longitudinal_window_extremos.csv"
# IMAGES_CSV = "csvs/adnimerged_longitudinal.csv"
REGIONS_DIR = "./images/regions"
SEG_DIR = "./images/seg"
BRAIN_MASK_DIR = "./images/brain_mask"
OUT_CSV = f"./csvs/{ab}/features_volumetric.csv"

LOG_EVERY = 1

# =========================
# ROI table
# =========================

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

SEG_LABEL_CSF = 1
SEG_LABEL_GM = 2
SEG_LABEL_WM = 3

FEATURE_COLS = [
    "mask_mm3",
    "gm_mm3",
    "gm_norm",
    "wm_mm3",
    "wm_norm",
    "csf_mm3",
    "csf_norm",
    "tissues_mm3",
    "tissues_norm",
]

ROWS_PER_SUBJECT_EXPECTED = 1 + len(ROI_TABLE)
FIELDNAMES = ["ID_IMG", "roi", "side", "label"] + FEATURE_COLS


def load_id_imgs(list_path: str) -> list[str]:
    with open(list_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID_IMG" not in reader.fieldnames:
            raise ValueError("CSV precisa ter coluna ID_IMG")
        seen: set[str] = set()
        ordered: list[str] = []
        for row in reader:
            raw = (row.get("ID_IMG") or "").strip()
            if raw and raw not in seen:
                seen.add(raw)
                ordered.append(raw)
    return ordered


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def validate_same_grid(
    id_img: str,
    ref_name: str,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
    other_name: str,
    other_shape: tuple[int, int, int],
    other_affine: np.ndarray,
) -> None:
    if ref_shape != other_shape:
        raise ValueError(
            f"[{id_img}] grid inconsistente: {other_name} shape={other_shape} "
            f"!= {ref_name} shape={ref_shape}"
        )
    if not np.allclose(ref_affine, other_affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"[{id_img}] affine de {other_name} difere de {ref_name}")


def format_value(v: float) -> str:
    if math.isnan(v) or math.isinf(v):
        return "NaN"
    return f"{v:.12e}"


def scan_csv_id_row_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = (row.get("ID_IMG") or "").strip()
            if iid:
                counts[iid] = counts.get(iid, 0) + 1
    return counts


def filter_csv_drop_ids(path: Path, drop_ids: set[str]) -> int:
    if not drop_ids or not path.is_file():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    with path.open(newline="", encoding="utf-8") as fin, tmp.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        for row in reader:
            iid = (row.get("ID_IMG") or "").strip()
            if iid in drop_ids:
                continue
            writer.writerow(row)
            kept += 1
    tmp.replace(path)
    return kept


def append_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    is_new = not path.is_file() or path.stat().st_size == 0
    with path.open("w" if is_new else "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="raise")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def process_image(id_img: str, regions_dir: Path, seg_dir: Path, brain_mask_dir: Path) -> list[dict[str, str]]:
    regions_path = regions_dir / f"{id_img}_regions.nii.gz"
    seg_path = seg_dir / f"{id_img}_seg.nii.gz"
    brain_mask_path = brain_mask_dir / f"{id_img}_brain_mask.nii.gz"

    for p in (regions_path, seg_path, brain_mask_path):
        if not p.is_file():
            raise FileNotFoundError(f"[{id_img}] arquivo ausente: {p}")

    img_regions = nib.load(str(regions_path))
    img_seg = nib.load(str(seg_path))
    img_brain_mask = nib.load(str(brain_mask_path))

    regions = np.asarray(img_regions.dataobj, dtype=np.int32)
    seg = np.asarray(img_seg.dataobj, dtype=np.int32)
    brain_mask = np.asarray(img_brain_mask.dataobj, dtype=np.float32)

    if regions.ndim != 3 or seg.ndim != 3 or brain_mask.ndim != 3:
        raise ValueError(f"[{id_img}] esperado volume 3D")

    validate_same_grid(
        id_img, "regions", regions.shape, img_regions.affine,
        "seg", seg.shape, img_seg.affine,
    )
    validate_same_grid(
        id_img, "regions", regions.shape, img_regions.affine,
        "brain_mask", brain_mask.shape, img_brain_mask.affine,
    )

    zooms = img_regions.header.get_zooms()[:3]
    voxel_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    brain_mask_bin = brain_mask > 0.0
    mask_mm3 = int(np.count_nonzero(brain_mask_bin)) * voxel_mm3

    v_csf_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_CSF))) * voxel_mm3
    v_gm_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_GM))) * voxel_mm3
    v_wm_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_WM))) * voxel_mm3

    batch: list[dict[str, str]] = []
    global_row: dict[str, str] = {
        "ID_IMG": id_img,
        "roi": "__global__",
        "side": "NA",
        "label": "NA",
        **{c: "" for c in FEATURE_COLS},
    }
    global_row["mask_mm3"] = format_value(mask_mm3)
    global_row["gm_mm3"] = format_value(v_gm_global)
    global_row["gm_norm"] = format_value(v_gm_global)
    global_row["wm_mm3"] = format_value(v_wm_global)
    global_row["wm_norm"] = format_value(v_wm_global)
    global_row["csf_mm3"] = format_value(v_csf_global)
    global_row["csf_norm"] = format_value(v_csf_global)
    global_row["tissues_mm3"] = format_value(mask_mm3)
    global_row["tissues_norm"] = format_value(mask_mm3)
    batch.append(global_row)

    for roi_name, side, label in ROI_TABLE:
        roi_mask = (regions == label) & brain_mask_bin
        roi_mm3 = int(np.count_nonzero(roi_mask)) * voxel_mm3

        row: dict[str, str] = {
            "ID_IMG": id_img,
            "roi": roi_name,
            "side": side,
            "label": str(label),
        }
        if roi_mm3 == 0:
            row.update({c: format_value(float("nan")) for c in FEATURE_COLS})
        else:
            v_csf = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_CSF))) * voxel_mm3
            v_gm = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_GM))) * voxel_mm3
            v_wm = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_WM))) * voxel_mm3
            v_roi_tissue = v_csf + v_gm + v_wm
            row["mask_mm3"] = format_value(roi_mm3)
            row["gm_mm3"] = format_value(v_gm)
            row["gm_norm"] = format_value(safe_div(v_gm, roi_mm3))
            row["wm_mm3"] = format_value(v_wm)
            row["wm_norm"] = format_value(safe_div(v_wm, roi_mm3))
            row["csf_mm3"] = format_value(v_csf)
            row["csf_norm"] = format_value(safe_div(v_csf, roi_mm3))
            row["tissues_mm3"] = format_value(v_roi_tissue)
            row["tissues_norm"] = format_value(safe_div(v_roi_tissue, roi_mm3))
        batch.append(row)

    return batch


def main() -> None:
    id_imgs = load_id_imgs(IMAGES_CSV)
    if not id_imgs:
        raise ValueError(f"Nenhum ID_IMG em {IMAGES_CSV}")

    regions_dir = Path(REGIONS_DIR).resolve()
    seg_dir = Path(SEG_DIR).resolve()
    brain_mask_dir = Path(BRAIN_MASK_DIR).resolve()
    out_csv = Path(OUT_CSV).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    skip_ids: set[str] = set()
    if out_csv.is_file() and out_csv.stat().st_size > 0:
        with out_csv.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None) or []
        if "value" in header or "feature_name" in header:
            raise ValueError(f"CSV em formato long/tidy: {out_csv}. Apague o arquivo ou use outro OUT_CSV.")

        counts = scan_csv_id_row_counts(out_csv)
        skip_ids = {iid for iid, c in counts.items() if c == ROWS_PER_SUBJECT_EXPECTED}
        partial = {iid for iid, c in counts.items() if 0 < c < ROWS_PER_SUBJECT_EXPECTED}
        if partial:
            print(f"Removendo parciais: {sorted(partial)}", flush=True)
            filter_csv_drop_ids(out_csv, partial)
        if skip_ids:
            print(
                f"Já no CSV: {len(skip_ids)} | a processar: {sum(1 for i in id_imgs if i not in skip_ids)}",
                flush=True,
            )

    t0 = time.perf_counter()
    total = len(id_imgs)
    processed = skipped = 0

    for idx, id_img in enumerate(id_imgs, start=1):
        if id_img in skip_ids:
            skipped += 1
            continue

        t_img = time.perf_counter()
        batch = process_image(id_img, regions_dir, seg_dir, brain_mask_dir)
        append_csv_rows(out_csv, batch)
        processed += 1

        if LOG_EVERY > 0 and (processed % LOG_EVERY) == 0:
            dt = time.perf_counter() - t_img
            print(
                f"[OK] {idx}/{total} IMG={id_img} rows={len(batch)} "
                f"dt={dt:.1f}s processed={processed} skipped={skipped}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"[DONE] processed={processed} skipped={skipped} elapsed={elapsed/60:.1f}min out={out_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()

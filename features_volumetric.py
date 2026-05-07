#!/usr/bin/env python3
"""
Calcula atributos volumetricos por ROI (24 labels) usando:
- regions: mapa de regioes (labels das ROIs)
- seg: segmentacao de tecidos (1=CSF, 2=GM, 3=WM)
- brain_mask: mascara binaria para normalizacao global

Saida:
- CSV wide:
  - 1 linha por (ID_IMG, roi, side, label)
  - colunas numéricas fixas (ex.: gm_mm3, gm_norm, wm_mm3, ...)
  - inclui 1 linha global por ID_IMG com roi="__global__" e feature mask_mm3
- Grava incrementalmente apos cada ID_IMG.
- Se o CSV de saida ja existir, le os ID_IMG e pula casos completos (retomada
  apos queda de energia). Linhas parciais sao removidas e reprocessadas.
- Use --overwrite para apagar o CSV e recomputar todos.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

ROI_TABLE: tuple[tuple[str, str, int], ...] = (
    # Subcortical (Inner Labels)
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
    
    # Cortical (Outer Labels)
    # ("entorhinal", "L", 1006),
    # ("entorhinal", "R", 2006),
    # ("parahippocampal", "L", 1016),
    # ("parahippocampal", "R", 2016),
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


def load_id_imgs(list_path: Path) -> list[str]:
    with list_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID_IMG" not in reader.fieldnames:
            raise SystemExit("CSV precisa ter cabecalho com coluna ID_IMG")
        seen: set[str] = set()
        ordered: list[str] = []
        for row in reader:
            raw = (row.get("ID_IMG") or "").strip()
            if not raw or raw in seen:
                continue
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
        raise SystemExit(
            f"[{id_img}] Grid inconsistente: {other_name} shape={other_shape} "
            f"!= {ref_name} shape={ref_shape}"
        )
    if not np.allclose(ref_affine, other_affine, rtol=0.0, atol=1e-5):
        raise SystemExit(
            f"[{id_img}] Grid inconsistente: affine de {other_name} difere de {ref_name}"
        )


def format_value(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "NaN"
    return f"{v:.12e}"


FEATURE_COLS = [
    "mask_mm3",  # apenas na linha __global__
    "gm_mm3",
    "gm_norm",
    "wm_mm3",
    "wm_norm",
    "csf_mm3",
    "csf_norm",
    "tissues_mm3",
    "tissues_norm",
]

# 1 linha global + 1 linha por ROI
ROWS_PER_SUBJECT_EXPECTED = 1 + len(ROI_TABLE)


def scan_csv_id_row_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = (row.get("ID_IMG") or "").strip()
            if iid:
                counts[iid] = counts.get(iid, 0) + 1
    return counts


def filter_csv_drop_ids(path: Path, fieldnames: list[str], drop_ids: set[str]) -> int:
    if not drop_ids or not path.is_file():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    with path.open(newline="", encoding="utf-8") as fin, tmp.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in reader:
            iid = (row.get("ID_IMG") or "").strip()
            if iid in drop_ids:
                continue
            writer.writerow(row)
            kept += 1
    tmp.replace(path)
    return kept


def append_csv_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    is_new = not path.is_file() or path.stat().st_size == 0
    mode = "w" if is_new else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calcula volumetria tecidual (CSF/GM/WM) por ROI e salva CSV wide (1 linha por ROI).",
    )
    ap.add_argument(
        "--list",
        type=Path,
        default=Path("/mnt/study-data/pgirardi/graphs/image_data_teste.txt"),
        help="CSV com coluna ID_IMG",
    )
    ap.add_argument(
        "--regions-dir",
        type=Path,
        default=Path("/mnt/study-data/pgirardi/graphs/images/regions"),
        help="Diretorio dos arquivos <ID_IMG>_regions.nii.gz",
    )
    ap.add_argument(
        "--seg-dir",
        type=Path,
        default=Path("/mnt/study-data/pgirardi/graphs/images/seg"),
        help="Diretorio dos arquivos <ID_IMG>_seg.nii.gz",
    )
    ap.add_argument(
        "--brain-mask-dir",
        type=Path,
        default=Path("/mnt/study-data/pgirardi/graphs/images/brain_mask"),
        help="Diretorio dos arquivos <ID_IMG>_brain_mask.nii.gz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/features_volumetric_teste.csv"
        ),
        help="Caminho do CSV de saida",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Logs detalhados por ROI.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Explicito: mesma logica de retomada quando o CSV existe (opcional; "
            "e o comportamento padrao se --out ja existir)."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Apaga o CSV de saida antes de rodar e reprocessa todos os ID_IMG.",
    )
    args = ap.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    id_imgs = load_id_imgs(args.list)
    if not id_imgs:
        raise SystemExit(f"Nenhum ID_IMG encontrado em {args.list}")

    args.regions_dir = args.regions_dir.resolve()
    args.seg_dir = args.seg_dir.resolve()
    args.brain_mask_dir = args.brain_mask_dir.resolve()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Wide: metadados + colunas de features
    fieldnames = ["ID_IMG", "roi", "side", "label"] + FEATURE_COLS

    skip_ids: set[str] = set()
    if args.overwrite and args.out.is_file():
        args.out.unlink()
        logger.info("Saida anterior removida (--overwrite).")

    use_existing_csv = args.out.is_file() and args.out.stat().st_size > 0
    if use_existing_csv:
        # Impede "retomada" em cima do CSV antigo (long/tidy).
        with args.out.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), None) or []
        if "value" in existing_header or "feature_name" in existing_header:
            raise SystemExit(
                f"CSV existente parece estar em formato long/tidy: {args.out}. "
                "Rode com --overwrite para regenerar em wide, ou escolha outro --out."
            )

        counts = scan_csv_id_row_counts(args.out)
        complete = {
            iid
            for iid, c in counts.items()
            if c == ROWS_PER_SUBJECT_EXPECTED
        }
        partial = {
            iid
            for iid, c in counts.items()
            if 0 < c < ROWS_PER_SUBJECT_EXPECTED
        }
        if partial:
            logger.warning(
                "Removendo linhas incompletas (reprocessamento): %s",
                sorted(partial),
            )
            filter_csv_drop_ids(args.out, fieldnames, partial)
        skip_ids = complete
        logger.info(
            "Retomada a partir de %s | ja completos=%s | a processar=%s",
            args.out,
            len(skip_ids),
            sum(1 for i in id_imgs if i not in skip_ids),
        )
    elif args.resume and not use_existing_csv:
        logger.info(
            "--resume sem CSV preexistente ou vazio: processando todos os ID_IMG."
        )

    t0 = time.perf_counter()
    total = len(id_imgs)
    total_written = 0
    logger.info(
        "Inicio | imagens=%s | regions=%s | seg=%s | brain_mask=%s | out=%s | "
        "overwrite=%s | retomada=%s",
        total,
        args.regions_dir,
        args.seg_dir,
        args.brain_mask_dir,
        args.out,
        args.overwrite,
        use_existing_csv,
    )

    for idx, id_img in enumerate(id_imgs, start=1):
        if id_img in skip_ids:
            logger.info("[%s/%s] PULADO (ja no CSV) %s", idx, total, id_img)
            continue

        regions_path = args.regions_dir / f"{id_img}_regions.nii.gz"
        seg_path = args.seg_dir / f"{id_img}_seg.nii.gz"
        brain_mask_path = args.brain_mask_dir / f"{id_img}_brain_mask.nii.gz"

        if not regions_path.is_file():
            raise SystemExit(f"[{id_img}] arquivo ausente: {regions_path}")
        if not seg_path.is_file():
            raise SystemExit(f"[{id_img}] arquivo ausente: {seg_path}")
        if not brain_mask_path.is_file():
            raise SystemExit(f"[{id_img}] arquivo ausente: {brain_mask_path}")

        logger.info("[%s/%s] INICIO %s", idx, total, id_img)
        batch: list[dict[str, str]] = []
        img_regions = nib.load(str(regions_path))
        img_seg = nib.load(str(seg_path))
        img_brain_mask = nib.load(str(brain_mask_path))

        regions = np.asarray(img_regions.dataobj, dtype=np.int32)
        seg = np.asarray(img_seg.dataobj, dtype=np.int32)
        brain_mask = np.asarray(img_brain_mask.dataobj, dtype=np.float32)

        if regions.ndim != 3 or seg.ndim != 3 or brain_mask.ndim != 3:
            raise SystemExit(
                f"[{id_img}] esperado volume 3D: "
                f"regions={regions.ndim}D, seg={seg.ndim}D, brain_mask={brain_mask.ndim}D"
            )

        validate_same_grid(
            id_img=id_img,
            ref_name="regions",
            ref_shape=regions.shape,
            ref_affine=img_regions.affine,
            other_name="seg",
            other_shape=seg.shape,
            other_affine=img_seg.affine,
        )
        validate_same_grid(
            id_img=id_img,
            ref_name="regions",
            ref_shape=regions.shape,
            ref_affine=img_regions.affine,
            other_name="brain_mask",
            other_shape=brain_mask.shape,
            other_affine=img_brain_mask.affine,
        )

        zooms = img_regions.header.get_zooms()[:3]
        voxel_mm3 = float(zooms[0] * zooms[1] * zooms[2])
        # Volume global de máscara (brain_mask) em mm³
        mask_voxels = int(np.count_nonzero(brain_mask > 0.0))
        mask_mm3 = mask_voxels * voxel_mm3

        # Volumes globais de tecidos (dentro do brain_mask)
        brain_mask_bin = brain_mask > 0.0
        csf_voxels_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_CSF)))
        gm_voxels_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_GM)))
        wm_voxels_global = int(np.count_nonzero(brain_mask_bin & (seg == SEG_LABEL_WM)))
        v_csf_global = csf_voxels_global * voxel_mm3
        v_gm_global = gm_voxels_global * voxel_mm3
        v_wm_global = wm_voxels_global * voxel_mm3

        global_row: dict[str, str] = {
            "ID_IMG": id_img,
            "roi": "__global__",
            "side": "NA",
            "label": "NA",
            **{c: "" for c in FEATURE_COLS},
        }
        global_row["mask_mm3"] = format_value(mask_mm3)
        # Para a linha global, preencher volumes globais e manter *_norm igual ao valor absoluto.
        global_row["gm_mm3"] = format_value(float(v_gm_global))
        global_row["gm_norm"] = format_value(float(v_gm_global))
        global_row["wm_mm3"] = format_value(float(v_wm_global))
        global_row["wm_norm"] = format_value(float(v_wm_global))
        global_row["csf_mm3"] = format_value(float(v_csf_global))
        global_row["csf_norm"] = format_value(float(v_csf_global))
        global_row["tissues_mm3"] = format_value(mask_mm3)
        global_row["tissues_norm"] = format_value(mask_mm3)
        batch.append(global_row)

        for roi_name, side, label in ROI_TABLE:
            roi_mask = regions == label
            roi_voxels = int(np.count_nonzero(roi_mask))
            roi_mm3 = roi_voxels * voxel_mm3

            csf_voxels = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_CSF)))
            gm_voxels = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_GM)))
            wm_voxels = int(np.count_nonzero(roi_mask & (seg == SEG_LABEL_WM)))

            v_csf = csf_voxels * voxel_mm3
            v_gm = gm_voxels * voxel_mm3
            v_wm = wm_voxels * voxel_mm3
            v_roi_tissue = v_csf + v_gm + v_wm

            # Normalização por volume da máscara desta linha:
            # - em ROIs: volume total da ROI (roi_mm3)
            # - em global: definido acima separadamente
            v_csf_div_mask = safe_div(v_csf, roi_mm3)
            v_gm_div_mask = safe_div(v_gm, roi_mm3)
            v_wm_div_mask = safe_div(v_wm, roi_mm3)
            v_roi_tissue_div_mask = safe_div(v_roi_tissue, roi_mm3)

            row: dict[str, str] = {
                "ID_IMG": id_img,
                "roi": roi_name,
                "side": side,
                "label": str(label),
                **{c: "" for c in FEATURE_COLS},
            }
            row["mask_mm3"] = format_value(float(roi_mm3))
            row["gm_mm3"] = format_value(float(v_gm))
            row["gm_norm"] = format_value(float(v_gm_div_mask))
            row["wm_mm3"] = format_value(float(v_wm))
            row["wm_norm"] = format_value(float(v_wm_div_mask))
            row["csf_mm3"] = format_value(float(v_csf))
            row["csf_norm"] = format_value(float(v_csf_div_mask))
            row["tissues_mm3"] = format_value(float(v_roi_tissue))
            row["tissues_norm"] = format_value(float(v_roi_tissue_div_mask))
            batch.append(row)

            logger.debug(
                "[%s] %s: csf=%.2f, gm=%.2f, wm=%.2f, total=%.2f",
                id_img,
                f"{roi_name}_{side}_{label}",
                v_csf,
                v_gm,
                v_wm,
                v_roi_tissue,
            )

        append_csv_rows(args.out, fieldnames, batch)
        total_written += len(batch)
        if len(batch) != ROWS_PER_SUBJECT_EXPECTED:
            logger.warning(
                "[%s] linhas=%s (esperado %s)",
                id_img,
                len(batch),
                ROWS_PER_SUBJECT_EXPECTED,
            )
        # logger.info("[%s/%s] OK %s | +%s linhas", idx, total, id_img, len(batch))

    dt = time.perf_counter() - t0
    logger.info(
        "Concluido em %.1fs | registros_escritos_nesta_exec=%s | csv=%s",
        dt,
        total_written,
        args.out,
    )


if __name__ == "__main__":
    main()

# How to run:
# Activate virtual environment: pyenvs graphs
# Run: python features_volumetric.py
# Retomar apos interrupcao: rode de novo o mesmo comando (o CSV e lido automaticamente).
# Recomputar tudo: python features_volumetric.py --overwrite
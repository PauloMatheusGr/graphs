#!/usr/bin/env python3
"""
Extrai atributos DVF longitudinais: follow-up → baseline do mesmo paciente.

Pré-requisito: 3_feat_gen_dvf.py → images/displacement_field_longitudinal/

Para cada visita follow-up (i2, i3):
  - domínio / ROI / brain_mask = baseline (i1)
  - DVF via inv_list = [affine, InverseWarp] do par moving→baseline
  - ID_IMG na linha = visita móvel (i2 ou i3); ref_tag = baseline_{i1}
  - sem linha para i1→i1

Saída (schema igual ao legado, path separado):
  csvs/cohorts/all_population/features_displacement_longitudinal.csv

Legado CN-template: 3_feat_dvf_old.py → features_displacement.csv

Downstream: 4_run_post_extract.py com DISP_FEATURES apontando para este CSV.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path

import ants
import pandas as pd

# =========================
# CONFIG (edite aqui)
# =========================

COHORT = "all_population"
COHORT_DIR = f"csvs/cohorts/{COHORT}"
IMAGES_CSV = f"{COHORT_DIR}/all_population.csv"

# Pastas SEPARADAS do legado (displacement_field / features_displacement.csv)
WARPS_DIR = "./images/displacement_field_longitudinal"
CLINIC_DIR = "./images/resampled_1.0mm"
REGIONS_DIR = "./images/regions"
BRAIN_MASK_DIR = "./images/brain_mask"

OUT_CSV = f"{COHORT_DIR}/features_displacement_longitudinal.csv"
RUN_DIR = os.path.join(WARPS_DIR, "features", COHORT)

RESUME = True
LOG_EVERY = 1

SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}

_OLD_PATH = Path(__file__).resolve().parent / "3_feat_dvf_old.py"


def _load_old():
    spec = importlib.util.spec_from_file_location("feat_dvf_old", _OLD_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"nao foi possivel carregar {_OLD_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_old = _load_old()
ROI_TABLE = _old.ROI_TABLE
POINT_CHUNK = _old.POINT_CHUNK


def clinic_path(img_id: str) -> str:
    return os.path.join(
        CLINIC_DIR, f"{img_id}_stripped_nlm_denoised_biascorrected.nii.gz"
    )


def inv_list_for(moving_id: str, baseline_id: str) -> list[str]:
    stem = f"{moving_id}_ref-{baseline_id}"
    return [
        os.path.join(WARPS_DIR, f"{stem}_0GenericAffine.mat"),
        os.path.join(WARPS_DIR, f"{stem}_1InverseWarp.nii.gz"),
    ]


def _sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MRI_DATE"] = pd.to_datetime(out["MRI_DATE"], errors="coerce")
    sort_cols = ["ID_PT"]
    if "slot" in out.columns:
        out["_slot_ord"] = out["slot"].astype(str).str.strip().map(SLOT_ORDER).fillna(99)
        sort_cols.append("_slot_ord")
    sort_cols.extend(["MRI_DATE", "ID_IMG"])
    return out.sort_values(sort_cols)


def baseline_id_by_pt(df: pd.DataFrame) -> dict[str, str]:
    sorted_df = _sort_chronological(df)
    out: dict[str, str] = {}
    for id_pt, g in sorted_df.groupby("ID_PT", sort=False):
        out[str(id_pt)] = str(g.iloc[0]["ID_IMG"]).strip()
    return out


def persist_run_metadata(
    *, run_meta_path: str, out_csv: str, done_keys_path: str, images_csv: str
) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "longitudinal_baseline_fixed",
        "inputs": {"images_csv": str(images_csv), "warps_dir": WARPS_DIR},
        "outputs": {"out_csv": str(out_csv), "done_keys_path": str(done_keys_path)},
        "env": {"DISPLACEMENT_POINT_CHUNK": POINT_CHUNK},
    }
    os.makedirs(os.path.dirname(run_meta_path), exist_ok=True)
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = pd.read_csv(IMAGES_CSV)
    required = {"ID_PT", "ID_IMG", "MRI_DATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IMAGES_CSV sem colunas: {sorted(missing)}")

    os.makedirs(RUN_DIR, exist_ok=True)
    done_keys_path = os.path.join(RUN_DIR, "done_keys.txt")
    run_meta_path = os.path.join(RUN_DIR, "run_meta.json")
    persist_run_metadata(
        run_meta_path=run_meta_path,
        out_csv=OUT_CSV,
        done_keys_path=done_keys_path,
        images_csv=IMAGES_CSV,
    )

    done = set()
    if RESUME:
        done |= _old.load_done_keys(done_keys_path)
        if os.path.isfile(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
            try:
                prev = pd.read_csv(OUT_CSV, usecols=["ID_IMG"])
                done |= set(prev["ID_IMG"].astype(str).str.strip().tolist())
            except Exception:
                pass
        print(f"[RESUME] follow-ups ja processados: {len(done)}", flush=True)

    base_by_pt = baseline_id_by_pt(df)
    df = _sort_chronological(df)

    t0 = time.time()
    processed = skipped = 0

    for id_pt, g in df.groupby("ID_PT", sort=False):
        baseline_id = base_by_pt.get(str(id_pt))
        if baseline_id is None:
            skipped += len(g)
            continue

        domain_path = clinic_path(baseline_id)
        regions_p = os.path.join(REGIONS_DIR, f"{baseline_id}_regions.nii.gz")
        bm_p = os.path.join(BRAIN_MASK_DIR, f"{baseline_id}_brain_mask.nii.gz")

        for r in g.itertuples(index=False):
            moving_id = str(r.ID_IMG).strip()
            if moving_id == baseline_id:
                continue

            if RESUME and moving_id in done:
                skipped += 1
                continue

            inv_list = inv_list_for(moving_id, baseline_id)
            if (
                not os.path.isfile(domain_path)
                or not os.path.isfile(regions_p)
                or not os.path.isfile(bm_p)
                or any(not os.path.isfile(p) for p in inv_list)
            ):
                skipped += 1
                continue

            t_img = time.time()
            domain_img = ants.image_read(domain_path)
            lj, m, dvg, ux, uy, uz, curl, strain_maps, refimg = (
                _old.compute_unitary_scalar_arrays(domain_img, inv_list)
            )
            spacing = tuple(map(float, refimg.spacing))
            strain_inf_fro = _old._infinitesimal_strain_fro_map(ux, uy, uz, spacing)
            labels = _old._load_and_resample_labelmap(regions_p, refimg)
            brain_mask = _old._load_and_resample_mask(bm_p, refimg)

            rows = []
            for roi, side, label in ROI_TABLE:
                lab = int(label)
                roi_mask = (labels == lab) & brain_mask
                row = _old._build_roi_feature_row(
                    id_pt=str(id_pt),
                    img_id=moving_id,
                    meta_row=r,
                    sex="NA",
                    age_range="NA",
                    roi=roi,
                    side=side,
                    label=lab,
                    roi_mask=roi_mask,
                    refimg=refimg,
                    lj=lj,
                    m=m,
                    dvg=dvg,
                    ux=ux,
                    uy=uy,
                    uz=uz,
                    curl=curl,
                    strain_maps=strain_maps,
                    strain_inf_fro=strain_inf_fro,
                )
                row["ref_tag"] = f"baseline_{baseline_id}"
                rows.append(row)

            if rows:
                _old.append_csv(pd.DataFrame(rows), OUT_CSV)

            if RESUME:
                _old.append_done_key(done_keys_path, moving_id)
                done.add(moving_id)

            processed += 1
            if LOG_EVERY > 0 and (processed % LOG_EVERY) == 0:
                dt = time.time() - t_img
                total_dt = time.time() - t0
                print(
                    f"[OK] moving={moving_id} fixed={baseline_id} PT={id_pt} "
                    f"rows={len(ROI_TABLE)} dt={dt:.1f}s "
                    f"processed={processed} skipped={skipped} "
                    f"elapsed={total_dt / 60:.1f}min",
                    flush=True,
                )

    total_dt = time.time() - t0
    print(
        f"[DONE] processed={processed} skipped={skipped} "
        f"elapsed={total_dt / 60:.1f}min out_csv={OUT_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gera warps ANTs longitudinais: follow-up → baseline do mesmo paciente.

Para cada ID_PT com visitas ordenadas (slot / MRI_DATE):
  - fixed  = imagem clínica do baseline (i1 = t0)
  - moving = i2 (t1) e i3 (t2)
  - saída  = affine + warp + inverseWarp em images/displacement_field_longitudinal/

Naming:
  {ID_IMG_moving}_ref-{ID_IMG_baseline}_0GenericAffine.mat
  {ID_IMG_moving}_ref-{ID_IMG_baseline}_1Warp.nii.gz
  {ID_IMG_moving}_ref-{ID_IMG_baseline}_1InverseWarp.nii.gz

Não gera warp i1→i1. Legado CN-template: 3_feat_gen_dvf_old.py → images/displacement_field/.

Extração de atributos: 3_feat_dvf.py (consome esta pasta).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import ants
import pandas as pd

# =========================
# CONFIG (edite aqui)
# =========================

COHORT = "all_population"
DEFAULT_IMAGES_CSV = f"csvs/cohorts/{COHORT}/all_population.csv"

DEFAULT_MIN_OUTPUT_BYTES = 1024

# Pastas SEPARADAS do legado (images/displacement_field/)
DEFAULT_TMPDIR = "./images/displacement_field_longitudinal/_tmp_ants"
clinic_dir = "./images/resampled_1.0mm"
warps_output = "./images/displacement_field_longitudinal"

SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}

os.makedirs(warps_output, exist_ok=True)

if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = os.path.abspath(DEFAULT_TMPDIR)
os.environ.setdefault("TMP", os.environ["TMPDIR"])
os.environ.setdefault("TEMP", os.environ["TMPDIR"])
os.makedirs(os.environ["TMPDIR"], exist_ok=True)


@dataclass(frozen=True)
class PatientBaseline:
    id_img: str
    clinic_path: str


def clinic_path_for(img_id: str) -> str:
    return os.path.join(
        clinic_dir, f"{img_id}_stripped_nlm_denoised_biascorrected.nii.gz"
    )


def warp_paths(moving_id: str, baseline_id: str) -> tuple[str, str, str]:
    stem = f"{moving_id}_ref-{baseline_id}"
    return (
        os.path.join(warps_output, f"{stem}_0GenericAffine.mat"),
        os.path.join(warps_output, f"{stem}_1Warp.nii.gz"),
        os.path.join(warps_output, f"{stem}_1InverseWarp.nii.gz"),
    )


def _file_nonempty(path: str, min_bytes: int) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def registration_bundle_complete(
    affine_out: str, warp_out: str, inv_warp_out: str, *, min_bytes: int
) -> bool:
    if not _file_nonempty(affine_out, 1):
        return False
    if not _file_nonempty(warp_out, min_bytes):
        return False
    if not _file_nonempty(inv_warp_out, min_bytes):
        return False
    return True


def remove_registration_bundle(
    affine_out: str, warp_out: str, inv_warp_out: str, *, reason: str
) -> None:
    for p in (affine_out, warp_out, inv_warp_out):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError as e:
            print(f"[WARN] nao foi possivel remover {p}: {e}")
    print(f"[RESUME] {reason}: arquivos de registro removidos para recomputar.")


def _sort_images_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MRI_DATE"] = pd.to_datetime(out["MRI_DATE"], errors="coerce")
    sort_cols = ["ID_PT"]
    if "slot" in out.columns:
        out["_slot_ord"] = out["slot"].astype(str).str.strip().map(SLOT_ORDER).fillna(99)
        sort_cols.append("_slot_ord")
    sort_cols.extend(["MRI_DATE", "ID_IMG"])
    return out.sort_values(sort_cols)


def build_patient_baseline_map(df_images: pd.DataFrame) -> dict[str, PatientBaseline]:
    """Primeira visita cronológica = fixed de todos os follow-ups do paciente."""
    df = _sort_images_chronologically(df_images)
    out: dict[str, PatientBaseline] = {}
    for id_pt, g in df.groupby("ID_PT", sort=False):
        r0 = g.iloc[0]
        bid = str(r0["ID_IMG"]).strip()
        out[str(id_pt)] = PatientBaseline(id_img=bid, clinic_path=clinic_path_for(bid))
    return out


def run_longitudinal_registrations(
    csv_images_path: str, *, min_output_bytes: int = DEFAULT_MIN_OUTPUT_BYTES
) -> None:
    df_imgs = pd.read_csv(csv_images_path)
    required = {"ID_PT", "ID_IMG", "MRI_DATE"}
    missing = required - set(df_imgs.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    baseline_by_pt = build_patient_baseline_map(df_imgs)
    df = _sort_images_chronologically(df_imgs)

    n_ok = n_skip = n_err = 0
    jobs: list[tuple[str, str, str, str]] = []  # id_pt, moving_id, baseline_id, moving_path

    for id_pt, g in df.groupby("ID_PT", sort=False):
        base = baseline_by_pt.get(str(id_pt))
        if base is None:
            continue
        for _, row in g.iterrows():
            moving_id = str(row["ID_IMG"]).strip()
            if moving_id == base.id_img:
                continue  # sem i1→i1
            jobs.append(
                (str(id_pt), moving_id, base.id_img, clinic_path_for(moving_id))
            )

    n_total = len(jobs)
    print(
        f"[START] pares follow-up→baseline={n_total} "
        f"pacientes={len(baseline_by_pt)} out={warps_output}",
        flush=True,
    )

    for idx, (id_pt, moving_id, baseline_id, moving_path) in enumerate(jobs, start=1):
        affine_out, warp_out, inv_warp_out = warp_paths(moving_id, baseline_id)
        fixed_path = clinic_path_for(baseline_id)

        if registration_bundle_complete(
            affine_out, warp_out, inv_warp_out, min_bytes=min_output_bytes
        ):
            print(
                f"[{idx}/{n_total}] [SKIP] {moving_id}→{baseline_id}: registro completo."
            )
            n_skip += 1
            continue

        if any(os.path.isfile(p) for p in (affine_out, warp_out, inv_warp_out)):
            remove_registration_bundle(
                affine_out,
                warp_out,
                inv_warp_out,
                reason=f"Registro incompleto {moving_id}_ref-{baseline_id}",
            )

        if not os.path.isfile(fixed_path):
            print(
                f"[{idx}/{n_total}] [SKIP] {moving_id}: baseline ausente: {fixed_path}"
            )
            n_skip += 1
            continue
        if not os.path.isfile(moving_path):
            print(
                f"[{idx}/{n_total}] [SKIP] {moving_id}: follow-up ausente: {moving_path}"
            )
            n_skip += 1
            continue

        print(
            f"[{idx}/{n_total}] [RUN] moving={moving_id} fixed={baseline_id} "
            f"(paciente={id_pt})",
            flush=True,
        )
        try:
            fixed_img = ants.image_read(fixed_path)
            moving_img = ants.image_read(moving_path)
            reg = ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform="SyN",
                interpolator="bspline",
            )

            fwd = reg.get("fwdtransforms", []) or []
            inv = reg.get("invtransforms", []) or []
            affine_src = next((p for p in fwd if str(p).endswith(".mat")), None)
            fwd_warp_src = next(
                (p for p in fwd if str(p).endswith(".nii") or str(p).endswith(".nii.gz")),
                None,
            )
            inv_warp_src = next(
                (p for p in inv if str(p).endswith(".nii") or str(p).endswith(".nii.gz")),
                None,
            )
            if affine_src is None or fwd_warp_src is None or inv_warp_src is None:
                print(
                    f"[{idx}/{n_total}] [ERROR] {moving_id}: transforms incompletos."
                )
                n_err += 1
                continue

            shutil.copy2(affine_src, affine_out)
            shutil.copy2(fwd_warp_src, warp_out)
            shutil.copy2(inv_warp_src, inv_warp_out)
            print(f"[{idx}/{n_total}] [OK] {moving_id} -> {warp_out}", flush=True)
            n_ok += 1
        except Exception as e:
            print(f"[{idx}/{n_total}] [ERROR] {moving_id}: {e}", flush=True)
            n_err += 1

    print(
        f"[DONE] registros longitudinais: ok={n_ok} skip={n_skip} err={n_err} "
        f"pares={n_total} out={warps_output}"
    )


if __name__ == "__main__":
    run_longitudinal_registrations(
        DEFAULT_IMAGES_CSV, min_output_bytes=DEFAULT_MIN_OUTPUT_BYTES
    )

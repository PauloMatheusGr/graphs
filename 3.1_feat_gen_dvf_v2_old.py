#!/usr/bin/env python3
"""
DVF v2: registra cada imagem clínica no template CN estratificado (sexo/idade baseline).

Pastas / saídas:
  warps:  images/displacement_field_v2/
  feats:  csvs/cohorts/all_population/features_displacement_v2.csv  (via 3_feat_dvf_v2.py)

Para cada ID_IMG em IMAGES_CSV:
  - fixed  = template CN (mesmo sexo e faixa etária do baseline do paciente)
  - moving = imagem clínica pré-processada (resampled_1.0mm)
  - saída  = affine + warp + inverse warp em images/displacement_field_v2/

Diferença vs v1 (3_feat_gen_dvf.py): fixed/moving invertidos → DVF no espaço do template.

DIAG/GROUP não entram na escolha do template: todos usam CN estratificado
por SEX e faixa etária da visita baseline.
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
# DEFAULT_IMAGES_CSV = "csvs/pilot_cn_ad_t0.csv"

DEFAULT_MIN_OUTPUT_BYTES = 1024

DEFAULT_TMPDIR = "./images/displacement_field_v2/_tmp_ants"

groupwise_dir = "images/groupwise/references"
clinic_dir = "images/resampled_1.0mm"
warps_output = "images/displacement_field_v3"

SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}

os.makedirs(warps_output, exist_ok=True)

if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = os.path.abspath(DEFAULT_TMPDIR)
os.environ.setdefault("TMP", os.environ["TMPDIR"])
os.environ.setdefault("TEMP", os.environ["TMPDIR"])
os.makedirs(os.environ["TMPDIR"], exist_ok=True)


@dataclass(frozen=True)
class BaselineReference:
    sex: str
    age: int
    age_range: str
    ref_path: str


def get_age_range(age: float) -> str:
    age = float(age)
    if 50 <= age <= 59.9:
        return "50-59"
    if 60 <= age <= 69.9:
        return "60-69"
    if 70 <= age <= 79.9:
        return "70-79"
    if 80 <= age <= 89.9:
        return "80-89"
    if 90 <= age <= 99.9:
        return "90-99"
    return "50-59" if age < 50 else "90-99"


def get_stratified_reference_path(sex: str, age_range: str) -> str:
    sex = str(sex).upper().strip()
    ref_filename = f"groupwise_DIAG-CN_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz"
    return os.path.join(groupwise_dir, ref_filename)


def subject_path_for(img_id: str) -> str:
    return os.path.join(
        clinic_dir, f"{img_id}_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
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


def build_baseline_reference_map(df_images: pd.DataFrame) -> dict[str, BaselineReference]:
    """
    Define o template CN estratificado por paciente a partir da visita baseline.

    Usa SEX e AGE da primeira visita (slot baseline, se disponível; senão MRI_DATE).
    DIAG e GROUP são ignorados de propósito: o referencial anatômico é sempre CN.
    """
    df = _sort_images_chronologically(df_images)
    ref_by_pt: dict[str, BaselineReference] = {}
    for id_pt, g in df.groupby("ID_PT", sort=False):
        r0 = g.iloc[0]
        sex = str(r0["SEX"]).upper().strip()
        age = int(r0["AGE"])
        age_range = get_age_range(age)
        ref_path = get_stratified_reference_path(sex=sex, age_range=age_range)
        ref_by_pt[str(id_pt)] = BaselineReference(
            sex=sex, age=age, age_range=age_range, ref_path=ref_path
        )
    return ref_by_pt


def run_individual_registrations(
    csv_images_path: str, *, min_output_bytes: int = DEFAULT_MIN_OUTPUT_BYTES
) -> None:
    df_imgs = pd.read_csv(csv_images_path)
    required = {"ID_PT", "ID_IMG", "SEX", "AGE", "MRI_DATE"}
    missing = required - set(df_imgs.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    ref_by_pt = build_baseline_reference_map(df_imgs)
    n_total = len(df_imgs)
    n_skip = n_ok = n_err = 0

    for idx, row in df_imgs.iterrows():
        img_id = str(row["ID_IMG"]).strip()
        id_pt = str(row["ID_PT"]).strip()
        ref = ref_by_pt.get(id_pt)
        if ref is None:
            print(f"[{idx + 1}/{n_total}] [SKIP] {img_id}: referencia baseline ausente.")
            n_skip += 1
            continue

        ref_tag = f"CN_SEX-{ref.sex}_AGE-{ref.age_range}"
        affine_out = os.path.join(warps_output, f"{img_id}_{ref_tag}_0GenericAffine.mat")
        warp_out = os.path.join(warps_output, f"{img_id}_{ref_tag}_1Warp.nii.gz")
        inv_warp_out = os.path.join(
            warps_output, f"{img_id}_{ref_tag}_1InverseWarp.nii.gz"
        )

        if registration_bundle_complete(
            affine_out, warp_out, inv_warp_out, min_bytes=min_output_bytes
        ):
            print(f"[{idx + 1}/{n_total}] [SKIP] {img_id}: registro completo ja existe.")
            n_skip += 1
            continue

        if any(os.path.isfile(p) for p in (affine_out, warp_out, inv_warp_out)):
            remove_registration_bundle(
                affine_out,
                warp_out,
                inv_warp_out,
                reason=f"Registro incompleto para {img_id}",
            )

        # v2: fixed = template CN, moving = sujeito
        fixed_path = ref.ref_path
        moving_path = subject_path_for(img_id)
        if not os.path.isfile(moving_path):
            print(f"[{idx + 1}/{n_total}] [SKIP] {img_id}: imagem clinica ausente: {moving_path}")
            n_skip += 1
            continue
        if not os.path.isfile(fixed_path):
            print(
                f"[{idx + 1}/{n_total}] [SKIP] {img_id}: template CN ausente: {fixed_path}"
            )
            n_skip += 1
            continue

        print(
            f"[{idx + 1}/{n_total}] [RUN] {img_id} "
            f"(paciente={id_pt}, fixed=template {ref_tag}, moving=sujeito, GROUP ignorado)"
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
                print(f"[{idx + 1}/{n_total}] [ERROR] {img_id}: transforms incompletos.")
                n_err += 1
                continue

            shutil.copy2(affine_src, affine_out)
            shutil.copy2(fwd_warp_src, warp_out)
            shutil.copy2(inv_warp_src, inv_warp_out)
            print(f"[{idx + 1}/{n_total}] [OK] {img_id} -> {warp_out}")
            n_ok += 1
        except Exception as e:
            print(f"[{idx + 1}/{n_total}] [ERROR] {img_id}: {e}")
            n_err += 1

    print(
        f"[DONE] registros v2: ok={n_ok} skip={n_skip} err={n_err} "
        f"total_linhas={n_total} out={warps_output}"
    )


if __name__ == "__main__":
    run_individual_registrations(
        DEFAULT_IMAGES_CSV, min_output_bytes=DEFAULT_MIN_OUTPUT_BYTES
    )

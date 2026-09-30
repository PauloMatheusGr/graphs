#!/usr/bin/env python3
"""
Gera warps ANTs (DVF): clínica → template groupwise estratificado (sexo/idade baseline).

Uso:
    python 3.1_feat_gen_dvf.py              # âncora CN (default)
    python 3.1_feat_gen_dvf.py --diag AD    # âncora AD
    python 3.1_feat_gen_dvf.py --src oasis --diag CN --ids-csv csvs/pilot/x.csv --shard 0/8

Saídas:
  CN → images/displacement_field_v3/
  AD → images/displacement_field_v3_ad/
  --src oasis → images/displacement_field_oasis_{cn,ad}/  (fixed=clínica, moving=template OASIS
    em MNI, SyNRA CC r4 100x70x50x20; o 1Warp fica na grade da imagem clínica)

Feats: 3.2_feat_dvf.py --diag {CN|AD}
DIAG/GROUP do paciente não escolhem o template — só SEX e idade baseline.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass

import time

import ants
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
import oasis_refs  # noqa: E402

COHORT = "all_population"
DEFAULT_IMAGES_CSV = f"csvs/cohorts/{COHORT}/all_population_True.csv"
DEFAULT_MIN_OUTPUT_BYTES = 1024
VALID_DIAG = ("CN", "AD")

groupwise_dir = "images/groupwise/references"
clinic_dir = "images/resampled_1.0mm"

SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}

# Set by configure_anchor()
ANCHOR_DIAG = "CN"
SRC = "adni"
warps_output = "images/displacement_field_v3"
SEED = 42


@dataclass(frozen=True)
class BaselineReference:
    sex: str
    age: int
    age_range: str
    ref_path: str


def configure_anchor(diag: str, src: str = "adni") -> None:
    global ANCHOR_DIAG, SRC, warps_output
    diag = str(diag).upper().strip()
    if diag not in VALID_DIAG:
        raise ValueError(f"diag={diag!r}; use {VALID_DIAG}")
    ANCHOR_DIAG = diag
    SRC = src
    if src == "oasis":
        warps_output = f"images/displacement_field_oasis_{diag.lower()}"
        tmp_default = f"./{warps_output}/_tmp_ants"
    elif diag == "CN":
        warps_output = "images/displacement_field_v3"
        tmp_default = "./images/displacement_field_v3/_tmp_ants"
    else:
        warps_output = "images/displacement_field_v3_ad"
        tmp_default = "./images/displacement_field_v3_ad/_tmp_ants"

    os.makedirs(warps_output, exist_ok=True)
    if not os.environ.get("TMPDIR"):
        os.environ["TMPDIR"] = os.path.abspath(tmp_default)
    os.environ.setdefault("TMP", os.environ["TMPDIR"])
    os.environ.setdefault("TEMP", os.environ["TMPDIR"])
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)


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
    ref_filename = (
        f"groupwise_DIAG-{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz"
    )
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
    """Template estratificado por SEX/idade baseline (âncora = ANCHOR_DIAG)."""
    if SRC == "oasis":
        return {
            pt: BaselineReference(
                sex=sex, age=-1, age_range=abin,
                ref_path=oasis_refs.template_path(ANCHOR_DIAG, sex, abin),
            )
            for pt, (sex, abin) in oasis_refs.baseline_by_pt(df_images).items()
        }
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


def _register(fixed_path: str, moving_path: str, *, verbose: bool) -> dict:
    fixed_img = ants.image_read(fixed_path)
    moving_img = ants.image_read(moving_path)
    if SRC == "oasis":
        # ponytail: antsRegistrationSyN[s,4] não serve, ANTsPy 0.6.3 força CC raio 2 no modo não-quick.
        reg = ants.registration(
            fixed=fixed_img, moving=moving_img, type_of_transform="SyNRA",
            syn_metric="CC", syn_sampling=4, reg_iterations=(100, 70, 50, 20),
            verbose=verbose,
        )
        fwd = reg["fwdtransforms"]
        assert sum(str(p).endswith(".mat") for p in fwd) == 1, fwd
        assert sum(str(p).endswith("Warp.nii.gz") for p in fwd) == 1, fwd
        return reg
    return ants.registration(
        fixed=fixed_img, moving=moving_img, type_of_transform="SyN", interpolator="bspline",
    )


def select_rows(df: pd.DataFrame, ids_csv: str | None, shard: str | None) -> pd.DataFrame:
    if ids_csv:
        ids = set(pd.read_csv(ids_csv)["ID_IMG"].astype(str).str.strip())
        df = df[df["ID_IMG"].astype(str).str.strip().isin(ids)]
        missing = ids - set(df["ID_IMG"].astype(str).str.strip())
        if missing:
            raise ValueError(f"{len(missing)} ID_IMG de {ids_csv} fora do CSV de imagens: {sorted(missing)[:5]}")
    if shard:
        k, n = map(int, shard.split("/"))
        assert 0 <= k < n, shard
        df = df.iloc[k::n]
    return df


def run_individual_registrations(
    csv_images_path: str,
    *,
    min_output_bytes: int = DEFAULT_MIN_OUTPUT_BYTES,
    ids_csv: str | None = None,
    shard: str | None = None,
    verbose_first: bool = False,
) -> None:
    df_all = pd.read_csv(csv_images_path)
    required = {"ID_PT", "ID_IMG", "SEX", "AGE", "MRI_DATE"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    # Baseline vem do CSV inteiro; o filtro só restringe quais imagens registrar.
    ref_by_pt = build_baseline_reference_map(df_all)
    df_imgs = select_rows(df_all, ids_csv, shard)
    n_total = len(df_imgs)
    n_skip = n_ok = n_err = 0
    times_csv = os.path.join(
        warps_output, f"reg_times_{shard.replace('/', 'of')}.csv" if shard else "reg_times.csv"
    )

    for idx, (_, row) in enumerate(df_imgs.iterrows()):
        img_id = str(row["ID_IMG"]).strip()
        id_pt = str(row["ID_PT"]).strip()
        ref = ref_by_pt.get(id_pt)
        if ref is None:
            print(f"[{idx + 1}/{n_total}] [SKIP] {img_id}: referencia baseline ausente.")
            n_skip += 1
            continue

        ref_tag = (
            oasis_refs.ref_tag(ANCHOR_DIAG, ref.sex, ref.age_range)
            if SRC == "oasis"
            else f"{ANCHOR_DIAG}_SEX-{ref.sex}_AGE-{ref.age_range}"
        )
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

        tpl_path, subj_path = ref.ref_path, subject_path_for(img_id)
        if SRC == "oasis":
            fixed_path, moving_path, roles = subj_path, tpl_path, "fixed=sujeito, moving=template"
        else:
            fixed_path, moving_path, roles = tpl_path, subj_path, "fixed=template, moving=sujeito"
        if not os.path.isfile(subj_path):
            print(f"[{idx + 1}/{n_total}] [SKIP] {img_id}: imagem clinica ausente: {subj_path}")
            n_skip += 1
            continue
        if not os.path.isfile(tpl_path):
            print(
                f"[{idx + 1}/{n_total}] [SKIP] {img_id}: "
                f"template {ANCHOR_DIAG} ausente: {tpl_path}"
            )
            n_skip += 1
            continue

        print(
            f"[{idx + 1}/{n_total}] [RUN] {img_id} "
            f"(paciente={id_pt}, ref={ref_tag}, {roles})",
            flush=True,
        )
        try:
            t_reg = time.time()
            reg = _register(fixed_path, moving_path, verbose=verbose_first and n_ok == 0)
            dt = time.time() - t_reg

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
            pd.DataFrame([{"ID_IMG": img_id, "ref_tag": ref_tag, "seconds": round(dt, 1),
                           "threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "")}]
                         ).to_csv(times_csv, mode="a", header=not os.path.isfile(times_csv), index=False)
            print(f"[{idx + 1}/{n_total}] [OK] {img_id} -> {warp_out} ({dt / 60:.1f} min)", flush=True)
            n_ok += 1
        except Exception as e:
            print(f"[{idx + 1}/{n_total}] [ERROR] {img_id}: {e}")
            n_err += 1

    print(
        f"[DONE] registros {ANCHOR_DIAG}: ok={n_ok} skip={n_skip} err={n_err} "
        f"total_linhas={n_total} out={warps_output}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DVF warps: clínica → template CN|AD")
    p.add_argument("--diag", default="CN", choices=VALID_DIAG, help="âncora normativa")
    p.add_argument("--src", default="adni", choices=("adni", "oasis"), help="origem dos templates")
    p.add_argument("--csv", default=DEFAULT_IMAGES_CSV, help="CSV de imagens")
    p.add_argument("--ids-csv", default=None, help="CSV com coluna ID_IMG para restringir")
    p.add_argument("--shard", default=None, help="k/n: processa linhas k::n")
    p.add_argument("--threads", type=int, default=1,
                   help="threads ITK (oasis); 1 = determinístico com --random-seed")
    p.add_argument("--verbose-first", action="store_true",
                   help="imprime o comando antsRegistration no primeiro registro")
    p.add_argument(
        "--min-bytes",
        type=int,
        default=DEFAULT_MIN_OUTPUT_BYTES,
        help="tamanho mínimo warp NIfTI",
    )
    args = p.parse_args(argv)
    if args.src == "oasis":
        ants.config.set_ants_deterministic(True, SEED)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    configure_anchor(args.diag, args.src)
    print(f"[INFO] SRC={SRC} ANCHOR={ANCHOR_DIAG} warps={warps_output} shard={args.shard}", flush=True)
    run_individual_registrations(
        args.csv, min_output_bytes=args.min_bytes, ids_csv=args.ids_csv,
        shard=args.shard, verbose_first=args.verbose_first,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

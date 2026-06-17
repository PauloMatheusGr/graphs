#!/usr/bin/env python3
"""Notebook 1 pós-extração: merge feat_*_all, long/wide, export ablation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ablation_prep import ROI_FILTER_DEFAULT, export_ablation_datasets, filter_rois, pivot_long_to_wide

BASE = Path("csvs/longitudinal_4_groups")
LONGITUDINAL = Path("csvs/adnimerged_longitudinal.csv")
MERGE_KEYS = ["ID_IMG", "roi", "side", "label"]

VOL_FEAT_COLS = [
    "mask_mm3", "gm_mm3", "gm_norm", "wm_mm3", "wm_norm",
    "csf_mm3", "csf_norm", "tissues_mm3", "tissues_norm",
]
RAD_SIZE_COLS = [
    "original_firstorder_Energy", "original_firstorder_TotalEnergy",
    "original_shape_MeshVolume", "original_shape_VoxelVolume",
    "original_shape_SurfaceArea", "original_shape_LeastAxisLength",
    "original_shape_MajorAxisLength", "original_shape_MinorAxisLength",
    "original_shape_Maximum2DDiameterColumn", "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice", "original_shape_Maximum3DDiameter",
]
VOL_SIZE_COLS = ["mask_mm3", "gm_mm3", "wm_mm3", "csf_mm3", "tissues_mm3"]


def normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ID_IMG"] = out["ID_IMG"].astype(str).str.strip()
    out["roi"] = out["roi"].astype(str).str.strip()
    out["side"] = out["side"].astype(str).str.strip()
    label_num = pd.to_numeric(out["label"], errors="coerce")
    out["label"] = label_num.map(lambda x: str(int(x)) if pd.notna(x) else "")
    return out


def merge_rad_vol_icv() -> Path:
    vol_df = normalize_merge_keys(pd.read_csv(BASE / "features_volumetric.csv"))
    rad_df = normalize_merge_keys(pd.read_csv(BASE / "features_radiomic.csv"))

    icv_df = (
        vol_df.loc[vol_df["roi"] == "__global__", ["ID_IMG", "mask_mm3"]]
        .drop_duplicates(subset=["ID_IMG"], keep="last")
        .rename(columns={"mask_mm3": "ICV_mask_mm3"})
    )
    icv_df["ICV_mask_mm3"] = pd.to_numeric(icv_df["ICV_mask_mm3"], errors="coerce")

    vol_roi = (
        vol_df.loc[vol_df["roi"] != "__global__", MERGE_KEYS + VOL_FEAT_COLS]
        .drop_duplicates(subset=MERGE_KEYS, keep="last")
    )
    for c in VOL_FEAT_COLS:
        vol_roi[c] = pd.to_numeric(vol_roi[c], errors="coerce")

    merged = rad_df.merge(vol_roi, on=MERGE_KEYS, how="left", validate="one_to_one")
    merged = merged.merge(icv_df, on="ID_IMG", how="left", validate="many_to_one")

    cols_to_norm = [c for c in RAD_SIZE_COLS + VOL_SIZE_COLS if c in merged.columns]
    merged[cols_to_norm] = pd.to_numeric(merged[cols_to_norm], errors="coerce")
    merged[cols_to_norm] = merged[cols_to_norm].div(merged["ICV_mask_mm3"], axis=0)

    out = BASE / "feat_rad_all.csv"
    merged.to_csv(out, index=False)
    print(f"[rad+vol] {out} shape={merged.shape}")
    return out


def add_cohort_meta() -> None:
    radiomics_merge = pd.read_csv(BASE / "feat_rad_all.csv")
    radiomics_merge["ID_IMG"] = radiomics_merge["ID_IMG"].astype(str).str.strip()

    cohort_cols = ["ID_IMG", "ID_PT", "GROUP", "SEX", "AGE", "MRI_DATE", "DIAG", "slot"]
    tech_cols = [
        "ID_IMG", "FIELD_STRENGTH", "MANUFACTURER", "MFG_MODEL",
        "MMSE_SCORE", "CDR_GLOBAL", "ADAS_SCORE", "FAQ_SCORE",
    ]
    longitudinal = pd.read_csv(LONGITUDINAL)
    meta_cols = list(dict.fromkeys(cohort_cols + tech_cols))
    meta_sub = (
        longitudinal[meta_cols]
        .assign(ID_IMG=lambda d: d["ID_IMG"].astype(str).str.strip())
        .drop_duplicates(subset=["ID_IMG"], keep="last")
    )
    merge = radiomics_merge.merge(meta_sub, on="ID_IMG", how="left", validate="many_to_one")
    batch_cols = ["MANUFACTURER", "MFG_MODEL", "FIELD_STRENGTH"]
    merge["batch"] = merge[batch_cols].astype(str).agg("_".join, axis=1)

    meta_order = [
        "ID_IMG", "roi", "side", "label",
        "ID_PT", "GROUP", "SEX", "AGE", "MRI_DATE", "DIAG", "slot",
        "FIELD_STRENGTH", "MANUFACTURER", "MFG_MODEL",
        "MMSE_SCORE", "CDR_GLOBAL", "ADAS_SCORE", "FAQ_SCORE", "batch",
    ]
    prefix = [c for c in meta_order if c in merge.columns]
    radiomic_cols = [c for c in merge.columns if c.startswith("original_")]
    vol_cols = [c for c in VOL_FEAT_COLS if c in merge.columns]
    icv_cols = [c for c in merge.columns if c == "ICV_mask_mm3"]
    known = set(prefix) | set(radiomic_cols) | set(vol_cols) | set(icv_cols)
    extra = [c for c in merge.columns if c not in known]
    merge = merge[prefix + radiomic_cols + vol_cols + icv_cols + extra]
    merge.to_csv(BASE / "feat_rad_all.csv", index=False)
    print(f"[meta rad] shape={merge.shape}")


def build_feat_disp_all() -> Path:
    def norm_keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ID_IMG"] = out["ID_IMG"].astype(str).str.strip()
        out["roi"] = out["roi"].astype(str).str.strip()
        out["side"] = out["side"].astype(str).str.strip()
        out["label"] = out["label"].astype(str).str.strip()
        return out

    df_disp = norm_keys(pd.read_csv(BASE / "features_displacement.csv"))
    meta_extra = ["ID_IMG", "slot", "FIELD_STRENGTH", "MANUFACTURER", "MFG_MODEL",
                  "MMSE_SCORE", "CDR_GLOBAL", "ADAS_SCORE", "FAQ_SCORE"]
    meta_sub = (
        pd.read_csv(LONGITUDINAL)[meta_extra]
        .assign(ID_IMG=lambda d: d["ID_IMG"].astype(str).str.strip())
        .drop_duplicates(subset=["ID_IMG"], keep="last")
    )
    overlap = [c for c in meta_sub.columns if c in df_disp.columns and c != "ID_IMG"]
    df_disp = df_disp.drop(columns=overlap, errors="ignore")
    df_all = df_disp.merge(meta_sub, on="ID_IMG", how="left", validate="many_to_one")
    batch_cols = ["MANUFACTURER", "MFG_MODEL", "FIELD_STRENGTH"]
    df_all["batch"] = df_all[batch_cols].astype(str).agg("_".join, axis=1)
    meta_order = [
        "ID_IMG", "roi", "side", "label",
        "ID_PT", "GROUP", "SEX", "AGE", "MRI_DATE", "DIAG", "slot", "ref_tag",
        "FIELD_STRENGTH", "MANUFACTURER", "MFG_MODEL",
        "MMSE_SCORE", "CDR_GLOBAL", "ADAS_SCORE", "FAQ_SCORE", "batch",
    ]
    prefix = [c for c in meta_order if c in df_all.columns]
    feat_cols = [c for c in df_all.columns if c not in prefix]
    df_all = df_all[prefix + feat_cols]
    out = BASE / "feat_disp_all.csv"
    df_all.to_csv(out, index=False)
    print(f"[disp] {out} shape={df_all.shape}")
    return out


def build_feat_merge_all() -> Path:
    overlap_meta = {
        "ID_PT", "SEX", "DIAG", "GROUP", "AGE", "MRI_DATE", "slot",
        "FIELD_STRENGTH", "MANUFACTURER", "MFG_MODEL",
        "MMSE_SCORE", "CDR_GLOBAL", "ADAS_SCORE", "FAQ_SCORE", "batch",
    }
    rad = pd.read_csv(BASE / "feat_rad_all.csv")
    rad = rad.drop(columns=[c for c in overlap_meta if c in rad.columns], errors="ignore")
    rad = normalize_merge_keys(rad)
    disp = normalize_merge_keys(pd.read_csv(BASE / "feat_disp_all.csv"))
    all_unit = rad.merge(disp, on=MERGE_KEYS, how="left", validate="one_to_one")
    if "MRI_DATE" in all_unit.columns:
        all_unit["MRI_DATE"] = (
            pd.to_datetime(all_unit["MRI_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        )
    out = BASE / "feat_merge_all.csv"
    all_unit.to_csv(out, index=False)
    print(f"[merge] {out} shape={all_unit.shape}")
    return out


def export_long_wide(roi: str = ROI_FILTER_DEFAULT) -> None:
    for name in ("merge", "rad", "disp"):
        src = pd.read_csv(BASE / f"feat_{name}_all.csv")
        long_df = filter_rois(src, roi)
        long_path = BASE / f"feat_{name}_all_{roi}_long.csv"
        long_df.to_csv(long_path, index=False)
        wide = pivot_long_to_wide(long_df)
        wide_path = BASE / f"feat_{name}_all_{roi}_wide.csv"
        wide.to_csv(wide_path, index=False)
        # aliases legados
        long_df.to_csv(BASE / f"feat_{name}_all_hipp.csv", index=False)
        print(f"[{name}] long={long_path.name} wide={wide_path.name} patients={len(wide)}")


def main() -> None:
    for p in (
        BASE / "features_volumetric.csv",
        BASE / "features_radiomic.csv",
        BASE / "features_displacement.csv",
    ):
        if not p.is_file():
            raise FileNotFoundError(f"Extração incompleta: {p}")

    merge_rad_vol_icv()
    add_cohort_meta()
    build_feat_disp_all()
    build_feat_merge_all()
    export_long_wide()
    paths = export_ablation_datasets(BASE)
    print(f"ablation export OK: {len(paths)} ficheiros")


if __name__ == "__main__":
    main()

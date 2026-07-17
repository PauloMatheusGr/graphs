"""Helpers partilhados: pivot long→wide T1/T2/T3 e export de CSVs para ablation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROI_FILTER_DEFAULT = "hippocampus"

# ComBat: fabricante + Tesla (sem MFG_MODEL — evita micro-batches no ADNI)
BATCH_COLS = ("MANUFACTURER", "FIELD_STRENGTH")


def assign_scanner_batch(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in BATCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas batch ausentes: {missing}")
    return df[list(BATCH_COLS)].astype(str).agg("_".join, axis=1)

SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}

META_COLS_WIDE = {
    "ID_PT",
    "GROUP",
    "SEX",
    "AGE",
    "ID_IMG",
    "MRI_DATE",
    "slot",
    "roi",
    "side",
    "label",
    "FIELD_STRENGTH",
    "MANUFACTURER",
    "MFG_MODEL",
    "batch",
    "DIAG",
    "MMSE_SCORE",
    "CDR_GLOBAL",
    "ADAS_SCORE",
    "FAQ_SCORE",
    "ref_tag",
    "ICV_mask_mm3",
}

VOL_FEAT_COLS = [
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

VOL_FEATURE_SUFFIXES = {
    "gm_norm",
    "wm_norm",
    "csf_norm",
}

SHAPE_RE = re.compile(r"^original_shape_")

TEXTURE_RE = re.compile(r"original_(glcm|gldm|glrlm|glszm|ngtdm)_")

DISP_PREFIX_DROP = (
    "centroid_",
    "ux_",
    "uy_",
    "uz_",
    "div_",
    "curlmag_",
    "mag_",
    "strain_trace_",
    "strain_vol_",
    "strain_det_",
    "strain_shear_ratio_",
    "strain_shear_energy_",
    "strain_shear_max_",
    "strain_l1_",
    "strain_l2_",
    "strain_l3_",
)

DISP_STAT_DROP = ("_n", "_variance", "_p05", "_p50", "_p95")


def filter_rois(
    df: pd.DataFrame,
    roi: str | list[str] = ROI_FILTER_DEFAULT,
    *,
    normalize_mri_date: bool = True,
) -> pd.DataFrame:
    if "roi" not in df.columns:
        raise KeyError("DataFrame precisa ter coluna 'roi'")
    rois = [roi] if isinstance(roi, str) else [str(r).strip() for r in roi]
    out = df.loc[df["roi"].astype(str).str.strip().isin(rois)].copy()
    if normalize_mri_date and "MRI_DATE" in out.columns:
        out["MRI_DATE"] = (
            pd.to_datetime(out["MRI_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        )
    return out


def wide_feat_regex(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(T[123])_")


def block_key_regex(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(T[123])_")


def pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in META_COLS_WIDE]
    out = df.copy()
    out["ROI_FULL"] = out["roi"].astype(str) + "_" + out["side"].astype(str)

    if "slot" in out.columns:
        out["_slot_ord"] = out["slot"].map(SLOT_ORDER).fillna(99)
        sort_cols = ["ID_PT", "ROI_FULL", "_slot_ord", "MRI_DATE", "ID_IMG"]
    else:
        out["MRI_DATE"] = pd.to_datetime(out["MRI_DATE"], errors="coerce")
        sort_cols = ["ID_PT", "ROI_FULL", "MRI_DATE", "ID_IMG"]

    out = out.sort_values(sort_cols)
    out["TIME"] = out.groupby(["ID_PT", "ROI_FULL"]).cumcount() + 1
    n_time = out.groupby(["ID_PT", "ROI_FULL"])["TIME"].max()
    bad = int((n_time != 3).sum())
    if bad:
        print(f"  aviso: {bad} grupos (ID_PT, ROI) sem exatamente 3 visitas")
    out["TIME"] = "T" + out["TIME"].astype(str)

    wide = out.pivot(
        index=["ID_PT", "GROUP", "SEX"],
        columns=["ROI_FULL", "TIME"],
        values=feature_cols,
    )
    wide.columns = [f"{roi}_{time}_{feat}" for feat, roi, time in wide.columns]
    wide = wide.reset_index()
    wide["SEX"] = wide["SEX"].map({"M": 0, "F": 1, 0: 0, 1: 1})
    return wide


def _select_vol_wide_columns(columns: list[str], roi: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(roi)}_[LR]_T[123]_(.+)$")
    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and m.group(1) in VOL_FEATURE_SUFFIXES:
            out.append(col)
    return out


def _select_shape_wide_columns(columns: list[str], roi: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(roi)}_[LR]_T[123]_(.+)$")
    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and SHAPE_RE.match(m.group(1)):
            out.append(col)
    return out


def _select_texture_wide_columns(columns: list[str], roi: str) -> list[str]:
    pat = wide_feat_regex(roi)
    return [c for c in columns if pat.match(c) and TEXTURE_RE.search(c)]


def _select_disp_wide_columns(columns: list[str], roi: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(roi)}_[LR]_T[123]_(.+)$")

    def keep(feat: str) -> bool:
        if any(feat.startswith(p) for p in DISP_PREFIX_DROP):
            return False
        if any(feat.endswith(s) for s in DISP_STAT_DROP):
            return False
        return True

    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and keep(m.group(1)):
            out.append(col)
    return out


def _filter_timepoint_columns(
    columns: list[str],
    *,
    roi: str,
    timepoints: tuple[str, ...],
) -> list[str]:
    if timepoints == ("T1", "T2", "T3"):
        return columns
    tp_pat = "|".join(re.escape(t) for t in timepoints)
    pat = re.compile(rf"^{re.escape(roi)}_[LR]_({tp_pat})_")
    return [c for c in columns if pat.match(c)]


def modality_wide_columns(
    columns: list[str] | pd.Index,
    modality: str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    timepoints: tuple[str, ...] = ("T1", "T2", "T3"),
) -> list[str]:
    """Mesmas colunas de feature que export_ablation_datasets grava em *_wide.csv."""
    cols = list(columns)
    if modality == "vol":
        out = _select_vol_wide_columns(cols, roi)
    elif modality == "shape":
        out = _select_shape_wide_columns(cols, roi)
    elif modality == "texture":
        out = _select_texture_wide_columns(cols, roi)
    elif modality == "disp":
        out = _select_disp_wide_columns(cols, roi)
    elif modality == "all":
        out = _select_vol_wide_columns(cols, roi)
        out += _select_shape_wide_columns(cols, roi)
        out += _select_texture_wide_columns(cols, roi)
        out += _select_disp_wide_columns(cols, roi)
        out = list(dict.fromkeys(out))
    else:
        raise ValueError(f"modalidade desconhecida: {modality}")
    return _filter_timepoint_columns(out, roi=roi, timepoints=timepoints)


def columns_matching_exclude(
    columns: list[str] | pd.Index,
    exclude_features: tuple[str, ...] | list[str] | None,
) -> list[str]:
    """Match exact long names or wide suffixes (…_token); never meta columns."""
    if not exclude_features:
        return []
    out: list[str] = []
    for c in columns:
        if c in META_COLS_WIDE:
            continue
        for ex in exclude_features:
            if not ex:
                continue
            if c == ex or c.endswith("_" + ex) or ex.endswith("_" + c):
                out.append(c)
                break
    return out


def drop_excluded_feature_columns(
    df: pd.DataFrame,
    exclude_features: tuple[str, ...] | list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop long/wide feature columns matching exclude tokens. Returns (df, dropped)."""
    drop = columns_matching_exclude(df.columns, exclude_features)
    if not drop:
        return df, []
    return df.drop(columns=drop), drop


def shape_long_from_rad_long(df_rad_long: pd.DataFrame) -> pd.DataFrame:
    meta = [c for c in df_rad_long.columns if c in META_COLS_WIDE or c in ("ID_IMG",)]
    feat = [c for c in df_rad_long.columns if SHAPE_RE.match(c)]
    icv = ["ICV_mask_mm3"] if "ICV_mask_mm3" in df_rad_long.columns else []
    cols = list(dict.fromkeys(meta + feat + icv))
    return df_rad_long[cols].copy()


def vol_long_from_rad_long(df_rad_long: pd.DataFrame) -> pd.DataFrame:
    meta = [c for c in df_rad_long.columns if c in META_COLS_WIDE or c in ("ID_IMG",)]
    feat = [c for c in VOL_FEAT_COLS if c in df_rad_long.columns]
    icv = ["ICV_mask_mm3"] if "ICV_mask_mm3" in df_rad_long.columns else []
    cols = list(dict.fromkeys(meta + feat + icv))
    return df_rad_long[cols].copy()


def export_ablation_long_only(
    rad: pd.DataFrame,
    disp: pd.DataFrame,
    merge: pd.DataFrame,
    base_dir: Path | str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
) -> dict[str, Path]:
    """Grava CSV long consumidos por ablation_runner (vol, shape, rad, disp, merge)."""
    ablation_dir = Path(base_dir) / "ablation" / roi
    ablation_dir.mkdir(parents=True, exist_ok=True)

    rad_long = filter_rois(rad, roi)
    disp_long = filter_rois(disp, roi)
    merge_long = filter_rois(merge, roi)
    vol_long = vol_long_from_rad_long(rad_long)
    shape_long = shape_long_from_rad_long(rad_long)

    paths: dict[str, Path] = {}
    for name, df in {
        "rad_long": rad_long,
        "disp_long": disp_long,
        "merge_long": merge_long,
        "vol_long": vol_long,
        "shape_long": shape_long,
    }.items():
        p = ablation_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        print(f"[ablation] {p.name} rows={len(df)}")
    return paths


def export_ablation_datasets(
    base_dir: Path | str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    write_wide: bool = False,
) -> dict[str, Path]:
    """Lê feat_*_all do disco; grava long em ablation/. Wide opcional (legado)."""
    base = Path(base_dir)
    rad = pd.read_csv(base / "feat_rad_all.csv")
    disp = pd.read_csv(base / "feat_disp_all.csv")
    merge = pd.read_csv(base / "feat_merge_all.csv")
    paths = export_ablation_long_only(rad, disp, merge, base, roi=roi)

    if not write_wide:
        return paths

    ablation_dir = base / "ablation" / roi
    rad_long = filter_rois(rad, roi)
    disp_long = filter_rois(disp, roi)
    merge_long = filter_rois(merge, roi)
    vol_long = vol_long_from_rad_long(rad_long)
    shape_long = shape_long_from_rad_long(rad_long)

    rad_wide = pivot_long_to_wide(rad_long)
    disp_wide = pivot_long_to_wide(disp_long)
    merge_wide = pivot_long_to_wide(merge_long)
    vol_wide = pivot_long_to_wide(vol_long)
    shape_wide = pivot_long_to_wide(shape_long)

    for name, df in {
        "rad_wide": rad_wide,
        "disp_wide": disp_wide,
        "merge_wide": merge_wide,
        "vol_wide": vol_wide,
        "shape_wide": shape_wide,
    }.items():
        p = ablation_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p

    meta = ["ID_PT", "GROUP", "SEX"]
    vol_cols = _select_vol_wide_columns(list(vol_wide.columns), roi)
    shape_cols = _select_shape_wide_columns(list(shape_wide.columns), roi)
    texture_cols = _select_texture_wide_columns(list(rad_wide.columns), roi)
    disp_cols = _select_disp_wide_columns(list(disp_wide.columns), roi)
    all_cols = _select_vol_wide_columns(list(merge_wide.columns), roi)
    all_cols += _select_shape_wide_columns(list(merge_wide.columns), roi)
    all_cols += _select_texture_wide_columns(list(merge_wide.columns), roi)
    all_cols += _select_disp_wide_columns(list(merge_wide.columns), roi)
    all_cols = list(dict.fromkeys(all_cols))

    for mod, (wide_df, cols) in {
        "vol": (vol_wide, vol_cols),
        "shape": (shape_wide, shape_cols),
        "texture": (rad_wide, texture_cols),
        "disp": (disp_wide, disp_cols),
        "all": (merge_wide, all_cols),
    }.items():
        out = wide_df[meta + cols].copy()
        p = ablation_dir / f"{mod}_wide.csv"
        out.to_csv(p, index=False)
        paths[f"{mod}_wide"] = p
        print(f"[{mod}] {p.name}: pacientes={len(out)} features={len(cols)}")

    return paths


if __name__ == "__main__":
    # ponytail: smoke — shape long + exclude-features matching
    import sys

    demo = pd.DataFrame(
        columns=[
            "ID_PT",
            "strain_fro_mean",
            "strain_fro_variance",
            "strain_fro_skewness",
            "strain_fro_kurtosis",
            "hippocampus_L_T1_strain_fro_variance",
        ]
    )
    hit = columns_matching_exclude(
        demo.columns,
        ("strain_fro_variance", "strain_fro_skewness", "strain_fro_kurtosis"),
    )
    assert hit == [
        "strain_fro_variance",
        "strain_fro_skewness",
        "strain_fro_kurtosis",
        "hippocampus_L_T1_strain_fro_variance",
    ], hit
    assert "ID_PT" not in hit
    dropped_df, dropped = drop_excluded_feature_columns(
        demo, ("strain_fro_variance", "strain_fro_skewness", "strain_fro_kurtosis"),
    )
    assert set(dropped) == set(hit)
    assert "strain_fro_mean" in dropped_df.columns
    print(f"ok: exclude-features matched {len(hit)} cols")

    base = Path("csvs/longitudinal_4_groups")
    rad_path = base / "ablation" / ROI_FILTER_DEFAULT / "rad_long.csv"
    if not rad_path.is_file():
        print(f"pulando self-check shape: {rad_path} ausente")
        sys.exit(0)
    rad = pd.read_csv(rad_path, nrows=1)
    n_shape = sum(1 for c in rad.columns if SHAPE_RE.match(c))
    assert n_shape == 14, f"esperado 14 shape cols, got {n_shape}"
    wide_n = len(modality_wide_columns(
        [f"hippocampus_L_T1_{c}" for c in rad.columns if SHAPE_RE.match(c)],
        "shape",
    ))
    assert wide_n == 14
    print(f"ok: {n_shape} shape long, modality_wide_columns shape={wide_n}")

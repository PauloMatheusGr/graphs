"""T1 + deltas longitudinais (D21, D31, D32) a partir de wide T1/T2/T3."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ablation_prep import (
    ROI_FILTER_DEFAULT,
    keep_disp_feat,
    keep_firstorder_feat,
    keep_shape_feat,
    keep_texture_feat,
    keep_vol_feat,
    modality_wide_columns as modality_wide_columns_absolute,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

DeltaKind = Literal["abs", "rel"]

META_WIDE = frozenset({"ID_PT", "GROUP", "SEX", "y"})
DELTA_EPS = 1e-8
DELTA_TIME_TOKENS = ("D21", "D31", "D32")
DELTA_TIME_TOKENS_LEGACY = ("D21", "D31", "SLOPE")
REPRESENTATION_TOKENS = ("T1", "D21", "D31", "D32")
REPRESENTATION_TOKENS_LEGACY = ("T1", "D21", "D31", "SLOPE")
REPRESENTATION_TOKENS_Q4 = ("T1", "D21", "D32")
REPRESENTATION_TOKENS_D21 = ("T1", "D21")
REPRESENTATION_TOKENS_Q5 = ("T1", "M", "A")
PROTOCOL_T1_DELTAS = "t1_deltas_abs"


def absolute_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(T[123])_(.+)$")


def representation_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_[LR]_(T1|D21|D31|D32|SLOPE|M|A)_(.+)$")


def absolute_delta(v_from: pd.Series, v_to: pd.Series) -> pd.Series:
    """Diferença T_to − T_from."""
    v_from = pd.to_numeric(v_from, errors="coerce").astype(float)
    v_to = pd.to_numeric(v_to, errors="coerce").astype(float)
    return v_to - v_from


def relative_delta(v_from: pd.Series, v_to: pd.Series, *, eps: float = DELTA_EPS) -> pd.Series:
    """(v_to − v_from) / v_from; NaN se |v_from| < eps."""
    v_from = pd.to_numeric(v_from, errors="coerce").astype(float)
    v_to = pd.to_numeric(v_to, errors="coerce").astype(float)
    diff = v_to - v_from
    denom = v_from.where(v_from.abs() >= eps)
    return diff / denom


def _pair_delta(
    v_from: pd.Series,
    v_to: pd.Series,
    *,
    delta_kind: DeltaKind,
) -> pd.Series:
    if delta_kind == "rel":
        return relative_delta(v_from, v_to)
    return absolute_delta(v_from, v_to)


def add_delta_columns(
    wide: pd.DataFrame,
    roi: str = ROI_FILTER_DEFAULT,
    *,
    include_t1: bool = True,
    include_absolute: bool = False,
    delta_kind: DeltaKind = "abs",
    include_slope: bool = False,
    include_ma: bool = False,
) -> pd.DataFrame:
    """D21=T2−T1, D31=T3−T1, D32=T3−T2 (abs default). Opcional T1, SLOPE legado (rel), M/A."""
    pat = absolute_col_pat(roi)
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for col in wide.columns:
        m = pat.match(col)
        if not m:
            continue
        side, feat = m.group(1), m.group(3)
        t = m.group(2)
        groups.setdefault((side, feat), {})[t] = col

    delta_cols: dict[str, pd.Series] = {}
    for (side, feat), times in groups.items():
        if not all(t in times for t in ("T1", "T2", "T3")):
            continue
        t1 = wide[times["T1"]]
        t2 = wide[times["T2"]]
        t3 = wide[times["T3"]]
        prefix = f"{roi}_{side}"
        d21 = _pair_delta(t1, t2, delta_kind=delta_kind)
        d31 = _pair_delta(t1, t3, delta_kind=delta_kind)
        d32 = _pair_delta(t2, t3, delta_kind=delta_kind)
        delta_cols[f"{prefix}_D21_{feat}"] = d21
        delta_cols[f"{prefix}_D31_{feat}"] = d31
        delta_cols[f"{prefix}_D32_{feat}"] = d32
        if include_ma:
            delta_cols[f"{prefix}_M_{feat}"] = (d21 + d32) / 2.0
            delta_cols[f"{prefix}_A_{feat}"] = d32 - d21
        if include_slope and delta_kind == "rel":
            delta_cols[f"{prefix}_SLOPE_{feat}"] = d31 / 2.0

    keep = [c for c in wide.columns if c in META_WIDE]
    if include_absolute:
        keep.extend(c for c in wide.columns if pat.match(c))
    elif include_t1:
        for times in groups.values():
            if all(t in times for t in ("T1", "T2", "T3")):
                keep.append(times["T1"])

    out = wide[keep].copy()
    if delta_cols:
        out = pd.concat([out, pd.DataFrame(delta_cols, index=wide.index)], axis=1)
    return out


def feature_tokens_for_delta_representation(representation: str) -> tuple[str, ...]:
    if representation == "deltas_only":
        return DELTA_TIME_TOKENS
    if representation == "t1_deltas_rel":
        return REPRESENTATION_TOKENS_LEGACY
    if representation in ("t1_deltas", "t1_deltas_abs"):
        return REPRESENTATION_TOKENS
    if representation == "t1_d21_d32":
        return REPRESENTATION_TOKENS_Q4
    if representation == "t1_d21":
        return REPRESENTATION_TOKENS_D21
    if representation == "t1_ma":
        return REPRESENTATION_TOKENS_Q5
    raise ValueError(f"representação delta desconhecida: {representation!r}")


def delta_kwargs_for_representation(representation: str) -> dict:
    if representation == "deltas_only":
        return {"include_t1": False, "delta_kind": "abs", "include_slope": False}
    if representation == "t1_deltas_rel":
        return {"include_t1": True, "delta_kind": "rel", "include_slope": True}
    if representation in ("t1_deltas", "t1_deltas_abs", "t1_d21_d32", "t1_d21"):
        return {"include_t1": True, "delta_kind": "abs", "include_slope": False}
    if representation == "t1_ma":
        return {
            "include_t1": True, "delta_kind": "abs", "include_slope": False,
            "include_ma": True,
        }
    raise ValueError(f"representação delta desconhecida: {representation!r}")


def _select_delta_columns(
    columns: Iterable[str],
    roi: str,
    *,
    feat_keep,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    pat = representation_col_pat(roi)
    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if not m:
            continue
        token, feat = m.group(1), m.group(2)
        if token not in feature_tokens:
            continue
        if feat_keep(feat):
            out.append(col)
    return out


def _select_vol_delta(
    columns: Iterable[str],
    roi: str,
    *,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    return _select_delta_columns(
        columns,
        roi,
        feat_keep=keep_vol_feat,
        feature_tokens=feature_tokens,
    )


def _select_shape_delta(
    columns: Iterable[str],
    roi: str,
    *,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    return _select_delta_columns(
        columns,
        roi,
        feat_keep=keep_shape_feat,
        feature_tokens=feature_tokens,
    )


def _select_texture_delta(
    columns: Iterable[str],
    roi: str,
    *,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    return _select_delta_columns(
        columns,
        roi,
        feat_keep=keep_texture_feat,
        feature_tokens=feature_tokens,
    )


def _select_disp_delta(
    columns: Iterable[str],
    roi: str,
    *,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    return _select_delta_columns(
        columns,
        roi,
        feat_keep=keep_disp_feat,
        feature_tokens=feature_tokens,
    )


def _select_firstorder_delta(
    columns: Iterable[str],
    roi: str,
    *,
    feature_tokens: tuple[str, ...],
) -> list[str]:
    return _select_delta_columns(
        columns,
        roi,
        feat_keep=keep_firstorder_feat,
        feature_tokens=feature_tokens,
    )


def modality_wide_columns(
    columns: list[str] | pd.Index,
    modality: str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    use_deltas: bool = False,
    feature_tokens: tuple[str, ...] | None = None,
) -> list[str]:
    if not use_deltas:
        return modality_wide_columns_absolute(columns, modality, roi=roi)

    tokens = feature_tokens or REPRESENTATION_TOKENS
    cols = list(columns)
    if modality == "vol":
        return _select_vol_delta(cols, roi, feature_tokens=tokens)
    if modality == "shape":
        return _select_shape_delta(cols, roi, feature_tokens=tokens)
    if modality == "texture":
        return _select_texture_delta(cols, roi, feature_tokens=tokens)
    if modality == "disp":
        return _select_disp_delta(cols, roi, feature_tokens=tokens)
    if modality == "firstorder":
        return _select_firstorder_delta(cols, roi, feature_tokens=tokens)
    if modality == "all":
        out = _select_vol_delta(cols, roi, feature_tokens=tokens)
        out += _select_shape_delta(cols, roi, feature_tokens=tokens)
        out += _select_texture_delta(cols, roi, feature_tokens=tokens)
        out += _select_disp_delta(cols, roi, feature_tokens=tokens)
        out += _select_firstorder_delta(cols, roi, feature_tokens=tokens)
        return list(dict.fromkeys(out))
    raise ValueError(f"modalidade desconhecida: {modality}")


if __name__ == "__main__":
    roi = ROI_FILTER_DEFAULT
    wide = pd.DataFrame(
        {
            "ID_PT": ["p1", "p2"],
            "GROUP": ["sMCI", "pMCI"],
            "SEX": [0, 1],
            f"{roi}_L_T1_gm_norm": [2.0, 1.0],
            f"{roi}_L_T2_gm_norm": [2.4, 1.2],
            f"{roi}_L_T3_gm_norm": [3.0, 1.5],
        }
    )
    out = add_delta_columns(wide, roi, delta_kind="abs")
    assert f"{roi}_L_D21_gm_norm" in out.columns
    assert f"{roi}_L_D32_gm_norm" in out.columns
    assert f"{roi}_L_SLOPE_gm_norm" not in out.columns
    assert abs(float(out[f"{roi}_L_D21_gm_norm"].iloc[0]) - 0.4) < 1e-9
    assert abs(float(out[f"{roi}_L_D31_gm_norm"].iloc[0]) - 1.0) < 1e-9
    assert abs(float(out[f"{roi}_L_D32_gm_norm"].iloc[0]) - 0.6) < 1e-9
    assert f"{roi}_L_T2_gm_norm" not in out.columns
    assert f"{roi}_L_T1_gm_norm" in out.columns

    dyn = add_delta_columns(wide, roi, include_t1=False, delta_kind="abs")
    assert f"{roi}_L_T1_gm_norm" not in dyn.columns
    assert len(modality_wide_columns(dyn.columns, "vol", roi=roi, use_deltas=True,
                                     feature_tokens=DELTA_TIME_TOKENS)) == 3

    rel = add_delta_columns(wide, roi, delta_kind="rel", include_slope=True)
    assert f"{roi}_L_SLOPE_gm_norm" in rel.columns
    assert abs(float(rel[f"{roi}_L_D21_gm_norm"].iloc[0]) - 0.2) < 1e-9

    tiny = relative_delta(pd.Series([1e-12]), pd.Series([1.0]))
    assert np.isnan(float(tiny.iloc[0]))
    n = len(modality_wide_columns(out.columns, "vol", roi=roi, use_deltas=True))
    assert n == 4  # L × (T1,D21,D31,D32) × gm_norm

    q4_cols = modality_wide_columns(
        out.columns, "vol", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_Q4,
    )
    assert len(q4_cols) == 3
    assert f"{roi}_L_D31_gm_norm" not in q4_cols

    d21_cols = modality_wide_columns(
        out.columns, "vol", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_D21,
    )
    assert len(d21_cols) == 2
    assert f"{roi}_L_T1_gm_norm" in d21_cols
    assert f"{roi}_L_D21_gm_norm" in d21_cols
    assert f"{roi}_L_D32_gm_norm" not in d21_cols

    ma = add_delta_columns(wide, roi, include_t1=True, delta_kind="abs", include_ma=True)
    m_col, a_col = f"{roi}_L_M_gm_norm", f"{roi}_L_A_gm_norm"
    assert m_col in ma.columns and a_col in ma.columns
    d21 = float(ma[f"{roi}_L_D21_gm_norm"].iloc[0])
    d32 = float(ma[f"{roi}_L_D32_gm_norm"].iloc[0])
    assert abs(float(ma[m_col].iloc[0]) - (d21 + d32) / 2.0) < 1e-9
    assert abs(float(ma[a_col].iloc[0]) - (d32 - d21)) < 1e-9
    q5_cols = modality_wide_columns(
        ma.columns, "vol", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_Q5,
    )
    assert set(q5_cols) == {f"{roi}_L_T1_gm_norm", m_col, a_col}

    wide_vol = pd.DataFrame(
        {
            "ID_PT": ["p1"],
            "GROUP": ["sMCI"],
            "SEX": [0],
            f"{roi}_L_T1_mask_mm3": [100.0],
            f"{roi}_L_T2_mask_mm3": [101.0],
            f"{roi}_L_T3_mask_mm3": [102.0],
            f"{roi}_L_T1_gm_norm": [2.0],
            f"{roi}_L_T2_gm_norm": [2.4],
            f"{roi}_L_T3_gm_norm": [3.0],
        }
    )
    vol_out = add_delta_columns(wide_vol, roi, delta_kind="abs")
    vol_cols = modality_wide_columns(vol_out.columns, "vol", roi=roi, use_deltas=True)
    assert f"{roi}_L_T1_mask_mm3" not in vol_cols
    assert f"{roi}_L_T1_gm_norm" in vol_cols
    assert f"{roi}_L_D32_gm_norm" in vol_cols

    q4_dummy = [
        f"{roi}_L_T1_original_glcm_Contrast",
        f"{roi}_L_D21_original_glcm_Contrast",
        f"{roi}_L_D21_original_glrlm_ShortRunEmphasis",
        f"{roi}_L_D21_original_firstorder_Mean",
        f"{roi}_L_D21_original_firstorder_Energy",
        f"{roi}_L_D21_original_shape_Sphericity",
        f"{roi}_L_D21_original_shape_MeshVolume",
        f"{roi}_L_D21_gm_norm",
    ]
    q4_tex = modality_wide_columns(
        q4_dummy, "texture", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_Q4,
    )
    q4_fo = modality_wide_columns(
        q4_dummy, "firstorder", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_Q4,
    )
    q4_shp = modality_wide_columns(
        q4_dummy, "shape", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_Q4,
    )
    assert f"{roi}_L_D21_original_glcm_Contrast" in q4_tex
    assert f"{roi}_L_D21_original_glrlm_ShortRunEmphasis" not in q4_tex
    assert f"{roi}_L_D21_original_firstorder_Mean" in q4_fo
    assert f"{roi}_L_D21_original_firstorder_Energy" not in q4_fo
    assert f"{roi}_L_D21_original_shape_Sphericity" in q4_shp
    assert f"{roi}_L_D21_original_shape_MeshVolume" not in q4_shp
    print("ablation_deltas self-check ok")

    # ponytail: paridade abs×4/3 em dados reais (se CSV existir)
    from pathlib import Path

    from ablation_prep import pivot_long_to_wide

    data_dir = Path("csvs/cohorts/36m_6m/ablation") / roi
    for mod, fname in (("vol", "vol_long.csv"), ("disp", "disp_long.csv"), ("shape", "shape_long.csv")):
        long_path = data_dir / fname
        if not long_path.is_file():
            continue
        wide = pivot_long_to_wide(pd.read_csv(long_path))
        n_abs = len(modality_wide_columns_absolute(wide.columns, mod, roi=roi))
        wide_d = add_delta_columns(wide, roi, include_t1=True, delta_kind="abs")
        n_delta = len(
            modality_wide_columns(wide_d.columns, mod, roi=roi, use_deltas=True),
        )
        expected = n_abs // 3 * 4
        assert n_delta == expected, f"{mod}: abs={n_abs} delta={n_delta} expected={expected}"
        n_q4 = len(modality_wide_columns(
            wide_d.columns, mod, roi=roi, use_deltas=True,
            feature_tokens=REPRESENTATION_TOKENS_Q4,
        ))
        assert n_q4 == n_abs, f"{mod}: abs={n_abs} q4={n_q4}"
        n_d21 = len(modality_wide_columns(
            wide_d.columns, mod, roi=roi, use_deltas=True,
            feature_tokens=REPRESENTATION_TOKENS_D21,
        ))
        assert n_d21 == n_abs // 3 * 2, f"{mod}: abs={n_abs} d21={n_d21}"
        wide_ma = add_delta_columns(wide, roi, include_t1=True, delta_kind="abs", include_ma=True)
        n_q5 = len(modality_wide_columns(
            wide_ma.columns, mod, roi=roi, use_deltas=True,
            feature_tokens=REPRESENTATION_TOKENS_Q5,
        ))
        assert n_q5 == n_abs, f"{mod}: abs={n_abs} q5={n_q5}"

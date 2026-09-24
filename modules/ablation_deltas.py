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
# ponytail: mean Gregorian month; upgrade = calendar-exact months if needed
DAYS_PER_MONTH = 30.436875
TIME_EPS_MONTHS = 1e-6
DELTA_TIME_TOKENS = ("D21", "D31", "D32")
DELTA_TIME_TOKENS_LEGACY = ("D21", "D31", "SLOPE")
REPRESENTATION_TOKENS = ("T1", "D21", "D31", "D32")
REPRESENTATION_TOKENS_LEGACY = ("T1", "D21", "D31", "SLOPE")
REPRESENTATION_TOKENS_Q4 = ("T1", "D21", "D32")
REPRESENTATION_TOKENS_D21 = ("T1", "D21")
REPRESENTATION_TOKENS_Q5 = ("T1", "M", "A")
REPRESENTATION_TOKENS_R10_R21 = ("T1", "R10", "R21")
REPRESENTATION_TOKENS_RATE02 = ("T1", "RATE02")
REPRESENTATION_TOKENS_OLS = ("T1", "BETA1")
RATE_TIME_TOKENS = ("R10", "R21", "RATE02", "BETA1")
RATE_REPRESENTATIONS = frozenset({"t1_r10_r21", "t1_rate02", "t1_ols"})
PROTOCOL_T1_DELTAS = "t1_deltas_abs"
_TOKEN_ALT = "T1|D21|D31|D32|SLOPE|M|A|R10|R21|RATE02|BETA1"


def absolute_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(T[123])_(.+)$")


def representation_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_[LR]_({_TOKEN_ALT})_(.+)$")


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
    include_t3_deltas: bool = True,
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
        required = ("T1", "T2", "T3") if include_t3_deltas else ("T1", "T2")
        if not all(t in times for t in required):
            continue
        t1 = wide[times["T1"]]
        t2 = wide[times["T2"]]
        prefix = f"{roi}_{side}"
        d21 = _pair_delta(t1, t2, delta_kind=delta_kind)
        delta_cols[f"{prefix}_D21_{feat}"] = d21
        if include_t3_deltas:
            t3 = wide[times["T3"]]
            d31 = _pair_delta(t1, t3, delta_kind=delta_kind)
            d32 = _pair_delta(t2, t3, delta_kind=delta_kind)
            delta_cols[f"{prefix}_D31_{feat}"] = d31
            delta_cols[f"{prefix}_D32_{feat}"] = d32
        if include_ma:
            if not include_t3_deltas:
                raise ValueError("M/A exige T3.")
            delta_cols[f"{prefix}_M_{feat}"] = (d21 + d32) / 2.0
            delta_cols[f"{prefix}_A_{feat}"] = d32 - d21
        if include_slope and delta_kind == "rel":
            if not include_t3_deltas:
                raise ValueError("SLOPE exige T3.")
            delta_cols[f"{prefix}_SLOPE_{feat}"] = d31 / 2.0

    keep = [c for c in wide.columns if c in META_WIDE]
    if include_absolute:
        keep.extend(c for c in wide.columns if pat.match(c))
    elif include_t1:
        for times in groups.values():
            required = ("T1", "T2", "T3") if include_t3_deltas else ("T1", "T2")
            if all(t in times for t in required):
                keep.append(times["T1"])

    out = wide[keep].copy()
    if delta_cols:
        out = pd.concat([out, pd.DataFrame(delta_cols, index=wide.index)], axis=1)
    return out


def visit_times_months(df_long: pd.DataFrame) -> pd.DataFrame:
    """Por ID_PT: t0=0, t1, t2 em meses desde baseline (mesma ordem slot/MRI_DATE do pivot)."""
    from ablation_prep import SLOT_ORDER

    if "ID_PT" not in df_long.columns or "MRI_DATE" not in df_long.columns:
        raise ValueError("visit_times_months exige ID_PT e MRI_DATE no long.")
    visits = df_long.copy()
    visits["ID_PT"] = visits["ID_PT"].astype(str)
    visits["MRI_DATE"] = pd.to_datetime(visits["MRI_DATE"], errors="coerce")
    if "slot" in visits.columns:
        visits["_slot_ord"] = visits["slot"].map(SLOT_ORDER).fillna(99)
        sort_cols = ["ID_PT", "_slot_ord", "MRI_DATE"]
        if "ID_IMG" in visits.columns:
            sort_cols.append("ID_IMG")
            visits = visits.drop_duplicates(["ID_PT", "ID_IMG"])
        else:
            visits = visits.drop_duplicates(["ID_PT", "slot"])
    else:
        sort_cols = ["ID_PT", "MRI_DATE"]
        if "ID_IMG" in visits.columns:
            sort_cols.append("ID_IMG")
            visits = visits.drop_duplicates(["ID_PT", "ID_IMG"])
        else:
            visits = visits.drop_duplicates(["ID_PT", "MRI_DATE"])
    visits = visits.sort_values(sort_cols)
    visits["_visit"] = visits.groupby("ID_PT").cumcount()

    rows: list[dict] = []
    for pid, g in visits.groupby("ID_PT", sort=False):
        g3 = g.nsmallest(3, "_visit") if "_visit" in g.columns else g.head(3)
        g3 = g3.sort_values("_visit")
        if len(g3) < 3 or g3["MRI_DATE"].isna().any():
            rows.append({"ID_PT": pid, "t0": np.nan, "t1": np.nan, "t2": np.nan})
            continue
        base = g3["MRI_DATE"].iloc[0]
        t1 = (g3["MRI_DATE"].iloc[1] - base).total_seconds() / (86400.0 * DAYS_PER_MONTH)
        t2 = (g3["MRI_DATE"].iloc[2] - base).total_seconds() / (86400.0 * DAYS_PER_MONTH)
        if not (np.isfinite(t1) and np.isfinite(t2) and t1 > TIME_EPS_MONTHS and t2 > t1 + TIME_EPS_MONTHS):
            rows.append({"ID_PT": pid, "t0": np.nan, "t1": np.nan, "t2": np.nan})
            continue
        rows.append({"ID_PT": pid, "t0": 0.0, "t1": float(t1), "t2": float(t2)})
    return pd.DataFrame(rows).set_index("ID_PT")


def ols_slope_three(
    t0: float, t1: float, t2: float,
    x0: float, x1: float, x2: float,
    *,
    eps: float = TIME_EPS_MONTHS,
) -> float:
    """β̂1 OLS de x~t com 3 pontos; NaN se denom≈0 ou não-finito."""
    t = np.asarray([t0, t1, t2], dtype=float)
    x = np.asarray([x0, x1, x2], dtype=float)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        return float("nan")
    t_bar = t.mean()
    x_bar = x.mean()
    denom = float(((t - t_bar) ** 2).sum())
    if denom < eps:
        return float("nan")
    return float(((t - t_bar) * (x - x_bar)).sum() / denom)


def _align_times_to_wide(wide: pd.DataFrame, times_months: pd.DataFrame) -> pd.DataFrame:
    if "ID_PT" not in wide.columns:
        raise ValueError("wide sem ID_PT para alinhar tempos.")
    tm = times_months.copy()
    if tm.index.name != "ID_PT" and "ID_PT" in tm.columns:
        tm = tm.set_index("ID_PT")
    tm.index = tm.index.astype(str)
    for col in ("t0", "t1", "t2"):
        if col not in tm.columns:
            raise ValueError(f"times_months sem coluna {col!r}")
    aligned = tm.reindex(wide["ID_PT"].astype(str).to_numpy())
    aligned.index = wide.index
    return aligned


def add_rate_columns(
    wide: pd.DataFrame,
    times_months: pd.DataFrame,
    roi: str = ROI_FILTER_DEFAULT,
    *,
    include_t1: bool = True,
    include_r10_r21: bool = False,
    include_rate02: bool = False,
    include_beta1: bool = False,
) -> pd.DataFrame:
    """R10/R21/RATE02/BETA1 com tempos reais (meses). Exige T1/T2/T3 no wide."""
    if not (include_r10_r21 or include_rate02 or include_beta1):
        raise ValueError("add_rate_columns: active pelo menos um de r10_r21/rate02/beta1")
    pat = absolute_col_pat(roi)
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for col in wide.columns:
        m = pat.match(col)
        if not m:
            continue
        side, feat = m.group(1), m.group(3)
        groups.setdefault((side, feat), {})[m.group(2)] = col

    aligned = _align_times_to_wide(wide, times_months)
    t0 = aligned["t0"].to_numpy(dtype=float)
    t1 = aligned["t1"].to_numpy(dtype=float)
    t2 = aligned["t2"].to_numpy(dtype=float)
    dt10 = t1 - t0
    dt21 = t2 - t1
    dt20 = t2 - t0
    ok10 = np.isfinite(dt10) & (dt10 > TIME_EPS_MONTHS)
    ok21 = np.isfinite(dt21) & (dt21 > TIME_EPS_MONTHS)
    ok20 = np.isfinite(dt20) & (dt20 > TIME_EPS_MONTHS)

    rate_cols: dict[str, pd.Series] = {}
    for (side, feat), times in groups.items():
        if not all(k in times for k in ("T1", "T2", "T3")):
            continue
        v0 = pd.to_numeric(wide[times["T1"]], errors="coerce").to_numpy(dtype=float)
        v1 = pd.to_numeric(wide[times["T2"]], errors="coerce").to_numpy(dtype=float)
        v2 = pd.to_numeric(wide[times["T3"]], errors="coerce").to_numpy(dtype=float)
        prefix = f"{roi}_{side}"
        if include_r10_r21:
            r10 = np.where(ok10, (v1 - v0) / dt10, np.nan)
            r21 = np.where(ok21, (v2 - v1) / dt21, np.nan)
            rate_cols[f"{prefix}_R10_{feat}"] = pd.Series(r10, index=wide.index)
            rate_cols[f"{prefix}_R21_{feat}"] = pd.Series(r21, index=wide.index)
        if include_rate02:
            r02 = np.where(ok20, (v2 - v0) / dt20, np.nan)
            rate_cols[f"{prefix}_RATE02_{feat}"] = pd.Series(r02, index=wide.index)
        if include_beta1:
            beta = np.array([
                ols_slope_three(t0[i], t1[i], t2[i], v0[i], v1[i], v2[i])
                for i in range(len(wide))
            ], dtype=float)
            rate_cols[f"{prefix}_BETA1_{feat}"] = pd.Series(beta, index=wide.index)

    keep = [c for c in wide.columns if c in META_WIDE]
    if include_t1:
        for times in groups.values():
            if all(k in times for k in ("T1", "T2", "T3")):
                keep.append(times["T1"])
    out = wide[keep].copy()
    if rate_cols:
        out = pd.concat([out, pd.DataFrame(rate_cols, index=wide.index)], axis=1)
    return out


def rate_kwargs_for_representation(representation: str) -> dict:
    if representation == "t1_r10_r21":
        return {"include_t1": True, "include_r10_r21": True}
    if representation == "t1_rate02":
        return {"include_t1": True, "include_rate02": True}
    if representation == "t1_ols":
        return {"include_t1": True, "include_beta1": True}
    raise ValueError(f"representação rate desconhecida: {representation!r}")


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
    if representation == "t1_r10_r21":
        return REPRESENTATION_TOKENS_R10_R21
    if representation == "t1_rate02":
        return REPRESENTATION_TOKENS_RATE02
    if representation == "t1_ols":
        return REPRESENTATION_TOKENS_OLS
    raise ValueError(f"representação delta desconhecida: {representation!r}")


def delta_kwargs_for_representation(representation: str) -> dict:
    if representation == "deltas_only":
        return {"include_t1": False, "delta_kind": "abs", "include_slope": False}
    if representation == "t1_deltas_rel":
        return {"include_t1": True, "delta_kind": "rel", "include_slope": True}
    if representation in ("t1_deltas", "t1_deltas_abs", "t1_d21_d32"):
        return {"include_t1": True, "delta_kind": "abs", "include_slope": False}
    if representation == "t1_d21":
        return {
            "include_t1": True,
            "delta_kind": "abs",
            "include_slope": False,
            "include_t3_deltas": False,
        }
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
    if modality in {"disp", "disp_ad", "disp_cnad"}:
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

    # Rate / OLS (revisor 2): V=[4000,4010,3940], t=[0,6.6,12.6] → β̂1 ≈ -4.66
    beta_ref = ols_slope_three(0.0, 6.6, 12.6, 4000.0, 4010.0, 3940.0)
    assert abs(beta_ref - (-370.0 / 79.44)) < 0.02, beta_ref
    times_demo = pd.DataFrame(
        {"t0": [0.0], "t1": [6.6], "t2": [12.6]},
        index=pd.Index(["p_demo"], name="ID_PT"),
    )
    wide_demo = pd.DataFrame(
        {
            "ID_PT": ["p_demo"],
            "GROUP": ["sMCI"],
            "SEX": [0],
            f"{roi}_L_T1_gm_norm": [4000.0],
            f"{roi}_L_T2_gm_norm": [4010.0],
            f"{roi}_L_T3_gm_norm": [3940.0],
        }
    )
    r_all = add_rate_columns(
        wide_demo, times_demo, roi,
        include_r10_r21=True, include_rate02=True, include_beta1=True,
    )
    assert abs(float(r_all[f"{roi}_L_R10_gm_norm"].iloc[0]) - (10.0 / 6.6)) < 1e-9
    assert abs(float(r_all[f"{roi}_L_R21_gm_norm"].iloc[0]) - (-70.0 / 6.0)) < 1e-9
    assert abs(float(r_all[f"{roi}_L_RATE02_gm_norm"].iloc[0]) - (-60.0 / 12.6)) < 1e-9
    assert abs(float(r_all[f"{roi}_L_BETA1_gm_norm"].iloc[0]) - beta_ref) < 1e-9
    assert feature_tokens_for_delta_representation("t1_r10_r21") == REPRESENTATION_TOKENS_R10_R21
    assert feature_tokens_for_delta_representation("t1_rate02") == REPRESENTATION_TOKENS_RATE02
    assert feature_tokens_for_delta_representation("t1_ols") == REPRESENTATION_TOKENS_OLS
    r10_cols = modality_wide_columns(
        r_all.columns, "vol", roi=roi, use_deltas=True,
        feature_tokens=REPRESENTATION_TOKENS_R10_R21,
    )
    assert set(r10_cols) == {
        f"{roi}_L_T1_gm_norm", f"{roi}_L_R10_gm_norm", f"{roi}_L_R21_gm_norm",
    }

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

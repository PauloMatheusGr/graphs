"""Representação T1 + deltas longitudinais relativos (D21, D31, SLOPE) a partir de wide T1/T2/T3."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ablation_prep import (
    DISP_PREFIX_DROP,
    DISP_STAT_DROP,
    ROI_FILTER_DEFAULT,
    SHAPE_RE,
    TEXTURE_RE,
    VOL_FEATURE_SUFFIXES,
    modality_wide_columns as modality_wide_columns_absolute,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

META_WIDE = frozenset({"ID_PT", "GROUP", "SEX", "y"})
DELTA_EPS = 1e-8
DELTA_TIME_TOKENS = ("D21", "D31", "SLOPE")
REPRESENTATION_TOKENS = ("T1", "D21", "D31", "SLOPE")
PROTOCOL_T1_DELTAS = "t1_deltas_rel"


def absolute_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(T[123])_(.+)$")


def representation_col_pat(roi: str = ROI_FILTER_DEFAULT) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_[LR]_(T1|D21|D31|SLOPE)_(.+)$")


def relative_delta(v_from: pd.Series, v_to: pd.Series, *, eps: float = DELTA_EPS) -> pd.Series:
    """Variação relativa (v_to − v_from) / v_from; NaN se |v_from| < eps."""
    v_from = pd.to_numeric(v_from, errors="coerce").astype(float)
    v_to = pd.to_numeric(v_to, errors="coerce").astype(float)
    diff = v_to - v_from
    denom = v_from.where(v_from.abs() >= eps)
    return diff / denom


def add_delta_columns(
    wide: pd.DataFrame,
    roi: str = ROI_FILTER_DEFAULT,
    *,
    include_t1: bool = True,
    include_absolute: bool = False,
) -> pd.DataFrame:
    """T1 + D21=(T2−T1)/T1, D31=(T3−T1)/T1, SLOPE=D31/2. Sem leakage entre pacientes."""
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
        d21 = relative_delta(t1, t2)
        d31 = relative_delta(t1, t3)
        delta_cols[f"{prefix}_D21_{feat}"] = d21
        delta_cols[f"{prefix}_D31_{feat}"] = d31
        delta_cols[f"{prefix}_SLOPE_{feat}"] = d31 / 2.0

    keep = [c for c in wide.columns if c in META_WIDE]
    if include_absolute:
        keep.extend(c for c in wide.columns if pat.match(c))
    elif include_t1:
        for times in groups.values():
            if "T1" in times:
                keep.append(times["T1"])

    out = wide[keep].copy()
    if delta_cols:
        out = pd.concat([out, pd.DataFrame(delta_cols, index=wide.index)], axis=1)
    return out


def _select_vol_delta(columns: Iterable[str], roi: str) -> list[str]:
    pat = representation_col_pat(roi)
    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and m.group(2) in VOL_FEATURE_SUFFIXES:
            out.append(col)
    return out


def _select_shape_delta(columns: Iterable[str], roi: str) -> list[str]:
    pat = representation_col_pat(roi)
    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and SHAPE_RE.match(m.group(2)):
            out.append(col)
    return out


def _select_texture_delta(columns: Iterable[str], roi: str) -> list[str]:
    pat = representation_col_pat(roi)
    return [c for c in columns if pat.match(c) and TEXTURE_RE.search(pat.match(c).group(2))]


def _select_disp_delta(columns: Iterable[str], roi: str) -> list[str]:
    pat = representation_col_pat(roi)

    def keep(feat: str) -> bool:
        if any(feat.startswith(p) for p in DISP_PREFIX_DROP):
            return False
        if any(feat.endswith(s) for s in DISP_STAT_DROP):
            return False
        return True

    out: list[str] = []
    for col in columns:
        m = pat.match(col)
        if m and keep(m.group(2)):
            out.append(col)
    return out


def modality_wide_columns(
    columns: list[str] | pd.Index,
    modality: str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    use_deltas: bool = False,
) -> list[str]:
    if not use_deltas:
        return modality_wide_columns_absolute(columns, modality, roi=roi)

    cols = list(columns)
    if modality == "vol":
        return _select_vol_delta(cols, roi)
    if modality == "shape":
        return _select_shape_delta(cols, roi)
    if modality == "texture":
        return _select_texture_delta(cols, roi)
    if modality == "disp":
        return _select_disp_delta(cols, roi)
    if modality == "all":
        out = _select_vol_delta(cols, roi)
        out += _select_shape_delta(cols, roi)
        out += _select_texture_delta(cols, roi)
        out += _select_disp_delta(cols, roi)
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
    out = add_delta_columns(wide, roi)
    assert f"{roi}_L_D21_gm_norm" in out.columns
    assert abs(float(out[f"{roi}_L_D21_gm_norm"].iloc[0]) - 0.2) < 1e-9
    assert abs(float(out[f"{roi}_L_D31_gm_norm"].iloc[0]) - 0.5) < 1e-9
    assert abs(float(out[f"{roi}_L_SLOPE_gm_norm"].iloc[0]) - 0.25) < 1e-9
    # absoluto seria D21=0.4 para T1=2.0; relativo = 0.2
    assert abs(float(out[f"{roi}_L_D21_gm_norm"].iloc[0]) - 0.4) > 0.1
    assert f"{roi}_L_T2_gm_norm" not in out.columns
    assert f"{roi}_L_T1_gm_norm" in out.columns
    tiny = relative_delta(pd.Series([1e-12]), pd.Series([1.0]))
    assert np.isnan(float(tiny.iloc[0]))
    n = len(modality_wide_columns(out.columns, "vol", roi=roi, use_deltas=True))
    assert n == 4  # L × (T1,D21,D31,SLOPE) × gm_norm
    print("ablation_deltas self-check ok")

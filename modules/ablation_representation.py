"""Protocolos de representação temporal: wide, t1_only, t1_deltas, deltas_only."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ablation_prep import ROI_FILTER_DEFAULT, modality_wide_columns

Representation = Literal["wide", "t1_only", "t1_deltas", "deltas_only", "t1_deltas_rel"]
REPRESENTATIONS: tuple[str, ...] = (
    "wide",
    "t1_only",
    "t1_deltas",
    "deltas_only",
    "t1_deltas_rel",
)
DELTA_REPRESENTATIONS = frozenset({"t1_deltas", "deltas_only", "t1_deltas_rel"})

RESULTS_ROOT_BY_PROTOCOL: dict[str, dict[str, str]] = {
    "abs": {
        "wide": "ablation_results",
        "t1_only": "ablation_results_t1_only",
        "t1_deltas": "ablation_results_deltas",
        "deltas_only": "ablation_results_deltas_only",
        "t1_deltas_rel": "ablation_results_deltas_rel",
    },
    "leaky": {
        "wide": "ablation_results_leaky",
        "t1_only": "ablation_results_leaky_t1_only",
        "t1_deltas": "ablation_results_leaky_deltas",
        "deltas_only": "ablation_results_leaky_deltas_only",
        "t1_deltas_rel": "ablation_results_leaky_deltas_rel",
    },
    "fusion": {
        "wide": "ablation_results_clinic_img",
        "t1_only": "ablation_results_clinic_img_t1_only",
        "t1_deltas": "ablation_results_clinic_img_deltas",
        "deltas_only": "ablation_results_clinic_img_deltas_only",
        "t1_deltas_rel": "ablation_results_clinic_img_deltas_rel",
    },
}


def parse_representation(value: str) -> Representation:
    v = value.strip().lower()
    if v not in REPRESENTATIONS:
        raise ValueError(f"representation inválida: {value!r} (use {' | '.join(REPRESENTATIONS)})")
    return v  # type: ignore[return-value]


def is_delta_representation(representation: str) -> bool:
    return representation in DELTA_REPRESENTATIONS


def apply_representation_wide(
    wide,
    representation: str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
):
    """Pós-pivot: deltas absolutos (default), dinâmica pura ou legado rel+SLOPE."""
    if not is_delta_representation(representation):
        return wide
    from ablation_deltas import add_delta_columns, delta_kwargs_for_representation

    return add_delta_columns(wide, roi, **delta_kwargs_for_representation(representation))


def feature_columns_for_representation(
    columns,
    modality: str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    representation: str = "wide",
) -> list[str]:
    if is_delta_representation(representation):
        from ablation_deltas import (
            feature_tokens_for_delta_representation,
            modality_wide_columns as modality_wide_columns_deltas,
        )

        return modality_wide_columns_deltas(
            columns,
            modality,
            roi=roi,
            use_deltas=True,
            feature_tokens=feature_tokens_for_delta_representation(representation),
        )
    if representation == "t1_only":
        return modality_wide_columns(columns, modality, roi=roi, timepoints=("T1",))
    return modality_wide_columns(columns, modality, roi=roi)


def resolve_stable_pool_min_timepoints(
    representation: str,
    value: int,
    *,
    log=None,
) -> int:
    """t1_only/deltas: filtro temporal ≥2 esvazia ou distorce o pool."""
    if representation in ("t1_only", *DELTA_REPRESENTATIONS) and value > 1:
        if log is not None:
            log.warning(
                "%s: stable-pool-min-timepoints %d → 0",
                representation,
                value,
            )
        return 0
    return value


def default_results_dir(
    base_dir: Path | str,
    modality: str,
    representation: str,
    *,
    protocol: str,
    results_dir: Path | str | None = None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    base = Path(base_dir)
    root_name = RESULTS_ROOT_BY_PROTOCOL[protocol].get(representation, "ablation_results")
    return base.parent.parent / root_name / modality


def default_fusion_results_dir(
    base_dir: Path | str,
    representation: str,
    *,
    results_dir: Path | str | None = None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    base = Path(base_dir)
    root_name = RESULTS_ROOT_BY_PROTOCOL["fusion"].get(
        representation, "ablation_results_clinic_img",
    )
    return base.parent.parent / root_name


if __name__ == "__main__":
    import pandas as pd

    roi = ROI_FILTER_DEFAULT
    wide = pd.DataFrame(
        {
            "ID_PT": ["p1"],
            "GROUP": ["sMCI"],
            "SEX": [0],
            f"{roi}_L_T1_gm_norm": [1.0],
            f"{roi}_L_T2_gm_norm": [1.1],
            f"{roi}_L_T3_gm_norm": [1.2],
        }
    )
    t1 = feature_columns_for_representation(
        wide.columns, "vol", roi=roi, representation="t1_only",
    )
    assert t1 == [f"{roi}_L_T1_gm_norm"]
    delta_wide = apply_representation_wide(wide, "t1_deltas", roi=roi)
    assert f"{roi}_L_D32_gm_norm" in delta_wide.columns
    dyn_wide = apply_representation_wide(wide, "deltas_only", roi=roi)
    assert f"{roi}_L_T1_gm_norm" not in dyn_wide.columns
    assert f"{roi}_L_D21_gm_norm" in dyn_wide.columns
    assert resolve_stable_pool_min_timepoints("t1_deltas", 2) == 0
    assert resolve_stable_pool_min_timepoints("deltas_only", 2) == 0
    assert resolve_stable_pool_min_timepoints("wide", 2) == 2
    print("ablation_representation self-check ok")

"""Protocolos de representação temporal: wide, t1_only, t1_deltas, deltas_only + fusion cross-mod."""

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

FUSION_MODALITIES: tuple[str, ...] = ("vol", "shape", "texture", "disp")
DEFAULT_FUSION_SPEC = "shape:t1_only,vol:deltas_only"
DEFAULT_LATE_FUSION_SPEC = "shape:t1_only,texture:t1_deltas"
FUSION_RESULTS_ROOT = "ablation_results_fusion"
LATE_FUSION_RESULTS_ROOT = "ablation_results_late_fusion"
FusionSlot = tuple[str, str]  # (modality, representation)

# Fingerprint curto: t1_shape__deltas_vol | t1_shape__t1_deltas_vol
_REP_SHORT: dict[str, str] = {
    "t1_only": "t1",
    "deltas_only": "deltas",
    "t1_deltas": "t1_deltas",
    "t1_deltas_rel": "t1_deltas_rel",
    "wide": "wide",
}

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


def parse_fusion_spec(value: str) -> tuple[FusionSlot, ...]:
    """Parse 'shape:t1_only,vol:deltas_only' → ((shape, t1_only), (vol, deltas_only))."""
    raw = (value or "").strip()
    if not raw:
        raise ValueError("fusion spec vazia")
    slots: list[FusionSlot] = []
    seen_mods: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"slot inválido {part!r}; use modality:representation "
                f"(ex. shape:t1_only,vol:deltas_only)"
            )
        mod, rep = part.split(":", 1)
        mod = mod.strip().lower()
        rep = parse_representation(rep.strip())
        if mod not in FUSION_MODALITIES:
            raise ValueError(
                f"modalidade fusion desconhecida: {mod!r} "
                f"(use {' | '.join(FUSION_MODALITIES)}; sem atalho 'all')"
            )
        if mod in seen_mods:
            raise ValueError(f"modalidade repetida no fusion: {mod!r}")
        seen_mods.add(mod)
        slots.append((mod, rep))
    if len(slots) < 2:
        raise ValueError("fusion exige ≥2 slots modality:rep")
    abs_delta = any(r in ("t1_deltas", "deltas_only") for _, r in slots)
    rel_delta = any(r == "t1_deltas_rel" for _, r in slots)
    if abs_delta and rel_delta:
        raise ValueError("fusion não mistura deltas abs (t1_deltas/deltas_only) com t1_deltas_rel")
    return tuple(slots)


def fusion_fingerprint(slots: tuple[FusionSlot, ...] | list[FusionSlot]) -> str:
    """shape:t1_only,vol:deltas_only → t1_shape__deltas_vol."""
    parts: list[str] = []
    for mod, rep in slots:
        short = _REP_SHORT.get(rep)
        if short is None:
            raise ValueError(f"representation sem short-name: {rep!r}")
        parts.append(f"{short}_{mod}")
    return "__".join(parts)


def fusion_label(slots: tuple[FusionSlot, ...] | list[FusionSlot]) -> str:
    return " ∪ ".join(f"{m}@{r}" for m, r in slots)


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


def apply_fusion_wide(
    wide,
    slots: tuple[FusionSlot, ...] | list[FusionSlot],
    *,
    roi: str = ROI_FILTER_DEFAULT,
):
    """Wide absoluto + colunas delta necessárias para todos os slots (include_absolute)."""
    reps = {r for _, r in slots}
    needs_abs_delta = bool(reps & {"t1_deltas", "deltas_only"})
    needs_rel_delta = "t1_deltas_rel" in reps
    if needs_abs_delta and needs_rel_delta:
        raise ValueError("fusion não mistura deltas abs com t1_deltas_rel")
    if needs_rel_delta:
        from ablation_deltas import add_delta_columns

        return add_delta_columns(
            wide, roi, include_t1=True, include_absolute=True,
            delta_kind="rel", include_slope=True,
        )
    if needs_abs_delta:
        from ablation_deltas import add_delta_columns

        return add_delta_columns(
            wide, roi, include_t1=True, include_absolute=True,
            delta_kind="abs", include_slope=False,
        )
    return wide


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


def feature_columns_for_fusion(
    columns,
    slots: tuple[FusionSlot, ...] | list[FusionSlot],
    *,
    roi: str = ROI_FILTER_DEFAULT,
) -> list[str]:
    out: list[str] = []
    for mod, rep in slots:
        out.extend(
            feature_columns_for_representation(columns, mod, roi=roi, representation=rep)
        )
    return list(dict.fromkeys(out))


def resolve_stable_pool_min_timepoints(
    representation: str,
    value: int,
    *,
    log=None,
) -> int:
    """t1_only/deltas/fusion: filtro temporal ≥2 esvazia ou distorce o pool."""
    if representation in ("t1_only", "fusion", *DELTA_REPRESENTATIONS) and value > 1:
        if log is not None:
            log.warning(
                "%s: stable-pool-min-timepoints %d → 0",
                representation,
                value,
            )
        return 0
    return value


def resolve_stable_pool_min_timepoints_for_fusion(
    slots: tuple[FusionSlot, ...] | list[FusionSlot],
    value: int,
    *,
    log=None,
) -> int:
    reps = {r for _, r in slots}
    if reps <= {"wide"}:
        return value
    return resolve_stable_pool_min_timepoints("fusion", value, log=log)


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
    """Clinic+img fusion roots (5_clinic_img.py; ex-5_baseline_comparison)."""
    if results_dir is not None:
        return Path(results_dir)
    base = Path(base_dir)
    root_name = RESULTS_ROOT_BY_PROTOCOL["fusion"].get(
        representation, "ablation_results_clinic_img",
    )
    return base.parent.parent / root_name


def default_crossmod_fusion_results_dir(
    base_dir: Path | str,
    slots: tuple[FusionSlot, ...] | list[FusionSlot],
    *,
    results_dir: Path | str | None = None,
) -> Path:
    """csvs/cohorts/{cohort}/ablation_results_fusion/{fingerprint}/."""
    if results_dir is not None:
        return Path(results_dir)
    base = Path(base_dir)
    return base.parent.parent / FUSION_RESULTS_ROOT / fusion_fingerprint(slots)


def default_late_fusion_results_dir(
    base_dir: Path | str,
    slots: tuple[FusionSlot, ...] | list[FusionSlot],
    *,
    results_dir: Path | str | None = None,
) -> Path:
    """csvs/cohorts/{cohort}/ablation_results_late_fusion/{fingerprint}/."""
    if results_dir is not None:
        return Path(results_dir)
    base = Path(base_dir)
    return base.parent.parent / LATE_FUSION_RESULTS_ROOT / fusion_fingerprint(slots)


def mono_results_dir_for_slot(
    base_dir: Path | str,
    slot: FusionSlot,
    *,
    protocol: str = "abs",
) -> Path:
    """Pasta mono-mod do slot (ex. …/ablation_results_t1_only/shape)."""
    mod, rep = slot
    return default_results_dir(base_dir, mod, rep, protocol=protocol)


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
            f"{roi}_L_T1_original_shape_Sphericity": [0.5],
            f"{roi}_L_T2_original_shape_Sphericity": [0.51],
            f"{roi}_L_T3_original_shape_Sphericity": [0.52],
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

    slots = parse_fusion_spec(DEFAULT_FUSION_SPEC)
    assert fusion_fingerprint(slots) == "t1_shape__deltas_vol"
    assert fusion_fingerprint(parse_fusion_spec("shape:t1_only,vol:t1_deltas")) == "t1_shape__t1_deltas_vol"
    fw = apply_fusion_wide(wide, slots, roi=roi)
    fcols = feature_columns_for_fusion(fw.columns, slots, roi=roi)
    assert f"{roi}_L_T1_original_shape_Sphericity" in fcols
    assert f"{roi}_L_D21_gm_norm" in fcols
    print("ablation_representation self-check ok")

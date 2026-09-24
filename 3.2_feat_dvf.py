#!/usr/bin/env python3
"""
DVF features (MBEC maps): mag / jac_det / strain_fro → CSV por âncora CN|AD.

Uso:
    python 3.2_feat_dvf.py                 # CN (default)
    python 3.2_feat_dvf.py --diag AD
    python 3.2_feat_dvf.py --self-check
    python 3.2_feat_dvf.py --diag AD --self-check

Pré-requisito: 3.1_feat_gen_dvf.py --diag {CN|AD}

Saídas:
  CN → features_displacement_v4.csv    | warps displacement_field_v3/
  AD → features_displacement_v4_ad.csv | warps displacement_field_v3_ad/

IO/ROI helpers: 3.2_feat_dvf_v2_old.py (lib arquivada, ainda importada).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import ants
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "3.2_feat_dvf_v2_old.py"
VALID_DIAG = ("CN", "AD")

_CURRENT: dict[str, np.ndarray] = {}
JAC_DET_PREFIX = "jac_det"
PAPER_MAP_PREFIXES = ("mag", JAC_DET_PREFIX, "strain_fro")
PAPER_MOMENT_KEYS = ("mean", "variance", "skewness", "kurtosis")

ROI_TABLE = (
    ("hippocampus", "L", 17),
    ("hippocampus", "R", 53),
    ("amygdala", "L", 18),
    ("amygdala", "R", 54),
    ("thalamus_proper", "L", 10),
    ("thalamus_proper", "R", 49),
    ("accumbens_area", "L", 26),
    ("accumbens_area", "R", 58),
    ("inf_lateral_ventricle", "L", 5),
    ("inf_lateral_ventricle", "R", 44),
    ("posterior_cingulate", "L", 1023),
    ("posterior_cingulate", "R", 2023),
    ("isthmus_cingulate", "L", 1010),
    ("isthmus_cingulate", "R", 2010),
    ("rostral_anterior_cingulate", "L", 1026),
    ("rostral_anterior_cingulate", "R", 2026),
    ("medial_orbitofrontal", "L", 1014),
    ("medial_orbitofrontal", "R", 2014),
    ("insula", "L", 1035),
    ("insula", "R", 2035),
)

# Filled by configure_anchor / _load_base
ANCHOR_DIAG = "CN"
base: ModuleType | None = None
_base_build_roi_feature_row = None


def _load_base() -> ModuleType:
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Lib DVF ausente: {BASE_SCRIPT} "
            "(esperado após arquivar 3.2_feat_dvf_v2.py → *_v2_old.py)"
        )
    spec = importlib.util.spec_from_file_location("feat_dvf_v2_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_anchor(diag: str) -> None:
    """Patch paths + ref_tag/template para CN ou AD."""
    global ANCHOR_DIAG, base, _base_build_roi_feature_row
    diag = str(diag).upper().strip()
    if diag not in VALID_DIAG:
        raise ValueError(f"diag={diag!r}; use {VALID_DIAG}")
    ANCHOR_DIAG = diag

    if base is None:
        base = _load_base()
        _base_build_roi_feature_row = base._build_roi_feature_row

    base.ROI_TABLE = ROI_TABLE

    if diag == "CN":
        base.WARPS_DIR = "./images/displacement_field_v3"
        base.OUT_CSV = f"{base.COHORT_DIR}/features_displacement_v4.csv"
    else:
        base.WARPS_DIR = "./images/displacement_field_v3_ad"
        base.OUT_CSV = f"{base.COHORT_DIR}/features_displacement_v4_ad.csv"
    base.RUN_DIR = str(Path(base.WARPS_DIR) / "features_v4" / base.COHORT)

    def _ref_tag(sex: str, age_range: str) -> str:
        return f"{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}"

    def template_path_for(sex: str, age_range: str) -> str:
        return os.path.join(
            base.GROUPWISE_DIR,
            f"groupwise_DIAG-{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz",
        )

    base._ref_tag = _ref_tag
    base.template_path_for = template_path_for


def nonlinear_warp_list_for(
    warps_dir: str,
    img_id: str,
    sex: str,
    age_range: str,
) -> list[str]:
    assert base is not None
    return [base.warp_path_for(warps_dir, img_id, sex, age_range)]


def _geometry_differences(
    field: ants.ANTsImage,
    template: ants.ANTsImage,
    *,
    atol: float = 1e-5,
) -> list[str]:
    differences: list[str] = []
    if tuple(field.shape) != tuple(template.shape):
        differences.append(f"shape={field.shape} != {template.shape}")
    if not np.allclose(field.spacing, template.spacing, atol=atol, rtol=0):
        differences.append(f"spacing={field.spacing} != {template.spacing}")
    if not np.allclose(field.origin, template.origin, atol=atol, rtol=0):
        differences.append(f"origin={field.origin} != {template.origin}")
    if not np.allclose(field.direction, template.direction, atol=atol, rtol=0):
        differences.append("direction diferente")
    return differences


def load_nonlinear_displacement(
    domain_img: ants.ANTsImage,
    warp_paths: list[str],
) -> ants.ANTsImage:
    if len(warp_paths) != 1:
        raise ValueError(
            "DVF espera exatamente um *_1Warp.nii.gz; "
            f"recebeu {warp_paths!r}"
        )
    warp_path = warp_paths[0]
    delta = ants.image_read(warp_path)
    arr = delta.numpy()
    if domain_img.dimension != 3 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Campo inválido em {warp_path}: esperado (X,Y,Z,3), "
            f"obtido shape={arr.shape}"
        )
    differences = _geometry_differences(delta, domain_img)
    if differences:
        raise ValueError(
            f"Geometria do warp incompatível com template em {warp_path}: "
            + "; ".join(differences)
        )
    return delta


def _append_map_stats(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    assert base is not None
    percentiles = base._stats_percentiles(values)
    row.update(base._feature_stats_columns(prefix, percentiles))
    moments = base._stats_moments(values)
    for key in ("variance", "skewness", "kurtosis"):
        row[f"{prefix}_{key}"] = moments[key]


def compute_unitary_scalar_arrays(
    domain_img: ants.ANTsImage,
    warp_paths: list[str],
):
    assert base is not None
    delta = load_nonlinear_displacement(domain_img, warp_paths)

    jac_det = ants.create_jacobian_determinant_image(
        domain_img, delta, do_log=False
    )
    logjac = ants.create_jacobian_determinant_image(
        domain_img, delta, do_log=True
    )
    mag = base.field_magnitude(delta)
    div = base.field_divergence(delta)
    ux_img, uy_img, uz_img = base.field_components(delta)
    curlmag = base.field_curl_magnitude(delta)

    arr = delta.numpy().astype(np.float32, copy=False)
    spacing = tuple(map(float, delta.spacing))
    strain_maps = base.strain_scalar_maps_from_displacement(
        arr[..., 0], arr[..., 1], arr[..., 2], spacing
    )
    strain_inf = base._infinitesimal_strain_fro_map(
        arr[..., 0], arr[..., 1], arr[..., 2], spacing
    )

    _CURRENT.clear()
    _CURRENT["jac_det"] = jac_det.numpy().astype(np.float32)
    _CURRENT["mag"] = mag.numpy().astype(np.float32)
    _CURRENT["strain_inf_fro"] = strain_inf

    return (
        logjac.numpy().astype(np.float32),
        mag.numpy().astype(np.float32),
        div.numpy().astype(np.float32),
        ux_img.numpy().astype(np.float32),
        uy_img.numpy().astype(np.float32),
        uz_img.numpy().astype(np.float32),
        curlmag.numpy().astype(np.float32),
        strain_maps,
        logjac,
    )


def build_roi_feature_row(**kwargs) -> dict[str, object]:
    assert _base_build_roi_feature_row is not None and base is not None
    row = _base_build_roi_feature_row(**kwargs)
    mask = kwargs["roi_mask"]
    if "jac_det" not in _CURRENT:
        raise RuntimeError("jac_det ausente: compute_unitary_scalar_arrays não rodou")
    _append_map_stats(row, JAC_DET_PREFIX, _CURRENT["jac_det"][mask])
    _append_map_stats(row, "strain_fro", _CURRENT["strain_inf_fro"][mask])
    mag_moments = base._stats_moments(_CURRENT["mag"][mask])
    for key in ("variance", "skewness", "kurtosis"):
        row[f"mag_{key}"] = mag_moments[key]
    return row


def persist_run_metadata(
    *,
    run_meta_path: str,
    out_csv: str,
    done_keys_path: str,
    images_csv: str,
) -> None:
    assert base is not None
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": f"dvf_v4_mbec_maps_{ANCHOR_DIAG.lower()}_only",
        "inputs": {
            "images_csv": str(images_csv),
            "warps_dir": str(base.WARPS_DIR),
            "warp_pattern": "*_1Warp.nii.gz",
            "anchor": ANCHOR_DIAG,
        },
        "outputs": {
            "out_csv": str(out_csv),
            "done_keys_path": str(done_keys_path),
        },
        "paper_alignment": {
            "reference": "MBEC-D-26-01909 (Andrade et al.) §§3.5–3.6",
            "maps": {
                "D": "mag = ||u||_2",
                "V": "jac_det = det(I + grad(u))",
                "S": "strain_fro = ||eps||_F, eps = 0.5 (H + H^T)",
            },
            "roi_stats": list(PAPER_MOMENT_KEYS),
        },
        "conventions": {
            "domain": f"{ANCHOR_DIAG} template (fixed)",
            "displacement": "forward SyN nonlinear warp only; affine excluded",
            "labels_to_template": "[forward warp, affine], nearest-neighbor",
            "n_rois": len(ROI_TABLE),
            "rois": [f"{name}_{side}" for name, side, _ in ROI_TABLE],
            "paper_prefixes": list(PAPER_MAP_PREFIXES),
        },
    }
    Path(run_meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(run_meta_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_extract() -> None:
    assert base is not None
    base.inv_list_for = nonlinear_warp_list_for
    base.compute_unitary_scalar_arrays = compute_unitary_scalar_arrays
    base._build_roi_feature_row = build_roi_feature_row
    base.persist_run_metadata = persist_run_metadata
    print(
        f"[INFO] ANCHOR={ANCHOR_DIAG} warps={base.WARPS_DIR} out={base.OUT_CSV}",
        flush=True,
    )
    base.main()


def self_check(diag: str = "CN") -> None:
    configure_anchor(diag)
    assert base is not None

    template = ants.from_numpy(np.zeros((3, 3, 3), dtype=np.float32))
    field = ants.from_numpy(
        np.zeros((3, 3, 3, 3), dtype=np.float32),
        has_components=True,
    )
    assert not _geometry_differences(field, template)

    shifted = ants.from_numpy(
        np.zeros((3, 3, 3, 3), dtype=np.float32),
        origin=(1.0, 0.0, 0.0),
        has_components=True,
    )
    assert any("origin=" in item for item in _geometry_differences(shifted, template))

    paths = nonlinear_warp_list_for(base.WARPS_DIR, "img", "F", "60-69")
    assert len(paths) == 1 and paths[0].endswith("_1Warp.nii.gz")
    assert f"{diag}_SEX-F_AGE-60-69_1Warp" in paths[0]
    assert "DIAG-" + diag in base.template_path_for("F", "60-69")

    grid_x = np.indices((3, 3, 3), dtype=np.float32)[0]
    zeros = np.zeros_like(grid_x)
    strain = base._infinitesimal_strain_fro_map(
        grid_x, zeros, zeros, (1.0, 1.0, 1.0)
    )
    np.testing.assert_allclose(strain, 1.0, atol=1e-6)

    delta = ants.from_numpy(
        np.stack([grid_x, zeros, zeros], axis=-1).astype(np.float32),
        has_components=True,
    )
    jac = ants.create_jacobian_determinant_image(template, delta, do_log=False)
    logjac = ants.create_jacobian_determinant_image(template, delta, do_log=True)
    jac_arr = jac.numpy()
    log_arr = logjac.numpy()
    assert float(np.nanmean(jac_arr)) > 1.0
    np.testing.assert_allclose(
        np.log(np.clip(jac_arr, 1e-6, None)),
        log_arr,
        atol=1e-4,
        rtol=1e-4,
    )

    assert len(ROI_TABLE) == 20
    assert PAPER_MAP_PREFIXES == ("mag", "jac_det", "strain_fro")
    if diag == "CN":
        assert base.OUT_CSV.endswith("features_displacement_v4.csv")
    else:
        assert base.OUT_CSV.endswith("features_displacement_v4_ad.csv")
    print(f"ok: 3.2_feat_dvf --diag {diag} (maps D/V/S, warp-only)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DVF features CN|AD (v4 maps)")
    p.add_argument("--diag", default="CN", choices=VALID_DIAG)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    if args.self_check:
        self_check(args.diag)
        return
    configure_anchor(args.diag)
    run_extract()


if __name__ == "__main__":
    main(sys.argv[1:])

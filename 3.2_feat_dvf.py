#!/usr/bin/env python3
"""
DVF features (MBEC): mag / jac_det / strain_fro por âncora CN|AD.

Uso:
    python 3.2_feat_dvf.py                 # CN (default)
    python 3.2_feat_dvf.py --diag AD
    python 3.2_feat_dvf.py --self-check

Pré-requisito: 3.1_feat_gen_dvf.py --diag {CN|AD}

Saídas:
  CN → features_displacement_v4.csv    | warps displacement_field_v3/
  AD → features_displacement_v4_ad.csv | warps displacement_field_v3_ad/

Modo --src oasis (3.1 --src oasis): domínio = imagem clínica, labels/máscara lidas direto.
  ROIs hipocampo núcleo / d2 / d4 / d8 (dilatação EDT em mm) / shell4 = d4 sem núcleo.
  Mapas jac_det, logjac, mag, strain_fro (ε infinitesimal).
  Sinal: fixed=sujeito, então jac_det > 1 = sujeito menor que o template (atrofia).
  CN → features_displacement_oasis_cn.csv | AD → features_displacement_oasis_ad.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
import oasis_refs  # noqa: E402

# =========================
# CONFIG
# =========================

COHORT = "all_population"
COHORT_DIR = f"csvs/cohorts/{COHORT}"
IMAGES_CSV = f"{COHORT_DIR}/all_population_True.csv"

GROUPWISE_DIR = "images/groupwise/references"
CLINIC_DIR = "./images/resampled_1.0mm"
REGIONS_DIR = "./images/regions"
BRAIN_MASK_DIR = "./images/brain_mask"

RESUME = True
LOG_EVERY = 1
POINT_CHUNK = int(os.environ.get("DISPLACEMENT_POINT_CHUNK", "400000"))
VALID_DIAG = ("CN", "AD")

# Set by configure_anchor()
ANCHOR_DIAG = "CN"
WARPS_DIR = "./images/displacement_field_v3"
OUT_CSV = f"{COHORT_DIR}/features_displacement_v4.csv"
RUN_DIR = os.path.join(WARPS_DIR, "features_v4", COHORT)

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

# Escalares derivados do tensor de strain (Green–Lagrange) por voxel — legado
STRAIN_SCALAR_NAMES = (
    "strain_trace",
    "strain_det",
    "strain_fro",
    "strain_vol",
    "strain_dev_norm",
    "strain_shear_max",
    "strain_l1",
    "strain_l2",
    "strain_l3",
    "strain_shear_ratio",
    "strain_shear_energy",
)


def load_done_keys(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                done.add(s)
    return done


def append_done_key(path: str, key: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(key + "\n")



def get_age_range(age):
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

def baseline_ref_tag_by_pt(df_images: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """(SEX, age_range) do baseline por ID_PT — bate com nomes dos warps v2."""
    df = df_images.copy()
    df["MRI_DATE"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    df = df.sort_values(["ID_PT", "MRI_DATE", "ID_IMG"])
    out: dict[str, tuple[str, str]] = {}
    for id_pt, g in df.groupby("ID_PT", sort=False):
        r0 = g.iloc[0]
        sex = str(r0["SEX"]).upper().strip()
        age_range = get_age_range(r0["AGE"])
        out[str(id_pt)] = (sex, age_range)
    return out


def template_path_for(sex: str, age_range: str) -> str:
    return os.path.join(
        GROUPWISE_DIR,
        f"groupwise_DIAG-{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz",
    )


def subject_path_for(clinic_dir: str, img_id: str) -> str:
    return os.path.join(
        clinic_dir, f"{img_id}_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
    )


def _ref_tag(sex: str, age_range: str) -> str:
    return f"{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}"


def inv_warp_path_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> str:
    return os.path.join(warps_dir, f"{img_id}_{_ref_tag(sex, age_range)}_1InverseWarp.nii.gz")


def warp_path_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> str:
    return os.path.join(warps_dir, f"{img_id}_{_ref_tag(sex, age_range)}_1Warp.nii.gz")


def affine_path_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> str:
    return os.path.join(warps_dir, f"{img_id}_{_ref_tag(sex, age_range)}_0GenericAffine.mat")


def inv_list_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> list[str]:
    """
    v2: fixed=template → inv maps template → sujeito.
    Mesma ordem de ficheiros que v1: [affine, InverseWarp].
    """
    return [
        affine_path_for(warps_dir, img_id, sex, age_range),
        inv_warp_path_for(warps_dir, img_id, sex, age_range),
    ]


def fwd_list_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> list[str]:
    """ANTs: moving(sujeito) → fixed(template): [Warp, Affine]."""
    return [
        warp_path_for(warps_dir, img_id, sex, age_range),
        affine_path_for(warps_dir, img_id, sex, age_range),
    ]


def warp_labelmap_to_template(
    label_path: str, template: ants.ANTsImage, fwd_list: list[str]
) -> np.ndarray:
    lab = ants.image_read(label_path)
    warped = ants.apply_transforms(
        fixed=template,
        moving=lab,
        transformlist=fwd_list,
        interpolator="nearestNeighbor",
    )
    return warped.numpy().astype(np.int32)


def warp_mask_to_template(
    mask_path: str, template: ants.ANTsImage, fwd_list: list[str]
) -> np.ndarray:
    m = ants.image_read(mask_path)
    warped = ants.apply_transforms(
        fixed=template,
        moving=m,
        transformlist=fwd_list,
        interpolator="nearestNeighbor",
    )
    return warped.numpy() > 0.5

def _feature_stats_columns(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    """Legado: mean, std, percentis."""
    return {
        f"{prefix}_n": stats["n"],
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_p05": stats["p05"],
        f"{prefix}_p50": stats["p50"],
        f"{prefix}_p95": stats["p95"],
    }


def _stats_moments(x: np.ndarray) -> dict[str, float]:
    """
    Momentos estatísticos ROI (Seção 3.5.4 / 3.6 do artigo):
      mean, variance (amostral), skewness, kurtosis (Fisher, normal → 0).
    """
    a = x.astype(np.float64, copy=False)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "variance": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }
    return {
        "n": float(a.size),
        "mean": float(np.mean(a)),
        "variance": float(np.var(a, ddof=1)),
        "skewness": float(skew(a, bias=False)),
        "kurtosis": float(kurtosis(a, bias=False, fisher=True)),
    }


def _stats_percentiles(x: np.ndarray) -> dict[str, float]:
    """Legado: mean, std, p05, p50, p95."""
    a = x.astype(np.float64, copy=False)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": float(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=0)),
        "p05": float(np.quantile(a, 0.05)),
        "p50": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
    }


def _append_article_moment_extras(
    row: dict[str, object],
    prefix: str,
    arr: np.ndarray,
    roi_mask: np.ndarray,
) -> None:
    """
    Acrescenta variance, skewness e kurtosis do artigo (Seção 3.5).

    mean/n/std/percentis vêm do bloco legado no mesmo prefixo quando aplicável.
    Para strain_fro, o legado usa Green–Lagrange e o artigo usa ε infinitesimal.
    """
    stats = _stats_moments(arr[roi_mask])
    for key in ("variance", "skewness", "kurtosis"):
        row[f"{prefix}_{key}"] = stats[key]


def _build_roi_feature_row(
    *,
    id_pt: str,
    img_id: str,
    meta_row,
    sex: str,
    age_range: str,
    roi: str,
    side: str,
    label: int,
    roi_mask: np.ndarray,
    refimg: ants.ANTsImage,
    lj: np.ndarray,
    m: np.ndarray,
    dvg: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    curl: np.ndarray,
    strain_maps: dict[str, np.ndarray],
    strain_inf_fro: np.ndarray,
) -> dict[str, object]:
    cx, cy, cz = _centroid_physical(roi_mask, refimg)

    row: dict[str, object] = {
        "ID_PT": id_pt,
        "ID_IMG": img_id,
        "DIAG": str(getattr(meta_row, "DIAG", "")),
        "GROUP": str(getattr(meta_row, "GROUP", "")),
        "SEX": str(getattr(meta_row, "SEX", "")),
        "AGE": float(getattr(meta_row, "AGE", np.nan)),
        "MRI_DATE": str(getattr(meta_row, "MRI_DATE", "")),
        "ref_tag": _ref_tag(sex, age_range),
        "roi": str(roi),
        "side": str(side),
        "label": str(label),
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "centroid_z": float(cz),
    }

    s_lj = _stats_percentiles(lj[roi_mask])
    s_m = _stats_percentiles(m[roi_mask])
    s_div = _stats_percentiles(dvg[roi_mask])
    s_ux = _stats_percentiles(ux[roi_mask])
    s_uy = _stats_percentiles(uy[roi_mask])
    s_uz = _stats_percentiles(uz[roi_mask])
    s_curl = _stats_percentiles(curl[roi_mask])

    row.update(
        {
            "logjac_n": s_lj["n"],
            "logjac_mean": s_lj["mean"],
            "logjac_std": s_lj["std"],
            "logjac_p05": s_lj["p05"],
            "logjac_p50": s_lj["p50"],
            "logjac_p95": s_lj["p95"],
            "mag_n": s_m["n"],
            "mag_mean": s_m["mean"],
            "mag_std": s_m["std"],
            "mag_p05": s_m["p05"],
            "mag_p50": s_m["p50"],
            "mag_p95": s_m["p95"],
            "div_n": s_div["n"],
            "div_mean": s_div["mean"],
            "div_std": s_div["std"],
            "div_p05": s_div["p05"],
            "div_p50": s_div["p50"],
            "div_p95": s_div["p95"],
            "ux_n": s_ux["n"],
            "ux_mean": s_ux["mean"],
            "ux_std": s_ux["std"],
            "ux_p05": s_ux["p05"],
            "ux_p50": s_ux["p50"],
            "ux_p95": s_ux["p95"],
            "uy_n": s_uy["n"],
            "uy_mean": s_uy["mean"],
            "uy_std": s_uy["std"],
            "uy_p05": s_uy["p05"],
            "uy_p50": s_uy["p50"],
            "uy_p95": s_uy["p95"],
            "uz_n": s_uz["n"],
            "uz_mean": s_uz["mean"],
            "uz_std": s_uz["std"],
            "uz_p05": s_uz["p05"],
            "uz_p50": s_uz["p50"],
            "uz_p95": s_uz["p95"],
            "curlmag_n": s_curl["n"],
            "curlmag_mean": s_curl["mean"],
            "curlmag_std": s_curl["std"],
            "curlmag_p05": s_curl["p05"],
            "curlmag_p50": s_curl["p50"],
            "curlmag_p95": s_curl["p95"],
        }
    )

    for strain_name in STRAIN_SCALAR_NAMES:
        row.update(
            _feature_stats_columns(
                strain_name, _stats_percentiles(strain_maps[strain_name][roi_mask])
            )
        )

    _append_article_moment_extras(row, "logjac", lj, roi_mask)
    _append_article_moment_extras(row, "mag", m, roi_mask)
    _append_article_moment_extras(row, "strain_fro", strain_inf_fro, roi_mask)

    # v4 paper maps: V=jac_det; S=strain infinitesimal (overwrite GL mix)
    if "jac_det" not in _CURRENT:
        raise RuntimeError("jac_det ausente: compute_unitary_scalar_arrays não rodou")
    for prefix, arr in (
        (JAC_DET_PREFIX, _CURRENT["jac_det"][roi_mask]),
        ("strain_fro", _CURRENT["strain_inf_fro"][roi_mask]),
    ):
        pct = _stats_percentiles(arr)
        row.update(_feature_stats_columns(prefix, pct))
        mom = _stats_moments(arr)
        for key in ("variance", "skewness", "kurtosis"):
            row[f"{prefix}_{key}"] = mom[key]
    mag_mom = _stats_moments(_CURRENT["mag"][roi_mask])
    for key in ("variance", "skewness", "kurtosis"):
        row[f"mag_{key}"] = mag_mom[key]

    return row


def _centroid_physical(mask: np.ndarray, ref_img: ants.ANTsImage) -> tuple[float, float, float]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    sp = np.array(ref_img.spacing, dtype=np.float64)
    org = np.array(ref_img.origin, dtype=np.float64)
    d = np.array(ref_img.direction, dtype=np.float64).reshape(3, 3)

    ijk = idx.astype(np.float64) + 0.5
    phys = (ijk * sp) @ d.T + org
    c = phys.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))

def field_magnitude(field: ants.ANTsImage) -> ants.ANTsImage:
    arr = field.numpy()
    mag = np.sqrt(np.sum(arr * arr, axis=-1))
    return ants.from_numpy(
        mag.astype(np.float32),
        origin=field.origin,
        spacing=field.spacing,
        direction=field.direction,
    )

def field_divergence(field: ants.ANTsImage) -> ants.ANTsImage:
    arr = field.numpy().astype(np.float32, copy=False)
    sx, sy, sz = map(float, field.spacing)
    dux_dx, dux_dy, dux_dz = np.gradient(arr[..., 0], sx, sy, sz, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(arr[..., 1], sx, sy, sz, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(arr[..., 2], sx, sy, sz, edge_order=1)
    div = dux_dx + duy_dy + duz_dz
    return ants.from_numpy(
        div.astype(np.float32),
        origin=field.origin,
        spacing=field.spacing,
        direction=field.direction,
    )

def field_components(field: ants.ANTsImage):
    arr = field.numpy().astype(np.float32, copy=False)
    ux = ants.from_numpy(arr[..., 0], origin=field.origin, spacing=field.spacing, direction=field.direction)
    uy = ants.from_numpy(arr[..., 1], origin=field.origin, spacing=field.spacing, direction=field.direction)
    uz = ants.from_numpy(arr[..., 2], origin=field.origin, spacing=field.spacing, direction=field.direction)
    return ux, uy, uz

def _deformation_gradient_from_components(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """H[i,j] = d(u_i)/d(x_j) com u = (ux, uy, uz)."""
    sx, sy, sz = spacing
    dux_dx, dux_dy, dux_dz = np.gradient(ux, sx, sy, sz, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(uy, sx, sy, sz, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(uz, sx, sy, sz, edge_order=1)
    return np.stack(
        [
            np.stack([dux_dx, dux_dy, dux_dz], axis=-1),
            np.stack([duy_dx, duy_dy, duy_dz], axis=-1),
            np.stack([duz_dx, duz_dy, duz_dz], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32, copy=False)


def _infinitesimal_strain_tensor(deformation_gradient: np.ndarray) -> np.ndarray:
    """ε = 0.5 * (H + H^T), H = grad(u) — strain infinitesimal do artigo (Eq. 6)."""
    return (0.5 * (deformation_gradient + np.swapaxes(deformation_gradient, -1, -2))).astype(
        np.float32, copy=False
    )


def _infinitesimal_strain_fro_map(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Mapa escalar S(x) = ||ε(x)||_F (Eq. 7), com ε infinitesimal."""
    h = _deformation_gradient_from_components(ux, uy, uz, spacing)
    eps = _infinitesimal_strain_tensor(h)
    return np.linalg.norm(eps, axis=(-2, -1)).astype(np.float32, copy=False)


def _green_lagrange_strain(deformation_gradient: np.ndarray) -> np.ndarray:
    """E = 0.5 * (F^T F - I), F = I + grad(u)."""
    eye = np.eye(3, dtype=np.float32)
    f = eye + deformation_gradient
    ft_f = np.einsum("...ki,...kj->...ij", f, f)
    return (0.5 * (ft_f - eye)).astype(np.float32, copy=False)


def strain_scalar_maps_from_displacement(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, spacing: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    """
    Invariantes escalares do tensor de Green–Lagrange no domínio da imagem clínica.
    Convenção alinhada ao itk.StrainImageFilter (GREENLAGRANGIAN) do registration-mni.ipynb.
    """
    h = _deformation_gradient_from_components(ux, uy, uz, spacing)
    e = _green_lagrange_strain(h)
    trace = np.trace(e, axis1=-2, axis2=-1)
    fro = np.linalg.norm(e, axis=(-2, -1))
    vol = (trace / 3.0).astype(np.float32, copy=False)
    dev = e - (vol[..., np.newaxis, np.newaxis] * np.eye(3, dtype=np.float32))
    dev_norm = np.linalg.norm(dev, axis=(-2, -1)).astype(np.float32, copy=False)
    eig = np.linalg.eigvalsh(e.astype(np.float64)).astype(np.float32)
    shear_max = (0.5 * (eig[..., 2] - eig[..., 0])).astype(np.float32, copy=False)
    fro_safe = fro + np.float32(1e-6)
    shear_ratio = np.where(fro > 0, dev_norm / fro_safe, 0.0).astype(np.float32, copy=False)
    shear_energy = (0.5 * dev_norm * dev_norm).astype(np.float32, copy=False)
    return {
        "strain_trace": trace.astype(np.float32, copy=False),
        "strain_det": np.linalg.det(e.astype(np.float64)).astype(np.float32),
        "strain_fro": fro.astype(np.float32, copy=False),
        "strain_vol": vol,
        "strain_dev_norm": dev_norm,
        "strain_shear_max": shear_max,
        "strain_l1": eig[..., 0],
        "strain_l2": eig[..., 1],
        "strain_l3": eig[..., 2],
        "strain_shear_ratio": shear_ratio,
        "strain_shear_energy": shear_energy,
    }


def field_curl_magnitude(field: ants.ANTsImage) -> ants.ANTsImage:
    arr = field.numpy().astype(np.float32, copy=False)
    sx, sy, sz = map(float, field.spacing)
    dux_dx, dux_dy, dux_dz = np.gradient(arr[..., 0], sx, sy, sz, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(arr[..., 1], sx, sy, sz, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(arr[..., 2], sx, sy, sz, edge_order=1)
    cx = duz_dy - duy_dz
    cy = dux_dz - duz_dx
    cz = duy_dx - dux_dy
    cmag = np.sqrt(cx * cx + cy * cy + cz * cz)
    return ants.from_numpy(
        cmag.astype(np.float32),
        origin=field.origin,
        spacing=field.spacing,
        direction=field.direction,
    )

def _load_and_resample_labelmap(path: str, target: ants.ANTsImage) -> np.ndarray:
    lab = ants.image_read(path)
    lab = ants.resample_image_to_target(lab, target, interp_type="nearestNeighbor")
    return lab.numpy().astype(np.int32)

def _load_and_resample_mask(path: str, target: ants.ANTsImage) -> np.ndarray:
    m = ants.image_read(path)
    m = ants.resample_image_to_target(m, target, interp_type="nearestNeighbor")
    return (m.numpy() > 0.5)

def index_grid_to_physical_points(domain_img: ants.ANTsImage) -> np.ndarray:
    shape = domain_img.shape
    dim = domain_img.dimension
    sp = np.array(domain_img.spacing, dtype=np.float64)
    org = np.array(domain_img.origin, dtype=np.float64)
    d = np.array(domain_img.direction, dtype=np.float64).reshape(dim, dim)
    grids = [np.arange(shape[i], dtype=np.float64) + 0.5 for i in range(dim)]
    idx = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, dim)
    scaled = idx * sp.reshape(1, dim)
    pts = (scaled @ d.T) + org.reshape(1, dim)
    return pts

def dataframe_points_xyz(pts_np: np.ndarray) -> pd.DataFrame:
    if pts_np.shape[1] == 3:
        return pd.DataFrame({"x": pts_np[:, 0], "y": pts_np[:, 1], "z": pts_np[:, 2]})
    raise ValueError("Esperado array (N, 3)")

def displacement_field_from_inv_list(domain_img: ants.ANTsImage, inv_list: list[str]) -> ants.ANTsImage:
    """
    Displacement (has_components=True) no DOMÍNIO do template (fixed v2):
    aplica inv_list (template → sujeito) aos pontos do template.
    Evita passar *_1Warp.nii.gz directo a create_jacobian_determinant_image.
    """
    if domain_img.dimension != 3:
        raise NotImplementedError("Apenas imagens 3D.")

    shape = domain_img.shape
    dim = domain_img.dimension
    n_vox = int(np.prod(shape))
    pts_flat = index_grid_to_physical_points(domain_img)
    disp = np.zeros((n_vox, dim), dtype=np.float64)

    start = 0
    while start < n_vox:
        end = min(start + POINT_CHUNK, n_vox)
        block = pts_flat[start:end]
        df_in = dataframe_points_xyz(block)
        df_out = ants.apply_transforms_to_points(3, df_in.copy(), inv_list)
        delta = df_out[["x", "y", "z"]].to_numpy(dtype=np.float64) - block
        disp[start:end, :] = delta
        start = end

    vec = disp.reshape(*(shape + (dim,)))
    return ants.from_numpy(
        vec.astype(np.float32),
        origin=domain_img.origin,
        spacing=domain_img.spacing,
        direction=domain_img.direction,
        has_components=True,
    )

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
    """Lê *_1Warp.nii.gz; falha se geometria ≠ template."""
    if len(warp_paths) != 1:
        raise ValueError(
            f"DVF espera exatamente um *_1Warp.nii.gz; recebeu {warp_paths!r}"
        )
    warp_path = warp_paths[0]
    delta = ants.image_read(warp_path)
    arr = delta.numpy()
    if domain_img.dimension != 3 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"Campo inválido em {warp_path}: esperado (X,Y,Z,3), obtido {arr.shape}"
        )
    differences = _geometry_differences(delta, domain_img)
    if differences:
        raise ValueError(
            f"Geometria do warp incompatível com template em {warp_path}: "
            + "; ".join(differences)
        )
    return delta


def nonlinear_warp_list_for(
    warps_dir: str, img_id: str, sex: str, age_range: str
) -> list[str]:
    return [warp_path_for(warps_dir, img_id, sex, age_range)]


def compute_unitary_scalar_arrays(domain_img: ants.ANTsImage, warp_paths: list[str]):
    """Mapas no template; displacement = só forward SyN (*_1Warp)."""
    delta = load_nonlinear_displacement(domain_img, warp_paths)
    jac_det = ants.create_jacobian_determinant_image(domain_img, delta, do_log=False)
    logjac = ants.create_jacobian_determinant_image(domain_img, delta, do_log=True)
    mag = field_magnitude(delta)
    div = field_divergence(delta)
    ux, uy, uz = field_components(delta)
    curlmag = field_curl_magnitude(delta)
    arr = delta.numpy().astype(np.float32, copy=False)
    spacing = tuple(map(float, delta.spacing))
    strain_maps = strain_scalar_maps_from_displacement(
        arr[..., 0], arr[..., 1], arr[..., 2], spacing
    )
    strain_inf = _infinitesimal_strain_fro_map(
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
        ux.numpy().astype(np.float32),
        uy.numpy().astype(np.float32),
        uz.numpy().astype(np.float32),
        curlmag.numpy().astype(np.float32),
        strain_maps,
        logjac,
    )

def append_csv(df: pd.DataFrame, out_csv_path: str) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    exists = os.path.isfile(out_csv_path) and os.path.getsize(out_csv_path) > 0
    df.to_csv(out_csv_path, mode="a", header=not exists, index=False, na_rep="NaN")


def configure_anchor(diag: str) -> None:
    global ANCHOR_DIAG, WARPS_DIR, OUT_CSV, RUN_DIR
    diag = str(diag).upper().strip()
    if diag not in VALID_DIAG:
        raise ValueError(f"diag={diag!r}; use {VALID_DIAG}")
    ANCHOR_DIAG = diag
    if diag == "CN":
        WARPS_DIR = "./images/displacement_field_v3"
        OUT_CSV = f"{COHORT_DIR}/features_displacement_v4.csv"
    else:
        WARPS_DIR = "./images/displacement_field_v3_ad"
        OUT_CSV = f"{COHORT_DIR}/features_displacement_v4_ad.csv"
    RUN_DIR = os.path.join(WARPS_DIR, "features_v4", COHORT)


def persist_run_metadata(
    *,
    run_meta_path: str,
    out_csv: str,
    done_keys_path: str,
    images_csv: str,
) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": f"dvf_mbec_maps_{ANCHOR_DIAG.lower()}_only",
        "inputs": {
            "images_csv": str(images_csv),
            "warps_dir": str(WARPS_DIR),
            "warp_pattern": "*_1Warp.nii.gz",
            "anchor": ANCHOR_DIAG,
        },
        "outputs": {"out_csv": str(out_csv), "done_keys_path": str(done_keys_path)},
        "paper_alignment": {
            "maps": {
                "D": "mag = ||u||_2",
                "V": "jac_det = det(I + grad(u))",
                "S": "strain_fro = ||eps||_F",
            },
            "roi_stats": list(PAPER_MOMENT_KEYS),
        },
        "env": {"DISPLACEMENT_POINT_CHUNK": POINT_CHUNK},
    }
    os.makedirs(os.path.dirname(run_meta_path), exist_ok=True)
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = pd.read_csv(IMAGES_CSV)
    required = {"ID_PT", "ID_IMG", "SEX", "AGE", "MRI_DATE"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"IMAGES_CSV sem colunas: {sorted(missing_cols)}")

    os.makedirs(RUN_DIR, exist_ok=True)
    done_keys_path = os.path.join(RUN_DIR, "done_keys.txt")
    run_meta_path = os.path.join(RUN_DIR, "run_meta.json")
    persist_run_metadata(
        run_meta_path=run_meta_path,
        out_csv=OUT_CSV,
        done_keys_path=done_keys_path,
        images_csv=IMAGES_CSV,
    )

    done: set[str] = set()
    if RESUME:
        done |= load_done_keys(done_keys_path)
        if os.path.isfile(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
            try:
                prev = pd.read_csv(OUT_CSV, usecols=["ID_IMG"])
                done |= set(prev["ID_IMG"].astype(str).str.strip().tolist())
            except Exception:
                pass
        print(f"[RESUME] imagens já processadas: {len(done)}", flush=True)

    ref_by_pt = baseline_ref_tag_by_pt(df)
    df["MRI_DATE"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    df = df.sort_values(["ID_PT", "MRI_DATE", "ID_IMG"])

    print(
        f"[INFO] ANCHOR={ANCHOR_DIAG} warps={WARPS_DIR} out={OUT_CSV}",
        flush=True,
    )

    t0 = time.time()
    processed = 0
    skipped = 0
    skip_reason = {"resume": 0, "no_ref": 0, "missing_inputs": 0}
    n_warp_files = sum(
        1
        for _root, _dirs, files in os.walk(WARPS_DIR)
        for f in files
        if f.endswith(("Warp.nii.gz", "InverseWarp.nii.gz", ".mat"))
    )
    if n_warp_files == 0:
        print(
            f"[WARN] warps vazios em {WARPS_DIR} — rode antes: "
            f"python 3.1_feat_gen_dvf.py --diag {ANCHOR_DIAG}",
            flush=True,
        )

    for r in df.itertuples(index=False):
        id_pt = str(r.ID_PT)
        img_id = str(r.ID_IMG).strip()

        if RESUME and img_id in done:
            skipped += 1
            skip_reason["resume"] += 1
            continue

        sex, age_range = ref_by_pt.get(id_pt, (None, None))
        if sex is None:
            skipped += 1
            skip_reason["no_ref"] += 1
            continue

        tpl_path = template_path_for(sex, age_range)
        subj_path = subject_path_for(CLINIC_DIR, img_id)
        warp_list = nonlinear_warp_list_for(WARPS_DIR, img_id, sex, age_range)
        fwd_list = fwd_list_for(WARPS_DIR, img_id, sex, age_range)
        regions_p = os.path.join(REGIONS_DIR, f"{img_id}_regions.nii.gz")
        bm_p = os.path.join(BRAIN_MASK_DIR, f"{img_id}_brain_mask.nii.gz")

        if (
            not os.path.isfile(tpl_path)
            or not os.path.isfile(subj_path)
            or not os.path.isfile(regions_p)
            or not os.path.isfile(bm_p)
            or any(not os.path.isfile(p) for p in warp_list)
            or any(not os.path.isfile(p) for p in fwd_list)
        ):
            skipped += 1
            skip_reason["missing_inputs"] += 1
            continue

        t_img = time.time()
        domain_img = ants.image_read(tpl_path)
        lj, m, dvg, ux, uy, uz, curl, strain_maps, refimg = compute_unitary_scalar_arrays(
            domain_img, warp_list
        )
        spacing = tuple(map(float, refimg.spacing))
        strain_inf_fro = _CURRENT["strain_inf_fro"]
        labels = warp_labelmap_to_template(regions_p, domain_img, fwd_list)
        brain_mask = warp_mask_to_template(bm_p, domain_img, fwd_list)

        rows = []
        for roi, side, label in ROI_TABLE:
            lab = int(label)
            roi_mask = (labels == lab) & brain_mask
            rows.append(
                _build_roi_feature_row(
                    id_pt=id_pt,
                    img_id=img_id,
                    meta_row=r,
                    sex=sex,
                    age_range=age_range,
                    roi=roi,
                    side=side,
                    label=lab,
                    roi_mask=roi_mask,
                    refimg=refimg,
                    lj=lj,
                    m=m,
                    dvg=dvg,
                    ux=ux,
                    uy=uy,
                    uz=uz,
                    curl=curl,
                    strain_maps=strain_maps,
                    strain_inf_fro=strain_inf_fro,
                )
            )

        if rows:
            append_csv(pd.DataFrame(rows), OUT_CSV)

        if RESUME:
            append_done_key(done_keys_path, img_id)
            done.add(img_id)

        processed += 1
        if LOG_EVERY > 0 and (processed % LOG_EVERY) == 0:
            dt = time.time() - t_img
            total_dt = time.time() - t0
            print(
                f"[OK] IMG={img_id} PT={id_pt} ref={_ref_tag(sex, age_range)} "
                f"domain=template rows={len(ROI_TABLE)} "
                f"dt={dt:.1f}s processed={processed} skipped={skipped} "
                f"elapsed={total_dt/60:.1f}min",
                flush=True,
            )

    total_dt = time.time() - t0
    print(
        f"[DONE] processed={processed} skipped={skipped} "
        f"reasons={skip_reason} "
        f"elapsed={total_dt/60:.1f}min out_csv={OUT_CSV}",
        flush=True,
    )


HIPPO_LABELS = (("L", 17), ("R", 53))
OASIS_DILATIONS_MM = (2, 4, 8)
OASIS_MAPS = ("jac_det", "logjac", "mag", "strain_fro")


def hippocampus_rois(
    labels: np.ndarray, brain: np.ndarray, spacing: tuple[float, float, float]
) -> list[tuple[str, str, int, np.ndarray]]:
    """Núcleo, dilatações em mm (EDT físico) e casca de 4 mm, sempre dentro do cérebro."""
    out = []
    for side, lab in HIPPO_LABELS:
        core = labels == lab
        dist = distance_transform_edt(~core, sampling=spacing)
        out.append(("hippocampus", side, lab, core & brain))
        for r in OASIS_DILATIONS_MM:
            out.append((f"hippocampus_d{r}", side, lab, (dist <= r) & brain))
        out.append(("hippocampus_shell4", side, lab, (dist <= 4) & ~core & brain))
    return out


def _map_stat_columns(prefix: str, x: np.ndarray) -> dict[str, float]:
    s = _stats_percentiles(x)
    m = _stats_moments(x)
    row = {f"{prefix}_{k}": s[k] for k in ("n", "mean", "std", "p05", "p50", "p95")}
    row.update({f"{prefix}_{k}": m[k] for k in ("variance", "skewness", "kurtosis")})
    return row


def compute_oasis_maps(domain_img: ants.ANTsImage, warp_path: str) -> dict[str, np.ndarray]:
    delta = load_nonlinear_displacement(domain_img, [warp_path])
    jac = ants.create_jacobian_determinant_image(domain_img, delta, do_log=False).numpy()
    arr = delta.numpy().astype(np.float32, copy=False)
    return {
        "jac_det": jac.astype(np.float32),
        "logjac": np.log(np.clip(jac, 1e-6, None)).astype(np.float32),
        "mag": np.sqrt((arr * arr).sum(-1)).astype(np.float32),
        "strain_fro": _infinitesimal_strain_fro_map(
            arr[..., 0], arr[..., 1], arr[..., 2], tuple(map(float, delta.spacing))
        ),
    }


def _read_on_domain(path: str, domain: ants.ANTsImage) -> np.ndarray:
    img = ants.image_read(path)
    diff = _geometry_differences(img, domain)
    if diff:
        raise ValueError(f"geometria de {path} difere da imagem clínica: {'; '.join(diff)}")
    return img.numpy()


def main_oasis(diag: str, ids_csv: str | None = None) -> None:
    warps_dir = f"./images/displacement_field_oasis_{diag.lower()}"
    out_csv = f"{COHORT_DIR}/features_displacement_oasis_{diag.lower()}.csv"
    run_dir = os.path.join(warps_dir, "features", COHORT)
    os.makedirs(run_dir, exist_ok=True)
    done_keys_path = os.path.join(run_dir, "done_keys.txt")
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "variant": f"dvf_oasis_{diag.lower()}",
            "inputs": {"images_csv": IMAGES_CSV, "warps_dir": warps_dir, "ids_csv": ids_csv},
            "outputs": {"out_csv": out_csv},
            "domain": "imagem clínica (fixed do SyNRA)",
            "sign": "jac_det > 1 = sujeito menor que o template (atrofia); inverso do disp ADNI",
            "rois": ["hippocampus", *[f"hippocampus_d{r}" for r in OASIS_DILATIONS_MM], "hippocampus_shell4"],
            "maps": list(OASIS_MAPS),
        }, f, ensure_ascii=False, indent=2)

    df = pd.read_csv(IMAGES_CSV)
    ref_by_pt = oasis_refs.baseline_by_pt(df)
    if ids_csv:
        ids = set(pd.read_csv(ids_csv)["ID_IMG"].astype(str).str.strip())
        df = df[df["ID_IMG"].astype(str).str.strip().isin(ids)]
    done = load_done_keys(done_keys_path) if RESUME else set()
    print(f"[INFO] oasis {diag} imagens={len(df)} feitas={len(done)} out={out_csv}", flush=True)

    t0 = time.time()
    processed = 0
    skip_reason = {"resume": 0, "no_ref": 0, "missing_inputs": 0}
    for r in df.itertuples(index=False):
        id_pt, img_id = str(r.ID_PT), str(r.ID_IMG).strip()
        if img_id in done:
            skip_reason["resume"] += 1
            continue
        if id_pt not in ref_by_pt:
            skip_reason["no_ref"] += 1
            continue
        sex, abin = ref_by_pt[id_pt]
        tag = oasis_refs.ref_tag(diag, sex, abin)
        warp_p = os.path.join(warps_dir, f"{img_id}_{tag}_1Warp.nii.gz")
        subj_p = subject_path_for(CLINIC_DIR, img_id)
        regions_p = os.path.join(REGIONS_DIR, f"{img_id}_regions.nii.gz")
        bm_p = os.path.join(BRAIN_MASK_DIR, f"{img_id}_brain_mask.nii.gz")
        if not all(os.path.isfile(p) for p in (warp_p, subj_p, regions_p, bm_p)):
            skip_reason["missing_inputs"] += 1
            continue

        t_img = time.time()
        domain = ants.image_read(subj_p)
        maps = compute_oasis_maps(domain, warp_p)
        labels = np.rint(_read_on_domain(regions_p, domain)).astype(np.int32)
        brain = _read_on_domain(bm_p, domain) > 0.5
        rows = []
        for roi, side, lab, mask in hippocampus_rois(labels, brain, tuple(map(float, domain.spacing))):
            cx, cy, cz = _centroid_physical(mask, domain)
            row = {
                "ID_PT": id_pt, "ID_IMG": img_id, "DIAG": str(getattr(r, "DIAG", "")),
                "GROUP": str(getattr(r, "GROUP", "")), "SEX": str(r.SEX), "AGE": float(r.AGE),
                "MRI_DATE": str(r.MRI_DATE), "ref_tag": tag, "roi": roi, "side": side,
                "label": str(lab), "centroid_x": cx, "centroid_y": cy, "centroid_z": cz,
            }
            for name in OASIS_MAPS:
                row.update(_map_stat_columns(name, maps[name][mask]))
            rows.append(row)
        append_csv(pd.DataFrame(rows), out_csv)
        append_done_key(done_keys_path, img_id)
        processed += 1
        print(f"[OK] IMG={img_id} ref={tag} rows={len(rows)} dt={time.time() - t_img:.1f}s "
              f"processed={processed}", flush=True)

    print(f"[DONE] oasis {diag} processed={processed} reasons={skip_reason} "
          f"elapsed={(time.time() - t0) / 60:.1f}min out={out_csv}", flush=True)


def self_check_oasis() -> None:
    labels = np.zeros((40, 40, 40), dtype=np.int32)
    labels[10:14, 10:14, 10:14] = 17
    labels[26:30, 26:30, 26:30] = 53
    rois = hippocampus_rois(labels, np.ones_like(labels, bool), (1.0, 1.0, 1.0))
    assert len(rois) == 10
    by = {(roi, side): m for roi, side, _, m in rois}
    for side in ("L", "R"):
        vols = [by[(k, side)].sum() for k in ("hippocampus", "hippocampus_d2", "hippocampus_d4", "hippocampus_d8")]
        assert vols[0] == 64 and vols == sorted(vols) and len(set(vols)) == 4, vols
        assert not (by[("hippocampus_shell4", side)] & by[("hippocampus", side)]).any()
        assert by[("hippocampus_shell4", side)].sum() == vols[2] - vols[0]
    row = _map_stat_columns("jac_det", np.array([1.0, 2.0, 3.0]))
    assert row["jac_det_mean"] == 2.0 and "jac_det_kurtosis" in row and len(row) == 9
    print("ok: 3.2_feat_dvf --src oasis (ROIs núcleo/d2/d4/d8/shell4, colunas)")


def self_check(diag: str = "CN") -> None:
    configure_anchor(diag)
    template = ants.from_numpy(np.zeros((3, 3, 3), dtype=np.float32))
    field = ants.from_numpy(
        np.zeros((3, 3, 3, 3), dtype=np.float32), has_components=True
    )
    assert not _geometry_differences(field, template)
    paths = nonlinear_warp_list_for(WARPS_DIR, "img", "F", "60-69")
    assert len(paths) == 1 and f"{diag}_SEX-F_AGE-60-69_1Warp" in paths[0]
    assert f"DIAG-{diag}" in template_path_for("F", "60-69")
    grid_x = np.indices((3, 3, 3), dtype=np.float32)[0]
    zeros = np.zeros_like(grid_x)
    strain = _infinitesimal_strain_fro_map(grid_x, zeros, zeros, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(strain, 1.0, atol=1e-6)
    assert len(ROI_TABLE) == 20
    if diag == "CN":
        assert OUT_CSV.endswith("features_displacement_v4.csv")
    else:
        assert OUT_CSV.endswith("features_displacement_v4_ad.csv")
    print(f"ok: 3.2_feat_dvf --diag {diag} (single-file, maps D/V/S)")


def cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DVF features CN|AD")
    p.add_argument("--diag", default="CN", choices=VALID_DIAG)
    p.add_argument("--src", default="adni", choices=("adni", "oasis"))
    p.add_argument("--ids-csv", default=None, help="CSV com coluna ID_IMG (só --src oasis)")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    if args.self_check:
        self_check_oasis() if args.src == "oasis" else self_check(args.diag)
        return
    if args.src == "oasis":
        main_oasis(args.diag, args.ids_csv)
        return
    configure_anchor(args.diag)
    main()


if __name__ == "__main__":
    cli(sys.argv[1:])

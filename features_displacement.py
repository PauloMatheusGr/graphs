#!/usr/bin/env python3
"""
Gera um CSV com atributos unitários do displacement vector field por IMAGEM.

Fluxo (objetivo, sem argparse/dataclass):
  1) Lê IMAGES_CSV (imagens unitárias).
  2) Inferir ref_tag baseline por paciente (para bater com os warps já gerados).
  3) Para cada ID_IMG:
       - carrega imagem clínica (fixed)
       - monta displacement field em memória usando inv_list = [affine, inverseWarp]
         (evita segfault no jacobiano ao usar *_1Warp.nii.gz diretamente)
       - calcula: logjac, mag, div, ux, uy, uz, curlmag
       - agrega stats por ROI (ROI_TABLE) e salva no OUT_CSV
  4) Retoma com done_keys.txt e também consultando o OUT_CSV.
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ants

# =========================
# CONFIG (edite aqui)
# =========================

ab = "abordagem_4_sMCI_pMCI"

IMAGES_CSV = "image_data_sMCI_pMCI.txt"

WARPS_DIR = "./images/displacement_field"
CLINIC_DIR = "./images/resampled_1.0mm"
REGIONS_DIR = "./images/regions"
BRAIN_MASK_DIR = "./images/brain_mask"

OUT_CSV = f"./csvs/{ab}/features_displacement_unitary.csv"
RUN_DIR = os.path.join(WARPS_DIR, "features_unitary", ab)

RESUME = True
LOG_EVERY = 1  # log a cada N imagens processadas

POINT_CHUNK = int(os.environ.get("DISPLACEMENT_POINT_CHUNK", "400000"))

# =========================
# ROI table (igual notebook)
# =========================
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


def persist_run_metadata(*, run_meta_path: str, out_csv: str, done_keys_path: str, images_csv: str) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"images_csv": str(images_csv)},
        "outputs": {"out_csv": str(out_csv), "done_keys_path": str(done_keys_path)},
        "env": {"DISPLACEMENT_POINT_CHUNK": POINT_CHUNK},
    }
    os.makedirs(os.path.dirname(run_meta_path), exist_ok=True)
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

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
    """
    Os warps foram gerados com base no baseline (primeira visita) do paciente.
    Então, para encontrar os arquivos em images/displacement_field/, usamos (SEX, age_range)
    do baseline por ID_PT.
    """
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

def fixed_path_for(clinic_dir: str, img_id: str) -> str:
    return os.path.join(
        clinic_dir, f"{img_id}_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
    )

def inv_warp_path_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> str:
    tag = f"CN_SEX-{sex}_AGE-{age_range}"
    return os.path.join(warps_dir, f"{img_id}_{tag}_1InverseWarp.nii.gz")

def affine_path_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> str:
    tag = f"CN_SEX-{sex}_AGE-{age_range}"
    return os.path.join(warps_dir, f"{img_id}_{tag}_0GenericAffine.mat")

def inv_list_for(warps_dir: str, img_id: str, sex: str, age_range: str) -> list[str]:
    """
    Mesma convenção do features_deltas_displacement.py:
      inv_list = [affine, inv_warp]
    """
    return [
        affine_path_for(warps_dir, img_id, sex, age_range),
        inv_warp_path_for(warps_dir, img_id, sex, age_range),
    ]

def _stats(x: np.ndarray) -> dict[str, float]:
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
    Constrói um displacement field (has_components=True) no DOMÍNIO da imagem clínica (fixed),
    aplicando somente o inv_list (fixed -> moving/template) aos pontos do domínio.

    Isso evita passar diretamente o *_1Warp.nii.gz para create_jacobian_determinant_image,
    que pode causar segfault em algumas builds do ANTs/ITK.
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

def compute_unitary_scalar_arrays(domain_img: ants.ANTsImage, inv_list: list[str]):
    """
    Mesma família de features do script de deltas, mas para o caso unitário.

    Este campo é construído no domínio da imagem clínica (fixed) aplicando a transformação
    inversa (fixed -> template) aos pontos. Isso evita segfault que ocorria ao passar
    diretamente o *_1Warp.nii.gz para create_jacobian_determinant_image.

    Retorna arrays float32: logjac, mag, div, ux, uy, uz, curlmag + ref_img para resample/centroide.
    """
    delta = displacement_field_from_inv_list(domain_img, inv_list)
    logjac = ants.create_jacobian_determinant_image(domain_img, delta, do_log=True)
    mag = field_magnitude(delta)
    div = field_divergence(delta)
    ux, uy, uz = field_components(delta)
    curlmag = field_curl_magnitude(delta)
    return (
        logjac.numpy().astype(np.float32),
        mag.numpy().astype(np.float32),
        div.numpy().astype(np.float32),
        ux.numpy().astype(np.float32),
        uy.numpy().astype(np.float32),
        uz.numpy().astype(np.float32),
        curlmag.numpy().astype(np.float32),
        logjac,  # ref_img
    )

def append_csv(df: pd.DataFrame, out_csv_path: str) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    exists = os.path.isfile(out_csv_path) and os.path.getsize(out_csv_path) > 0
    df.to_csv(out_csv_path, mode="a", header=not exists, index=False)

def main():
    df = pd.read_csv(IMAGES_CSV)

    required = {"ID_PT", "ID_IMG", "SEX", "AGE", "MRI_DATE"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"IMAGES_CSV não tem colunas obrigatórias: {sorted(missing_cols)}")

    os.makedirs(RUN_DIR, exist_ok=True)
    done_keys_path = os.path.join(RUN_DIR, "done_keys.txt")
    run_meta_path = os.path.join(RUN_DIR, "run_meta.json")
    persist_run_metadata(
        run_meta_path=run_meta_path,
        out_csv=OUT_CSV,
        done_keys_path=done_keys_path,
        images_csv=IMAGES_CSV,
    )

    done = set()
    if RESUME:
        done |= load_done_keys(done_keys_path)
        if os.path.isfile(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
            try:
                prev = pd.read_csv(OUT_CSV, usecols=["ID_IMG"])
                done |= set(prev["ID_IMG"].astype(str).str.strip().tolist())
            except Exception:
                pass
        print(f"[RESUME] imagens já processadas: {len(done)}", flush=True)

    # baseline ref_tag por paciente para bater com warps em disco
    ref_by_pt = baseline_ref_tag_by_pt(df)

    df["MRI_DATE"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    df = df.sort_values(["ID_PT", "MRI_DATE", "ID_IMG"])

    t0 = time.time()
    processed = 0
    skipped = 0

    for r in df.itertuples(index=False):
        id_pt = str(r.ID_PT)
        img_id = str(r.ID_IMG).strip()

        if RESUME and img_id in done:
            skipped += 1
            continue

        sex, age_range = ref_by_pt.get(id_pt, (None, None))
        if sex is None:
            skipped += 1
            continue

        fixed_p = fixed_path_for(CLINIC_DIR, img_id)
        inv_list = inv_list_for(WARPS_DIR, img_id, sex, age_range)
        regions_p = os.path.join(REGIONS_DIR, f"{img_id}_regions.nii.gz")
        bm_p = os.path.join(BRAIN_MASK_DIR, f"{img_id}_brain_mask.nii.gz")

        if (
            not os.path.isfile(fixed_p)
            or not os.path.isfile(regions_p)
            or any(not os.path.isfile(p) for p in inv_list)
        ):
            skipped += 1
            continue

        t_img = time.time()

        domain_img = ants.image_read(fixed_p)
        lj, m, dvg, ux, uy, uz, curl, refimg = compute_unitary_scalar_arrays(domain_img, inv_list)
        labels = _load_and_resample_labelmap(regions_p, refimg)
        brain_mask = _load_and_resample_mask(bm_p, refimg) if os.path.isfile(bm_p) else None

        rows = []
        for roi, side, label in ROI_TABLE:
            lab = int(label)
            roi_mask = labels == lab
            if brain_mask is not None:
                roi_mask = roi_mask & brain_mask

            cx, cy, cz = _centroid_physical(roi_mask, refimg)

            s_lj = _stats(lj[roi_mask])
            s_m = _stats(m[roi_mask])
            s_div = _stats(dvg[roi_mask])
            s_ux = _stats(ux[roi_mask])
            s_uy = _stats(uy[roi_mask])
            s_uz = _stats(uz[roi_mask])
            s_curl = _stats(curl[roi_mask])

            rows.append(
                {
                    "ID_PT": id_pt,
                    "ID_IMG": img_id,
                    "DIAG": str(getattr(r, "DIAG", "")),
                    "GROUP": str(getattr(r, "GROUP", "")),
                    "SEX": str(getattr(r, "SEX", "")),
                    "AGE": float(getattr(r, "AGE", np.nan)),
                    "MRI_DATE": str(getattr(r, "MRI_DATE", "")),
                    "ref_tag": f"CN_SEX-{sex}_AGE-{age_range}",
                    "roi": str(roi),
                    "side": str(side),
                    "label": str(label),
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "centroid_z": float(cz),
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
                f"[OK] IMG={img_id} PT={id_pt} ref=CN_SEX-{sex}_AGE-{age_range} rows={len(ROI_TABLE)} "
                f"dt={dt:.1f}s processed={processed} skipped={skipped} elapsed={total_dt/60:.1f}min",
                flush=True,
            )

    total_dt = time.time() - t0
    print(f"[DONE] processed={processed} skipped={skipped} elapsed={total_dt/60:.1f}min out_csv={OUT_CSV}", flush=True)

if __name__ == "__main__":
    main()
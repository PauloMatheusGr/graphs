import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime

import ants
import numpy as np
import pandas as pd

# =========================
# Defaults (edite aqui)
# =========================
DEFAULT_IMAGES_CSV = "image_data_sMCI_pMCI.txt"
DEFAULT_COMBINATIONS_CSV = "cj_data_abordagem_4_sMCI_pMCI.txt"

# Fases: "1" (warps), "2" (features), "both"
DEFAULT_PHASE = "both"

# Tamanho mínimo para considerar NIfTI completo (em bytes)
DEFAULT_MIN_OUTPUT_BYTES = 1024

# Temporários (evita /tmp do sistema; respeita TMPDIR)
DEFAULT_TMPDIR = "./images/displacement_field/_tmp_ants"

# Templates CN estratificados (Imagens MÓVEIS)
groupwise_dir = "./images/groupwise"
# Imagens clínicas pré-processadas (Imagens FIXAS)
clinic_dir = "./images/resampled_1.0mm"
# Labels/máscaras/ROIs
labels_images_root = "./images"
regions_dir = "./images/regions"
brain_mask_dir = "./images/brain_mask"

# Saídas (warps são compartilhados entre abordagens)
warps_output = "./images/displacement_field"
csvs_root = "./csvs"

os.makedirs(warps_output, exist_ok=True)

if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = os.path.abspath(DEFAULT_TMPDIR)
os.environ.setdefault("TMP", os.environ["TMPDIR"])
os.environ.setdefault("TEMP", os.environ["TMPDIR"])
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Pontos por lote (evita pico de RAM)
POINT_CHUNK = int(os.environ.get("DISPLACEMENT_POINT_CHUNK", "400000"))

# Máscara cerebral opcional: env DISPLACEMENT_BRAIN_MASK
BRAIN_MASK_TEMPLATE = os.environ.get("DISPLACEMENT_BRAIN_MASK", "").strip()

# Tabela de ROIs (igual ao notebook)
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


@dataclass(frozen=True)
class BaselineReference:
    sex: str
    age: int
    age_range: str
    ref_path: str


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


def get_stratified_reference_path(sex, age_range):
    sex = str(sex).upper().strip()
    ref_filename = f"groupwise_DIAG-CN_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz"
    return os.path.join(groupwise_dir, ref_filename)


def fixed_path_for(img_id):
    return os.path.join(
        clinic_dir, f"{img_id}_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
    )


def warp_prefix(img_id: str, ref_tag: str) -> str:
    return os.path.join(warps_output, f"{img_id}_{ref_tag}")


def fwd_inv_paths_for(img_id: str, ref_tag: str):
    p = warp_prefix(img_id, ref_tag)
    fwd_warp = f"{p}_1Warp.nii.gz"
    affine = f"{p}_0GenericAffine.mat"
    inv_warp = f"{p}_1InverseWarp.nii.gz"
    fwd_list = [fwd_warp, affine]
    inv_list = [affine, inv_warp]
    return fwd_warp, affine, inv_warp, fwd_list, inv_list


def index_grid_to_physical_points(domain_img):
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


def apply_fwd_then_inv(pts_df: pd.DataFrame, fwd_list: list, inv_list: list):
    mid = ants.apply_transforms_to_points(3, pts_df.copy(), fwd_list)
    out = ants.apply_transforms_to_points(3, mid, inv_list)
    return out


def relative_displacement_field(domain_img, fwd_list_a: list, inv_list_b: list):
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
        df_out = apply_fwd_then_inv(df_in, fwd_list_a, inv_list_b)
        delta = df_out[["x", "y", "z"]].to_numpy(dtype=np.float64) - block
        disp[start:end, :] = delta
        start = end
    vec = disp.reshape(*(shape + (dim,)))
    field = ants.from_numpy(
        vec.astype(np.float32),
        origin=domain_img.origin,
        spacing=domain_img.spacing,
        direction=domain_img.direction,
        has_components=True,
    )
    return field


def field_magnitude(field):
    arr = field.numpy()
    mag = np.sqrt(np.sum(arr * arr, axis=-1))
    return ants.from_numpy(
        mag.astype(np.float32),
        origin=field.origin,
        spacing=field.spacing,
        direction=field.direction,
    )


def field_divergence(field):
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


def field_components(field):
    arr = field.numpy().astype(np.float32, copy=False)
    ux = ants.from_numpy(
        arr[..., 0], origin=field.origin, spacing=field.spacing, direction=field.direction
    )
    uy = ants.from_numpy(
        arr[..., 1], origin=field.origin, spacing=field.spacing, direction=field.direction
    )
    uz = ants.from_numpy(
        arr[..., 2], origin=field.origin, spacing=field.spacing, direction=field.direction
    )
    return ux, uy, uz


def field_curl_magnitude(field):
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


def load_brain_mask_for_domain(img_id_baseline: str, domain_img):
    cand = os.path.join(
        labels_images_root, "brain_mask", f"{img_id_baseline}_brain_mask.nii.gz"
    )
    path = cand if os.path.isfile(cand) else None
    if path is None and BRAIN_MASK_TEMPLATE and os.path.isfile(BRAIN_MASK_TEMPLATE):
        path = BRAIN_MASK_TEMPLATE
    if path is None:
        return None
    m = ants.image_read(path)
    m = ants.resample_image_to_target(m, domain_img, interp_type="nearestNeighbor")
    return m


def _infer_ab_from_cj_path(cj_path: str) -> str:
    """
    Tenta inferir 'ab' a partir do nome 'cj_data_<ab>.txt'.
    Se não casar, retorna 'custom'.
    """
    base = os.path.basename(str(cj_path))
    if base.startswith("cj_data_") and base.endswith(".txt"):
        return base[len("cj_data_") : -len(".txt")]
    return "custom"


def _file_nonempty(path: str, min_bytes: int) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def registration_bundle_complete(
    affine_out: str, warp_out: str, inv_warp_out: str, *, min_bytes: int
) -> bool:
    if not _file_nonempty(affine_out, 1):
        return False
    if not _file_nonempty(warp_out, min_bytes):
        return False
    if not _file_nonempty(inv_warp_out, min_bytes):
        return False
    return True


def remove_registration_bundle(
    affine_out: str, warp_out: str, inv_warp_out: str, *, reason: str
) -> None:
    for p in (affine_out, warp_out, inv_warp_out):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError as e:
            print(f"[WARN] nao foi possivel remover {p}: {e}")
    print(f"[RESUME] {reason}: arquivos de registro removidos para recomputar.")


def build_baseline_reference_map(df_images: pd.DataFrame):
    df = df_images.copy()
    df["MRI_DATE"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    df = df.sort_values(["ID_PT", "MRI_DATE", "ID_IMG"])
    ref_by_pt = {}
    for id_pt, g in df.groupby("ID_PT", sort=False):
        r0 = g.iloc[0]
        sex = str(r0["SEX"]).upper().strip()
        age = int(r0["AGE"])
        age_range = get_age_range(age)
        ref_path = get_stratified_reference_path(sex=sex, age_range=age_range)
        ref_by_pt[str(id_pt)] = BaselineReference(
            sex=sex, age=age, age_range=age_range, ref_path=ref_path
        )
    return ref_by_pt


def run_individual_registrations(csv_images_path, *, min_output_bytes: int = 1024):
    df_imgs = pd.read_csv(csv_images_path)
    ref_by_pt = build_baseline_reference_map(df_imgs)

    for idx, row in df_imgs.iterrows():
        img_id = str(row["ID_IMG"])
        id_pt = str(row["ID_PT"])
        ref = ref_by_pt[id_pt]
        ref_tag = f"CN_SEX-{ref.sex}_AGE-{ref.age_range}"

        affine_out = os.path.join(
            warps_output, f"{img_id}_{ref_tag}_0GenericAffine.mat"
        )
        warp_out = os.path.join(warps_output, f"{img_id}_{ref_tag}_1Warp.nii.gz")
        inv_warp_out = os.path.join(
            warps_output, f"{img_id}_{ref_tag}_1InverseWarp.nii.gz"
        )

        if registration_bundle_complete(
            affine_out, warp_out, inv_warp_out, min_bytes=min_output_bytes
        ):
            print(f"[{idx}] [SKIP] {img_id}: registro completo ja existe.")
            continue
        partial = any(os.path.isfile(p) for p in (affine_out, warp_out, inv_warp_out))
        if partial:
            remove_registration_bundle(
                affine_out,
                warp_out,
                inv_warp_out,
                reason=f"Registro incompleto para {img_id}",
            )

        fixed_path = fixed_path_for(img_id)
        moving_path = ref.ref_path
        if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
            print(f"[SKIP] Arquivo não encontrado para {img_id}")
            continue

        print(f"[{idx}] Registrando Template em {img_id}...")
        fixed_img = ants.image_read(fixed_path)
        moving_img = ants.image_read(moving_path)

        reg = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform="SyN",
            interpolator="bspline",
        )

        fwd = reg.get("fwdtransforms", []) or []
        inv = reg.get("invtransforms", []) or []
        affine_src = next((p for p in fwd if str(p).endswith(".mat")), None)
        fwd_warp_src = next(
            (p for p in fwd if str(p).endswith(".nii") or str(p).endswith(".nii.gz")),
            None,
        )
        inv_warp_src = next(
            (p for p in inv if str(p).endswith(".nii") or str(p).endswith(".nii.gz")),
            None,
        )
        if affine_src is None or fwd_warp_src is None or inv_warp_src is None:
            print(f"[ERROR] {img_id}: transforms incompletos: fwd={fwd} inv={inv}")
            continue

        shutil.copy2(affine_src, affine_out)
        shutil.copy2(fwd_warp_src, warp_out)
        shutil.copy2(inv_warp_src, inv_warp_out)


def _stats(x: np.ndarray) -> dict[str, float]:
    """
    Mesmo formato do notebook:
      n, mean, std, p05, p50, p95
    """
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


def _load_and_resample_labelmap(path: str, target: ants.ANTsImage) -> np.ndarray:
    lab = ants.image_read(path)
    lab = ants.resample_image_to_target(lab, target, interp_type="nearestNeighbor")
    return lab.numpy().astype(np.int32)


def _load_and_resample_mask(path: str, target: ants.ANTsImage) -> np.ndarray:
    m = ants.image_read(path)
    m = ants.resample_image_to_target(m, target, interp_type="nearestNeighbor")
    return (m.numpy() > 0.5)


def compute_pair_scalar_arrays(domain_img, fwd_list_a: list, inv_list_b: list):
    """
    Calcula os 7 mapas (em memória) e devolve arrays numpy float32:
      logjac, mag, div, ux, uy, uz, curlmag
    """
    delta = relative_displacement_field(domain_img, fwd_list_a, inv_list_b)
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
        logjac,  # ref_img (para resample/centroide)
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


def write_parquet_part(df: pd.DataFrame, *, dataset_dir: str, part_prefix: str) -> str:
    """
    Grava um arquivo part-*.parquet.
    Observação: isso cria um "dataset" (diretório com vários parts), robusto para retomada.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{part_prefix}_{ts}_{os.getpid()}.parquet"
    final_path = os.path.join(dataset_dir, fn)
    tmp_path = os.path.join(dataset_dir, f".{fn}.partial")

    try:
        df.to_parquet(tmp_path, index=False)
    except Exception as e:
        raise RuntimeError(
            "Falha ao escrever Parquet. Verifique se 'pyarrow' (ou 'fastparquet') está instalado."
        ) from e
    os.replace(tmp_path, final_path)
    return final_path


def append_csv(df: pd.DataFrame, out_csv_path: str) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    exists = os.path.isfile(out_csv_path) and os.path.getsize(out_csv_path) > 0
    df.to_csv(out_csv_path, mode="a", header=not exists, index=False)


def persist_run_metadata(
    *,
    run_meta_path: str,
    done_keys_path: str,
    parquet_dataset_dir: str,
    out_csv_path: str,
):
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "defaults": {
            "DEFAULT_IMAGES_CSV": DEFAULT_IMAGES_CSV,
            "DEFAULT_COMBINATIONS_CSV": DEFAULT_COMBINATIONS_CSV,
            "DEFAULT_PHASE": DEFAULT_PHASE,
            "DEFAULT_MIN_OUTPUT_BYTES": DEFAULT_MIN_OUTPUT_BYTES,
            "POINT_CHUNK": POINT_CHUNK,
        },
        "env": {
            "TMPDIR": os.environ.get("TMPDIR"),
            "DISPLACEMENT_BRAIN_MASK": BRAIN_MASK_TEMPLATE if BRAIN_MASK_TEMPLATE else "",
        },
        "outputs": {
            "done_keys_path": done_keys_path,
            "parquet_dataset_dir": parquet_dataset_dir,
            "out_csv_path": out_csv_path,
        },
    }
    try:
        with open(run_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] nao foi possivel gravar metadata: {e}")


def process_longitudinal_combinations_to_features(
    csv_comb_path,
    csv_images_path,
    *,
    min_output_bytes: int = 1024,
):
    """
    Opção A: compatível com o notebook (features por ROI).

    Saída "long":
      1 linha por (ID_PT, COMBINATION_NUMBER, TRIPLET_IDX, pair, roi, side, label)
      com estatísticas de: logjac, mag, div, ux, uy, uz, curlmag

    Retomada (por triplet):
      - Se key estiver em done_keys.txt, pula o triplet
      - Cada triplet gera:
        - 1 part Parquet no dataset_dir (todas as linhas do triplet)
        - append no CSV final (mesmas linhas)
    """
    ab = _infer_ab_from_cj_path(csv_comb_path)
    features_output_dir = os.path.join(warps_output, "features", ab)
    os.makedirs(features_output_dir, exist_ok=True)

    done_keys_path = os.path.join(features_output_dir, "done_keys.txt")
    run_meta_path = os.path.join(features_output_dir, "run_meta.json")
    parquet_dataset_dir = os.path.join(features_output_dir, "parquet_dataset")
    os.makedirs(parquet_dataset_dir, exist_ok=True)

    out_csv_path = os.path.join(csvs_root, ab, f"features_displacement_abordagem_4_sMCI_pMCI.csv")

    persist_run_metadata(
        run_meta_path=run_meta_path,
        done_keys_path=done_keys_path,
        parquet_dataset_dir=parquet_dataset_dir,
        out_csv_path=out_csv_path,
    )

    done = load_done_keys(done_keys_path)

    df_comb = pd.read_csv(csv_comb_path)
    df_imgs = pd.read_csv(csv_images_path)
    ref_by_pt = build_baseline_reference_map(df_imgs)

    if "MRI_DATE" in df_comb.columns:
        df_comb["MRI_DATE"] = pd.to_datetime(df_comb["MRI_DATE"], errors="coerce")

    for (id_pt, comb_id), group in df_comb.groupby(["ID_PT", "COMBINATION_NUMBER"]):
        sort_cols = []
        if "MRI_DATE" in group.columns:
            sort_cols.append("MRI_DATE")
        if "TIME_PROG" in group.columns:
            sort_cols.append("TIME_PROG")
        sort_cols.append("ID_IMG")
        group = group.sort_values(sort_cols)

        ids_all = group["ID_IMG"].astype(str).values
        if len(ids_all) < 3:
            continue
        if (len(ids_all) % 3) != 0:
            continue

        for triplet_idx in range(len(ids_all) // 3):
            ids = ids_all[triplet_idx * 3 : (triplet_idx + 1) * 3]
            if len(ids) != 3:
                continue

            key = f"{id_pt}|{comb_id}|t{triplet_idx}"
            if key in done:
                continue

            ref = ref_by_pt.get(str(id_pt))
            if ref is None:
                print(f"[SKIP] {id_pt} comb={comb_id} triplet={triplet_idx}: referencia CN indisponivel.")
                continue

            if not os.path.exists(ref.ref_path):
                print(
                    f"[SKIP] {id_pt} comb={comb_id} triplet={triplet_idx}: ref nao encontrada: {ref.ref_path}"
                )
                continue

            ref_tag = f"CN_SEX-{ref.sex}_AGE-{ref.age_range}"

            # Warps necessários (para ids[0], ids[1], ids[2])
            missing = []
            fwd_inv_by_id = {}
            for sid in ids:
                fw, aff, iw, fwd_l, inv_l = fwd_inv_paths_for(sid, ref_tag)
                for p in (fw, aff, iw):
                    if not os.path.isfile(p):
                        missing.append(p)
                fwd_inv_by_id[sid] = (fwd_l, inv_l)
            if missing:
                print(
                    f"[SKIP] {id_pt} comb={comb_id} triplet={triplet_idx}: warp(s) ausentes: {missing}"
                )
                continue

            # Domínios
            fixed_baseline = fixed_path_for(ids[0])
            fixed_t2 = fixed_path_for(ids[1])
            if not os.path.isfile(fixed_baseline) or not os.path.isfile(fixed_t2):
                print(
                    f"[SKIP] {id_pt} comb={comb_id} triplet={triplet_idx}: imagens clinicas ausentes."
                )
                continue

            try:
                domain_img_1 = ants.image_read(fixed_baseline)
                domain_img_2 = ants.image_read(fixed_t2)

                fwd1, _ = fwd_inv_by_id[ids[0]]
                fwd2, _ = fwd_inv_by_id[ids[1]]
                _, inv2 = fwd_inv_by_id[ids[1]]
                _, inv3 = fwd_inv_by_id[ids[2]]

                # Carrega region maps (necessários) e brain mask (opcional) igual ao notebook:
                # - para 12/13: usa i1 no domínio do baseline
                # - para 23: usa i2 no domínio do tempo2
                regions12_p = os.path.join(regions_dir, f"{str(ids[0]).strip()}_regions.nii.gz")
                regions23_p = os.path.join(regions_dir, f"{str(ids[1]).strip()}_regions.nii.gz")
                if not os.path.isfile(regions12_p) or not os.path.isfile(regions23_p):
                    print(
                        f"[SKIP] {id_pt} comb={comb_id} triplet={triplet_idx}: regions ausentes (i1/i2)."
                    )
                    continue

                # Computa os arrays dos 3 pares (sem salvar NIfTI)
                lj12, m12, d12, ux12, uy12, uz12, c12, refimg12 = compute_pair_scalar_arrays(
                    domain_img_1, fwd1, inv2
                )
                lj13, m13, d13, ux13, uy13, uz13, c13, refimg13 = compute_pair_scalar_arrays(
                    domain_img_1, fwd1, inv3
                )
                lj23, m23, d23, ux23, uy23, uz23, c23, refimg23 = compute_pair_scalar_arrays(
                    domain_img_2, fwd2, inv3
                )

                labels12 = _load_and_resample_labelmap(regions12_p, refimg12)
                labels23 = _load_and_resample_labelmap(regions23_p, refimg23)

                bm12_p = os.path.join(brain_mask_dir, f"{str(ids[0]).strip()}_brain_mask.nii.gz")
                bm23_p = os.path.join(brain_mask_dir, f"{str(ids[1]).strip()}_brain_mask.nii.gz")
                brain_mask12 = (
                    _load_and_resample_mask(bm12_p, refimg12) if os.path.isfile(bm12_p) else None
                )
                brain_mask23 = (
                    _load_and_resample_mask(bm23_p, refimg23) if os.path.isfile(bm23_p) else None
                )

                rows: list[dict[str, object]] = []
                for roi, side, label in ROI_TABLE:
                    lab = int(label)
                    roi_mask12 = labels12 == lab
                    if brain_mask12 is not None:
                        roi_mask12 = roi_mask12 & brain_mask12

                    roi_mask23 = labels23 == lab
                    if brain_mask23 is not None:
                        roi_mask23 = roi_mask23 & brain_mask23

                    centroid_x12, centroid_y12, centroid_z12 = _centroid_physical(
                        roi_mask12, refimg12
                    )
                    centroid_x23, centroid_y23, centroid_z23 = _centroid_physical(
                        roi_mask23, refimg23
                    )

                    def pack_row(pair: str, centroid_xyz, roi_mask, arrays):
                        a_lj, a_m, a_div, a_ux, a_uy, a_uz, a_curl = arrays
                        s_lj = _stats(a_lj[roi_mask])
                        s_m = _stats(a_m[roi_mask])
                        s_div = _stats(a_div[roi_mask])
                        s_ux = _stats(a_ux[roi_mask])
                        s_uy = _stats(a_uy[roi_mask])
                        s_uz = _stats(a_uz[roi_mask])
                        s_curl = _stats(a_curl[roi_mask])

                        cx, cy, cz = centroid_xyz
                        return {
                            "ID_PT": str(id_pt),
                            "COMBINATION_NUMBER": int(comb_id),
                            "TRIPLET_IDX": int(triplet_idx),
                            "pair": str(pair),
                            "ID_IMG_i1": str(ids[0]).strip(),
                            "ID_IMG_i2": str(ids[1]).strip(),
                            "ID_IMG_i3": str(ids[2]).strip(),
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

                    # pair 12 e 13 usam ROI do i1 (roi_mask12 / centroid12)
                    rows.append(
                        pack_row(
                            "12",
                            (centroid_x12, centroid_y12, centroid_z12),
                            roi_mask12,
                            (lj12, m12, d12, ux12, uy12, uz12, c12),
                        )
                    )
                    rows.append(
                        pack_row(
                            "13",
                            (centroid_x12, centroid_y12, centroid_z12),
                            roi_mask12,
                            (lj13, m13, d13, ux13, uy13, uz13, c13),
                        )
                    )
                    # pair 23 usa ROI do i2 (roi_mask23 / centroid23)
                    rows.append(
                        pack_row(
                            "23",
                            (centroid_x23, centroid_y23, centroid_z23),
                            roi_mask23,
                            (lj23, m23, d23, ux23, uy23, uz23, c23),
                        )
                    )

                df_rows = pd.DataFrame(rows)
                write_parquet_part(
                    df_rows,
                    dataset_dir=parquet_dataset_dir,
                    part_prefix=f"pt{str(id_pt)}_c{int(comb_id)}_t{int(triplet_idx)}",
                )
                append_csv(df_rows, out_csv_path)

                # Marca como concluído somente após persistências (parquet + csv)
                append_done_key(done_keys_path, key)
                done.add(key)

                print(
                    f"OK: {id_pt} comb={comb_id} triplet={triplet_idx} -> displacement ROI features"
                )

            except Exception as e:
                print(f"[ERROR] {id_pt} comb={comb_id} triplet={triplet_idx}: {e}")
                continue


if __name__ == "__main__":
    comb_path = DEFAULT_COMBINATIONS_CSV
    images_csv = DEFAULT_IMAGES_CSV

    if DEFAULT_PHASE in ("1", "both"):
        run_individual_registrations(images_csv, min_output_bytes=DEFAULT_MIN_OUTPUT_BYTES)
    if DEFAULT_PHASE in ("2", "both"):
        process_longitudinal_combinations_to_features(
            comb_path, images_csv, min_output_bytes=DEFAULT_MIN_OUTPUT_BYTES
        )

    print(
        "[DONE] features_displacement.py terminou. "
        f"phase={DEFAULT_PHASE} images_csv={images_csv} combinations_csv={comb_path}"
    )


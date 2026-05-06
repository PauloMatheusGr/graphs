#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Groupwise registration (ANTsPy) para construir um template médio por estrato.

Uso:
    python groupwise_ants.py <min_age> <max_age> <sex>

Exemplo:
    python groupwise_ants.py 50 60 F
"""

import time
import sys
from pathlib import Path
import pandas as pd
import ants
import numpy as np
import os
import tempfile
import shutil

# Opcional: barra de progresso
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

# =========================
# CONFIGURAÇÕES FIXAS (AJUSTE SE NECESSÁRIO)
# =========================
CSV_PATH = "../datasets/output/adni/adnimerged.csv"
IMAGES_DIR = Path("/mnt/databases/mri/adni/preproc/3-biasfield")
SUFFIX = "_stripped_nlm_denoised_biascorrected.nii.gz"

# Referência para histogram matching (MNI)
REF_MNI = Path(
    "/mnt/study-data/pgirardi/preproc/atlases/templates/"
    "mni152_2009c_template.nii.gz"
)

# Máximo de imagens por estrato
N_MAX = 20

# Reprodutibilidade da amostragem
RANDOM_SEED = 7

# Registro
TYPE_OF_TRANSFORM = "SyN"  # ex.: "SyNCC" pode ser mais lento; use para melhor robustez se necessário
N_ITER_TEMPLATE = 5
VERBOSE = True

# Diagnóstico do estrato (ajuste se necessário)
DIAG_FILTER = "CN"

# Padronização espacial (correção do corte)
REORIENT_TO = "RAS"
PAD_VOXELS = (20, 20, 20)

# Grid isotrópico desejado para reduzir "espessura de fatia"
DESIRED_SPACING = (1.0, 1.0, 1.0)

# ANTs resample_image interp_type: 0=linear, 1=nearest, 2=gaussian,
# 3=windowed sinc, 4=bspline (adequado para T1 contínuas).
INTERPOLATOR = 4

# Saídas
OUT_DIR = Path(f"./adni/{DIAG_FILTER}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TEMP DIR (ANTs/ITK)
# =========================
# Se True, mantém os temporários para depuração (não recomendado em produção)
KEEP_TMP_ANTS = False

# Diretório base onde os temporários por execução serão criados
TMP_ANTS_BASE = OUT_DIR / "_tmp_ants"
TMP_ANTS_BASE.mkdir(parents=True, exist_ok=True)

if not os.access(TMP_ANTS_BASE, os.W_OK):
    raise PermissionError(f"Sem permissão de escrita em TMP_ANTS_BASE: {TMP_ANTS_BASE}")

# =========================
# FUNÇÕES
# =========================


def usage_and_exit(code: int = 1):
    print(
        "Uso:\n"
        "  python groupwise_ants.py <min_age> <max_age> <sex>\n\n"
        "Onde:\n"
        "  <min_age>  idade mínima (numérico)\n"
        "  <max_age>  idade máxima (numérico)\n"
        "  <sex>      'F' ou 'M'\n\n"
        "Exemplo:\n"
        "  python groupwise_ants.py 50 60 F\n"
    )
    sys.exit(code)


def parse_args(argv):
    if len(argv) != 4:
        usage_and_exit(1)

    try:
        min_age = float(argv[1])
        max_age = float(argv[2])
    except ValueError:
        print("Erro: <min_age> e <max_age> devem ser numéricos.")
        usage_and_exit(1)

    sex = str(argv[3]).strip().upper()
    if sex not in {"M", "F"}:
        print("Erro: <sex> deve ser 'M' ou 'F'.")
        usage_and_exit(1)

    if min_age > max_age:
        print("Erro: <min_age> não pode ser maior que <max_age>.")
        usage_and_exit(1)

    return min_age, max_age, sex


def filter_stratum(df: pd.DataFrame, sex: str, age_min: float, age_max: float, n_max: int, seed: int) -> pd.DataFrame:
    dff = df.copy()
    dff["AGE"] = pd.to_numeric(dff["AGE"], errors="coerce")
    dff["DIAG"] = dff["DIAG"].astype(str).str.strip().str.upper()
    dff["SEX"] = dff["SEX"].astype(str).str.strip().str.upper()

    sex = str(sex).strip().upper()

    dff = dff[
        (dff["DIAG"] == DIAG_FILTER) &
        (dff["SEX"] == sex) &
        (dff["AGE"].notna()) &
        (dff["AGE"] >= float(age_min)) &
        (dff["AGE"] <= float(age_max))
    ].copy()

    if len(dff) == 0:
        raise RuntimeError(
            f"Nenhuma imagem encontrada após aplicar filtros: DIAG={DIAG_FILTER}, SEX={sex}, AGE=[{age_min},{age_max}]."
        )

    if len(dff) > n_max:
        dff = dff.sample(n=n_max, replace=False, random_state=seed)

    dff = dff.sort_values(["AGE", "ID_IMG"]).reset_index(drop=True)
    return dff


def resolve_paths(subset: pd.DataFrame, images_dir: Path, suffix: str):
    missing = []
    id_to_path = {}

    for img_id in subset["ID_IMG"].astype(str):
        p = images_dir / f"{img_id}{suffix}"
        if not p.exists():
            missing.append(img_id)
        else:
            id_to_path[img_id] = p

    subset_ok = subset[subset["ID_IMG"].astype(str).isin(id_to_path.keys())].copy().reset_index(drop=True)
    return subset_ok, id_to_path, missing


def _extent_mm(img: ants.ANTsImage):
    shp = img.shape
    sp = img.spacing
    return (float(shp[0] * sp[0]), float(shp[1] * sp[1]), float(shp[2] * sp[2]))


def _extent_mm_single(img: ants.ANTsImage):
    shp = img.shape
    sp = img.spacing
    return (float(shp[0] * sp[0]), float(shp[1] * sp[1]), float(shp[2] * sp[2]))


def _choose_target_with_padding(imgs):
    extents = [_extent_mm(im) for im in imgs]
    idx = max(range(len(imgs)), key=lambda i: extents[i][0] * extents[i][1] * extents[i][2])
    target = imgs[idx]
    target = ants.pad_image(target, pad_width=list(PAD_VOXELS))
    return target, idx, extents


def brain_mask_safe(im: ants.ANTsImage) -> ants.ANTsImage:
    return ants.get_mask(im)


def robust_rescale_brain(im: ants.ANTsImage, mask: ants.ANTsImage, low=0.5, high=99.5) -> ants.ANTsImage:
    arr = im.numpy()
    m = mask.numpy() > 0
    marr = arr[m]

    if marr.size < 1000:
        return im

    p1, p2 = np.percentile(marr, [low, high])
    if not np.isfinite(p1) or not np.isfinite(p2) or p2 <= p1:
        return im

    arr = np.clip(arr, p1, p2)
    arr = (arr - p1) / (p2 - p1 + 1e-8)

    out = ants.from_numpy(arr, origin=im.origin, spacing=im.spacing, direction=im.direction)
    return out


def min_max_norm(img: ants.ANTsImage) -> ants.ANTsImage:
    data = img.numpy()
    dmin, dmax = data.min(), data.max()
    if dmax <= dmin:
        return img.new_image_like(np.zeros_like(data, dtype=np.float64))
    return img.new_image_like((data - dmin) / (dmax - dmin))


def load_hist_match_reference(ref_path: Path):
    """
    Carrega template MNI uma vez: máscara, ref normalizada, escala robusta (p99.9 no cérebro).
    """
    ref_image = ants.image_read(str(ref_path))
    mask_ref = ants.get_mask(ref_image)
    ref_norm = min_max_norm(ref_image)

    ref_data = ref_image.numpy()
    mask_arr = mask_ref.numpy().astype(bool, copy=False)
    if mask_arr.any():
        ref_scale = float(np.percentile(ref_data[mask_arr], 99.9))
    else:
        ref_scale = float(np.max(ref_data))
    return mask_ref, ref_norm, ref_scale


def histogram_match(image: ants.ANTsImage, mask_ref, ref_norm, ref_scale: float) -> ants.ANTsImage:
    """
    Casamento mascarado (ants.histogram_match_image2) + reescala + clip.
    Replica o pipeline de resample.py::histogram_match.
    """
    mask_src = ants.get_mask(image)
    src_norm = min_max_norm(image)
    matched = ants.histogram_match_image2(
        source_image=src_norm,
        reference_image=ref_norm,
        source_mask=mask_src,
        reference_mask=mask_ref,
        match_points=32,
        transform_domain_size=512,
    )
    out = matched * ref_scale
    arr = np.clip(np.asarray(out.numpy()), 0.0, ref_scale)
    return out.new_image_like(arr)


def resample_isotropic_mm(img: ants.ANTsImage, spacing_mm: float, interp_type: int = INTERPOLATOR) -> ants.ANTsImage:
    """
    Reamostra para grade isotrópica (spacing igual em todas as dimensões).
    Replica a abordagem de resample.py::resample_isotropic_mm.
    """
    dim = img.dimension
    params = tuple(float(spacing_mm) for _ in range(dim))
    return ants.resample_image(
        img,
        params,
        use_voxels=False,
        interp_type=interp_type,
    )


def prealign_to_target(moving: ants.ANTsImage, target: ants.ANTsImage,
                       moving_mask: ants.ANTsImage, target_mask: ants.ANTsImage) -> ants.ANTsImage:
    """
    Pré-alinhamento em cadeia: Rigid → Affine (antes do groupwise SyN).
    Recalcula máscara na imagem rigidamente alinhada para o Affine.
    """
    reg_rigid = ants.registration(
        fixed=target,
        moving=moving,
        type_of_transform="Rigid",
        fixed_mask=target_mask,
        moving_mask=moving_mask,
        verbose=False,
    )
    warped_rigid = reg_rigid["warpedmovout"]
    mask_after_rigid = brain_mask_safe(warped_rigid)

    reg_affine = ants.registration(
        fixed=target,
        moving=warped_rigid,
        type_of_transform="Affine",
        fixed_mask=target_mask,
        moving_mask=mask_after_rigid,
        verbose=False,
    )
    return reg_affine["warpedmovout"]


def build_groupwise_template(image_paths, n_iter: int, type_of_transform: str, verbose: bool):
    t_global = time.time()

    # 0) Carregar referência do histogram matching (MNI)
    if not REF_MNI.exists():
        raise FileNotFoundError(f"Template MNI (REF_MNI) não encontrado: {REF_MNI}")
    mask_ref, ref_norm, ref_scale = load_hist_match_reference(REF_MNI)

    # 1) Ler, reorientar, histogram matching + resample isotrópico (1mm) e validar (3D)
    t0 = time.time()
    imgs = []

    for p in tqdm(image_paths, desc="Lendo imagens", unit="img"):
        im = ants.image_read(str(p))
        im = ants.reorient_image2(im, orientation=REORIENT_TO)

        if len(im.shape) != 3:
            raise RuntimeError(f"Imagem não-3D detectada: {Path(p).name} | shape={im.shape}")

        im = histogram_match(im, mask_ref, ref_norm, ref_scale)
        im = resample_isotropic_mm(im, spacing_mm=float(DESIRED_SPACING[0]), interp_type=INTERPOLATOR)
        imgs.append(im)

    print(f"[OK] Leitura + reorientação + HM(MNI) + resample(1mm, Bspline) | {(time.time()-t0)/60:.2f} min", flush=True)

    # Estatísticas de spacing após reorientação
    spacings = np.array([im.spacing for im in imgs], dtype=float)
    print("\n[SPACING] Estatísticas (mm) após HM + resample isotrópico")
    print("  min :", spacings.min(axis=0))
    print("  mean:", spacings.mean(axis=0))
    print("  max :", spacings.max(axis=0))

    for i, (p, im) in enumerate(zip(image_paths, imgs)):
        print(f"  {i:02d} | {Path(p).name} | spacing={im.spacing} | shape={im.shape}")

    if len(imgs) < 2:
        raise RuntimeError("Número insuficiente de imagens para groupwise registration (>=2).")

    # 2) Target com padding e reamostragem isotrópica
    t0 = time.time()
    target, target_idx, extents = _choose_target_with_padding(imgs)

    target = ants.resample_image(
        target,
        list(DESIRED_SPACING),
        use_voxels=False,
        interp_type=INTERPOLATOR,
    )

    if verbose:
        e_before = extents[target_idx]
        e_after = _extent_mm_single(target)
        print("\n[Padronização espacial]", flush=True)
        print(f"Orientação: {REORIENT_TO}", flush=True)
        print(f"Target escolhido (idx={target_idx}) | extensão original (mm): {e_before}", flush=True)
        print(f"Padding (voxels): {PAD_VOXELS}", flush=True)
        print(f"Target reamostrado p/ {DESIRED_SPACING} | extensão (mm): {e_after}", flush=True)
        print(f"Target shape: {target.shape} | spacing: {target.spacing}", flush=True)

    print(f"[OK] Target (padding + resample isotrópico) | {(time.time()-t0)/60:.2f} min", flush=True)

    # 3) Reamostrar todas as imagens para o grid do target
    t0 = time.time()
    def _resample_to_target(im: ants.ANTsImage, tgt: ants.ANTsImage) -> ants.ANTsImage:
        try:
            return ants.resample_image_to_target(im, tgt, interp_type="bspline")
        except Exception:
            return ants.resample_image_to_target(im, tgt, interp_type="linear")

    imgs = [_resample_to_target(im, target) for im in imgs]
    print(f"[OK] Reamostragem p/ target | {(time.time()-t0)/60:.2f} min", flush=True)

    sp2 = np.array([im.spacing for im in imgs], dtype=float)
    print("\n[SPACING] Após resample_image_to_target (deve bater com target.spacing)")
    print("  target.spacing:", target.spacing)
    print("  min :", sp2.min(axis=0))
    print("  mean:", sp2.mean(axis=0))
    print("  max :", sp2.max(axis=0))

    # 4) Máscaras
    t0 = time.time()
    masks = [brain_mask_safe(im) for im in imgs]
    target_mask = brain_mask_safe(target)
    print(f"[OK] Máscaras iniciais | {(time.time()-t0)/60:.2f} min", flush=True)

    # 5) Normalização robusta
    t0 = time.time()
    imgs = [robust_rescale_brain(im, mk, low=0.5, high=99.5) for im, mk in zip(imgs, masks)]
    print(f"[OK] Normalização robusta | {(time.time()-t0)/60:.2f} min", flush=True)

    # Barra global (registros)
    n_imgs = len(imgs)
    total_regs = 2 * n_imgs + (n_iter * n_imgs)  # Rigid + Affine + groupwise
    pbar = tqdm(total=total_regs, desc="Registros totais (Rigid + Affine + Groupwise)", unit="reg", ncols=110)

    # 6) Pré-alinhamento (Rigid → Affine)
    t0 = time.time()
    imgs_prealigned = []
    for im, mk in zip(imgs, masks):
        warped = prealign_to_target(im, target, mk, target_mask)
        imgs_prealigned.append(warped)
        pbar.update(2)
    print(f"[OK] Pré-alinhamento (Rigid + Affine) | {(time.time()-t0)/60:.2f} min", flush=True)

    # Máscaras pós pré-alinhamento
    t0 = time.time()
    masks_prealigned = [brain_mask_safe(im) for im in imgs_prealigned]
    print(f"[OK] Máscaras pós pré-alinhamento | {(time.time()-t0)/60:.2f} min", flush=True)

    # 7) Template inicial
    t0 = time.time()
    template = ants.average_images(imgs_prealigned)
    template = _resample_to_target(template, target)
    print(f"[OK] Template inicial | {(time.time()-t0)/60:.2f} min", flush=True)

    # 8) Loop groupwise
    for it in range(1, n_iter + 1):
        if verbose:
            print(f"\n[Groupwise] Iteração {it}/{n_iter} | Transformação: {type_of_transform}", flush=True)

        t_it = time.time()
        template_mask = brain_mask_safe(template)
        warped_list = []

        for img, moving_mask in zip(imgs_prealigned, masks_prealigned):
            reg = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform=type_of_transform,
                fixed_mask=template_mask,
                moving_mask=moving_mask,
                verbose=False
            )
            warped_list.append(reg["warpedmovout"])
            pbar.update(1)

        template = ants.average_images(warped_list)
        template = _resample_to_target(template, target)

        if verbose:
            print(f"[OK] Iteração {it} | {(time.time()-t_it)/60:.2f} min", flush=True)

    pbar.close()

    # 9) Normaliza template final para visualização
    t0 = time.time()
    final_mask = brain_mask_safe(template)
    template = robust_rescale_brain(template, final_mask, low=2.0, high=98.0)
    print(f"[OK] Normalização template final | {(time.time()-t0)/60:.2f} min", flush=True)

    print(f"\n[OK] Tempo total (build_groupwise_template): {(time.time()-t_global)/60:.2f} min", flush=True)
    return template


def _set_tmp_env(tmp_dir: Path):
    """
    Força o diretório temporário usado por bibliotecas (ANTs/ITK via ANTsPy e Python tempfile).
    """
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)

    tempfile.tempdir = str(tmp_dir)


def _make_run_tmp_dir(base_dir: Path) -> Path:
    """
    Cria um diretório temporário único por execução, dentro de base_dir.
    """
    return Path(tempfile.mkdtemp(prefix="ants_", dir=str(base_dir)))


def main(argv):
    t_start = time.time()

    run_tmp = _make_run_tmp_dir(TMP_ANTS_BASE)
    _set_tmp_env(run_tmp)

    print(f"[TMP] Usando diretório temporário: {run_tmp}", flush=True)
    print(f"[TMP] tempfile.gettempdir(): {tempfile.gettempdir()}", flush=True)

    def _cleanup_tmp():
        if KEEP_TMP_ANTS:
            print(f"[TMP] KEEP_TMP_ANTS=True, mantendo: {run_tmp}", flush=True)
            return
        shutil.rmtree(run_tmp, ignore_errors=True)

    try:
        age_min, age_max, sex = parse_args(argv)

        df = pd.read_csv(CSV_PATH)
        required = {"ID_IMG", "SEX", "AGE", "DIAG"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV não contém colunas obrigatórias: {missing_cols}")

        subset = filter_stratum(df, sex, age_min, age_max, N_MAX, RANDOM_SEED)
        subset_ok, id_to_path, missing_ids = resolve_paths(subset, IMAGES_DIR, SUFFIX)

        print(f"Filtro: DIAG={DIAG_FILTER} | SEX={sex} | AGE=[{age_min}, {age_max}] | N_MAX={N_MAX}", flush=True)
        print(f"Selecionadas (após amostragem): {len(subset)}", flush=True)
        print(f"Com arquivo encontrado: {len(subset_ok)}", flush=True)

        if missing_ids:
            print(f"Aviso: {len(missing_ids)} IDs sem arquivo correspondente. Ex.: {missing_ids[:10]}", flush=True)

        if len(subset_ok) < 2:
            raise RuntimeError("Após resolver caminhos, restaram menos de 2 imagens. Verifique IDs e diretório.")

        tag = f"DIAG-{DIAG_FILTER}_SEX-{sex}_AGE-{int(age_min)}-{int(age_max)}_N-{len(subset_ok)}"

        selected_csv = OUT_DIR / f"selected_{tag}.csv"
        subset_ok.to_csv(selected_csv, index=False)
        print(f"CSV selecionado: {selected_csv}", flush=True)

        image_paths = [id_to_path[str(i)] for i in subset_ok["ID_IMG"].astype(str)]

        template = build_groupwise_template(
            image_paths=image_paths,
            n_iter=N_ITER_TEMPLATE,
            type_of_transform=TYPE_OF_TRANSFORM,
            verbose=VERBOSE
        )

        out_template = OUT_DIR / f"groupwise_{tag}_template.nii.gz"
        ants.image_write(template, str(out_template))

        t_total = time.time() - t_start
        print(f"\nTempo total de execução (script): {t_total/60:.2f} min ({t_total:.1f} s)", flush=True)

        print("\nConcluído.", flush=True)
        print("Template final:", out_template, flush=True)

    finally:
        _cleanup_tmp()


if __name__ == "__main__":
    main(sys.argv)


# How to run:
# python groupwise_ants.py 50.0 59.0 F
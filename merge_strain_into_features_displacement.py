import os
from datetime import datetime

import ants
import numpy as np
import pandas as pd

import features_displacement as fd


# =========================
# Defaults (edite aqui)
# =========================
DEFAULT_IMAGES_CSV = fd.DEFAULT_IMAGES_CSV
DEFAULT_COMBINATIONS_CSV = "cj_data_abordagem_1_sMCI_pMCI.txt"
DEFAULT_BASE_FEATURES_CSV = os.path.join(
    fd.csvs_root, "abordagem_1_sMCI_pMCI", "features_displacement_abordagem_1_sMCI_pMCI.csv"
)


KEY_COLS = [
    "ID_PT",
    "COMBINATION_NUMBER",
    "TRIPLET_IDX",
    "pair",
    "roi",
    "side",
    "label",
]


def _fmt_s(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def _progress_line(*, done: int, total: int, started_at_ts: float) -> str:
    now = datetime.now().timestamp()
    elapsed = now - started_at_ts
    rate = (done / elapsed) if elapsed > 0 else 0.0
    rem = ((total - done) / rate) if rate > 0 else float("inf")
    pct = (100.0 * done / total) if total > 0 else 0.0
    return (
        f"[PROGRESS] triplets {done}/{total} ({pct:.1f}%) "
        f"elapsed={_fmt_s(elapsed)} "
        f"rate={rate:.3f} triplets/s "
        f"eta={_fmt_s(rem) if np.isfinite(rem) else '??'}"
    )


def strain_from_displacement(delta_field: ants.ANTsImage):
    """
    Small-strain tensor: e = 0.5*(grad(u) + grad(u)^T)
    Retorna arrays float32:
      exx, eyy, ezz, exy, exz, eyz, enormF
    """
    u = delta_field.numpy().astype(np.float32, copy=False)  # (..., 3)
    sx, sy, sz = map(float, delta_field.spacing)

    dux_dx, dux_dy, dux_dz = np.gradient(u[..., 0], sx, sy, sz, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(u[..., 1], sx, sy, sz, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(u[..., 2], sx, sy, sz, edge_order=1)

    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyz = 0.5 * (duy_dz + duz_dy)

    enorm = np.sqrt(
        exx * exx
        + eyy * eyy
        + ezz * ezz
        + 2.0 * (exy * exy + exz * exz + eyz * eyz)
    )

    return (
        exx.astype(np.float32, copy=False),
        eyy.astype(np.float32, copy=False),
        ezz.astype(np.float32, copy=False),
        exy.astype(np.float32, copy=False),
        exz.astype(np.float32, copy=False),
        eyz.astype(np.float32, copy=False),
        enorm.astype(np.float32, copy=False),
    )


def compute_pair_strain_arrays(domain_img, fwd_list_a: list, inv_list_b: list):
    delta = fd.relative_displacement_field(domain_img, fwd_list_a, inv_list_b)
    return strain_from_displacement(delta)


def _pack_strain_stats(roi_mask: np.ndarray, strain_arrays: tuple[np.ndarray, ...]) -> dict[str, float]:
    exx, eyy, ezz, exy, exz, eyz, en = strain_arrays

    s_exx = fd._stats(exx[roi_mask])
    s_eyy = fd._stats(eyy[roi_mask])
    s_ezz = fd._stats(ezz[roi_mask])
    s_exy = fd._stats(exy[roi_mask])
    s_exz = fd._stats(exz[roi_mask])
    s_eyz = fd._stats(eyz[roi_mask])
    s_en = fd._stats(en[roi_mask])

    # Colunas só de strain (sem duplicar as features já existentes no CSV base)
    out = {
        "strain_exx_n": s_exx["n"],
        "strain_exx_mean": s_exx["mean"],
        "strain_exx_std": s_exx["std"],
        "strain_exx_p05": s_exx["p05"],
        "strain_exx_p50": s_exx["p50"],
        "strain_exx_p95": s_exx["p95"],
        "strain_eyy_n": s_eyy["n"],
        "strain_eyy_mean": s_eyy["mean"],
        "strain_eyy_std": s_eyy["std"],
        "strain_eyy_p05": s_eyy["p05"],
        "strain_eyy_p50": s_eyy["p50"],
        "strain_eyy_p95": s_eyy["p95"],
        "strain_ezz_n": s_ezz["n"],
        "strain_ezz_mean": s_ezz["mean"],
        "strain_ezz_std": s_ezz["std"],
        "strain_ezz_p05": s_ezz["p05"],
        "strain_ezz_p50": s_ezz["p50"],
        "strain_ezz_p95": s_ezz["p95"],
        "strain_exy_n": s_exy["n"],
        "strain_exy_mean": s_exy["mean"],
        "strain_exy_std": s_exy["std"],
        "strain_exy_p05": s_exy["p05"],
        "strain_exy_p50": s_exy["p50"],
        "strain_exy_p95": s_exy["p95"],
        "strain_exz_n": s_exz["n"],
        "strain_exz_mean": s_exz["mean"],
        "strain_exz_std": s_exz["std"],
        "strain_exz_p05": s_exz["p05"],
        "strain_exz_p50": s_exz["p50"],
        "strain_exz_p95": s_exz["p95"],
        "strain_eyz_n": s_eyz["n"],
        "strain_eyz_mean": s_eyz["mean"],
        "strain_eyz_std": s_eyz["std"],
        "strain_eyz_p05": s_eyz["p05"],
        "strain_eyz_p50": s_eyz["p50"],
        "strain_eyz_p95": s_eyz["p95"],
        "strain_enormF_n": s_en["n"],
        "strain_enormF_mean": s_en["mean"],
        "strain_enormF_std": s_en["std"],
        "strain_enormF_p05": s_en["p05"],
        "strain_enormF_p50": s_en["p50"],
        "strain_enormF_p95": s_en["p95"],
    }
    return out


def _count_triplets(df_comb: pd.DataFrame) -> int:
    total = 0
    for _, group in df_comb.groupby(["ID_PT", "COMBINATION_NUMBER"]):
        ids_all = group["ID_IMG"].astype(str).values
        if len(ids_all) >= 3 and (len(ids_all) % 3) == 0:
            total += (len(ids_all) // 3)
    return int(total)


def _triplet_key(id_pt: str, comb_id: int, triplet_idx: int) -> str:
    return f"{id_pt}|{int(comb_id)}|t{int(triplet_idx)}"


def _checkpoint_dir_for_outputs(*, base_csv_path: str, out_csv_path: str | None) -> str:
    """
    Diretório de checkpoint (Parquet parts + done_keys) derivado do CSV final.
    """
    if out_csv_path is None:
        base_dir = os.path.dirname(base_csv_path)
        base_name = os.path.basename(base_csv_path)
        out_csv_path = os.path.join(base_dir, base_name.replace(".csv", "_with_strain.csv"))
    out_dir = os.path.dirname(out_csv_path)
    stem = os.path.splitext(os.path.basename(out_csv_path))[0]
    return os.path.join(out_dir, f".{stem}_checkpoint")


def _list_parquet_parts(parts_dir: str) -> list[str]:
    if not os.path.isdir(parts_dir):
        return []
    files = [
        os.path.join(parts_dir, fn)
        for fn in os.listdir(parts_dir)
        if fn.endswith(".parquet") and (not fn.startswith("."))
    ]
    files.sort()
    return files


def _concat_parquet_parts(parts: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, axis=0, ignore_index=True)


def process_triplets_with_checkpoints(
    *,
    base_df: pd.DataFrame,
    csv_comb_path: str,
    csv_images_path: str,
    checkpoint_dir: str,
    log_every_triplets: int,
):
    df_comb = pd.read_csv(csv_comb_path)
    df_imgs = pd.read_csv(csv_images_path)
    ref_by_pt = fd.build_baseline_reference_map(df_imgs)

    if "MRI_DATE" in df_comb.columns:
        df_comb["MRI_DATE"] = pd.to_datetime(df_comb["MRI_DATE"], errors="coerce")

    os.makedirs(checkpoint_dir, exist_ok=True)
    parts_dir = os.path.join(checkpoint_dir, "merged_parts")
    os.makedirs(parts_dir, exist_ok=True)
    done_keys_path = os.path.join(checkpoint_dir, "done_keys.txt")

    done = fd.load_done_keys(done_keys_path)

    total_triplets = _count_triplets(df_comb)
    started_at_ts = datetime.now().timestamp()
    done_triplets = 0
    skipped_no_ref = 0
    skipped_missing_warps = 0
    skipped_missing_inputs = 0
    errors = 0
    skipped_already_done = 0

    log_every_triplets = int(log_every_triplets) if int(log_every_triplets) > 0 else 10
    print(
        f"[INFO] total_triplets={total_triplets} log_every_triplets={log_every_triplets} "
        f"checkpoint_dir={checkpoint_dir} already_done={len(done)}",
        flush=True,
    )

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

        ref = ref_by_pt.get(str(id_pt))
        if ref is None or (not os.path.exists(ref.ref_path)):
            skipped_no_ref += (len(ids_all) // 3)
            continue
        ref_tag = f"CN_SEX-{ref.sex}_AGE-{ref.age_range}"

        for triplet_idx in range(len(ids_all) // 3):
            ids = ids_all[triplet_idx * 3 : (triplet_idx + 1) * 3]
            if len(ids) != 3:
                continue

            key = _triplet_key(str(id_pt), int(comb_id), int(triplet_idx))
            if key in done:
                skipped_already_done += 1
                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done}",
                        flush=True,
                    )
                continue

            # Warps necessários
            fwd_inv_by_id = {}
            missing = []
            for sid in ids:
                fw, aff, iw, fwd_l, inv_l = fd.fwd_inv_paths_for(sid, ref_tag)
                for p in (fw, aff, iw):
                    if not os.path.isfile(p):
                        missing.append(p)
                fwd_inv_by_id[sid] = (fwd_l, inv_l)
            if missing:
                skipped_missing_warps += 1
                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done}",
                        flush=True,
                    )
                continue

            fixed_baseline = fd.fixed_path_for(ids[0])
            fixed_t2 = fd.fixed_path_for(ids[1])
            if not os.path.isfile(fixed_baseline) or not os.path.isfile(fixed_t2):
                skipped_missing_inputs += 1
                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done}",
                        flush=True,
                    )
                continue

            regions12_p = os.path.join(fd.regions_dir, f"{str(ids[0]).strip()}_regions.nii.gz")
            regions23_p = os.path.join(fd.regions_dir, f"{str(ids[1]).strip()}_regions.nii.gz")
            if not os.path.isfile(regions12_p) or not os.path.isfile(regions23_p):
                skipped_missing_inputs += 1
                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done}",
                        flush=True,
                    )
                continue

            try:
                domain_img_1 = ants.image_read(fixed_baseline)
                domain_img_2 = ants.image_read(fixed_t2)

                fwd1, _ = fwd_inv_by_id[ids[0]]
                fwd2, _ = fwd_inv_by_id[ids[1]]
                _, inv2 = fwd_inv_by_id[ids[1]]
                _, inv3 = fwd_inv_by_id[ids[2]]

                strain12 = compute_pair_strain_arrays(domain_img_1, fwd1, inv2)
                strain13 = compute_pair_strain_arrays(domain_img_1, fwd1, inv3)
                strain23 = compute_pair_strain_arrays(domain_img_2, fwd2, inv3)

                labels12 = fd._load_and_resample_labelmap(regions12_p, domain_img_1)
                labels23 = fd._load_and_resample_labelmap(regions23_p, domain_img_2)

                bm12_p = os.path.join(fd.brain_mask_dir, f"{str(ids[0]).strip()}_brain_mask.nii.gz")
                bm23_p = os.path.join(fd.brain_mask_dir, f"{str(ids[1]).strip()}_brain_mask.nii.gz")
                brain_mask12 = (
                    fd._load_and_resample_mask(bm12_p, domain_img_1) if os.path.isfile(bm12_p) else None
                )
                brain_mask23 = (
                    fd._load_and_resample_mask(bm23_p, domain_img_2) if os.path.isfile(bm23_p) else None
                )

                strain_rows: list[dict[str, object]] = []
                for roi, side, label in fd.ROI_TABLE:
                    lab = int(label)

                    roi_mask12 = labels12 == lab
                    if brain_mask12 is not None:
                        roi_mask12 = roi_mask12 & brain_mask12

                    roi_mask23 = labels23 == lab
                    if brain_mask23 is not None:
                        roi_mask23 = roi_mask23 & brain_mask23

                    base_key = {
                        "ID_PT": str(id_pt),
                        "COMBINATION_NUMBER": int(comb_id),
                        "TRIPLET_IDX": int(triplet_idx),
                        "roi": str(roi),
                        "side": str(side),
                        "label": str(label),
                    }

                    strain_rows.append(
                        {
                            **base_key,
                            "pair": "12",
                            **_pack_strain_stats(roi_mask12, strain12),
                        }
                    )
                    strain_rows.append(
                        {
                            **base_key,
                            "pair": "13",
                            **_pack_strain_stats(roi_mask12, strain13),
                        }
                    )
                    strain_rows.append(
                        {
                            **base_key,
                            "pair": "23",
                            **_pack_strain_stats(roi_mask23, strain23),
                        }
                    )

                strain_df = pd.DataFrame(strain_rows)
                strain_df["ID_PT"] = strain_df["ID_PT"].astype(str)
                strain_df["pair"] = strain_df["pair"].astype(str)
                strain_df["roi"] = strain_df["roi"].astype(str)
                strain_df["side"] = strain_df["side"].astype(str)
                strain_df["label"] = strain_df["label"].astype(str)

                base_slice = base_df[
                    (base_df["ID_PT"].astype(str) == str(id_pt))
                    & (base_df["COMBINATION_NUMBER"].astype(int) == int(comb_id))
                    & (base_df["TRIPLET_IDX"].astype(int) == int(triplet_idx))
                ].copy()

                if base_slice.empty:
                    raise RuntimeError("base_slice vazio: triplet não existe no CSV base")

                merged_part = base_slice.merge(strain_df, on=KEY_COLS, how="left", validate="one_to_one")

                part_path = fd.write_parquet_part(
                    merged_part,
                    dataset_dir=parts_dir,
                    part_prefix=f"pt{str(id_pt)}_c{int(comb_id)}_t{int(triplet_idx)}",
                )
                fd.append_done_key(done_keys_path, key)
                done.add(key)

                print(f"[CKPT] saved {part_path} key={key} rows={len(merged_part)}", flush=True)

                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done}",
                        flush=True,
                    )
            except Exception as e:
                errors += 1
                done_triplets += 1
                if (done_triplets % log_every_triplets) == 0:
                    print(
                        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
                        + f" | skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
                        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} "
                        f"skipped_already_done={skipped_already_done} last_error={type(e).__name__}",
                        flush=True,
                    )
                continue

    print(
        _progress_line(done=done_triplets, total=total_triplets, started_at_ts=started_at_ts)
        + f" | DONE process_triplets: skipped_no_ref={skipped_no_ref} skipped_missing_warps={skipped_missing_warps} "
        f"skipped_missing_inputs={skipped_missing_inputs} errors={errors} skipped_already_done={skipped_already_done}",
        flush=True,
    )


def merge_strain_into_base_csv(
    *,
    base_csv_path: str,
    csv_comb_path: str,
    csv_images_path: str,
    out_csv_path: str | None = None,
):
    base_df = pd.read_csv(base_csv_path)

    # garante chaves existentes
    for c in KEY_COLS:
        if c not in base_df.columns:
            raise RuntimeError(f"Coluna obrigatória ausente no CSV base: {c}")

    if out_csv_path is None:
        base_dir = os.path.dirname(base_csv_path)
        base_name = os.path.basename(base_csv_path)
        out_csv_path = os.path.join(base_dir, base_name.replace(".csv", "_with_strain.csv"))

    log_every = int(os.environ.get("DISP_MERGE_STRAIN_LOG_EVERY", "10"))
    checkpoint_dir = os.environ.get("DISP_MERGE_STRAIN_CKPT_DIR", "").strip() or _checkpoint_dir_for_outputs(
        base_csv_path=base_csv_path, out_csv_path=out_csv_path
    )

    # normaliza tipos para merge estável (uma vez)
    base_df = base_df.copy()
    base_df["ID_PT"] = base_df["ID_PT"].astype(str)
    base_df["pair"] = base_df["pair"].astype(str)
    base_df["roi"] = base_df["roi"].astype(str)
    base_df["side"] = base_df["side"].astype(str)
    base_df["label"] = base_df["label"].astype(str)

    process_triplets_with_checkpoints(
        base_df=base_df,
        csv_comb_path=csv_comb_path,
        csv_images_path=csv_images_path,
        checkpoint_dir=checkpoint_dir,
        log_every_triplets=log_every,
    )

    parts = _list_parquet_parts(os.path.join(checkpoint_dir, "merged_parts"))
    merged = _concat_parquet_parts(parts)
    if merged.empty:
        raise RuntimeError(
            "merged vazio após checkpoints. Verifique se houve erros/skips ou se o checkpoint_dir está correto."
        )

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    merged.to_csv(out_csv_path, index=False)
    return out_csv_path, merged


if __name__ == "__main__":
    comb_path = os.environ.get("DISP_MERGE_STRAIN_COMBINATIONS_CSV", DEFAULT_COMBINATIONS_CSV)
    images_csv = os.environ.get("DISP_MERGE_STRAIN_IMAGES_CSV", DEFAULT_IMAGES_CSV)
    base_csv = os.environ.get("DISP_MERGE_STRAIN_BASE_CSV", DEFAULT_BASE_FEATURES_CSV)
    out_csv = os.environ.get("DISP_MERGE_STRAIN_OUT_CSV", "").strip() or None

    print(
        "[START] merge_strain_into_features_displacement.py "
        f"base={base_csv} comb={comb_path} images={images_csv} out={out_csv or '(auto)'} "
        f"at={datetime.now().isoformat(timespec='seconds')}"
    )
    out_path, _ = merge_strain_into_base_csv(
        base_csv_path=base_csv,
        csv_comb_path=comb_path,
        csv_images_path=images_csv,
        out_csv_path=out_csv,
    )
    print(f"[DONE] wrote: {out_path}")


"""NeuroComBat por fold externo (fit no treino, transform em treino ∪ teste).

Usa ``neuroCombat`` no treino e ``neuroCombatFromTraining`` nas imagens de teste
com batches já vistos no fit.
Covariáveis: batch, AGE, SEX. Unidade estatística: uma linha wide por ``ID_IMG_ref``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from neuroCombat.neuroCombat import neuroCombatFromTraining

import exp_utils as u

MIN_BATCH_SAMPLES = 3


def combat_feature_columns(feat_names: list[str]) -> list[str]:
    """Radiomics/volume para ComBat (SEX fica só nas covariáveis)."""
    return [c for c in feat_names if c != "SEX"]


def _wide_col_key(row: pd.Series, feat: str) -> str:
    pair = str(row["pair"]).strip()
    roi = str(row["roi"]) if "roi" in row.index else "NA"
    side = str(row["side"]) if "side" in row.index else "NA"
    label = str(row["label"]) if "label" in row.index else "NA"
    return f"{pair}|{roi}|{side}|{label}|{feat}"


def image_refs_for_sample_indices(
    df: pd.DataFrame,
    group_key: list[str],
    pair_order: list[str],
    sample_indices: np.ndarray,
) -> set[str]:
    idx_set = {int(i) for i in np.asarray(sample_indices, dtype=int)}
    refs: set[str] = set()
    for sample_idx, block in u.iter_triplet_blocks(df, group_key, pair_order):
        if sample_idx not in idx_set:
            continue
        refs.update(block["ID_IMG_ref"].astype(str).str.strip().tolist())
    return refs


def _build_wide_table(
    df: pd.DataFrame,
    image_refs: set[str],
    combat_feats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Retorna (wide_features, covars, image_ids) para imagens em ``image_refs``."""
    ref_list = sorted(str(r).strip() for r in image_refs)
    sub = df[df["ID_IMG_ref"].astype(str).str.strip().isin(ref_list)].copy()
    if sub.empty:
        raise ValueError("Nenhuma linha no CSV para os ID_IMG_ref pedidos.")

    feat_by_img: dict[str, dict[str, float]] = {}
    cov_by_img: dict[str, dict[str, Any]] = {}

    for img_ref, g in sub.groupby(sub["ID_IMG_ref"].astype(str).str.strip(), sort=True):
        feat_row: dict[str, float] = {}
        for _, r in g.iterrows():
            for feat in combat_feats:
                key = _wide_col_key(r, feat)
                val = pd.to_numeric(r[feat], errors="coerce")
                if pd.isna(val):
                    continue
                feat_row[key] = float(val)

        batch = g["batch"].iloc[0]
        age = pd.to_numeric(g["AGE"].iloc[0], errors="coerce")
        if pd.isna(batch) or not np.isfinite(float(age)):
            continue
        try:
            sex_val = float(u.encode_sex_column(g["SEX"])[0])
        except ValueError:
            continue

        feat_by_img[str(img_ref)] = feat_row
        cov_by_img[str(img_ref)] = {
            "batch": str(batch),
            "AGE": float(age),
            "SEX": sex_val,
        }

    if not feat_by_img:
        raise ValueError("Nenhuma imagem com batch/AGE/SEX/features completos para ComBat.")

    col_order = sorted({k for row in feat_by_img.values() for k in row})
    rows: list[dict[str, float]] = []
    img_ids: list[str] = []
    cov_rows: list[dict[str, Any]] = []
    for img_ref in sorted(feat_by_img):
        feat_row = feat_by_img[img_ref]
        values = [feat_row.get(k, np.nan) for k in col_order]
        if not np.all(np.isfinite(values)):
            continue
        rows.append(dict(zip(col_order, values, strict=True)))
        img_ids.append(img_ref)
        cov_rows.append(cov_by_img[img_ref])

    if not rows:
        raise ValueError("Nenhuma imagem com todas as features preenchidas para ComBat.")

    wide = pd.DataFrame(rows, index=img_ids).sort_index()
    covars = pd.DataFrame(cov_rows, index=img_ids).sort_index()
    return wide, covars, img_ids


def _write_wide_back(
    df_out: pd.DataFrame,
    wide_harmonized: pd.DataFrame,
    combat_feats: list[str],
    image_refs: set[str],
) -> None:
    ref_set = {str(r).strip() for r in image_refs}
    lookup = wide_harmonized.to_dict("index")
    mask = df_out["ID_IMG_ref"].astype(str).str.strip().isin(ref_set)
    for idx in df_out.index[mask]:
        row = df_out.loc[idx]
        img = str(row["ID_IMG_ref"]).strip()
        if img not in lookup:
            continue
        vals = lookup[img]
        for feat in combat_feats:
            key = _wide_col_key(row, feat)
            if key not in vals:
                continue
            val = vals[key]
            if np.isfinite(val):
                df_out.at[idx, feat] = val


def _run_neurocombat_train(
    wide_train: pd.DataFrame,
    cov_train: pd.DataFrame,
    train_ids: list[str],
) -> tuple[pd.DataFrame, dict]:
    dat = wide_train.loc[train_ids].to_numpy(dtype=np.float64).T
    covars = cov_train.loc[train_ids].copy()
    res = neuroCombat(
        dat=dat,
        covars=covars,
        batch_col="batch",
        categorical_cols=["SEX"],
        continuous_cols=["AGE"],
        eb=True,
        parametric=True,
        mean_only=False,
    )
    wide_h = pd.DataFrame(
        np.asarray(res["data"], dtype=np.float64).T,
        index=train_ids,
        columns=wide_train.columns,
    )
    return wide_h, res["estimates"]


def _run_neurocombat_apply(
    wide: pd.DataFrame,
    cov: pd.DataFrame,
    image_ids: list[str],
    estimates: dict,
) -> pd.DataFrame:
    dat = wide.loc[image_ids].to_numpy(dtype=np.float64).T
    batch = cov.loc[image_ids, "batch"].to_numpy(dtype=str)
    res = neuroCombatFromTraining(dat=dat, batch=batch, estimates=estimates)
    return pd.DataFrame(
        np.asarray(res["data"], dtype=np.float64).T,
        index=image_ids,
        columns=wide.columns,
    )


def harmonize_dataframe_fold(
    df: pd.DataFrame,
    *,
    exp_md_path: Path,
    train_image_refs: set[str],
    transform_image_refs: set[str],
    fold_id: int = 0,
) -> pd.DataFrame:
    """Fit ComBat no treino do fold; transforma imagens de treino ∪ teste."""
    if not train_image_refs:
        raise ValueError("train_image_refs vazio — impossível ajustar ComBat.")
    if not transform_image_refs:
        return df.copy()

    feat_names = u.parse_feature_columns(exp_md_path)
    combat_feats = combat_feature_columns(feat_names)
    if not combat_feats:
        raise ValueError("Nenhuma coluna radiomica para harmonizar.")

    for col in ("ID_IMG_ref", "batch", "AGE", "SEX", "pair", "roi", "side", "label"):
        if col not in df.columns:
            raise ValueError(f"Coluna {col!r} ausente no CSV (necessária para ComBat).")

    df_out = df.copy()
    wide_all, cov_all, all_ids = _build_wide_table(df, transform_image_refs, combat_feats)
    train_ids = sorted(
        img for img in wide_all.index if str(img) in {str(r).strip() for r in train_image_refs}
    )
    wide_train = wide_all.loc[train_ids]
    cov_train = cov_all.loc[train_ids]

    train_batches = cov_train["batch"].value_counts()
    if len(train_batches) < 2:
        warnings.warn(
            f"Fold {fold_id + 1}: ComBat ignorado (<2 batches no treino: "
            f"{train_batches.to_dict()}).",
            stacklevel=2,
        )
        return df_out

    small = train_batches[train_batches < MIN_BATCH_SAMPLES]
    if len(small):
        warnings.warn(
            f"Fold {fold_id + 1}: batches com <{MIN_BATCH_SAMPLES} imagens no treino: "
            f"{small.to_dict()}",
            stacklevel=2,
        )

    fitted_batches = set(cov_train["batch"].astype(str).tolist())
    known_mask = cov_all["batch"].astype(str).isin(fitted_batches)
    unknown_imgs = [img for img, ok in zip(all_ids, known_mask.tolist()) if not ok]
    if unknown_imgs:
        warnings.warn(
            f"Fold {fold_id + 1}: {len(unknown_imgs)} imagem(ns) com batch só no teste "
            f"(mantidas sem harmonização).",
            stacklevel=2,
        )

    known_ids = [img for img, ok in zip(all_ids, known_mask.tolist()) if ok]
    if len(known_ids) < 2:
        warnings.warn(
            f"Fold {fold_id + 1}: ComBat ignorado (<2 imagens harmonizáveis).",
            stacklevel=2,
        )
        return df_out

    train_known = [img for img in train_ids if img in set(known_ids)]
    if len(train_known) < 2:
        warnings.warn(
            f"Fold {fold_id + 1}: ComBat ignorado (<2 imagens no treino harmonizável).",
            stacklevel=2,
        )
        return df_out

    wide_h_train, estimates = _run_neurocombat_train(wide_train, cov_train, train_known)

    test_only = [img for img in known_ids if img not in set(train_known)]
    wide_parts = [wide_h_train]
    if test_only:
        wide_h_test = _run_neurocombat_apply(wide_all, cov_all, test_only, estimates)
        wide_parts.append(wide_h_test)
    wide_h = pd.concat(wide_parts)

    _write_wide_back(df_out, wide_h, combat_feats, set(known_ids))

    if fold_id == 0:
        print(
            f"Fold 1 — NeuroComBat: fit em {len(train_known)} imagens, "
            f"transform em {len(known_ids)} imagens "
            f"({len(unknown_imgs)} sem batch visto no treino)."
        )
    return df_out


def harmonize_fold_for_samples(
    df: pd.DataFrame,
    *,
    exp_md_path: Path,
    group_key: list[str],
    pair_order: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: int = 0,
) -> pd.DataFrame:
    train_refs = image_refs_for_sample_indices(df, group_key, pair_order, train_idx)
    test_refs = image_refs_for_sample_indices(df, group_key, pair_order, test_idx)
    transform_refs = train_refs | test_refs
    return harmonize_dataframe_fold(
        df,
        exp_md_path=exp_md_path,
        train_image_refs=train_refs,
        transform_image_refs=transform_refs,
        fold_id=fold_id,
    )

"""Harmonização NeuroComBat por fold (exp1/exp2) com neurocombat-sklearn.

Wide por ID_IMG_ref (ROI nas colunas), fit no treino externo do fold, transform em
todas as imagens do fold antes de load_tensor (delta_rate ou baseline_rate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import exp_utils as u


def _patch_neurocombat_sklearn_one_hot() -> None:
    """sklearn >= 1.2 removeu o argumento sparse= de OneHotEncoder."""
    import neurocombat_sklearn.neurocombat_sklearn as _ncs
    from sklearn.preprocessing import OneHotEncoder as _SklearnOHE

    class _CompatOneHotEncoder(_SklearnOHE):
        def __init__(
            self,
            categories: str = "auto",
            drop: str | None = None,
            sparse: bool = False,
            dtype: type | np.dtype = np.float64,
            handle_unknown: str = "error",
        ):
            self.sparse = sparse
            super().__init__(
                categories=categories,
                drop=drop,
                sparse_output=sparse,
                dtype=dtype,
                handle_unknown=handle_unknown,
            )

    _ncs.OneHotEncoder = _CompatOneHotEncoder


try:
    _patch_neurocombat_sklearn_one_hot()
    from neurocombat_sklearn import CombatModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "neurocombat-sklearn é necessário para RUN_NEUROCOMBAT=1. "
        "Instale com: pip install -r requirements-neurocombat.txt"
    ) from e

GROUP_KEY_DEFAULT = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
ROI_DEDUP_KEYS = ["ID_IMG_ref", "roi", "side", "label"]
IMAGE_COVAR_KEYS = ["batch", "AGE", "SEX", "ID_IMG_ref"]


def env_bool(key: str, default: bool) -> bool:
    import os

    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes")


def read_features_long_csv(csv_path: Path) -> pd.DataFrame:
    """Carrega CSV long (deltas ou unitários); remove MRI_DATE se existir."""
    df = pd.read_csv(csv_path)
    if "MRI_DATE" in df.columns:
        df = df.drop(columns=["MRI_DATE"])
    return df


read_unitary_csv = read_features_long_csv


def triplet_keys_in_tensor_order(
    df: pd.DataFrame, group_key: list[str]
) -> list[tuple[Any, ...]]:
    """Mesma ordem que load_tensor_from_dataframe (groupby sort=False)."""
    keys: list[tuple[Any, ...]] = []
    for keys_row, _g in df.groupby(group_key, sort=False):
        if not isinstance(keys_row, tuple):
            keys_row = (keys_row,)
        keys.append(keys_row)
    return keys


def triplet_meta(
    df: pd.DataFrame,
    group_key: list[str],
    pair_order: list[str],
    *,
    require_sex: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rótulos e grupos por triplet (sem construir tensor)."""
    y_map = {"sMCI": 0, "pMCI": 1}
    y_out: list[int] = []
    groups_out: list[str] = []
    sex_out: list[int] = []
    has_sex = "SEX" in df.columns
    if require_sex and not has_sex:
        raise ValueError("Coluna SEX ausente no CSV (necessária com require_sex=True).")

    for _keys, g in df.groupby(group_key, sort=False):
        rows = []
        for p in pair_order:
            gp = g[g["pair"].astype(str).str.strip() == p]
            if gp.empty:
                rows = None
                break
            rows.append(gp)
        if rows is None:
            continue
        block = pd.concat(rows, axis=0)
        if len(block) != 60:
            continue
        grp = str(block["GROUP"].iloc[0])
        if grp not in y_map:
            raise ValueError(f"GROUP inesperado: {grp!r}")
        y_out.append(y_map[grp])
        groups_out.append(str(block["ID_PT"].iloc[0]))
        if has_sex:
            sex_out.append(int(u.encode_sex_column(block["SEX"])[0]))
        else:
            sex_out.append(0)

    if not y_out:
        raise RuntimeError("Nenhum triplet válido (60 linhas) para meta.")
    return (
        np.asarray(y_out, dtype=np.int32),
        np.asarray(groups_out, dtype=object),
        np.asarray(sex_out, dtype=np.int8),
    )


def _meta_and_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    key_cols = [
        "ID_PT",
        "COMBINATION_NUMBER",
        "TRIPLET_IDX",
        "pair",
        "ID_IMG_i1",
        "ID_IMG_i2",
        "ID_IMG_i3",
        "ref_tag",
        "roi",
        "side",
        "label",
    ]
    extra_cols = ["t12", "t13", "t23", "GROUP", "SEX", "DIAG", "AGE", "TIME_PROG"]
    technical_cols = [
        "ID_IMG_ref",
        "FIELD_STRENGTH",
        "SLICE_THICKNESS",
        "MANUFACTURER",
        "MFG_MODEL",
        "batch",
    ]
    meta_cols = [c for c in key_cols + extra_cols + technical_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("Nenhuma coluna de feature encontrada no CSV unitário.")
    return meta_cols, feature_cols


def _roi_slot_name(roi: str, side: str, label: Any) -> str:
    return f"{roi}|{side}|{label}"


def _image_table_dedup(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Uma linha por (ID_IMG_ref, roi, side, label) com covariáveis por imagem."""
    work = df.copy()
    work["ID_IMG_ref"] = work["ID_IMG_ref"].astype(str).str.strip()
    for c in ("roi", "side"):
        work[c] = work[c].astype(str).str.strip()
    work["_roi_slot"] = [
        _roi_slot_name(r, s, lb)
        for r, s, lb in zip(work["roi"], work["side"], work["label"], strict=True)
    ]

    dup_mask = work.duplicated(subset=ROI_DEDUP_KEYS, keep=False)
    if dup_mask.any():
        agg = work.loc[dup_mask].groupby(ROI_DEDUP_KEYS)[feature_cols]
        for key, sub in agg:
            if sub.nunique().max() > 1:
                print(
                    f"Aviso ComBat: valores divergentes para {key}; "
                    "usando primeira ocorrência."
                )

    keep_cols = ROI_DEDUP_KEYS + ["batch", "AGE", "SEX"] + feature_cols + ["_roi_slot"]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in work.columns))
    dedup = work.drop_duplicates(subset=ROI_DEDUP_KEYS, keep="first")
    return dedup[keep_cols]


def _wide_from_image_table(img: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    covar = img.groupby("ID_IMG_ref", sort=False).first()[["batch", "AGE", "SEX"]]
    feat = img.pivot_table(
        index="ID_IMG_ref",
        columns="_roi_slot",
        values=feature_cols,
        aggfunc="first",
    )
    if isinstance(feat.columns, pd.MultiIndex):
        feat.columns = [f"{slot}__{metric}" for metric, slot in feat.columns]
    else:
        feat.columns = [str(c) for c in feat.columns]
    wide = covar.join(feat, how="inner")
    wide.index.name = "ID_IMG_ref"
    return wide.reset_index()


def _encode_sex_for_combat(series: pd.Series) -> np.ndarray:
    return u.encode_sex_column(series).reshape(-1, 1)


def _fit_transform_wide(
    wide: pd.DataFrame,
    train_images: set[str],
    *,
    min_batch_count: int = 3,
) -> pd.DataFrame:
    feat_cols = [c for c in wide.columns if c not in IMAGE_COVAR_KEYS]
    work = wide.copy()
    work["ID_IMG_ref"] = work["ID_IMG_ref"].astype(str)
    work["AGE"] = pd.to_numeric(work["AGE"], errors="coerce")

    mask = (
        work["batch"].notna()
        & work["AGE"].notna()
        & work["SEX"].notna()
        & work[feat_cols].notna().all(axis=1)
    )
    if (~mask).any():
        n_drop = int((~mask).sum())
        print(f"ComBat: imagens excluídas (batch/AGE/SEX/features NaN): {n_drop}")

    work = work.loc[mask].reset_index(drop=True)
    train_mask = work["ID_IMG_ref"].isin(train_images)
    train_df = work.loc[train_mask]
    if train_df.empty:
        raise ValueError("ComBat: nenhuma imagem de treino válida após filtro.")

    n_batch = train_df["batch"].nunique()
    if n_batch < 2:
        raise ValueError(
            f"ComBat requer >= 2 batches no treino do fold; encontrados: {n_batch}"
        )

    batch_counts = train_df["batch"].value_counts()
    small = batch_counts[batch_counts < min_batch_count]
    if len(small):
        print(
            "Aviso ComBat: batches com <3 imagens no treino do fold:",
            small.to_dict(),
        )

    X_tr = train_df[feat_cols].to_numpy(dtype=float)
    batch_le = LabelEncoder()
    sites_tr = batch_le.fit_transform(train_df["batch"].astype(str)).reshape(-1, 1)
    sex_tr = _encode_sex_for_combat(train_df["SEX"])
    age_tr = train_df[["AGE"]].to_numpy(dtype=float)

    model = CombatModel()
    model.fit(
        X_tr,
        sites_tr,
        discrete_covariates=sex_tr,
        continuous_covariates=age_tr,
    )

    batch_all = work["batch"].astype(str)
    seen_mask = batch_all.isin(batch_le.classes_)
    unseen_batch = sorted(set(batch_all.unique()) - set(batch_le.classes_))
    if unseen_batch:
        print(
            "Aviso ComBat: batch(es) só no teste do fold (sem harmonizar essas imagens):",
            unseen_batch,
        )

    X_out = work[feat_cols].to_numpy(dtype=float)
    if seen_mask.any():
        sites_seen = batch_le.transform(batch_all[seen_mask]).reshape(-1, 1)
        X_seen = work.loc[seen_mask, feat_cols].to_numpy(dtype=float)
        sex_seen = _encode_sex_for_combat(work.loc[seen_mask, "SEX"])
        age_seen = work.loc[seen_mask, ["AGE"]].to_numpy(dtype=float)
        X_harm_seen = model.transform(
            X_seen,
            sites_seen,
            discrete_covariates=sex_seen,
            continuous_covariates=age_seen,
        )
        X_out[seen_mask.to_numpy()] = X_harm_seen

    out = work[["ID_IMG_ref", "batch", "AGE", "SEX"]].copy()
    out[feat_cols] = X_out
    return out


def _long_keys_frame(df: pd.DataFrame, group_key: list[str]) -> pd.DataFrame:
    sub = df[list(group_key)].drop_duplicates()
    for c in group_key:
        sub[c] = sub[c].astype(str) if c == "ID_PT" else sub[c]
    return sub


def _triplet_image_sets(
    df_long: pd.DataFrame,
    keys: list[tuple[Any, ...]],
    triplet_idx: np.ndarray,
    group_key: list[str],
) -> tuple[set[tuple[Any, ...]], set[str]]:
    triplet_idx = np.asarray(triplet_idx, dtype=int)
    sel_keys = {keys[i] for i in triplet_idx}
    gcols = list(group_key)
    sub = df_long.copy()
    if "ID_PT" in gcols:
        sub["ID_PT"] = sub["ID_PT"].astype(str)
    key_frame = sub[gcols].apply(lambda row: tuple(row[c] for c in gcols), axis=1)
    mask = key_frame.isin(sel_keys)
    images = set(sub.loc[mask, "ID_IMG_ref"].astype(str).str.strip().unique())
    return sel_keys, images


def harmonize_unitary_long_fold(
    df_long: pd.DataFrame,
    train_triplet_idx: np.ndarray,
    test_triplet_idx: np.ndarray,
    group_key: list[str],
    *,
    enabled: bool,
) -> pd.DataFrame:
    """Devolve cópia do long com features harmonizadas (wide ComBat por imagem)."""
    if not enabled:
        return df_long

    _meta_cols, feature_cols = _meta_and_feature_cols(df_long)
    keys = triplet_keys_in_tensor_order(df_long, group_key)
    train_triplet_idx = np.asarray(train_triplet_idx, dtype=int)
    test_triplet_idx = np.asarray(test_triplet_idx, dtype=int)
    _train_keys, train_images = _triplet_image_sets(
        df_long, keys, train_triplet_idx, group_key
    )
    _test_keys, test_images = _triplet_image_sets(
        df_long, keys, test_triplet_idx, group_key
    )
    fold_images = train_images | test_images
    if not train_images:
        raise ValueError("ComBat: nenhum ID_IMG_ref no treino do fold.")
    if not fold_images:
        raise ValueError("ComBat: nenhum ID_IMG_ref no fold (treino+teste).")

    img = _image_table_dedup(df_long, feature_cols)
    img = img[img["ID_IMG_ref"].astype(str).str.strip().isin(fold_images)]
    wide = _wide_from_image_table(img, feature_cols)
    wide_h = _fit_transform_wide(wide, train_images)

    harm_feat = wide_h.set_index("ID_IMG_ref")
    harm_long = harm_feat.reset_index().melt(
        id_vars=["ID_IMG_ref"],
        var_name="_wide_col",
        value_name="_val",
    )
    harm_long[["_roi_slot", "_feat"]] = harm_long["_wide_col"].str.split(
        "__", n=1, expand=True
    )
    harm_long = harm_long.drop(columns=["_wide_col"])
    harm_long = harm_long.rename(columns={"_feat": "feature", "_val": "harm_val"})

    out = df_long.copy()
    out["ID_IMG_ref"] = out["ID_IMG_ref"].astype(str).str.strip()
    for c in ("roi", "side"):
        out[c] = out[c].astype(str).str.strip()
    out["_roi_slot"] = [
        _roi_slot_name(r, s, lb)
        for r, s, lb in zip(out["roi"], out["side"], out["label"], strict=True)
    ]
    out["_row_id"] = np.arange(len(out), dtype=np.int64)

    merge_keys = ["ID_IMG_ref", "_roi_slot"]
    harm_sub = harm_long.pivot_table(
        index=merge_keys, columns="feature", values="harm_val", aggfunc="first"
    ).reset_index()

    merged = out[merge_keys + ["_row_id"]].merge(harm_sub, on=merge_keys, how="left")
    merged = merged.sort_values("_row_id", kind="mergesort")
    for feat in feature_cols:
        if feat in merged.columns:
            upd = merged[feat].notna()
            out.loc[upd.to_numpy(), feat] = merged.loc[upd, feat].to_numpy(dtype=float)

    out = out.drop(columns=["_roi_slot", "_row_id"])
    missing = out[feature_cols].isna().any(axis=1).sum()
    if missing:
        print(
            f"Aviso ComBat: {int(missing)} linhas long sem feature harmonizada "
            "(imagem filtrada ou slot ausente)."
        )
    return out


def load_cv_assets(
    csv_path: Path,
    exp_md_path: Path,
    pair_order: list[str],
    group_key: list[str],
    *,
    run_neurocombat: bool,
    require_sex: bool = False,
    temporal_mode: str = "baseline_rate",
    dt_epsilon: float = 0.5,
) -> dict[str, Any]:
    """Carrega dados para CV (exp1/exp2): tensor global ou df_long + meta (ComBat por fold)."""
    if run_neurocombat:
        df_long = read_features_long_csv(csv_path)
        y, groups, sex = triplet_meta(
            df_long, group_key, pair_order, require_sex=require_sex
        )
        return {
            "df_long": df_long,
            "X_3d": None,
            "y": y,
            "groups": groups,
            "sex": sex,
            "feat_names": None,
            "slot_labels": None,
            "temporal_mode": temporal_mode,
            "dt_epsilon": dt_epsilon,
            "pair_order": pair_order,
            "group_key": group_key,
            "exp_md_path": exp_md_path,
            "require_sex": require_sex,
        }
    X_3d, y, groups, sex, feat_names, slot_labels = u.load_tensor(
        csv_path,
        exp_md_path,
        pair_order,
        group_key,
        require_sex=require_sex,
        temporal_mode=temporal_mode,
        dt_epsilon=dt_epsilon,
    )
    return {
        "df_long": None,
        "X_3d": X_3d,
        "y": y,
        "groups": groups,
        "sex": sex,
        "feat_names": feat_names,
        "slot_labels": slot_labels,
        "temporal_mode": temporal_mode,
        "dt_epsilon": dt_epsilon,
        "pair_order": pair_order,
        "group_key": group_key,
        "exp_md_path": exp_md_path,
        "require_sex": require_sex,
    }


def fold_tensor_from_assets(
    assets: dict[str, Any],
    train_triplet_idx: np.ndarray,
    test_triplet_idx: np.ndarray,
    *,
    run_neurocombat: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Tensor 3D do fold (harmonizado ou reutiliza tensor pré-carregado)."""
    if run_neurocombat:
        assert assets["df_long"] is not None
        return load_fold_tensor(
            assets["df_long"],
            train_triplet_idx,
            test_triplet_idx,
            assets["exp_md_path"],
            assets["pair_order"],
            assets["group_key"],
            run_neurocombat=True,
            require_sex=assets["require_sex"],
            temporal_mode=assets["temporal_mode"],
            dt_epsilon=assets["dt_epsilon"],
        )
    assert assets["X_3d"] is not None
    return assets["X_3d"], assets["feat_names"], assets["slot_labels"]


def load_fold_tensor(
    df_long: pd.DataFrame,
    train_triplet_idx: np.ndarray,
    test_triplet_idx: np.ndarray,
    exp_md_path: Path,
    pair_order: list[str],
    group_key: list[str],
    *,
    run_neurocombat: bool,
    require_sex: bool = False,
    temporal_mode: str = "baseline_rate",
    dt_epsilon: float = 0.5,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Harmoniza (opcional) e constrói tensor 3D para um fold."""
    df_fold = harmonize_unitary_long_fold(
        df_long,
        train_triplet_idx,
        test_triplet_idx,
        group_key,
        enabled=run_neurocombat,
    )
    X_3d, _y, _g, _s, feat_names, slot_labels = u.load_tensor_from_dataframe(
        df_fold,
        exp_md_path,
        pair_order,
        group_key,
        require_sex=require_sex,
        temporal_mode=temporal_mode,
        dt_epsilon=dt_epsilon,
    )
    return X_3d, feat_names, slot_labels


# Retrocompatibilidade (nomes antigos).
exp2_load_cv_assets = load_cv_assets
load_exp2_fold_tensor = load_fold_tensor

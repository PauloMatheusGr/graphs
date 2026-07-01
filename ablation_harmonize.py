"""NeuroComBat por fold para ablation (formato long, unidade ID_IMG).

Fit no treino do fold; transform em treino ∪ teste.
Covariáveis: batch, AGE, SEX (sem GROUP).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Any, Iterator

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from neuroCombat.neuroCombat import neuroCombatFromTraining

MIN_BATCH_SAMPLES = 5


def pool_small_batches(
    cov: pd.DataFrame,
    train_ids: list[str],
    *,
    min_n: int = MIN_BATCH_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Funde batches raros no treino em OTHER_<tesla> para EB estável."""
    out = cov.copy()
    batches = out["batch"].astype(str)
    train_counts = batches.loc[train_ids].value_counts()
    small = {str(b) for b, n in train_counts.items() if n < min_n}
    if not small:
        return out, {}

    def _remap(batch: str) -> str:
        if batch not in small:
            return batch
        # ponytail: pool por FIELD_STRENGTH (último segmento de MANUFACTURER_FIELD)
        return f"OTHER_{batch.rsplit('_', 1)[-1]}"

    mapping = {b: _remap(b) for b in batches.unique()}
    merged = {k: v for k, v in mapping.items() if k != v}
    out["batch"] = batches.map(mapping)
    return out, merged


@contextlib.contextmanager
def _quiet_combat(quiet: bool) -> Iterator[None]:
    if not quiet:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

META_COLS = frozenset(
    {
        "ID_IMG",
        "ID_PT",
        "GROUP",
        "SEX",
        "AGE",
        "MRI_DATE",
        "DIAG",
        "slot",
        "roi",
        "side",
        "label",
        "FIELD_STRENGTH",
        "MANUFACTURER",
        "MFG_MODEL",
        "batch",
        "ref_tag",
        "MMSE_SCORE",
        "CDR_GLOBAL",
        "ADAS_SCORE",
        "FAQ_SCORE",
        "ICV_mask_mm3",
    }
)


def _encode_sex(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy(dtype=np.float64)
        if np.isfinite(v).all() and np.isin(v, (0.0, 1.0)).all():
            return v
    upper = series.astype(str).str.strip().str.upper()
    mapped = upper.map({"F": 0.0, "M": 1.0, "0": 0.0, "1": 1.0})
    if mapped.isna().any():
        bad = sorted(upper[mapped.isna()].unique().tolist()[:10])
        raise ValueError(f"SEX inesperado: {bad}")
    return mapped.to_numpy(dtype=np.float64)


def _wide_col_key(row: pd.Series, feat: str) -> str:
    roi = str(row["roi"]) if "roi" in row.index else "NA"
    side = str(row["side"]) if "side" in row.index else "NA"
    label = str(row["label"]) if "label" in row.index else "NA"
    return f"{roi}|{side}|{label}|{feat}"


def infer_feature_columns(df: pd.DataFrame, feature_cols: list[str] | None = None) -> list[str]:
    if feature_cols is not None:
        return [c for c in feature_cols if c in df.columns]
    return [
        c
        for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _build_wide_table(
    df: pd.DataFrame,
    image_ids: set[str],
    combat_feats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ref_list = sorted(str(r).strip() for r in image_ids)
    sub = df[df["ID_IMG"].astype(str).str.strip().isin(ref_list)].copy()
    if sub.empty:
        raise ValueError("Nenhuma linha no CSV para os ID_IMG pedidos.")

    feat_by_img: dict[str, dict[str, float]] = {}
    cov_by_img: dict[str, dict[str, Any]] = {}

    for img_id, g in sub.groupby(sub["ID_IMG"].astype(str).str.strip(), sort=True):
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
            sex_val = float(_encode_sex(g["SEX"])[0])
        except ValueError:
            continue

        feat_by_img[str(img_id)] = feat_row
        cov_by_img[str(img_id)] = {
            "batch": str(batch),
            "AGE": float(age),
            "SEX": sex_val,
        }

    if not feat_by_img:
        raise ValueError("Nenhuma imagem com batch/AGE/SEX/features completos para ComBat.")

    col_order = sorted({k for row in feat_by_img.values() for k in row})
    rows: list[dict[str, float]] = []
    img_ids_out: list[str] = []
    cov_rows: list[dict[str, Any]] = []
    for img_id in sorted(feat_by_img):
        feat_row = feat_by_img[img_id]
        values = [feat_row.get(k, np.nan) for k in col_order]
        if not np.all(np.isfinite(values)):
            continue
        rows.append(dict(zip(col_order, values, strict=True)))
        img_ids_out.append(img_id)
        cov_rows.append(cov_by_img[img_id])

    if not rows:
        raise ValueError("Nenhuma imagem com todas as features preenchidas para ComBat.")

    wide = pd.DataFrame(rows, index=img_ids_out).sort_index()
    covars = pd.DataFrame(cov_rows, index=img_ids_out).sort_index()
    return wide, covars, img_ids_out


def _write_wide_back(
    df_out: pd.DataFrame,
    wide_harmonized: pd.DataFrame,
    combat_feats: list[str],
    image_ids: set[str],
) -> None:
    ref_set = {str(r).strip() for r in image_ids}
    lookup = wide_harmonized.to_dict("index")
    mask = df_out["ID_IMG"].astype(str).str.strip().isin(ref_set)
    for idx in df_out.index[mask]:
        row = df_out.loc[idx]
        img = str(row["ID_IMG"]).strip()
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
    *,
    quiet: bool = True,
) -> tuple[pd.DataFrame, dict]:
    dat = wide_train.loc[train_ids].to_numpy(dtype=np.float64).T
    covars = cov_train.loc[train_ids].copy()
    with _quiet_combat(quiet):
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
    *,
    quiet: bool = True,
) -> pd.DataFrame:
    dat = wide.loc[image_ids].to_numpy(dtype=np.float64).T
    batch = cov.loc[image_ids, "batch"].to_numpy(dtype=str)
    with _quiet_combat(quiet):
        res = neuroCombatFromTraining(dat=dat, batch=batch, estimates=estimates)
    return pd.DataFrame(
        np.asarray(res["data"], dtype=np.float64).T,
        index=image_ids,
        columns=wide.columns,
    )


def harmonize_long_fold(
    df_long: pd.DataFrame,
    *,
    train_id_imgs: set[str],
    transform_id_imgs: set[str],
    feature_cols: list[str] | None = None,
    fold_id: int = 0,
    quiet: bool = True,
) -> pd.DataFrame:
    """Fit ComBat nas imagens de treino; harmoniza imagens de transform (treino ∪ teste)."""
    if not train_id_imgs:
        raise ValueError("train_id_imgs vazio — impossível ajustar ComBat.")
    if not transform_id_imgs:
        return df_long.copy()

    for col in ("ID_IMG", "batch", "AGE", "SEX", "roi", "side", "label"):
        if col not in df_long.columns:
            raise ValueError(f"Coluna {col!r} ausente no CSV long (necessária para ComBat).")

    combat_feats = infer_feature_columns(df_long, feature_cols)
    if not combat_feats:
        warnings.warn(f"[fold {fold_id}] Nenhuma feature numérica para ComBat; retornando original.")
        return df_long.copy()

    df_out = df_long.copy()
    wide_all, cov_all, _ = _build_wide_table(df_out, transform_id_imgs, combat_feats)
    train_ids = sorted(
        img for img in wide_all.index if str(img) in {str(r).strip() for r in train_id_imgs}
    )
    if not train_ids:
        warnings.warn(f"[fold {fold_id}] Nenhuma imagem de treino com dados ComBat; retornando original.")
        return df_out

    cov_all, pooled = pool_small_batches(cov_all, train_ids)
    if pooled:
        warnings.warn(
            f"[fold {fold_id}] batches fundidos (n<{MIN_BATCH_SAMPLES} no treino): {pooled}"
        )

    # train_batches = cov_all.loc[train_ids]["batch"].value_counts()
    # if len(train_batches) < 2:
    #     warnings.warn(
    #         f"[fold {fold_id}] ComBat precisa de >=2 batches no treino; "
    #         f"encontrados {len(train_batches)}. Retornando original."
    #     )
    #     return df_out

    # small = train_batches[train_batches < MIN_BATCH_SAMPLES]
    # if not small.empty:
    #     warnings.warn(
    #         f"[fold {fold_id}] ComBat ignorado — batch no treino ainda com "
    #         f"<{MIN_BATCH_SAMPLES} amostras após merge: {small.to_dict()}"
    #     )
    #     return df_out

    # transform_ids = sorted(
    #     img for img in wide_all.index if str(img) in {str(r).strip() for r in transform_id_imgs}
    # )
    # test_only_batches = set(cov_all.loc[transform_ids, "batch"]) - set(
    #     cov_all.loc[train_ids, "batch"]
    # )
    # if test_only_batches:
    #     warnings.warn(
    #         f"[fold {fold_id}] batches só no teste (não harmonizados): {sorted(test_only_batches)}"
    #     )

    # wide_train = wide_all.loc[train_ids]
    # cov_train = cov_all.loc[train_ids]
    # wide_h_train, estimates = _run_neurocombat_train(
    #     wide_train, cov_train, train_ids, quiet=quiet
    # )

    # harmonized_parts: list[pd.DataFrame] = [wide_h_train]
    # apply_ids = [i for i in transform_ids if i not in train_ids]
    # if apply_ids:
    #     apply_ok = [i for i in apply_ids if cov_all.loc[i, "batch"] in train_batches.index]
    #     if apply_ok:
    #         wide_h_apply = _run_neurocombat_apply(
    #             wide_all, cov_all, apply_ok, estimates, quiet=quiet
    #         )
    #         harmonized_parts.append(wide_h_apply)

    # wide_harmonized = pd.concat(harmonized_parts)
    # harmonized_ids = set(wide_harmonized.index.astype(str))
    # _write_wide_back(df_out, wide_harmonized, combat_feats, harmonized_ids)
    # return df_out
    train_batches = cov_all.loc[train_ids]["batch"].value_counts()

    # Batches com amostras suficientes no treino para fit estável
    good_batches = {str(b) for b, n in train_batches.items() if n >= MIN_BATCH_SAMPLES}
    small = train_batches[train_batches < MIN_BATCH_SAMPLES]
    if not small.empty:
        warnings.warn(
            f"[fold {fold_id}] batches excluídos do ComBat "
            f"(<{MIN_BATCH_SAMPLES} imgs no treino): {small.to_dict()}"
        )

    if len(good_batches) < 2:
        warnings.warn(
            f"[fold {fold_id}] ComBat precisa de >=2 batches 'grandes' no treino; "
            f"encontrados {len(good_batches)} (de {len(train_batches)} no total). "
            f"Retornando original."
        )
        return df_out

    transform_ids = sorted(
        img for img in wide_all.index
        if str(img) in {str(r).strip() for r in transform_id_imgs}
    )

    def _batch(img: str) -> str:
        return str(cov_all.loc[img, "batch"])

    # Imagens harmonizáveis = batch ∈ good_batches
    known_ids = [img for img in transform_ids if _batch(img) in good_batches]
    train_known = [img for img in train_ids if img in set(known_ids)]

    unknown_imgs = [img for img in transform_ids if img not in set(known_ids)]
    if unknown_imgs:
        warnings.warn(
            f"[fold {fold_id}] {len(unknown_imgs)} imagem(ns) sem harmonização "
            f"(batch pequeno no treino ou batch só no teste)."
        )

    if len(known_ids) < 2:
        warnings.warn(
            f"[fold {fold_id}] ComBat ignorado (<2 imagens harmonizáveis). "
            f"Retornando original."
        )
        return df_out

    if len(train_known) < 2:
        warnings.warn(
            f"[fold {fold_id}] ComBat ignorado (<2 imagens no treino harmonizável). "
            f"Retornando original."
        )
        return df_out

    wide_train = wide_all.loc[train_known]
    cov_train = cov_all.loc[train_known]
    wide_h_train, estimates = _run_neurocombat_train(
        wide_train, cov_train, train_known, quiet=quiet
    )

    test_only = [img for img in known_ids if img not in set(train_known)]
    harmonized_parts: list[pd.DataFrame] = [wide_h_train]
    if test_only:
        wide_h_apply = _run_neurocombat_apply(
            wide_all, cov_all, test_only, estimates, quiet=quiet
        )
        harmonized_parts.append(wide_h_apply)

    wide_harmonized = pd.concat(harmonized_parts)
    _write_wide_back(df_out, wide_harmonized, combat_feats, set(known_ids))
    return df_out

def image_ids_for_patients(df_long: pd.DataFrame, patient_ids: set[str]) -> set[str]:
    mask = df_long["ID_PT"].astype(str).str.strip().isin({str(p).strip() for p in patient_ids})
    return set(df_long.loc[mask, "ID_IMG"].astype(str).str.strip().tolist())


if __name__ == "__main__":
    # ponytail: smoke helpers + skip ComBat quando batch único no treino
    cov_demo = pd.DataFrame(
        {"batch": ["A_1.5", "A_1.5", "B_3.0", "B_3.0", "B_3.0", "C_1.5"]},
        index=["i1", "i2", "i3", "i4", "i5", "i6"],
    )
    pooled, merged = pool_small_batches(cov_demo, ["i1", "i2", "i6"], min_n=3)
    assert merged == {"A_1.5": "OTHER_1.5", "C_1.5": "OTHER_1.5"}
    assert pooled.loc["i1", "batch"] == "OTHER_1.5"
    assert pooled.loc["i3", "batch"] == "B_3.0"

    rng = np.random.default_rng(0)
    rows = []
    for img, pt in [("I1", "P1"), ("I2", "P1"), ("I3", "P2")]:
        rows.append(
            {
                "ID_IMG": img,
                "ID_PT": pt,
                "GROUP": "CN",
                "batch": "only_batch",
                "AGE": 70.0,
                "SEX": "M",
                "roi": "hippocampus",
                "side": "L",
                "label": "1",
                "f1": float(rng.normal()),
            }
        )
    df = pd.DataFrame(rows)
    out = harmonize_long_fold(
        df,
        train_id_imgs={"I1", "I2"},
        transform_id_imgs={"I1", "I2", "I3"},
        feature_cols=["f1"],
        fold_id=0,
    )
    assert out["f1"].tolist() == df["f1"].tolist()
    assert image_ids_for_patients(df, {"P1"}) == {"I1", "I2"}
    print("ablation_harmonize self-check OK")

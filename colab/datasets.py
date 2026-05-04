from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripletKey:
    """Chave canônica de um conjunto (triplet)."""

    id_pt: str
    combination_number: str
    triplet_idx: str | None = None


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def read_triplets_table(path: str | Path) -> pd.DataFrame:
    """
    Lê uma tabela long onde cada linha é (ID_PT, COMBINATION_NUMBER, ID_IMG, ...).

    Exemplo (como `cj_data_abordagem_teste.txt`):
    - várias linhas por (ID_PT, COMBINATION_NUMBER) (tipicamente 3 imagens: i1, i2, i3)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Triplets file não encontrado: {p}")
    df = pd.read_csv(p)
    required = {"ID_PT", "COMBINATION_NUMBER", "ID_IMG"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Triplets file sem colunas obrigatórias: {sorted(missing)}")
    return df


def read_radiomics_merge(path: str | Path) -> pd.DataFrame:
    """
    Lê radiômica por imagem×ROI×lado. Chave esperada:
    (ID_IMG, roi, side, label)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Radiomics merge não encontrado: {p}")
    df = pd.read_csv(p)
    required = {"ID_IMG", "roi", "side", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Radiomics merge sem colunas obrigatórias: {sorted(missing)}")
    return df


def build_triplet_sequence_from_radiomics(
    triplets_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
    *,
    id_pt_col: str = "ID_PT",
    comb_col: str = "COMBINATION_NUMBER",
    img_col: str = "ID_IMG",
    group_col: str = "GROUP",
    sex_col: str = "SEX",
    time_col: str = "MRI_DATE",
    roi_cols: tuple[str, str, str, str] = ("roi", "side", "label", "ID_IMG"),
    drop_incomplete_triplets: bool = True,
    rois: list[str] | None = None,
    labels: list[int] | None = None,
    max_triplets: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Monta sequência temporal real i1->i2->i3 usando radiomics por imagem.

    Saída:
    - X: (n_triplets, seq_len=3, n_features)
    - y: (n_triplets,) binário (0/1) a partir de GROUP
    - sex: (n_triplets, 1) 0/1 (F/M)
    - groups: (n_triplets,) = ID_PT (para StratifiedGroupKFold)
    - feature_names: lista de features (raízes) no mesmo order do eixo n_features

    Observação:
    - Radiomics é nível imagem×ROI×lado. Para virar um vetor por imagem, fazemos pivot wide
      com colunas do tipo "roi=<r>|side=<s>|label=<l>|<feature>" e então montamos
      a sequência (i1,i2,i3) por conjunto.
    """
    rois = rois or []
    labels = labels or []

    tdf = triplets_df.copy()
    for c in (id_pt_col, comb_col, img_col):
        tdf[c] = tdf[c].astype(str)

    # ordena imagens por data (ou por ordem no arquivo se não houver data)
    if time_col in tdf.columns:
        tdf["_time"] = pd.to_datetime(tdf[time_col], errors="coerce")
    else:
        tdf["_time"] = pd.NaT

    rdf = radiomics_df.copy()
    rdf["ID_IMG"] = rdf["ID_IMG"].astype(str)
    rdf["roi"] = rdf["roi"].astype(str).str.strip()
    rdf["side"] = rdf["side"].astype(str).str.strip()
    rdf["label"] = pd.to_numeric(rdf["label"], errors="coerce").astype("Int64")

    if rois:
        rdf = rdf[rdf["roi"].isin([r.strip() for r in rois])].copy()
    if labels:
        rdf = rdf[rdf["label"].isin([int(x) for x in labels])].copy()

    # features numéricas (radiomics + ICV etc), excluindo chaves
    ignore = {"ID_IMG", "roi", "side", "label"}
    feature_cols = [
        c for c in rdf.columns if c not in ignore and pd.api.types.is_numeric_dtype(rdf[c])
    ]
    if not feature_cols:
        raise ValueError("Nenhuma coluna numérica de radiomics encontrada.")

    # wide por imagem: uma linha por ID_IMG
    prefix = (
        "roi=" + rdf["roi"].astype(str)
        + "|side=" + rdf["side"].astype(str)
        + "|label=" + rdf["label"].astype(str)
    )
    melt = rdf[["ID_IMG"]].copy()
    melt["prefix"] = prefix
    # melt para pivot colunas com nome único
    m = rdf[["ID_IMG"]].copy()
    m["prefix"] = prefix
    m_long = rdf[["ID_IMG"]].copy()
    m_long["prefix"] = prefix
    m_long = m_long.join(rdf[feature_cols])
    m_long = m_long.melt(id_vars=["ID_IMG", "prefix"], value_vars=feature_cols, var_name="feat", value_name="value")
    m_long["col"] = m_long["prefix"].astype(str) + "|" + m_long["feat"].astype(str)
    wide_img = (
        m_long.pivot_table(index="ID_IMG", columns="col", values="value", aggfunc="first")
        .sort_index(axis=1)
        .reset_index()
    )

    # junta triplets -> radiomics wide por imagem
    tdf2 = tdf.merge(wide_img, how="left", left_on=img_col, right_on="ID_IMG")
    # remove a coluna duplicada de join
    if "ID_IMG_y" in tdf2.columns:
        tdf2 = tdf2.drop(columns=["ID_IMG_y"])

    # coleta nomes das features (todas as colunas numéricas wide)
    wide_feature_names = [c for c in tdf2.columns if c not in set(tdf.columns) | {"ID_IMG"}]
    wide_feature_names = [c for c in wide_feature_names if c != "ID_IMG_x"]
    wide_feature_names = sorted(set(wide_feature_names))
    if not wide_feature_names:
        raise ValueError("Falha ao construir features wide por imagem a partir de radiomics.")

    rows: list[tuple[np.ndarray, str, str, str]] = []
    # cada grupo é um triplet
    group_cols = [id_pt_col, comb_col]
    if "TRIPLET_IDX" in tdf2.columns:
        group_cols.append("TRIPLET_IDX")

    for _, g in tdf2.groupby(group_cols, sort=False):
        g = g.sort_values("_time", kind="stable") if "_time" in g.columns else g
        # pega as 3 primeiras imagens (se houver >3 por alguma razão)
        g = g.head(3).copy()
        if drop_incomplete_triplets and len(g) != 3:
            continue

        # y/sex por conjunto (assumimos constantes no grupo)
        if group_col not in g.columns or sex_col not in g.columns:
            raise KeyError(f"Triplets file precisa conter {group_col} e {sex_col} para montar y/sex.")
        y_val = str(g[group_col].iloc[0])
        sex_val = str(g[sex_col].iloc[0]).upper()
        id_pt = str(g[id_pt_col].iloc[0])

        X_seq = g[wide_feature_names].to_numpy(dtype=np.float32, copy=True)
        if X_seq.shape[0] != 3:
            # se drop_incomplete_triplets=False, ainda precisamos de 3 passos
            # para manter shape consistente; aqui optamos por pular.
            continue
        rows.append((X_seq, y_val, sex_val, id_pt))

        if max_triplets is not None and len(rows) >= int(max_triplets):
            break

    if not rows:
        raise ValueError("Nenhum triplet foi montado. Verifique se há 3 imagens por conjunto.")

    X = np.stack([r[0] for r in rows], axis=0)
    y_raw = np.array([r[1] for r in rows], dtype=object)
    sex_raw = np.array([r[2] for r in rows], dtype=object)
    groups = np.array([r[3] for r in rows], dtype=object)

    # encode sex
    sex = pd.Series(sex_raw.astype(str)).str.upper().map({"F": 0, "M": 1}).to_numpy()
    if (sex < 0).any():
        bad = sorted(pd.Series(sex_raw.astype(str)).unique().tolist())
        raise ValueError(f"Valores inesperados em SEX: {bad} (esperado F/M)")
    sex = sex.astype(np.float32).reshape(-1, 1)

    # encode y: pMCI como positivo se existir
    classes = sorted(pd.Series(y_raw.astype(str)).unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Esperado GROUP binário, encontrei {len(classes)} classes: {classes}")
    pos = "pMCI" if "pMCI" in classes else sorted(classes)[-1]
    y = (pd.Series(y_raw.astype(str)) == pos).astype(int).to_numpy(dtype=np.int32)

    return X.astype(np.float32), y, sex, groups.astype(str), wide_feature_names


def build_wide_tabular_from_long_pairs(
    df_long: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"),
    group_col: str = "GROUP",
    sex_col: str = "SEX",
    roi_col: str = "roi",
    side_col: str = "side",
    pair_col: str = "pair",
    drop_cols: Iterable[str] = (),
    feature_prefix_whitelist: tuple[str, ...] = (
        "logjac_",
        "mag_",
        "div_",
        "ux_",
        "uy_",
        "uz_",
        "curlmag_",
        "strain_",
        "original_",
    ),
    include_dt_as_feature: bool = True,
) -> pd.DataFrame:
    """
    Converte CSV long (linhas com roi/side/pair) em wide (1 linha por conjunto),
    criando colunas do tipo:
      roi=<roi>|side=<side>|pair=<pair>|<feature>

    Mantém também:
    - GROUP, SEX, ID_PT e chave do conjunto (key_cols) como colunas.

    Use para: sklearn tabular / MLP (flatten).
    """
    df = df_long.copy()

    # garante presença das colunas chave
    req = set(key_cols) | {group_col, sex_col, roi_col, side_col, pair_col}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"CSV long sem colunas obrigatórias para wide: {sorted(missing)}")

    # decide features: numéricas e com prefixo permitido
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist()]
    drop_set = set(drop_cols) | set(key_cols) | {group_col, roi_col, side_col, pair_col}
    if include_dt_as_feature:
        # mapeia t12/t13/t23 -> dt por linha via pair, mas armazenaremos como feature separada
        # (dt entra no whitelist via nome 'dt')
        pass
    feat_cols = []
    for c in numeric_cols:
        if c in drop_set:
            continue
        if c.startswith(feature_prefix_whitelist):
            feat_cols.append(c)
    feat_cols = sorted(set(feat_cols))
    if not feat_cols:
        raise ValueError("Não encontrei colunas de features numéricas com prefixos esperados.")

    # cria dt se possível
    work = df[list(dict.fromkeys(list(key_cols) + [group_col, sex_col, roi_col, side_col, pair_col] + feat_cols))].copy()
    if include_dt_as_feature and all(c in df.columns for c in ("t12", "t13", "t23")):
        time_map = {"12": "t12", "13": "t13", "23": "t23"}
        p = work[pair_col].astype(str)
        dt = np.full((len(work),), np.nan, dtype=np.float32)
        for pair, tcol in time_map.items():
            mask = p == str(pair)
            if mask.any():
                dt[mask] = pd.to_numeric(df.loc[mask, tcol], errors="coerce").astype(np.float32).to_numpy()
        work["dt"] = dt
        feat_cols2 = feat_cols + ["dt"]
    else:
        feat_cols2 = feat_cols

    prefix = (
        "roi=" + work[roi_col].astype(str).str.strip()
        + "|side=" + work[side_col].astype(str).str.strip()
        + "|pair=" + work[pair_col].astype(str).str.strip()
    )
    long = work[list(key_cols)].copy()
    long[group_col] = work[group_col].astype(str)
    long[sex_col] = work[sex_col].astype(str)
    long["prefix"] = prefix
    long = long.join(work[feat_cols2])
    long = long.melt(id_vars=list(key_cols) + [group_col, sex_col, "prefix"], value_vars=feat_cols2, var_name="feat", value_name="value")
    long["col"] = long["prefix"].astype(str) + "|" + long["feat"].astype(str)

    wide = (
        long.pivot_table(index=list(key_cols) + [group_col, sex_col], columns="col", values="value", aggfunc="first")
        .sort_index(axis=1)
        .reset_index()
    )
    return wide


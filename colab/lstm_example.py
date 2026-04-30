# python colab/lstm_example.py \
#   --csv "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv" \
#   --sequence-source pairs \
#   --pair-order 12,13,23 \
#   --n-splits 5 \
#   --inner-fold 5 \
#   --epochs 50 \
#   --batch-size 64 \
#   --seed 42 \
#   --device cpu \
#   --balance downsample \
#   --shap \
#   --shap-outdir "/mnt/study-data/pgirardi/graphs/colab/outputs"

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf


DEFAULT_CSV = (
    "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/"
    "all_delta_features_neurocombat.csv"
)


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def filter_by_roi_label(
    df: pd.DataFrame,
    *,
    rois: list[str] | None = None,
    labels: list[int] | None = None,
    roi_label: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filtra linhas pelo(s) ROI(s) e/ou label(s) no CSV (colunas 'roi' e 'label').

    - rois: valores em df['roi'] (ex.: 'hippocampus,amygdala')
    - labels: inteiros em df['label'] (ex.: 17,53)
    - roi_label: pares 'roi:label' (ex.: 'hippocampus:17,hippocampus:53')
    """
    if ("roi" not in df.columns) or ("label" not in df.columns):
        raise ValueError("CSV precisa ter colunas 'roi' e 'label' para filtrar ROIs.")

    rois = rois or []
    labels = labels or []
    roi_label = roi_label or []

    out = df.copy()
    out["roi"] = out["roi"].astype(str).str.strip()
    out["_label_int"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out[out["_label_int"].notna()].copy()
    out["_label_int"] = out["_label_int"].astype(int)

    if roi_label:
        pairs: list[tuple[str, int]] = []
        for item in roi_label:
            if ":" not in item:
                raise ValueError(
                    f"Use o formato roi:label em --roi-label. Recebi: {item}"
                )
            r, lab = item.split(":", 1)
            pairs.append((r.strip(), int(lab.strip())))

        mask = False
        for r, lab in pairs:
            mask = mask | ((out["roi"] == r) & (out["_label_int"] == lab))
        out = out[mask].copy()
        return out.drop(columns=["_label_int"])

    mask = True
    if rois:
        mask = mask & out["roi"].isin([r.strip() for r in rois])
    if labels:
        mask = mask & out["_label_int"].isin([int(x) for x in labels])

    out = out[mask].copy()
    return out.drop(columns=["_label_int"])


def _build_temporal_tensor(
    df: pd.DataFrame,
    *,
    timesteps: list[str],
    drop_if_nan: bool,
) -> tuple[np.ndarray, list[str]]:
    """
    Constrói X com shape (n_amostras, seq_len, n_features) a partir de colunas
    com sufixos temporais (ex.: *_base, *_follow, *_delta).

    Retorna X e a lista de "roots" (features base sem sufixo).
    """
    # Só faz sentido para colunas numéricas.
    num_cols = set(df.select_dtypes(include=["number"]).columns.tolist())

    roots_by_ts: dict[str, set[str]] = {}
    for ts in timesteps:
        suffix = f"_{ts}"
        cols_ts = [c for c in num_cols if c.endswith(suffix)]
        roots_by_ts[ts] = {c[: -len(suffix)] for c in cols_ts}

    # Roots comuns a todos os timesteps selecionados.
    common_roots = sorted(set.intersection(*(roots_by_ts[ts] for ts in timesteps)))
    if not common_roots:
        raise ValueError(
            "Não encontrei features numéricas com sufixos temporais comuns para "
            f"timesteps={timesteps}. Ex.: colunas '*_base' e '*_follow'."
        )

    seq = []
    for ts in timesteps:
        cols = [f"{r}_{ts}" for r in common_roots]
        X_ts = df[cols].to_numpy(dtype=np.float32, copy=True)
        seq.append(X_ts)

    X = np.stack(seq, axis=1)  # (n, seq_len, n_features)

    if drop_if_nan:
        mask = np.isfinite(X).all(axis=(1, 2))
        X = X[mask]

    return X, common_roots


def _build_pairwise_triplet_tensor(
    df: pd.DataFrame,
    *,
    pair_order: list[str],
    group_cols: list[str],
    drop_incomplete: bool,
    drop_if_nan: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Constrói X com shape (n_amostras, 3, n_features) usando AS LINHAS como passos temporais.

    Cada amostra corresponde a um (paciente, tripla, ROI, lado, ...) e possui 3 linhas:
    pair=12, pair=13, pair=23 (ou outra ordem definida em pair_order).

    As features por passo são:
    - radiomics *_delta e *_absdelta (numéricas)
    - atributos do campo de deformação (centroid_*, logjac_*, mag_*, div_*, ux_*, uy_*, uz_*, curlmag_*)
    - uma feature de tempo "dt" extraída de t12/t13/t23 conforme o par
    """
    required = set(group_cols + ["pair", "GROUP", "SEX", "ID_PT"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas necessárias para sequência por pares: {sorted(missing)}")

    # Features numéricas candidatas (exclui base/follow porque não são "passos" aqui)
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist()]
    delta_cols = [c for c in num_cols if c.endswith("_delta") or c.endswith("_absdelta")]

    deform_prefixes = (
        "centroid_",
        "logjac_",
        "mag_",
        "div_",
        "ux_",
        "uy_",
        "uz_",
        "curlmag_",
    )
    deform_cols = [c for c in num_cols if c.startswith(deform_prefixes)]

    # t12/t13/t23 viram uma única coluna dt por timestep
    time_cols = [c for c in ("t12", "t13", "t23") if c in df.columns]
    base_feature_cols = sorted(set(delta_cols + deform_cols))

    if not base_feature_cols:
        raise ValueError(
            "Não encontrei colunas numéricas *_delta/_absdelta ou deformações (logjac_*, mag_*, ...)."
        )

    # Mapeia pair -> qual coluna t usar
    time_map = {"12": "t12", "13": "t13", "23": "t23"}
    available_pairs = set(df["pair"].astype(str).unique().tolist())

    rows = []
    meta_rows: list[dict[str, object]] = []
    for keys, g in df.groupby(group_cols, sort=False):
        g2 = g.copy()
        g2["pair"] = g2["pair"].astype(str)

        if drop_incomplete and not all(p in set(g2["pair"]) for p in pair_order):
            continue

        # monta (3, F)
        step_vecs = []
        ok = True
        for p in pair_order:
            gp = g2[g2["pair"] == p]
            if gp.empty:
                ok = False
                break
            # se houver duplicata, pega a primeira
            r = gp.iloc[0]
            vec = r[base_feature_cols].to_numpy(dtype=np.float32, copy=True)

            dt = np.float32(np.nan)
            if p in time_map and time_map[p] in time_cols:
                dt = np.float32(r[time_map[p]])
            vec = np.concatenate([vec, np.array([dt], dtype=np.float32)], axis=0)
            step_vecs.append(vec)

        if not ok:
            continue

        X_seq = np.stack(step_vecs, axis=0)  # (3, F+1)
        y_val = str(g2["GROUP"].iloc[0])
        sex_val = str(g2["SEX"].iloc[0])
        id_pt = str(g2["ID_PT"].iloc[0])

        rows.append((X_seq, y_val, sex_val, id_pt))
        # keys é tuple alinhada a group_cols (ou escalar se len==1)
        if not isinstance(keys, tuple):
            keys_t = (keys,)
        else:
            keys_t = keys
        meta = {c: v for c, v in zip(group_cols, keys_t)}
        meta_rows.append(meta)

    if not rows:
        raise ValueError(
            f"Nenhuma sequência foi montada. Pairs disponíveis: {sorted(available_pairs)}. "
            f"Esperado conter: {pair_order}."
        )

    X = np.stack([r[0] for r in rows], axis=0)
    y = np.array([r[1] for r in rows], dtype=object)
    sex = np.array([r[2] for r in rows], dtype=object)
    id_pt = np.array([r[3] for r in rows], dtype=object)
    meta_df = pd.DataFrame(meta_rows)

    feature_cols = base_feature_cols + ["dt"]

    if drop_if_nan:
        mask = np.isfinite(X).all(axis=(1, 2))
        X, y, sex, id_pt = X[mask], y[mask], sex[mask], id_pt[mask]

    return X, y, sex, id_pt, meta_df, feature_cols


def _zscore_fit_transform_3d(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score por feature usando apenas treino.
    Normaliza todos os timesteps com o mesmo scaler por feature.
    """
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n_train * seq_len, n_feat))

    X_train_z = scaler.transform(X_train.reshape(n_train * seq_len, n_feat)).reshape(
        n_train, seq_len, n_feat
    )
    n_test = X_test.shape[0]
    X_test_z = scaler.transform(X_test.reshape(n_test * seq_len, n_feat)).reshape(
        n_test, seq_len, n_feat
    )
    return X_train_z.astype(np.float32), X_test_z.astype(np.float32)


def _zscore_fit_transform_3d_from_fit(
    X_fit: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score por feature usando APENAS o conjunto de fit (treino interno).
    Aplica o mesmo scaler em val e test.
    """
    n_fit, seq_len, n_feat = X_fit.shape
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(n_fit * seq_len, n_feat))

    X_fit_z = scaler.transform(X_fit.reshape(n_fit * seq_len, n_feat)).reshape(
        n_fit, seq_len, n_feat
    )
    n_val = X_val.shape[0]
    X_val_z = scaler.transform(X_val.reshape(n_val * seq_len, n_feat)).reshape(
        n_val, seq_len, n_feat
    )
    n_test = X_test.shape[0]
    X_test_z = scaler.transform(X_test.reshape(n_test * seq_len, n_feat)).reshape(
        n_test, seq_len, n_feat
    )
    return (
        X_fit_z.astype(np.float32),
        X_val_z.astype(np.float32),
        X_test_z.astype(np.float32),
    )


def _make_model(tf_module, seq_len: int, n_feat: int):
    keras = tf_module.keras
    layers = keras.layers
    models = keras.models

    x_in = keras.Input(shape=(seq_len, n_feat), name="x")
    sex_in = keras.Input(shape=(1,), name="sex")

    x = layers.Masking(mask_value=0.0)(x_in)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([x, sex_in])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[x_in, sex_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treina uma LSTM para classificar sMCI vs pMCI usando features temporais "
            "no formato *_base / *_follow (opcional *_delta)."
        )
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Caminho do CSV.")
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="Lista de ROIs (coluna roi), separadas por vírgula. Ex.: hippocampus,amygdala",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Lista de labels (coluna label), separados por vírgula. Ex.: 17,53",
    )
    parser.add_argument(
        "--roi-label",
        type=str,
        default="",
        help="Lista de pares roi:label. Ex.: hippocampus:17,hippocampus:53",
    )
    parser.add_argument(
        "--sequence-source",
        choices=["columns", "pairs"],
        default="pairs",
        help=(
            "Como construir a sequência temporal. "
            "'columns' usa colunas *_base/_follow/_delta; "
            "'pairs' usa 3 linhas por tripla (pair=12/13/23) como passos temporais."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default="base,follow",
        help="Lista de timesteps (sufixos) separados por vírgula. Ex.: base,follow ou base,follow,delta",
    )
    parser.add_argument(
        "--pair-order",
        type=str,
        default="12,13,23",
        help="Ordem dos pares quando --sequence-source pairs. Ex.: 12,13,23",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Folds do StratifiedGroupKFold.")
    parser.add_argument(
        "--inner-fold",
        type=int,
        default=5,
        help=(
            "Número de folds internos para separar validação dentro do treino (nested). "
            "Usa o primeiro split como validação (sem vazamento por ID_PT)."
        ),
    )
    parser.add_argument(
        "--balance",
        choices=["none", "downsample"],
        default="none",
        help=(
            "Balanceamento aplicado SOMENTE no treino (fit). "
            "'none' não balanceia; "
            "'downsample' faz downsampling por paciente (ID_PT) balanceando GROUP+SEX."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50, help="Épocas de treino por fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help=(
            "Device para treino. Se der erro cuDNN/driver, use '--device cpu'. "
            "Para GPU, combine com '--gpu-index' e '--memory-growth'."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Qual GPU usar quando --device gpu (0, 1, ...).",
    )
    parser.add_argument(
        "--memory-growth",
        action="store_true",
        help="Habilita memory growth (reduz chance de erro por alocação de VRAM).",
    )
    parser.add_argument(
        "--drop-nan",
        action="store_true",
        help="Remove linhas com NaN/inf nas features temporais.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Gera SHAP por fold e agrega (média/DP) para atributos e ROI/label.",
    )
    parser.add_argument(
        "--shap-folds",
        type=str,
        default="all",
        help="Quais folds calcular SHAP: 'all' (default) ou lista tipo '1,3,5'.",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=300,
        help="Quantas amostras (sequências) usar no SHAP (para acelerar).",
    )
    parser.add_argument(
        "--shap-background",
        type=int,
        default=200,
        help="Quantas amostras do treino usar como background no SHAP.",
    )
    parser.add_argument(
        "--shap-outdir",
        type=str,
        default=str(Path(__file__).parent / "outputs"),
        help="Diretório para salvar CSVs de SHAP.",
    )
    args = parser.parse_args()

    # Seleção de device: execute isso o mais cedo possível.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")
    else:
        # Use uma única GPU e (opcionalmente) memory growth.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")

    # Import do TensorFlow DEPOIS de configurar env vars (principal para GPU).
    import tensorflow as tf

    # Configura device visível e memory growth após o import.
    if args.device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    else:
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.set_visible_devices(gpus[0], "GPU")
                if args.memory_growth:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    rois = _parse_csv_list(args.roi)
    labels = [int(x) for x in _parse_csv_list(args.label)]
    roi_label = _parse_csv_list(args.roi_label)
    if rois or labels or roi_label:
        before = df.shape
        df = filter_by_roi_label(df, rois=rois, labels=labels, roi_label=roi_label)
        print(f"[ROI] filtro aplicado: {before} -> {df.shape}")

    required = {"ID_PT", "GROUP", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    if args.sequence_source == "pairs":
        pair_order = [p.strip() for p in args.pair_order.split(",") if p.strip()]
        group_cols = [
            c
            for c in ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX", "roi", "side", "label"]
            if c in df.columns
        ]
        X, y_raw, sex_raw, id_pt_raw, meta_df, feat_cols = _build_pairwise_triplet_tensor(
            df,
            pair_order=pair_order,
            group_cols=group_cols,
            drop_incomplete=True,
            drop_if_nan=args.drop_nan,
        )
        print(f"X shape: {X.shape}  (seq_len={X.shape[1]}, n_feat={X.shape[2]})")
        print(f"Pair order: {pair_order}")
        print(f"Features por passo: {len(feat_cols)}")
        # Target/sex/groups a partir das sequências
        y_raw = y_raw.astype(str)
        sex_s = pd.Series(sex_raw.astype(str)).str.upper().map({"F": 0, "M": 1}).to_numpy()
        if (sex_s < 0).any():
            bad = sorted(pd.Series(sex_raw.astype(str)).unique().tolist())
            raise ValueError(f"Valores inesperados em SEX: {bad} (esperado F/M)")
        sex = sex_s.astype(np.float32).reshape(-1, 1)
        groups = id_pt_raw.astype(str)
        strat_col = (pd.Series(y_raw).astype(str) + "_" + pd.Series(sex_raw).astype(str)).to_numpy()
    else:
        # Sequência baseada em colunas *_base/_follow/_delta
        timesteps = [t.strip() for t in args.timesteps.split(",") if t.strip()]
        X, roots = _build_temporal_tensor(df, timesteps=timesteps, drop_if_nan=args.drop_nan)
        y_raw = df["GROUP"].astype(str).to_numpy()
        sex = df["SEX"].astype(str).str.upper().map({"F": 0, "M": 1}).fillna(-1).to_numpy()
        if (sex < 0).any():
            bad = sorted(df.loc[sex < 0, "SEX"].astype(str).unique().tolist())
            raise ValueError(f"Valores inesperados em SEX: {bad} (esperado F/M)")
        sex = sex.astype(np.float32).reshape(-1, 1)
        groups = df["ID_PT"].astype(str).to_numpy()
        strat_col = (df["GROUP"].astype(str) + "_" + df["SEX"].astype(str)).to_numpy()
        print(f"X shape: {X.shape}  (seq_len={X.shape[1]}, n_feat={X.shape[2]})")
        print(f"Timesteps: {timesteps}")
        print(f"Features (roots) comuns: {len(roots)}")
        meta_df = (
            df[["roi", "label"]].copy()
            if ("roi" in df.columns and "label" in df.columns)
            else pd.DataFrame()
        )

    classes = sorted(np.unique(y_raw).tolist())
    if len(classes) != 2:
        raise ValueError(f"Esperado GROUP binário, encontrei {len(classes)} classes: {classes}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    pos_label_name = "pMCI" if "pMCI" in le.classes_.tolist() else le.classes_[1]
    print(f"Classes: {le.classes_.tolist()}  (pos='{pos_label_name}')")

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    X_dummy = np.zeros((len(y), 1), dtype=np.int8)

    fold_metrics: list[dict] = []
    shap_feat_rows: list[dict[str, object]] = []
    shap_roi_rows: list[dict[str, object]] = []

    shap_folds: set[int] = set()
    if str(args.shap_folds).strip().lower() == "all":
        shap_folds = set(range(1, int(args.n_splits) + 1))
    else:
        for x in _parse_csv_list(str(args.shap_folds)):
            shap_folds.add(int(x))

    run_dir = Path(args.shap_outdir) / f"lstm_shap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    if args.shap:
        run_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, (tr_idx, te_idx) in enumerate(
        splitter.split(X_dummy, strat_col, groups), start=1
    ):
        print(f"\n=== Fold {fold_idx}/{args.n_splits} ===")
        # stats por paciente / "imagens" (quando existirem colunas no df original)
        if args.sequence_source == "pairs":
            # meta_df tem chaves por amostra (inclui ID_PT). Se tiver colunas ID_IMG_* no CSV, também imprime.
            def _count_unique_images_from_meta(meta: pd.DataFrame) -> int:
                img_cols = [c for c in ["ID_IMG_i1", "ID_IMG_i2", "ID_IMG_i3", "ID_IMG_ref"] if c in meta.columns]
                if not img_cols:
                    return 0
                s: set[str] = set()
                for c in img_cols:
                    s |= set(meta[c].astype(str).tolist())
                s.discard("nan"); s.discard("None"); s.discard("")
                return len(s)

            meta_tr = meta_df.iloc[tr_idx] if not meta_df.empty else pd.DataFrame()
            meta_te = meta_df.iloc[te_idx] if not meta_df.empty else pd.DataFrame()
            if not meta_tr.empty:
                msg = f"[outer_train] samples={len(tr_idx)} ID_PT={meta_tr['ID_PT'].astype(str).nunique()}"
                nimg = _count_unique_images_from_meta(meta_tr)
                if nimg:
                    msg += f" unique_images={nimg}"
                print(msg)
            else:
                print(f"[outer_train] samples={len(tr_idx)}")
            if not meta_te.empty:
                msg = f"[outer_test] samples={len(te_idx)} ID_PT={meta_te['ID_PT'].astype(str).nunique()}"
                nimg = _count_unique_images_from_meta(meta_te)
                if nimg:
                    msg += f" unique_images={nimg}"
                print(msg)
            else:
                print(f"[outer_test] samples={len(te_idx)}")
        else:
            # modo columns: cada linha é uma amostra
            df_tr = df.iloc[tr_idx]
            df_te = df.iloc[te_idx]
            print(f"[outer_train] rows={len(df_tr)} ID_PT={df_tr['ID_PT'].astype(str).nunique()}")
            print(f"[outer_test] rows={len(df_te)} ID_PT={df_te['ID_PT'].astype(str).nunique()}")

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sex_tr, sex_te = sex[tr_idx], sex[te_idx]

        # Split interno (validação) respeitando grupos por paciente
        inner = StratifiedGroupKFold(
            n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
        )
        inner_tr, inner_va = next(inner.split(X_dummy[tr_idx], strat_col[tr_idx], groups[tr_idx]))
        # índices globais de pacientes no fit/val
        idx_fit = tr_idx[inner_tr]
        idx_val = tr_idx[inner_va]

        if args.balance == "downsample":
            # downsample por paciente dentro do fit
            df_fit = df.iloc[idx_fit].copy()
            df_fit["strat"] = df_fit["GROUP"].astype(str) + "_" + df_fit["SEX"].astype(str)
            pt_strat = df_fit.groupby("ID_PT", sort=False)["strat"].first()
            rng = np.random.RandomState(int(args.seed))
            pts_by_strat: dict[str, list[str]] = {}
            for pt, st in pt_strat.items():
                pts_by_strat.setdefault(str(st), []).append(str(pt))
            min_n = min(len(v) for v in pts_by_strat.values())
            selected_pts: set[str] = set()
            for st, pts in pts_by_strat.items():
                pts = list(pts)
                rng.shuffle(pts)
                selected_pts.update(pts[:min_n])
            idx_fit = idx_fit[df_fit["ID_PT"].astype(str).isin(selected_pts).to_numpy()]

        fit_mask = np.isin(tr_idx, idx_fit)
        val_mask = np.isin(tr_idx, idx_val)
        X_fit, X_val = X_tr[fit_mask], X_tr[val_mask]
        y_fit, y_val = y_tr[fit_mask], y_tr[val_mask]
        sex_fit, sex_val = sex_tr[fit_mask], sex_tr[val_mask]

        # z-score SEM vazamento: fit no X_fit e aplica em val/test
        X_fit, X_val, X_te = _zscore_fit_transform_3d_from_fit(X_fit, X_val, X_te)

        print(
            f"[fit] samples={len(X_fit)} ID_PT={len(set(groups[idx_fit]))} | "
            f"[val] samples={len(X_val)} ID_PT={len(set(groups[idx_val]))} | "
            f"[test] samples={len(X_te)} ID_PT={len(set(groups[te_idx]))}"
        )

        # Sem balanceamento adicional por default (evita dupla correção).
        # Se quiser class_weight no futuro, reintroduza como flag separada.
        class_weight = None

        model = _make_model(tf, seq_len=X_tr.shape[1], n_feat=X_tr.shape[2])
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=8, restore_best_weights=True
            )
        ]

        model.fit(
            {"x": X_fit, "sex": sex_fit},
            y_fit,
            validation_data=({"x": X_val, "sex": sex_val}, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            class_weight=class_weight,
            callbacks=cb,
        )

        y_proba = model.predict({"x": X_te, "sex": sex_te}, verbose=0).reshape(-1)
        y_pred = (y_proba >= 0.5).astype(np.int32)

        acc = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, pos_label=1)
        auc = roc_auc_score(y_te, y_proba)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "f1_pos": float(f1),
                "roc_auc_pos": float(auc),
            }
        )

        print(
            f"\nFold {fold_idx}/{args.n_splits} | "
            f"acc={acc:.4f} bacc={bacc:.4f} f1={f1:.4f} auc={auc:.4f}"
        )

        if args.shap and (fold_idx in shap_folds):
            try:
                import shap  # type: ignore
            except Exception as e:
                raise SystemExit(
                    "SHAP não está instalado. Instale com: pip install shap\n"
                    f"Erro original: {type(e).__name__}: {e}"
                )
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as e:
                raise SystemExit(
                    "matplotlib é necessário para salvar plots do SHAP. Instale com: pip install matplotlib\n"
                    f"Erro original: {type(e).__name__}: {e}"
                )

            n_bg = max(1, min(int(args.shap_background), X_tr.shape[0]))
            n_samp = max(1, min(int(args.shap_samples), X_te.shape[0]))
            bg = X_tr[:n_bg]
            X_eval = X_te[:n_samp]
            sex_eval = sex_te[:n_samp]

            meta_eval = (
                meta_df.iloc[te_idx].reset_index(drop=True).iloc[:n_samp]
                if not meta_df.empty
                else pd.DataFrame()
            )

            seq_len = X_eval.shape[1]
            n_feat = X_eval.shape[2]

            # SHAP chama o modelo com batches de tamanhos variados (masks por linha).
            # Para evitar mismatch de batch entre x e sex, empacotamos sex como a última "feature"
            # e separamos dentro do predict.
            def predict_proba_pos(x_flat_with_sex: np.ndarray) -> np.ndarray:
                x_flat = x_flat_with_sex[:, :-1]
                sex_b = x_flat_with_sex[:, -1].astype(np.float32, copy=False).reshape(-1, 1)
                x_3d = x_flat.reshape((x_flat.shape[0], seq_len, n_feat))
                return model.predict({"x": x_3d, "sex": sex_b}, verbose=0).reshape(-1)

            bg_flat = bg.reshape((bg.shape[0], seq_len * n_feat))
            X_flat = X_eval.reshape((X_eval.shape[0], seq_len * n_feat))
            # anexa sexo como última feature
            bg_flat = np.concatenate([bg_flat, sex_tr[:n_bg].astype(np.float32)], axis=1)
            X_flat = np.concatenate([X_flat, sex_eval.astype(np.float32)], axis=1)

            explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg_flat))
            shap_values = explainer(X_flat)
            sv = np.asarray(shap_values.values)
            abs_sv = np.abs(sv)
            sample_mean_abs = abs_sv.mean(axis=1)

            # nomes das features: "<feat>@<step>"
            if args.sequence_source == "pairs":
                step_tags = pair_order
            else:
                step_tags = list(range(seq_len))

            feat_names: list[str] = []
            for step_i, tag in enumerate(step_tags):
                for c in feat_cols:
                    feat_names.append(f"{c}@{tag}")
            feat_names.append("sex")

            feat_rank = (
                pd.DataFrame(
                    {"feature": feat_names, "mean_abs_shap": abs_sv.mean(axis=0).tolist()}
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            feat_csv = run_dir / f"shap_feature_importance_lstm_fold_{fold_idx:02d}.csv"
            feat_rank.to_csv(feat_csv, index=False)
            print(f"[SHAP] feature importance -> {feat_csv}")

            for _, r in feat_rank.iterrows():
                shap_feat_rows.append(
                    {
                        "fold": int(fold_idx),
                        "feature": str(r["feature"]),
                        "mean_abs_shap": float(r["mean_abs_shap"]),
                    }
                )

            if not meta_eval.empty and ("roi" in meta_eval.columns) and ("label" in meta_eval.columns):
                roi_meta = meta_eval[["roi", "label"]].copy()
                roi_meta["sample_mean_abs_shap"] = sample_mean_abs[: len(roi_meta)]
                roi_rank = (
                    roi_meta.groupby(["roi", "label"], dropna=False)["sample_mean_abs_shap"]
                    .mean()
                    .reset_index()
                    .sort_values("sample_mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                roi_csv = run_dir / f"shap_roi_importance_lstm_fold_{fold_idx:02d}.csv"
                roi_rank.to_csv(roi_csv, index=False)
                print(f"[SHAP] ROI/label importance -> {roi_csv}")
                for _, rr in roi_rank.iterrows():
                    shap_roi_rows.append(
                        {
                            "fold": int(fold_idx),
                            "roi": rr["roi"],
                            "label": rr["label"],
                            "mean_sample_abs_shap": float(rr["sample_mean_abs_shap"]),
                        }
                    )

            # Plots (salva PNGs; útil em ambiente sem display)
            try:
                # força nomes reais
                try:
                    shap_values.feature_names = feat_names
                except Exception:
                    pass
                shap_values_named = shap.Explanation(
                    values=np.asarray(shap_values.values),
                    base_values=shap_values.base_values,
                    data=X_flat,
                    feature_names=feat_names,
                )
                shap.plots.bar(shap_values_named, max_display=30, show=False)
                plt.tight_layout()
                p = run_dir / f"shap_bar_lstm_fold_{fold_idx:02d}.png"
                plt.savefig(p, dpi=200)
                plt.close()
                print(f"[SHAP] bar plot -> {p}")
            except Exception as e:
                print(f"[SHAP][WARN] falha ao gerar bar plot: {e}")

            try:
                shap.plots.beeswarm(shap_values_named, max_display=30, show=False)
                plt.tight_layout()
                p = run_dir / f"shap_beeswarm_lstm_fold_{fold_idx:02d}.png"
                plt.savefig(p, dpi=200)
                plt.close()
                print(f"[SHAP] beeswarm plot -> {p}")
            except Exception as e:
                print(f"[SHAP][WARN] falha ao gerar beeswarm plot: {e}")

            # Plot das principais ROIs (rank por roi,label)
            try:
                if "roi_rank" in locals():
                    topn = min(20, len(roi_rank))
                    if topn > 0:
                        roi_plot = roi_rank.head(topn).copy()
                        roi_plot["roi_label"] = roi_plot["roi"].astype(str) + ":" + roi_plot["label"].astype(str)
                        plt.figure(figsize=(10, 6))
                        plt.barh(roi_plot["roi_label"][::-1], roi_plot["sample_mean_abs_shap"][::-1])
                        plt.xlabel("mean(|SHAP|) por amostra (média por roi,label)")
                        plt.title("Top ROIs por contribuição SHAP (LSTM)")
                        plt.tight_layout()
                        p = run_dir / f"roi_bar_lstm_fold_{fold_idx:02d}.png"
                        plt.savefig(p, dpi=200)
                        plt.close()
                        print(f"[SHAP] ROI bar plot -> {p}")
            except Exception as e:
                print(f"[SHAP][WARN] falha ao gerar ROI bar plot: {e}")

    m = pd.DataFrame(fold_metrics)
    print("\n=== Resumo CV ===")
    print(m[["accuracy", "balanced_accuracy", "f1_pos", "roc_auc_pos"]].agg(["mean", "std"]))

    if args.shap and shap_feat_rows:
        sf = pd.DataFrame(shap_feat_rows)
        sf.to_csv(run_dir / "shap_feature_importance_all_folds_long.csv", index=False)
        sf_agg = (
            sf.groupby(["feature"])["mean_abs_shap"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={"mean": "mean_abs_shap_mean", "std": "mean_abs_shap_std", "count": "n_folds_present"}
            )
            .sort_values("mean_abs_shap_mean", ascending=False)
            .reset_index(drop=True)
        )
        sf_agg.to_csv(run_dir / "shap_feature_importance_agg.csv", index=False)
        print(f"[SHAP] agregado features -> {run_dir / 'shap_feature_importance_agg.csv'}")

    if args.shap and shap_roi_rows:
        sr = pd.DataFrame(shap_roi_rows)
        sr.to_csv(run_dir / "shap_roi_importance_all_folds_long.csv", index=False)
        sr_agg = (
            sr.groupby(["roi", "label"])["mean_sample_abs_shap"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_sample_abs_shap_mean",
                    "std": "mean_sample_abs_shap_std",
                    "count": "n_folds_present",
                }
            )
            .sort_values("mean_sample_abs_shap_mean", ascending=False)
            .reset_index(drop=True)
        )
        sr_agg.to_csv(run_dir / "shap_roi_importance_agg.csv", index=False)
        print(f"[SHAP] agregado ROI -> {run_dir / 'shap_roi_importance_agg.csv'}")


if __name__ == "__main__":
    main()


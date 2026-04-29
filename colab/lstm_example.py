# How to run:
    
#     python /mnt/study-data/pgirardi/graphs/colab/lstm_example.py \
#   --sequence-source pairs \
#   --pair-order 12,13,23 \
#   --device cpu

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
    "/mnt/study-data/pgirardi/graphs/csvs/abordagem_4_teste/"
    "features_all_abordagem_4_teste.csv"
)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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

    if not rows:
        raise ValueError(
            f"Nenhuma sequência foi montada. Pairs disponíveis: {sorted(available_pairs)}. "
            f"Esperado conter: {pair_order}."
        )

    X = np.stack([r[0] for r in rows], axis=0)
    y = np.array([r[1] for r in rows], dtype=object)
    sex = np.array([r[2] for r in rows], dtype=object)
    id_pt = np.array([r[3] for r in rows], dtype=object)

    feature_cols = base_feature_cols + ["dt"]

    if drop_if_nan:
        mask = np.isfinite(X).all(axis=(1, 2))
        X, y, sex, id_pt = X[mask], y[mask], sex[mask], id_pt[mask]

    return X, y, sex, id_pt, feature_cols


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

    required = {"ID_PT", "GROUP", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    if args.sequence_source == "pairs":
        pair_order = [p.strip() for p in args.pair_order.split(",") if p.strip()]
        group_cols = [c for c in ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX", "roi", "side"] if c in df.columns]
        X, y_raw, sex_raw, id_pt_raw, feat_cols = _build_pairwise_triplet_tensor(
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
    for fold_idx, (tr_idx, te_idx) in enumerate(
        splitter.split(X_dummy, strat_col, groups), start=1
    ):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sex_tr, sex_te = sex[tr_idx], sex[te_idx]

        # z-score por fold (fit no treino)
        X_tr, X_te = _zscore_fit_transform_3d(X_tr, X_te)

        # Class weights simples para desbalanceamento
        n0 = int((y_tr == 0).sum())
        n1 = int((y_tr == 1).sum())
        if n0 == 0 or n1 == 0:
            raise ValueError(
                f"Fold {fold_idx}: uma classe sumiu no treino (n0={n0}, n1={n1})."
            )
        class_weight = {0: (n0 + n1) / (2 * n0), 1: (n0 + n1) / (2 * n1)}

        model = _make_model(tf, seq_len=X_tr.shape[1], n_feat=X_tr.shape[2])
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=8, restore_best_weights=True
            )
        ]

        model.fit(
            {"x": X_tr, "sex": sex_tr},
            y_tr,
            validation_split=0.1,
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

    m = pd.DataFrame(fold_metrics)
    print("\n=== Resumo CV ===")
    print(m[["accuracy", "balanced_accuracy", "f1_pos", "roc_auc_pos"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()


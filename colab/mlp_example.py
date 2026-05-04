# python colab/mlp_example.py \
#   --csv "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv" \
#   --input-mode wide_from_long \
#   --device cpu \
#   --n-splits 5 \
#   --inner-fold 5 \
#   --seed 42 \
#   --kbest 600 \
#   --epochs 80 \
#   --batch-size 64 \
#   --balance downsample \
#   --shap \
#   --shap-samples 200 \
#   --shap-background 150 \
#   --shap-outdir "/mnt/study-data/pgirardi/graphs/colab/outputs"

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from colab.datasets import build_wide_tabular_from_long_pairs


DEFAULT_CSV = (
    "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/"
    "all_delta_features_neurocombat.csv"
)


def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _downsample_patients_by_group_sex(
    df_train: pd.DataFrame,
    *,
    id_col: str = "ID_PT",
    group_col: str = "GROUP",
    sex_col: str = "SEX",
    seed: int = 123,
) -> pd.DataFrame:
    if group_col not in df_train.columns or sex_col not in df_train.columns or id_col not in df_train.columns:
        raise ValueError("Downsample requer colunas ID_PT, GROUP e SEX.")
    td = df_train.copy()
    td["_strat"] = td[group_col].astype(str) + "_" + td[sex_col].astype(str)
    pt_strat = td.groupby(id_col, sort=False)["_strat"].first()
    rng = np.random.RandomState(int(seed))
    pts_by_strat: dict[str, list[str]] = {}
    for pt, st in pt_strat.items():
        pts_by_strat.setdefault(str(st), []).append(str(pt))
    min_n = min(len(v) for v in pts_by_strat.values())
    selected_pts: set[str] = set()
    for st, pts in pts_by_strat.items():
        pts = list(pts)
        rng.shuffle(pts)
        selected_pts.update(pts[:min_n])
    return td[td[id_col].astype(str).isin(selected_pts)].drop(columns=["_strat"])


def _zscore_fit_transform_2d_from_fit(
    X_fit: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(X_fit)
    return (
        scaler.transform(X_fit).astype(np.float32, copy=False),
        scaler.transform(X_val).astype(np.float32, copy=False),
        scaler.transform(X_test).astype(np.float32, copy=False),
    )


def _make_model(tf_module, n_feat: int, *, hidden: list[int], dropout: float, lr: float):
    keras = tf_module.keras
    layers = keras.layers
    models = keras.models

    x_in = keras.Input(shape=(n_feat,), name="x")
    sex_in = keras.Input(shape=(1,), name="sex")

    x = x_in
    for h in hidden:
        x = layers.Dense(int(h), activation="relu")(x)
        if float(dropout) > 0.0:
            x = layers.Dropout(float(dropout))(x)

    x = layers.Concatenate()([x, sex_in])
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[x_in, sex_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLP (deep tabular) para classificar sMCI vs pMCI. "
            "Suporta construir wide+flatten a partir do CSV long (roi/side/pair por linha) "
            "e nested CV por paciente (ID_PT)."
        )
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Caminho do CSV.")
    parser.add_argument(
        "--input-mode",
        choices=["as_is_rows", "wide_from_long"],
        default="wide_from_long",
        help=(
            "'as_is_rows': cada linha do CSV é uma amostra (comportamento tipo cnn_example). "
            "'wide_from_long': faz pivot para 1 linha por conjunto (ID_PT,COMBINATION_NUMBER,TRIPLET_IDX)."
        ),
    )
    parser.add_argument("--kbest", type=int, default=600, help="Número de features (KBest) no treino externo.")
    parser.add_argument("--epochs", type=int, default=80, help="Épocas de treino por fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Folds externos do StratifiedGroupKFold.")
    parser.add_argument("--inner-fold", type=int, default=5, help="Folds internos (val) dentro do treino externo.")
    parser.add_argument(
        "--balance",
        choices=["none", "downsample"],
        default="none",
        help="Balanceamento aplicado SOMENTE no fit (treino interno).",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        default="512,128",
        help="Lista de tamanhos de camadas ocultas. Ex.: 512,128",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout nas camadas ocultas.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate do Adam.")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device para treino. Se der erro cuDNN/driver, use '--device cpu'.",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="Qual GPU usar quando --device gpu.")
    parser.add_argument(
        "--memory-growth",
        action="store_true",
        help="Habilita memory growth (reduz chance de erro por alocação de VRAM).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Gera SHAP no fold 1 (para acelerar) e salva ranking de atributos.",
    )
    parser.add_argument("--shap-samples", type=int, default=200, help="Amostras no SHAP (teste).")
    parser.add_argument("--shap-background", type=int, default=150, help="Background no SHAP (treino).")
    parser.add_argument(
        "--shap-outdir",
        type=str,
        default=str(Path(__file__).parent / "outputs"),
        help="Diretório para salvar SHAP.",
    )
    args = parser.parse_args()

    # Configure TF device selection BEFORE importing tensorflow.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

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
    df_raw = pd.read_csv(csv_path)

    if args.input_mode == "wide_from_long":
        df = build_wide_tabular_from_long_pairs(df_raw)
        # em wide_from_long, ID_PT continua existindo (para groups)
    else:
        df = df_raw.copy()

    required = {"ID_PT", "GROUP", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    # y
    y_raw = df["GROUP"].astype(str).to_numpy()
    classes = sorted(np.unique(y_raw).tolist())
    if len(classes) != 2:
        raise ValueError(f"Esperado GROUP binário, encontrei {len(classes)} classes: {classes}")
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    pos_label_name = "pMCI" if "pMCI" in le.classes_.tolist() else le.classes_[1]
    print(f"Classes: {le.classes_.tolist()}  (pos='{pos_label_name}')")

    # sex
    sex = df["SEX"].astype(str).str.upper().map({"F": 0, "M": 1}).fillna(-1).to_numpy()
    if (sex < 0).any():
        bad = sorted(df.loc[sex < 0, "SEX"].astype(str).unique().tolist())
        raise ValueError(f"Valores inesperados em SEX: {bad} (esperado F/M)")
    sex = sex.astype(np.float32).reshape(-1, 1)

    # groups + strat
    groups = df["ID_PT"].astype(str).to_numpy()
    strat_col = (df["GROUP"].astype(str) + "_" + df["SEX"].astype(str)).to_numpy()

    # X: numéricas exceto colunas meta
    ignore = {"GROUP", "SEX", "ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"}
    X_df = df[[c for c in df.columns if c not in ignore]].select_dtypes(include=["number"])
    if X_df.empty:
        raise ValueError("Nenhuma feature numérica encontrada após remover colunas meta.")
    X_all = X_df.to_numpy(dtype=np.float32, copy=True)
    feat_names = list(X_df.columns)

    hidden = [int(x) for x in _parse_csv_list(args.hidden)]
    if not hidden:
        raise ValueError("--hidden precisa ter pelo menos 1 camada. Ex.: 512,128")

    splitter = StratifiedGroupKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    X_dummy = np.zeros((len(y), 1), dtype=np.int8)

    fold_rows: list[dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(X_dummy, strat_col, groups), start=1):
        print(f"\n=== Fold {fold_idx}/{int(args.n_splits)} ===")
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sex_tr, sex_te = sex[tr_idx], sex[te_idx]

        # Split interno (val)
        inner = StratifiedGroupKFold(n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed))
        inner_tr, inner_va = next(inner.split(X_dummy[tr_idx], strat_col[tr_idx], groups[tr_idx]))
        idx_fit = tr_idx[inner_tr]
        idx_val = tr_idx[inner_va]

        df_fit = df.iloc[idx_fit].copy()
        if args.balance == "downsample":
            df_fit = _downsample_patients_by_group_sex(df_fit, seed=int(args.seed) + int(fold_idx))
            idx_fit = df_fit.index.to_numpy(dtype=np.int64)

        # Masks dentro do pool de treino externo
        fit_mask = np.isin(tr_idx, idx_fit)
        val_mask = np.isin(tr_idx, idx_val)

        X_fit = X_tr[fit_mask]
        y_fit = y_tr[fit_mask]
        sex_fit = sex_tr[fit_mask]

        X_val = X_tr[val_mask]
        y_val = y_tr[val_mask]
        sex_val = sex_tr[val_mask]

        # KBest SEM vazamento: fit no conjunto fit
        k = max(1, min(int(args.kbest), X_fit.shape[1]))
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_fit, y_fit)
        mask = selector.get_support()
        selected_names = [n for n, keep in zip(feat_names, mask) if bool(keep)]

        X_fit = X_fit[:, mask]
        X_val = X_val[:, mask]
        X_te2 = X_te[:, mask]

        # z-score SEM vazamento: fit no fit e aplica em val/test
        X_fit, X_val, X_te2 = _zscore_fit_transform_2d_from_fit(X_fit, X_val, X_te2)

        model = _make_model(
            tf,
            n_feat=X_fit.shape[1],
            hidden=hidden,
            dropout=float(args.dropout),
            lr=float(args.lr),
        )
        cb = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)
        ]
        model.fit(
            {"x": X_fit, "sex": sex_fit},
            y_fit,
            validation_data=({"x": X_val, "sex": sex_val}, y_val),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            verbose=1,
            callbacks=cb,
        )

        y_proba = model.predict({"x": X_te2, "sex": sex_te}, verbose=0).reshape(-1)
        y_pred = (y_proba >= 0.5).astype(np.int32)

        acc = float(accuracy_score(y_te, y_pred))
        bacc = float(balanced_accuracy_score(y_te, y_pred))
        f1 = float(f1_score(y_te, y_pred, pos_label=1))
        auc = float(roc_auc_score(y_te, y_proba))
        fold_rows.append({"fold": float(fold_idx), "acc": acc, "bacc": bacc, "f1": f1, "auc": auc})
        print(f"[fold {fold_idx}] acc={acc:.4f} bacc={bacc:.4f} f1={f1:.4f} auc={auc:.4f} | kbest={k}")

        # SHAP (somente fold 1 para acelerar)
        if args.shap and fold_idx == 1:
            try:
                import shap  # type: ignore
            except Exception as e:
                raise SystemExit(
                    "SHAP não está instalado. Instale com: pip install shap\n"
                    f"Erro original: {type(e).__name__}: {e}"
                )
            out_dir = Path(args.shap_outdir) / f"mlp_shap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir.mkdir(parents=True, exist_ok=True)

            n_bg = max(1, min(int(args.shap_background), X_fit.shape[0]))
            n_samp = max(1, min(int(args.shap_samples), X_te2.shape[0]))
            bg = X_fit[:n_bg]
            X_eval = X_te2[:n_samp]
            sex_eval = sex_te[:n_samp]

            # empacota sexo como última feature para evitar mismatch de batch sizes
            def predict_proba_pos(x_with_sex: np.ndarray) -> np.ndarray:
                x = x_with_sex[:, :-1]
                sx = x_with_sex[:, -1].astype(np.float32, copy=False).reshape(-1, 1)
                return model.predict({"x": x, "sex": sx}, verbose=0).reshape(-1)

            bg2 = np.concatenate([bg, sex_fit[:n_bg].astype(np.float32)], axis=1)
            X2 = np.concatenate([X_eval, sex_eval.astype(np.float32)], axis=1)
            feat_names_shap = selected_names + ["sex"]

            explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg2))
            shap_values = explainer(X2)
            sv = np.asarray(shap_values.values)
            abs_sv = np.abs(sv)
            feat_rank = (
                pd.DataFrame({"feature": feat_names_shap, "mean_abs_shap": abs_sv.mean(axis=0).tolist()})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            feat_rank.to_csv(out_dir / "shap_feature_importance_mlp_fold01.csv", index=False)
            print(f"[SHAP] feature rank -> {out_dir / 'shap_feature_importance_mlp_fold01.csv'}")

    m = pd.DataFrame(fold_rows)
    print("\n=== Resumo CV ===")
    print(m[["acc", "bacc", "f1", "auc"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()


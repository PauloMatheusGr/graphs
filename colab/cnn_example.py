# python colab/cnn_example.py \
#   --csv "/mnt/study-data/pgirardi/graphs/csvs/abordagem_teste/all_delta_features_neurocombat.csv" \
#   --device cpu \
#   --n-splits 5 \
#   --inner-fold 5 \
#   --seed 42 \
#   --kbest 100 \
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    Filtra linhas pelo(s) ROI(s) e/ou label(s).

    - rois: valores em df['roi'] (ex.: 'hippocampus,amygdala')
    - labels: inteiros em df['label'] (ex.: 17,53)
    - roi_label: pares 'roi:label' (ex.: 'hippocampus:17,hippocampus:53')

    Regras:
    - Se roi_label for fornecido, ele define o filtro (OR entre pares).
    - Caso contrário, aplica (roi in rois) AND (label in labels) quando ambos existirem.
    """
    if ("roi" not in df.columns) or ("label" not in df.columns):
        raise ValueError("CSV precisa ter colunas 'roi' e 'label' para filtrar ROIs.")

    rois = rois or []
    labels = labels or []
    roi_label = roi_label or []

    out = df.copy()
    out["roi"] = out["roi"].astype(str).str.strip()

    # label pode vir como str/float; normaliza em int.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina um CNN 1D simples a partir de um CSV de features."
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Caminho do CSV.")
    parser.add_argument("--kbest", type=int, default=100, help="Número de features (KBest).")
    parser.add_argument("--epochs", type=int, default=100, help="Épocas de treino.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção do teste.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do split.")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=0,
        help=(
            "Se > 1, usa cross-validation externa (folds). "
            "Se <= 1, usa um único holdout por grupos (ID_PT) com --test-size."
        ),
    )
    parser.add_argument(
        "--inner-fold",
        type=int,
        default=5,
        help=(
            "Número de folds internos para separar validação dentro do treino (nested). "
            "Usa o primeiro split como validação para early stopping."
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
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device para treino. Se der erro cuDNN/driver, use '--device cpu'.",
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
        help="Quantas amostras do teste usar no SHAP (para acelerar).",
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

    # Seleção de device: configure env vars ANTES de importar TF.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf
    from tensorflow.keras import layers, models

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

    pd_feat = pd.read_csv(csv_path)
    rois = _parse_csv_list(args.roi)
    labels = [int(x) for x in _parse_csv_list(args.label)]
    roi_label = _parse_csv_list(args.roi_label)
    if rois or labels or roi_label:
        before = pd_feat.shape
        pd_feat = filter_by_roi_label(pd_feat, rois=rois, labels=labels, roi_label=roi_label)
        print(f"[ROI] filtro aplicado: {before} -> {pd_feat.shape}")

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)

    if "GROUP" not in pd_feat.columns:
        raise KeyError(
            "Coluna target 'GROUP' não encontrada no CSV. "
            f"Colunas disponíveis: {list(pd_feat.columns)[:30]}..."
        )

    # Features numéricas (float) e target
    X_df = pd_feat.select_dtypes(include=["float"])
    # for col in ("t12", "t13"):
    #     if col in X_df.columns:
    #         X_df = X_df.drop([col], axis=1)
    y = pd_feat["GROUP"]

    y_target = pd_feat["GROUP"].to_numpy()

    # Encode se não for binário numérico
    if y_target.dtype.kind in {"U", "S", "O"} or len(np.unique(y_target)) > 2:
        y_target = LabelEncoder().fit_transform(y_target)

    idx_all = np.arange(len(pd_feat), dtype=np.int64)

    # helpers de estratificação/grupos
    strat_col = (
        (pd_feat["GROUP"].astype(str) + "_" + pd_feat["SEX"].astype(str)).to_numpy()
        if "SEX" in pd_feat.columns
        else pd_feat["GROUP"].astype(str).to_numpy()
    )
    if "ID_PT" not in pd_feat.columns:
        raise KeyError(
            "Para evitar vazamento por paciente, este script exige a coluna 'ID_PT' no CSV."
        )
    groups = pd_feat["ID_PT"].astype(str).to_numpy()

    is_binary = len(np.unique(y_target)) == 2
    if not is_binary:
        raise ValueError(
            "Esse script está configurado para classificação binária no pós-processamento "
            "(threshold 0.5). Seu target parece ter mais de 2 classes."
        )

    def make_model(n_feat: int):
        m = models.Sequential(
            [
                tf.keras.Input(shape=(n_feat, 1)),
                layers.Conv1D(32, 2, activation="relu"),
                layers.Flatten(),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m

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

    def _unique_image_ids_from_rows(df_slice: pd.DataFrame) -> int:
        img_cols = [c for c in ["ID_IMG", "ID_IMG_ref", "ID_IMG_i1", "ID_IMG_i2", "ID_IMG_i3"] if c in df_slice.columns]
        if not img_cols:
            return 0
        ids: set[str] = set()
        for c in img_cols:
            ids |= set(df_slice[c].astype(str).tolist())
        ids.discard("nan")
        ids.discard("None")
        ids.discard("")
        return len(ids)

    def _print_split_stats(tag: str, idx: np.ndarray) -> None:
        df_slice = pd_feat.iloc[idx]
        n_rows = int(len(df_slice))
        n_pt = int(df_slice["ID_PT"].astype(str).nunique())
        n_img = _unique_image_ids_from_rows(df_slice)
        msg = f"[{tag}] rows={n_rows} ID_PT={n_pt}"
        if n_img:
            msg += f" unique_images={n_img}"
        if "GROUP" in df_slice.columns:
            g = df_slice["GROUP"].astype(str).value_counts().to_dict()
            msg += f" GROUP={g}"
        if "SEX" in df_slice.columns:
            s = df_slice["SEX"].astype(str).value_counts().to_dict()
            msg += f" SEX={s}"
        print(msg)

    def _downsample_train_idx(idx: np.ndarray, *, seed: int) -> np.ndarray:
        """
        Downsample por paciente: seleciona um subconjunto de ID_PT para que cada estrato GROUP+SEX
        tenha o mesmo número de pacientes (igual ao mínimo).
        Retorna índices de LINHAS pertencentes aos pacientes selecionados.
        """
        train_df = pd_feat.iloc[idx].copy()
        if "GROUP" not in train_df.columns or "SEX" not in train_df.columns:
            raise KeyError("Para downsample é necessário ter colunas GROUP e SEX no CSV.")
        train_df["strat"] = train_df["GROUP"].astype(str) + "_" + train_df["SEX"].astype(str)
        # 1 linha por paciente para contagem por estrato
        pt_strat = train_df.groupby("ID_PT", sort=False)["strat"].first()
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
        mask = train_df["ID_PT"].astype(str).isin(selected_pts).to_numpy()
        return idx[mask]

    fold_metrics: list[dict[str, float]] = []
    shap_feat_rows: list[dict[str, object]] = []
    shap_roi_rows: list[dict[str, object]] = []

    shap_folds: set[int] = set()
    if str(args.shap_folds).strip().lower() == "all":
        shap_folds = set(range(1, int(args.n_splits) + 1)) if int(args.n_splits) > 1 else {1}
    else:
        for x in _parse_csv_list(str(args.shap_folds)):
            shap_folds.add(int(x))

    run_dir = Path(args.shap_outdir) / f"cnn_shap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    if args.shap:
        run_dir.mkdir(parents=True, exist_ok=True)

    if int(args.n_splits) and int(args.n_splits) > 1:
        # Loop externo (CV)
        outer = StratifiedGroupKFold(
            n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed)
        )
        outer_split = outer.split(idx_all.reshape(-1, 1), strat_col, groups)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_split, start=1):
            print(f"\n=== Fold {fold_idx}/{int(args.n_splits)} ===")
            _print_split_stats("outer_train", tr_idx)
            _print_split_stats("outer_test", te_idx)

            # KBest SEM vazamento: fit apenas no treino externo
            X_tr_df = X_df.iloc[tr_idx]
            y_tr_raw = pd_feat["GROUP"].iloc[tr_idx]
            selector = SelectKBest(score_func=f_classif, k=int(args.kbest))
            selector.fit(X_tr_df, y_tr_raw)
            selected_features = X_tr_df.columns[selector.get_support()]

            X_sel = pd_feat[selected_features].to_numpy()
            if y_target.dtype.kind in {"U", "S", "O"} or len(np.unique(y_target)) > 2:
                y_enc = LabelEncoder().fit_transform(y_target)
            else:
                y_enc = y_target.astype(int, copy=False)

            X_tr2d, X_te2d = X_sel[tr_idx], X_sel[te_idx]
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            # Loop interno (validação) para early stopping (sem vazamento por ID_PT quando disponível)
            inner = StratifiedGroupKFold(
                n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
            )
            inner_tr, inner_va = next(
                inner.split(X_tr2d, strat_col[tr_idx], groups[tr_idx])
            )
            idx_fit = tr_idx[inner_tr]
            idx_val = tr_idx[inner_va]
            if args.balance == "downsample":
                idx_fit = _downsample_train_idx(idx_fit, seed=int(args.seed))
            _print_split_stats("fit", idx_fit)
            _print_split_stats("val", idx_val)

            # Re-indexa fit/val para arrays 2d
            fit_mask = np.isin(tr_idx, idx_fit)
            val_mask = np.isin(tr_idx, idx_val)
            X_fit_2d = X_tr2d[fit_mask]
            y_fit = y_tr[fit_mask]
            X_val_2d = X_tr2d[val_mask]
            y_val = y_tr[val_mask]

            # z-score SEM vazamento: fit no conjunto de fit
            X_fit_2d, X_val_2d, X_te2d = _zscore_fit_transform_2d_from_fit(
                X_fit_2d, X_val_2d, X_te2d
            )

            X_fit = X_fit_2d.reshape((X_fit_2d.shape[0], X_fit_2d.shape[1], 1))
            X_val = X_val_2d.reshape((X_val_2d.shape[0], X_val_2d.shape[1], 1))
            X_test = X_te2d.reshape((X_te2d.shape[0], X_te2d.shape[1], 1))

            model = make_model(n_feat=X_tr2d.shape[1])
            cb = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ]
            model.fit(
                X_fit,
                y_fit,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                callbacks=cb,
            )

            y_pred_proba = model.predict(X_test, verbose=0).reshape(-1)
            y_pred = (y_pred_proba > 0.5).astype(int)

            acc = float(accuracy_score(y_te, y_pred))
            prec = float(precision_score(y_te, y_pred, average="weighted"))
            rec = float(recall_score(y_te, y_pred, average="weighted"))
            f1 = float(f1_score(y_te, y_pred, average="weighted"))
            fold_metrics.append(
                {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
            )
            print(
                f"\nFold {fold_idx}/{int(args.n_splits)} | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
            )

            if args.shap and (int(fold_idx) in shap_folds):
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

                feat_names = list(selected_features)

                n_samp = max(1, min(int(args.shap_samples), X_te2d.shape[0]))
                X_eval = X_te2d[:n_samp]
                idx_eval = te_idx[:n_samp]
                n_bg = max(1, min(int(args.shap_background), X_eval.shape[0]))
                bg = X_eval[:n_bg]

                def predict_proba_pos(x_2d: np.ndarray) -> np.ndarray:
                    x_3d = x_2d.reshape((x_2d.shape[0], x_2d.shape[1], 1))
                    return model.predict(x_3d, verbose=0).reshape(-1)

                explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg))
                shap_values = explainer(X_eval)
                try:
                    shap_values.feature_names = feat_names
                except Exception:
                    pass
                shap_values = shap.Explanation(
                    values=np.asarray(shap_values.values),
                    base_values=shap_values.base_values,
                    data=X_eval,
                    feature_names=feat_names,
                )

                sv = np.asarray(shap_values.values)
                abs_sv = np.abs(sv)
                sample_mean_abs = abs_sv.mean(axis=1)

                feat_rank = (
                    pd.DataFrame(
                        {"feature": feat_names, "mean_abs_shap": abs_sv.mean(axis=0).tolist()}
                    )
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                feat_csv = run_dir / f"shap_feature_importance_cnn_fold_{fold_idx:02d}.csv"
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

                if ("roi" in pd_feat.columns) and ("label" in pd_feat.columns):
                    meta = pd_feat.iloc[idx_eval][["roi", "label"]].copy()
                    meta["sample_mean_abs_shap"] = sample_mean_abs[: len(meta)]
                    roi_rank = (
                        meta.groupby(["roi", "label"], dropna=False)["sample_mean_abs_shap"]
                        .mean()
                        .reset_index()
                        .sort_values("sample_mean_abs_shap", ascending=False)
                        .reset_index(drop=True)
                    )
                    roi_csv = run_dir / f"shap_roi_importance_cnn_fold_{fold_idx:02d}.csv"
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

                try:
                    shap.plots.bar(shap_values, max_display=30, show=False)
                    plt.tight_layout()
                    p = run_dir / f"shap_bar_cnn_fold_{fold_idx:02d}.png"
                    plt.savefig(p, dpi=200)
                    plt.close()
                except Exception:
                    pass

                try:
                    shap.plots.beeswarm(shap_values, max_display=30, show=False)
                    plt.tight_layout()
                    p = run_dir / f"shap_beeswarm_cnn_fold_{fold_idx:02d}.png"
                    plt.savefig(p, dpi=200)
                    plt.close()
                except Exception:
                    pass

        m = pd.DataFrame(fold_metrics)
        print("\n=== Resumo CV (média ± std) ===")
        print(m.agg(["mean", "std"]))
    else:
        # Holdout simples (comportamento antigo)
        # Split por grupos (ID_PT) para evitar vazamento paciente->teste
        gss = GroupShuffleSplit(
            n_splits=1, test_size=float(args.test_size), random_state=int(args.seed)
        )
        tr_idx, te_idx = next(gss.split(idx_all.reshape(-1, 1), strat_col, groups))
        print("\n=== Holdout ===")
        _print_split_stats("train_pool", tr_idx)
        _print_split_stats("test", te_idx)

        # KBest SEM vazamento: fit apenas no treino
        X_tr_df = X_df.iloc[tr_idx]
        y_tr_raw = pd_feat["GROUP"].iloc[tr_idx]
        selector = SelectKBest(score_func=f_classif, k=int(args.kbest))
        selector.fit(X_tr_df, y_tr_raw)
        selected_features = X_tr_df.columns[selector.get_support()]

        X_sel = pd_feat[selected_features].to_numpy()
        if y_target.dtype.kind in {"U", "S", "O"} or len(np.unique(y_target)) > 2:
            y_enc = LabelEncoder().fit_transform(y_target)
        else:
            y_enc = y_target.astype(int, copy=False)

        X_tr2d, X_te2d = X_sel[tr_idx], X_sel[te_idx]
        y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

        # Split interno por grupos para validação (early stopping)
        inner = StratifiedGroupKFold(
            n_splits=int(args.inner_fold), shuffle=True, random_state=int(args.seed)
        )
        inner_tr, inner_va = next(inner.split(X_tr2d, strat_col[tr_idx], groups[tr_idx]))

        idx_fit = tr_idx[inner_tr]
        idx_val = tr_idx[inner_va]
        if args.balance == "downsample":
            idx_fit = _downsample_train_idx(idx_fit, seed=int(args.seed))
        _print_split_stats("fit", idx_fit)
        _print_split_stats("val", idx_val)

        fit_mask = np.isin(tr_idx, idx_fit)
        val_mask = np.isin(tr_idx, idx_val)
        X_fit_2d = X_tr2d[fit_mask]
        y_fit = y_tr[fit_mask]
        X_val_2d = X_tr2d[val_mask]
        y_val = y_tr[val_mask]

        # z-score SEM vazamento: fit no conjunto de fit
        X_fit_2d, X_val_2d, X_te2d = _zscore_fit_transform_2d_from_fit(
            X_fit_2d, X_val_2d, X_te2d
        )

        X_fit = X_fit_2d.reshape((X_fit_2d.shape[0], X_fit_2d.shape[1], 1))
        X_val = X_val_2d.reshape((X_val_2d.shape[0], X_val_2d.shape[1], 1))
        X_test = X_te2d.reshape((X_te2d.shape[0], X_te2d.shape[1], 1))

        model = make_model(n_feat=X_tr2d.shape[1])
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ]
        model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=cb,
        )

        y_pred_proba = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (y_pred_proba > 0.5).astype(int)

        print("\n--- Metricas ---")
        print(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}")
        print(f"Precision: {precision_score(y_te, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_te, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(y_te, y_pred, average='weighted'):.4f}")

        mse = mean_squared_error(y_te, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)

        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2-Score: {r2:.4f}")

        if args.shap:
            # Holdout = fold 1
            fold_idx = 1
            if int(fold_idx) in shap_folds:
                try:
                    import shap  # type: ignore
                except Exception as e:
                    raise SystemExit(
                        "SHAP não está instalado. Instale com: pip install shap\n"
                        f"Erro original: {type(e).__name__}: {e}"
                    )

                feat_names = list(selected_features)
                n_samp = max(1, min(int(args.shap_samples), X_te2d.shape[0]))
                X_eval = X_te2d[:n_samp]
                idx_eval = te_idx[:n_samp]
                n_bg = max(1, min(int(args.shap_background), X_eval.shape[0]))
                bg = X_eval[:n_bg]

                def predict_proba_pos(x_2d: np.ndarray) -> np.ndarray:
                    x_3d = x_2d.reshape((x_2d.shape[0], x_2d.shape[1], 1))
                    return model.predict(x_3d, verbose=0).reshape(-1)

                explainer = shap.Explainer(predict_proba_pos, shap.maskers.Independent(bg))
                shap_values = explainer(X_eval)
                shap_values = shap.Explanation(
                    values=np.asarray(shap_values.values),
                    base_values=shap_values.base_values,
                    data=X_eval,
                    feature_names=feat_names,
                )

                sv = np.asarray(shap_values.values)
                abs_sv = np.abs(sv)
                sample_mean_abs = abs_sv.mean(axis=1)

                feat_rank = (
                    pd.DataFrame(
                        {"feature": feat_names, "mean_abs_shap": abs_sv.mean(axis=0).tolist()}
                    )
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                feat_csv = run_dir / "shap_feature_importance_cnn_holdout.csv"
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

                if ("roi" in pd_feat.columns) and ("label" in pd_feat.columns):
                    meta = pd_feat.iloc[idx_eval][["roi", "label"]].copy()
                    meta["sample_mean_abs_shap"] = sample_mean_abs[: len(meta)]
                    roi_rank = (
                        meta.groupby(["roi", "label"], dropna=False)["sample_mean_abs_shap"]
                        .mean()
                        .reset_index()
                        .sort_values("sample_mean_abs_shap", ascending=False)
                        .reset_index(drop=True)
                    )
                    roi_csv = run_dir / "shap_roi_importance_cnn_holdout.csv"
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

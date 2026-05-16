"""Pipeline LSTM partilhado por exp1_lstm.py e exp2_lstm.py (espelho exp*_xgboost.py)."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import exp1_utils as u
import matplotlib.pyplot as plt
import numpy as np
import optuna
import shap
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# XLA auto-jit em RNNs na GPU falha com CUDNN_STATUS_NOT_INITIALIZED neste host (driver 570 + TF 2.21).
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_DISABLE_JIT_COMPILE", "1")


def _apply_lstm_device_env() -> None:
    """Deve correr antes de `import tensorflow` (ver exp1_lstm / exp2_lstm)."""
    device = os.environ.get("LSTM_DEVICE", "gpu").strip().lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if device != "gpu":
        raise ValueError(f"LSTM_DEVICE inválido: {device!r} (use 'cpu' ou 'gpu').")
    gpu_ix = os.environ.get("LSTM_GPU_INDEX", "").strip()
    if gpu_ix:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ix


_apply_lstm_device_env()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _configure_tensorflow() -> str:
    """Retorna rótulo do dispositivo efetivo ('cpu' ou descrição da GPU)."""
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    device = os.environ.get("LSTM_DEVICE", "gpu").strip().lower()
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(
            "Aviso: LSTM_DEVICE=gpu mas nenhuma GPU visível; a usar CPU. "
            "Verifique LSTM_GPU_INDEX e processos na GPU."
        )
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu (fallback)"
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    names = [getattr(g, "name", str(g)) for g in gpus]
    return f"gpu ({', '.join(names)})"


_LSTM_RUNTIME_DEVICE = _configure_tensorflow()

RANDOM_STATE = 42
CORR_THR = 0.9
VAR_THR = 0.0
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)
TOP_K_ROI = 10
TOP_K_ATTR = 20
OPTUNA_LSTM_TRIALS = 20
INNER_NCV_SPLITS = 5
EPOCHS_MAX = 100
EARLY_STOPPING_PATIENCE = 10
SHAP_BACKGROUND = 40
SHAP_SAMPLES = 60


@dataclass(frozen=True)
class LstmExperimentConfig:
    exp_name: str  # "exp1" | "exp2"
    csv_path: Path
    exp_md_path: Path
    pair_order: list[str]
    temporal_mode: str
    dt_epsilon: float
    model_slug: str = "lstm"
    downsample_group_sex: bool = True
    title_prefix: str = "LSTM"


def _class_weight_dict(y_tr: np.ndarray) -> dict[int, float] | None:
    y_tr = np.asarray(y_tr, dtype=int)
    n0 = int((y_tr == 0).sum())
    n1 = int((y_tr == 1).sum())
    if n0 == 0 or n1 == 0:
        return None
    total = float(len(y_tr))
    return {0: total / (2.0 * n0), 1: total / (2.0 * n1)}


def _make_lstm(seq_len: int, n_feat: int, hp: dict[str, Any]) -> keras.Model:
    units = int(hp["units"])
    dropout = float(hp["dropout"])
    lr = float(hp["learning_rate"])
    inp = keras.Input(shape=(seq_len, n_feat), name="x")
    # use_cudnn=False: implementação genérica (evita CudnnRNNV3 quando cuDNN não inicializa).
    x = layers.LSTM(units, dropout=dropout, use_cudnn=False)(inp)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
        jit_compile=False,
    )
    return model


def _predict_proba_pos(model: keras.Model, X_seq: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X_seq, verbose=0), dtype=np.float64).reshape(-1)


def _fit_lstm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    hp: dict[str, Any],
    *,
    epochs: int,
    patience: int,
    verbose: int = 0,
) -> tuple[keras.Model, keras.callbacks.History]:
    seq_len = int(X_tr.shape[1])
    n_feat = int(X_tr.shape[2])
    model = _make_lstm(seq_len, n_feat, hp)
    cw = _class_weight_dict(y_tr)
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=int(hp["batch_size"]),
        class_weight=cw,
        callbacks=cb,
        verbose=verbose,
    )
    return model, hist


def _fit_lstm_optuna(
    X_3d: np.ndarray,
    y: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    X_refit_tr_seq: np.ndarray,
    X_refit_val_seq: np.ndarray,
    refit_tr_idx: np.ndarray,
    refit_val_idx: np.ndarray,
    *,
    fold_id: int,
) -> tuple[keras.Model, dict[str, Any], float, keras.callbacks.History]:
    seed = RANDOM_STATE + 211 * int(fold_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hp = {
            "units": trial.suggest_int("units", 16, 96, step=16),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        aucs: list[float] = []
        for in_tr, in_va in inner_splits:
            X_tr_seq, X_va_seq = u.seq_scaled_train_val(
                X_3d, in_tr, in_va, corr_thr=CORR_THR, var_thr=VAR_THR
            )
            model, _ = _fit_lstm(
                X_tr_seq.astype(np.float32),
                y[in_tr],
                X_va_seq.astype(np.float32),
                y[in_va],
                hp,
                epochs=EPOCHS_MAX,
                patience=EARLY_STOPPING_PATIENCE,
                verbose=0,
            )
            proba = _predict_proba_pos(model, X_va_seq.astype(np.float32))
            y_va = y[in_va]
            if len(np.unique(y_va)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_va, proba)))
            keras.backend.clear_session()
        if not aucs:
            return float("-inf")
        return float(np.mean(aucs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_LSTM_TRIALS, show_progress_bar=False)

    best_hp = dict(study.best_trial.params)
    model, hist = _fit_lstm(
        X_refit_tr_seq.astype(np.float32),
        y[refit_tr_idx],
        X_refit_val_seq.astype(np.float32),
        y[refit_val_idx],
        best_hp,
        epochs=EPOCHS_MAX,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=0,
    )
    return model, best_hp, float(study.best_value), hist


def _shap_abs_mean_lstm(
    model: keras.Model,
    X_te_flat: np.ndarray,
    X_bg_flat: np.ndarray,
    *,
    seq_len: int,
    n_feat: int,
) -> np.ndarray:
    n_flat = seq_len * n_feat
    if X_te_flat.shape[1] != n_flat:
        raise ValueError(
            f"SHAP: esperado {n_flat} colunas achatadas, recebido {X_te_flat.shape[1]}."
        )

    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_seq = np.asarray(x, dtype=np.float32).reshape(-1, seq_len, n_feat)
        return _predict_proba_pos(model, x_seq)

    n_bg = max(1, min(SHAP_BACKGROUND, len(X_bg_flat)))
    n_samp = max(1, min(SHAP_SAMPLES, len(X_te_flat)))
    bg = X_bg_flat[:n_bg]
    X_eval = X_te_flat[:n_samp]
    n_features = int(X_eval.shape[1])
    min_evals = 2 * n_features + 1
    explainer = shap.Explainer(predict_fn, shap.maskers.Independent(bg))
    shap_values = explainer(X_eval, max_evals=min_evals)
    sv = np.asarray(shap_values.values, dtype=np.float64)
    return np.abs(sv).mean(axis=0)


def run_lstm_experiment(cfg: LstmExperimentConfig) -> None:
    t0 = time.perf_counter()
    colab_dir = Path(__file__).resolve().parent
    run_dir_fn: Callable[..., Path] = (
        u.exp1_run_dir if cfg.exp_name == "exp1" else u.exp2_run_dir
    )
    run_dir = run_dir_fn(
        colab_dir,
        downsample_group_sex=cfg.downsample_group_sex,
        model_slug=cfg.model_slug,
    )
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"

    X_3d, y, groups, sex, feat_names, slot_labels = u.load_tensor(
        cfg.csv_path,
        cfg.exp_md_path,
        cfg.pair_order,
        ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"],
        require_sex=cfg.downsample_group_sex,
        temporal_mode=cfg.temporal_mode,
        dt_epsilon=cfg.dt_epsilon,
    )
    n_raw = X_3d.shape[2]
    n_samples = len(y)

    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    dummy = np.zeros(len(y), dtype=np.int8)

    acc_folds: list[float] = []
    auc_folds: list[float] = []
    f1_folds: list[float] = []

    y_oof = np.full(n_samples, -1, dtype=np.int32)
    pred_oof = np.full(n_samples, -1, dtype=np.int32)
    proba_oof = np.full(n_samples, np.nan, dtype=np.float64)
    y_splits: list[np.ndarray] = []
    score_splits: list[np.ndarray] = []
    fold_test_fold_ids: list[np.ndarray] = []
    fold_test_y: list[np.ndarray] = []
    fold_test_score: list[np.ndarray] = []
    metrics_rows: list[dict[str, float | int]] = []
    outer_fold_assign = np.full(n_samples, -1, dtype=np.int32)

    shap_roi: dict[str, float] = defaultdict(float)
    shap_attr: dict[str, float] = defaultdict(float)
    best_shap_n = -1
    best_shap: tuple[keras.Model, np.ndarray, np.ndarray, int, int] | None = None

    prefix = cfg.title_prefix
    print(
        f"LSTM — dispositivo: {_LSTM_RUNTIME_DEVICE} "
        f"(LSTM_DEVICE={os.environ.get('LSTM_DEVICE', 'gpu')!r}, "
        f"LSTM_GPU_INDEX={os.environ.get('LSTM_GPU_INDEX', '')!r} ou todas, "
        f"use_cudnn=False, XLA auto-jit desligado; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(todas)')!r})"
    )
    if cfg.downsample_group_sex:
        print("Downsample ativo: treino externo por paciente (estratos y×SEX).")

    for fold_id, (train_idx, test_idx) in enumerate(sgk.split(dummy, y, groups)):
        train_idx = np.asarray(train_idx, dtype=int)
        n_tr0 = len(train_idx)
        if cfg.downsample_group_sex:
            train_idx = u.downsample_train_indices(
                train_idx, groups, y, sex, seed=RANDOM_STATE + 31 * fold_id
            )
        if fold_id == 0:
            print(
                f"Fold 1 — treino externo: {n_tr0} -> {len(train_idx)} amostras"
                + (" (após downsample)." if cfg.downsample_group_sex else ".")
            )

        inner_splits = u.inner_cv_splits(
            train_idx,
            y,
            groups,
            fold_id=fold_id,
            n_splits_requested=INNER_NCV_SPLITS,
            random_state=RANDOM_STATE,
        )
        tr_fit_idx, val_idx = u.inner_train_val(
            train_idx, y, groups, fold_id=fold_id, random_state=RANDOM_STATE
        )

        X_trf = X_3d[tr_fit_idx].reshape(-1, X_3d.shape[2])
        keep_corr = u.corr_keep_indices(X_trf, CORR_THR)
        n_after_corr = len(keep_corr)
        X_trf_c = X_trf[:, keep_corr]
        vt = VarianceThreshold(threshold=VAR_THR)
        vt.fit(X_trf_c)
        keep_var = np.where(vt.get_support())[0]
        keep_final = keep_corr[keep_var]
        n_after_var = len(keep_final)

        def apply_cols(A: np.ndarray) -> np.ndarray:
            return A[:, :, keep_final]

        X_train_fit = apply_cols(X_3d[tr_fit_idx])
        X_val = apply_cols(X_3d[val_idx])
        X_test = apply_cols(X_3d[test_idx])
        n_tr = len(tr_fit_idx) * X_3d.shape[1]
        scaler = StandardScaler()
        scaler.fit(X_train_fit.reshape(n_tr, -1))

        def scale(A: np.ndarray) -> np.ndarray:
            s = A.shape
            return scaler.transform(A.reshape(-1, s[-1])).reshape(s)

        X_train_fit = scale(X_train_fit)
        X_val = scale(X_val)
        X_test = scale(X_test)

        X_train_seq = u.panels_to_seq(X_train_fit)
        X_val_seq = u.panels_to_seq(X_val)
        X_test_seq = u.panels_to_seq(X_test)
        seq_len = int(X_train_seq.shape[1])
        n_feat_seq = int(X_train_seq.shape[2])

        X_train_flat = X_train_fit.reshape(len(tr_fit_idx), -1)
        X_val_flat = X_val.reshape(len(val_idx), -1)
        X_test_flat = X_test.reshape(len(test_idx), -1)

        if fold_id == 0:
            fig, ax = plt.subplots()
            ax.bar(
                ["Raw", "Após correlação", "Após variância"],
                [n_raw, n_after_corr, n_after_var],
            )
            ax.set_ylabel("Nº atributos")
            fig.tight_layout()
            u.save_pdf(fig, fig_dir / "feature_counts.pdf")
            u.save_feature_counts_fold0_csv(
                tab_dir / "feature_counts_fold0.csv",
                n_raw=n_raw,
                n_after_corr=n_after_corr,
                n_after_variance=n_after_var,
            )

        model, best_params, best_val_auc, hist = _fit_lstm_optuna(
            X_3d,
            y,
            inner_splits,
            X_train_seq,
            X_val_seq,
            tr_fit_idx,
            val_idx,
            fold_id=fold_id,
        )
        print(
            f"Fold {fold_id + 1}/5 — Optuna (AUC val interna média em {len(inner_splits)} folds NCV="
            f"{best_val_auc:.4f}): {best_params}"
        )

        if fold_id == 0:
            h = hist.history
            epochs = np.arange(1, len(h.get("loss", [])) + 1, dtype=np.int32)
            cols: dict[str, np.ndarray] = {}
            if "loss" in h:
                cols["loss"] = np.asarray(h["loss"], dtype=np.float64)
            if "val_loss" in h:
                cols["val_loss"] = np.asarray(h["val_loss"], dtype=np.float64)
            if "val_auc" in h:
                cols["val_auc"] = np.asarray(h["val_auc"], dtype=np.float64)
            if "accuracy" in h:
                cols["accuracy"] = np.asarray(h["accuracy"], dtype=np.float64)
            if "val_accuracy" in h:
                cols["val_accuracy"] = np.asarray(h["val_accuracy"], dtype=np.float64)
            u.save_training_curve_csv(tab_dir / "training_curves_fold0.csv", epochs, cols)
            u.plot_training_curves_keras_pdf(
                tab_dir / "training_curves_fold0.csv",
                fig_dir / "training_curves.pdf",
                title=(
                    f"{prefix} — fold 1/5 (holdout tr_fit|val após NCV interno na seleção de "
                    "hiperparâmetros)"
                ),
            )

        proba_te = _predict_proba_pos(model, X_test_seq.astype(np.float32))
        pred_te = (proba_te >= 0.5).astype(np.int32)
        outer_fold_assign[test_idx] = int(fold_id)
        y_oof[test_idx] = y[test_idx]
        pred_oof[test_idx] = pred_te
        proba_oof[test_idx] = proba_te
        y_splits.append(np.asarray(y[test_idx], dtype=np.int32))
        score_splits.append(np.asarray(proba_te, dtype=np.float64))
        fold_test_fold_ids.append(np.full(len(test_idx), int(fold_id), dtype=np.int32))
        fold_test_y.append(np.asarray(y[test_idx], dtype=np.int32))
        fold_test_score.append(np.asarray(proba_te, dtype=np.float64))

        sm = _shap_abs_mean_lstm(
            model,
            X_test_flat,
            X_train_flat,
            seq_len=seq_len,
            n_feat=n_feat_seq,
        )
        u.accumulate_flat_importance(
            shap_roi, shap_attr, sm, keep_final, feat_names, slot_labels
        )

        if len(test_idx) > best_shap_n:
            best_shap_n = len(test_idx)
            best_shap = (model, X_test_flat.copy(), keep_final.copy(), seq_len, n_feat_seq)

        acc, auc, f1 = u.binary_metrics_from_proba(y[test_idx], pred_te, proba_te)
        metrics_rows.append({"fold": int(fold_id) + 1, "acc": acc, "auc": auc, "f1": f1})
        acc_folds.append(acc)
        auc_folds.append(auc)
        f1_folds.append(f1)
        print(f"Fold {fold_id + 1}/5 — teste: acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        keras.backend.clear_session()

    acc_a = np.asarray(acc_folds, dtype=np.float64)
    auc_a = np.asarray(auc_folds, dtype=np.float64)
    f1_a = np.asarray(f1_folds, dtype=np.float64)
    suffix = " | treino com downsample GROUP×SEX." if cfg.downsample_group_sex else "."
    print(
        "Resumo 5-fold SGK (média ± dp) — teste: "
        f"acc={acc_a.mean():.4f} ± {acc_a.std(ddof=0):.4f}, "
        f"AUC={np.nanmean(auc_a):.4f} ± {np.nanstd(auc_a):.4f}, "
        f"F1={f1_a.mean():.4f} ± {f1_a.std(ddof=0):.4f}"
        f"{suffix}"
    )

    u.save_metrics_per_fold_csv(tab_dir / "metrics_per_fold.csv", metrics_rows)
    u.save_fold_test_scores_csv(
        tab_dir / "fold_test_scores.csv",
        np.concatenate(fold_test_fold_ids),
        np.concatenate(fold_test_y),
        np.concatenate(fold_test_score),
    )
    mask = y_oof >= 0
    u.save_oof_predictions_csv(
        tab_dir / "oof_predictions.csv",
        np.flatnonzero(mask),
        y_oof[mask],
        pred_oof[mask],
        proba_oof[mask],
        outer_fold_assign[mask],
        groups[mask].astype(str),
    )

    n_f = float(len(acc_folds))
    shap_roi_m = {k: v / n_f for k, v in shap_roi.items()}
    shap_attr_m = {k: v / n_f for k, v in shap_attr.items()}
    u.save_importance_long_csv(
        tab_dir / "importance_shap_roi_mean.csv",
        list(shap_roi_m.keys()),
        [float(shap_roi_m[k]) for k in shap_roi_m],
    )
    u.save_importance_long_csv(
        tab_dir / "importance_shap_attr_mean.csv",
        list(shap_attr_m.keys()),
        [float(shap_attr_m[k]) for k in shap_attr_m],
    )

    u.plot_confusion_oof_pdf(
        y_oof[mask],
        pred_oof[mask],
        fig_dir / "confusion_oof.pdf",
        title=f"{prefix} — matriz de confusão agregada (predições OOF, 5-fold SGK)",
    )
    u.plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        fig_dir / "roc_cv.pdf",
        fig_dir / "pr_cv.pdf",
        title_prefix=prefix,
        fpr_grid=FPR_GRID,
        rec_grid=REC_GRID,
    )
    u.plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        fig_dir / "metrics_box_cv.pdf",
        title=f"{prefix} — distribuição das métricas no teste (5 folds)",
    )
    u.plot_top_bars_pdf(
        shap_roi_m,
        fig_dir / "shap_top_roi.pdf",
        title=f"{prefix} — |SHAP| médio no teste, agregado por ROI (média dos 5 folds)",
        top_k=TOP_K_ROI,
        xlabel="Média da média |SHAP| por coluna (por fold)",
    )
    u.plot_top_bars_pdf(
        shap_attr_m,
        fig_dir / "shap_top_attr.pdf",
        title=f"{prefix} — |SHAP| médio no teste, agregado por nome de atributo",
        top_k=TOP_K_ATTR,
        xlabel="Média da média |SHAP| por coluna (por fold)",
    )

    if best_shap is not None:
        b_model, X_b, kf, sl, nf = best_shap
        sm = _shap_abs_mean_lstm(b_model, X_b, X_b, seq_len=sl, n_feat=nf)
        nc = len(kf)
        flat_names: list[str] = []
        for fi in range(X_b.shape[1]):
            slot = fi // nc
            j = fi % nc
            roi = u.roi_from_slot_label(slot_labels[slot])
            fn = feat_names[int(kf[j])]
            flat_names.append(f"{roi}|{fn}"[:72])
        top_m = min(25, len(sm))
        top_idx = np.argsort(-sm)[:top_m]
        fig_sh, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            np.array(flat_names, dtype=object)[top_idx][::-1],
            sm[top_idx][::-1],
        )
        ax.set_xlabel("Média |SHAP|")
        ax.set_title(
            f"{prefix} — top features (fold com maior conjunto de teste; Kernel SHAP)"
        )
        fig_sh.tight_layout()
        u.save_pdf(fig_sh, fig_dir / "shap_summary.pdf")

    elapsed = time.perf_counter() - t0
    print(f"Tempo total: {elapsed:.1f} s — artefactos em {run_dir}")
    u.write_run_meta_json(
        run_dir,
        model_slug=cfg.model_slug,
        downsample_group_sex=cfg.downsample_group_sex,
        duration_seconds=elapsed,
        extra={
            "inner_ncv_splits": INNER_NCV_SPLITS,
            "optuna_trials": OPTUNA_LSTM_TRIALS,
            "temporal_mode": cfg.temporal_mode,
            "dt_epsilon": cfg.dt_epsilon,
            "lstm_seq_len": u.PANEL_SEQ_STEPS,
            "csv_schema": (
                "metrics_per_fold, oof_predictions, fold_test_scores, "
                "importance_shap_*, training_curves_fold0"
            ),
        },
    )

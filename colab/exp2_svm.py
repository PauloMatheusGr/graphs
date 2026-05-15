"""exp2: SVM linear (LinearSVC) tabular — mesmo pipeline que exp2_xgboost (Optuna + NCV interno).

Nested CV: Optuna maximiza a média da AUC em StratifiedGroupKFold internos; o modelo final
ajusta-se em tr_fit e avalia-se em val (holdout) como no XGBoost. AUC usa decision_function.
Downsample opcional no treino externo (GROUP×SEX).
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import exp1_utils as u
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"
EXP2_PATH = ROOT / "exp2.md"
MODEL_SLUG = "svm"
PAIR_ORDER = ["1", "2", "3"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
TEMPORAL_MODE = "baseline_rate"
DT_EPSILON = 0.5
CORR_THR = 0.9
VAR_THR = 0.0
RANDOM_STATE = 42
DOWNSAMPLE_GROUP_SEX = True
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)
TOP_K_ROI = 10
TOP_K_ATTR = 20
OPTUNA_SVM_TRIALS = 30
INNER_NCV_SPLITS = 5


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_linear_svc_optuna(
    X_3d: np.ndarray,
    y: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    X_refit_tr_flat: np.ndarray,
    X_refit_val_flat: np.ndarray,
    refit_tr_idx: np.ndarray,
    refit_val_idx: np.ndarray,
    *,
    fold_id: int,
) -> tuple[LinearSVC, dict[str, Any], float]:
    """Optuna em C (log); objetivo = média AUC nos folds internos (decision_function)."""
    seed = RANDOM_STATE + 131 * int(fold_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        aucs: list[float] = []
        for in_tr, in_va in inner_splits:
            X_tr_flat, X_va_flat = u.flat_scaled_tabular_train_val(
                X_3d, in_tr, in_va, corr_thr=CORR_THR, var_thr=VAR_THR
            )
            clf = LinearSVC(
                C=float(C),
                dual="auto",
                max_iter=50_000,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_tr_flat, y[in_tr])
            scores = clf.decision_function(X_va_flat)
            y_va = y[in_va]
            if len(np.unique(y_va)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_va, scores)))
        if not aucs:
            return float("-inf")
        return float(np.mean(aucs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_SVM_TRIALS, show_progress_bar=False)

    best_C = float(study.best_trial.params["C"])
    model = LinearSVC(
        C=best_C,
        dual="auto",
        max_iter=50_000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_refit_tr_flat, y[refit_tr_idx])
    return model, dict(study.best_trial.params), float(study.best_value)


def main() -> None:
    t0 = time.perf_counter()
    run_dir = u.exp2_run_dir(
        COLAB_DIR,
        downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
        model_slug=MODEL_SLUG,
    )
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"

    X_3d, y, groups, sex, feat_names, slot_labels = u.load_tensor(
        CSV_PATH,
        EXP2_PATH,
        PAIR_ORDER,
        GROUP_KEY,
        require_sex=DOWNSAMPLE_GROUP_SEX,
        temporal_mode=TEMPORAL_MODE,
        dt_epsilon=DT_EPSILON,
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

    imp_roi: dict[str, float] = defaultdict(float)
    imp_attr: dict[str, float] = defaultdict(float)

    if DOWNSAMPLE_GROUP_SEX:
        print("Downsample ativo: treino externo por paciente (estratos y×SEX).")

    for fold_id, (train_idx, test_idx) in enumerate(sgk.split(dummy, y, groups)):
        train_idx = np.asarray(train_idx, dtype=int)
        n_tr0 = len(train_idx)
        if DOWNSAMPLE_GROUP_SEX:
            train_idx = u.downsample_train_indices(
                train_idx, groups, y, sex, seed=RANDOM_STATE + 31 * fold_id
            )
        if fold_id == 0:
            print(
                f"Fold 1 — treino externo: {n_tr0} -> {len(train_idx)} amostras"
                + (" (após downsample)." if DOWNSAMPLE_GROUP_SEX else ".")
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
        n_tr = len(tr_fit_idx) * 60
        scaler = StandardScaler()
        scaler.fit(X_train_fit.reshape(n_tr, -1))

        def scale(A: np.ndarray) -> np.ndarray:
            s = A.shape
            return scaler.transform(A.reshape(-1, s[-1])).reshape(s)

        X_train_fit = scale(X_train_fit)
        X_val = scale(X_val)
        X_test = scale(X_test)

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

        model, best_params, best_val_auc = _fit_linear_svc_optuna(
            X_3d,
            y,
            inner_splits,
            X_train_flat,
            X_val_flat,
            tr_fit_idx,
            val_idx,
            fold_id=fold_id,
        )
        print(
            f"Fold {fold_id + 1}/5 — Optuna SVM linear (AUC val interna média em {len(inner_splits)} folds NCV="
            f"{best_val_auc:.4f}): {best_params}"
        )

        df_te = model.decision_function(X_test_flat)
        proba_te = _sigmoid(df_te)
        pred_te = model.predict(X_test_flat).astype(np.int32, copy=False)
        outer_fold_assign[test_idx] = int(fold_id)
        y_oof[test_idx] = y[test_idx]
        pred_oof[test_idx] = pred_te
        proba_oof[test_idx] = proba_te
        y_splits.append(np.asarray(y[test_idx], dtype=np.int32))
        score_splits.append(np.asarray(proba_te, dtype=np.float64))
        fold_test_fold_ids.append(
            np.full(len(test_idx), int(fold_id), dtype=np.int32)
        )
        fold_test_y.append(np.asarray(y[test_idx], dtype=np.int32))
        fold_test_score.append(np.asarray(proba_te, dtype=np.float64))

        coef_abs = np.abs(np.asarray(model.coef_, dtype=np.float64).ravel())
        u.accumulate_flat_importance(
            imp_roi, imp_attr, coef_abs, keep_final, feat_names, slot_labels
        )

        acc, auc, f1 = u.binary_metrics_from_proba(
            y[test_idx], pred_te, proba_te
        )
        metrics_rows.append(
            {"fold": int(fold_id) + 1, "acc": acc, "auc": auc, "f1": f1}
        )
        acc_folds.append(acc)
        auc_folds.append(auc)
        f1_folds.append(f1)
        print(
            f"Fold {fold_id + 1}/5 — teste: acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}"
        )

    acc_a = np.asarray(acc_folds, dtype=np.float64)
    auc_a = np.asarray(auc_folds, dtype=np.float64)
    f1_a = np.asarray(f1_folds, dtype=np.float64)
    suffix = " | treino com downsample GROUP×SEX." if DOWNSAMPLE_GROUP_SEX else "."
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
    roi_m = {k: v / n_f for k, v in imp_roi.items()}
    attr_m = {k: v / n_f for k, v in imp_attr.items()}
    u.save_importance_long_csv(
        tab_dir / "importance_coef_roi_mean.csv",
        list(roi_m.keys()),
        [float(roi_m[k]) for k in roi_m],
    )
    u.save_importance_long_csv(
        tab_dir / "importance_coef_attr_mean.csv",
        list(attr_m.keys()),
        [float(attr_m[k]) for k in attr_m],
    )

    u.plot_confusion_oof_pdf(
        y_oof[mask],
        pred_oof[mask],
        fig_dir / "confusion_oof.pdf",
        title="SVM linear — matriz de confusão agregada (predições OOF, 5-fold SGK)",
    )
    u.plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        fig_dir / "roc_cv.pdf",
        fig_dir / "pr_cv.pdf",
        title_prefix="SVM linear",
        fpr_grid=FPR_GRID,
        rec_grid=REC_GRID,
    )
    u.plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        fig_dir / "metrics_box_cv.pdf",
        title="SVM linear — distribuição das métricas no teste (5 folds)",
    )
    u.plot_top_bars_pdf(
        roi_m,
        fig_dir / "coef_top_roi.pdf",
        title="SVM linear — |coef| médio no teste, agregado por ROI (média dos 5 folds)",
        top_k=TOP_K_ROI,
        xlabel="Média do |coef| por coluna (por fold)",
    )
    u.plot_top_bars_pdf(
        attr_m,
        fig_dir / "coef_top_attr.pdf",
        title="SVM linear — |coef| médio no teste, agregado por atributo",
        top_k=TOP_K_ATTR,
        xlabel="Média do |coef| por coluna (por fold)",
    )

    elapsed = time.perf_counter() - t0
    print(f"Tempo total: {elapsed:.1f} s — artefactos em {run_dir}")
    u.write_run_meta_json(
        run_dir,
        model_slug=MODEL_SLUG,
        downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
        duration_seconds=elapsed,
        extra={
            "inner_ncv_splits": INNER_NCV_SPLITS,
            "optuna_trials": OPTUNA_SVM_TRIALS,
            "temporal_mode": TEMPORAL_MODE,
            "dt_epsilon": DT_EPSILON,
        },
    )


if __name__ == "__main__":
    main()

"""exp2: ROCKET + regressão logística L1 (saga) com Optuna e nested CV interno.

Nested CV: Optuna maximiza a média da AUC em K folds StratifiedGroupKFold dentro
do treino externo; ROCKET é reajustado uma vez por fold interno (sem vazamento).
Downsample opcional no treino externo por paciente (GROUP×SEX).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import exp1_utils as u
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"
EXP2_PATH = ROOT / "exp2.md"
MODEL_SLUG = "rocket"
PAIR_ORDER = ["1", "2", "3"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
TEMPORAL_MODE = "baseline_rate"
DT_EPSILON = 0.5
CORR_THR = 0.9
VAR_THR = 0.0
RANDOM_STATE = 42
NUM_KERNELS = 2_000
# Optuna: média da AUC em INNER_NCV_SPLITS folds internos; C em escala log.
OPTUNA_ROCKET_TRIALS = 30
INNER_NCV_SPLITS = 5
# Grade só para figura diagnóstica (fold 1): acurácia vs log10(C).
C_DIAG_GRID = np.logspace(-4, 4, 17)
DOWNSAMPLE_GROUP_SEX = True
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)


def _fit_l1_logreg_optuna(
    y: np.ndarray,
    inner_z_pairs: list[tuple[np.ndarray, np.ndarray]],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    Z_refit_tr: np.ndarray,
    y_refit_tr: np.ndarray,
    *,
    fold_id: int,
) -> tuple[LogisticRegression, dict[str, Any], float]:
    """Objetivo Optuna = média da AUC nos folds internos (features ROCKET já transformadas)."""
    seed = RANDOM_STATE + 113 * int(fold_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        aucs: list[float] = []
        for (Ztr, Zva), (in_tr, in_va) in zip(inner_z_pairs, inner_splits):
            clf = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=float(C),
                max_iter=10_000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            clf.fit(Ztr, y[in_tr])
            y_va = y[in_va]
            if len(np.unique(y_va)) < 2:
                continue
            proba = clf.predict_proba(Zva)[:, 1]
            aucs.append(float(roc_auc_score(y_va, proba)))
        if not aucs:
            return float("-inf")
        return float(np.mean(aucs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_ROCKET_TRIALS, show_progress_bar=False)

    best_C = float(study.best_trial.params["C"])
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=best_C,
        max_iter=10_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(Z_refit_tr, y_refit_tr)
    return clf, dict(study.best_trial.params), float(study.best_value)


def main() -> None:
    t0 = time.perf_counter()
    run_dir = u.exp2_run_dir(
        COLAB_DIR,
        downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
        model_slug=MODEL_SLUG,
    )
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"

    X_3d, y, groups, sex, _feat_names, _slot_labels = u.load_tensor(
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
    score_oof = np.full(n_samples, np.nan, dtype=np.float64)
    y_splits: list[np.ndarray] = []
    score_splits: list[np.ndarray] = []
    fold_test_fold_ids: list[np.ndarray] = []
    fold_test_y: list[np.ndarray] = []
    fold_test_score: list[np.ndarray] = []
    metrics_rows: list[dict[str, float | int]] = []
    outer_fold_assign = np.full(n_samples, -1, dtype=np.int32)

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
        inner_z_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (in_tr, in_va) in enumerate(inner_splits):
            Xr_tr, Xr_va = u.prepare_scaled_rocket_inputs(
                X_3d, in_tr, in_va, corr_thr=CORR_THR, var_thr=VAR_THR
            )
            rk = Rocket(
                num_kernels=NUM_KERNELS,
                random_state=RANDOM_STATE + 17 * fold_id + i,
            )
            rk.fit(Xr_tr)
            inner_z_pairs.append((rk.transform(Xr_tr), rk.transform(Xr_va)))

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

        Xr_tr = np.transpose(X_train_fit, (0, 2, 1))
        Xr_val = np.transpose(X_val, (0, 2, 1))
        Xr_te = np.transpose(X_test, (0, 2, 1))

        rocket = Rocket(num_kernels=NUM_KERNELS, random_state=RANDOM_STATE)
        rocket.fit(Xr_tr)
        Z_tr = rocket.transform(Xr_tr)
        Z_val = rocket.transform(Xr_val)
        Z_te = rocket.transform(Xr_te)

        clf, best_params, best_val_auc = _fit_l1_logreg_optuna(
            y,
            inner_z_pairs,
            inner_splits,
            Z_tr,
            y[tr_fit_idx],
            fold_id=fold_id,
        )
        print(
            f"Fold {fold_id + 1}/5 — Optuna L1 (AUC val interna média em {len(inner_splits)} folds NCV="
            f"{best_val_auc:.4f}): {best_params}"
        )

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

            acc_c: list[float] = []
            for C in C_DIAG_GRID:
                m = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=float(C),
                    max_iter=10_000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                m.fit(Z_tr, y[tr_fit_idx])
                acc_c.append(accuracy_score(y[val_idx], m.predict(Z_val)))

            fig2, ax2 = plt.subplots()
            ax2.plot(np.log10(C_DIAG_GRID), acc_c, marker="o")
            ax2.set_xlabel("log10(C)")
            ax2.set_ylabel("acurácia (validação)")
            fig2.suptitle(
                "Fold 1/5 — regressão logística L1 (saga) em features ROCKET (sem épocas de treino)."
            )
            fig2.tight_layout()
            u.save_pdf(fig2, fig_dir / "l1_C_validation_curve.pdf")
            u.save_training_curve_csv(
                tab_dir / "l1_C_diagnostic_fold0.csv",
                np.log10(C_DIAG_GRID).astype(np.float64),
                {"accuracy_val": np.asarray(acc_c, dtype=np.float64)},
            )

        pred_te = clf.predict(Z_te)
        sc_te = clf.predict_proba(Z_te)[:, 1]
        outer_fold_assign[test_idx] = int(fold_id)
        y_oof[test_idx] = y[test_idx]
        pred_oof[test_idx] = pred_te.astype(np.int32, copy=False)
        score_oof[test_idx] = sc_te
        y_splits.append(np.asarray(y[test_idx], dtype=np.int32))
        score_splits.append(np.asarray(sc_te, dtype=np.float64))
        fold_test_fold_ids.append(
            np.full(len(test_idx), int(fold_id), dtype=np.int32)
        )
        fold_test_y.append(np.asarray(y[test_idx], dtype=np.int32))
        fold_test_score.append(np.asarray(sc_te, dtype=np.float64))

        acc, auc, f1 = u.binary_metrics_from_proba(y[test_idx], pred_te, sc_te)
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
        score_oof[mask],
        outer_fold_assign[mask],
        groups[mask].astype(str),
    )

    u.plot_confusion_oof_pdf(
        y_oof[mask],
        pred_oof[mask],
        fig_dir / "confusion_oof.pdf",
        title="ROCKET+L1 (Optuna) — matriz de confusão agregada (predições OOF, 5-fold SGK)",
        cmap="Greens",
    )
    u.plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        fig_dir / "roc_cv.pdf",
        fig_dir / "pr_cv.pdf",
        title_prefix="ROCKET+L1 (Optuna)",
        fpr_grid=FPR_GRID,
        rec_grid=REC_GRID,
        roc_scope_label="teste",
        pr_scope_label="teste",
    )
    u.plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        fig_dir / "metrics_box_cv.pdf",
        title="ROCKET+L1 (Optuna) — distribuição das métricas no teste (5 folds)",
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
            "optuna_trials": OPTUNA_ROCKET_TRIALS,
            "temporal_mode": TEMPORAL_MODE,
            "dt_epsilon": DT_EPSILON,
        },
    )


if __name__ == "__main__":
    main()

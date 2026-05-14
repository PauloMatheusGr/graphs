"""exp1: XGBoost tabular a partir de exp1.md (Optuna + nested CV interno + early stopping).

Nested CV: o Optuna maximiza a média da AUC em vários StratifiedGroupKFold internos
dentro do treino externo; o refit final usa tr_fit/val (holdout) como antes.
Downsample opcional no treino externo por paciente (GROUP×SEX).
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import exp1_utils as u
import matplotlib.pyplot as plt
import optuna
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv"
EXP1_PATH = ROOT / "exp1.md"
MODEL_SLUG = "xgboost"
PAIR_ORDER = ["12", "13", "23"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
CORR_THR = 0.9
# VarianceThreshold(0.0) remove apenas colunas constantes no treino (exp1: baixa variância).
VAR_THR = 0.0
RANDOM_STATE = 42
# True: antes do split interno, subsampling de pacientes no treino do fold
# para min(# pacientes) por estrato GROUP×SEX (F/M × sMCI/pMCI).
DOWNSAMPLE_GROUP_SEX = False
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)
TOP_K_ROI = 15
TOP_K_ATTR = 20
# Optuna: objetivo = média da AUC nos folds do NCV interno (StratifiedGroupKFold).
OPTUNA_XGB_TRIALS = 25
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_MAX = 500
# Folds internos dentro do treino externo (média da AUC no Optuna). 3 equilibra rigor e custo.
INNER_NCV_SPLITS = 5


def _scale_pos_weight_ratio(y_tr: np.ndarray) -> float:
    """Razão neg/pos para scale_pos_weight (classe 0 = negativa, 1 = positiva)."""
    y_tr = np.asarray(y_tr).astype(int, copy=False)
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / max(n_pos, 1))


def _xgb_train_params_from_optuna_dict(bp: dict[str, Any], *, base_spw: float) -> dict[str, Any]:
    d = dict(bp)
    spw_mul = float(d.pop("spw_mul"))
    lr = float(d.pop("learning_rate"))
    reg_l = float(d.pop("reg_lambda"))
    return {
        "max_depth": int(d["max_depth"]),
        "eta": lr,
        "subsample": float(d["subsample"]),
        "colsample_bytree": float(d["colsample_bytree"]),
        "lambda": reg_l,
        "min_child_weight": float(d["min_child_weight"]),
        "gamma": float(d["gamma"]),
        "scale_pos_weight": base_spw * spw_mul,
    }


def _xgb_train_booster_early(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    train_params: dict[str, Any],
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    evals_result: dict[str, Any] | None,
) -> xgb.Booster:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    full = {"objective": "binary:logistic", "eval_metric": "logloss", **train_params}
    return xgb.train(
        full,
        dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        evals_result=evals_result,
    )


def _booster_predict_proba_pos(bst: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dm = xgb.DMatrix(X)
    bi = bst.best_iteration
    if bi is None:
        it = (0, bst.num_boosted_rounds())
    else:
        it = (0, int(bi) + 1)
    return np.asarray(bst.predict(dm, iteration_range=it), dtype=np.float64)


def _xgbc_from_booster(bst: xgb.Booster) -> XGBClassifier:
    clf = XGBClassifier()
    clf.load_model(bst.save_raw("json"))
    return clf


def _fit_xgb_optuna(
    X_3d: np.ndarray,
    y: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    X_refit_tr_flat: np.ndarray,
    X_refit_val_flat: np.ndarray,
    refit_tr_idx: np.ndarray,
    refit_val_idx: np.ndarray,
    *,
    fold_id: int,
) -> tuple[XGBClassifier, dict[str, Any], float, dict[str, Any]]:
    """Optuna com objetivo = média da AUC nos folds internos (NCV); refit com early stopping em refit val."""
    seed = RANDOM_STATE + 97 * int(fold_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        bp_t = {
            "spw_mul": trial.suggest_float("spw_mul", 0.25, 4.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        aucs: list[float] = []
        for in_tr, in_va in inner_splits:
            X_tr_flat, X_va_flat = u.flat_scaled_tabular_train_val(
                X_3d, in_tr, in_va, corr_thr=CORR_THR, var_thr=VAR_THR
            )
            base_spw = _scale_pos_weight_ratio(y[in_tr])
            native = _xgb_train_params_from_optuna_dict(bp_t, base_spw=base_spw)
            bst = _xgb_train_booster_early(
                X_tr_flat,
                y[in_tr],
                X_va_flat,
                y[in_va],
                native,
                num_boost_round=N_ESTIMATORS_MAX,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                evals_result=None,
            )
            proba = _booster_predict_proba_pos(bst, X_va_flat)
            y_va = y[in_va]
            if len(np.unique(y_va)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_va, proba)))
        if not aucs:
            return float("-inf")
        return float(np.mean(aucs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_XGB_TRIALS, show_progress_bar=False)

    best_bp = dict(study.best_trial.params)
    base_spw_final = _scale_pos_weight_ratio(y[refit_tr_idx])
    native_final = _xgb_train_params_from_optuna_dict(best_bp, base_spw=base_spw_final)
    evals_res: dict[str, Any] = {}
    bst_final = _xgb_train_booster_early(
        X_refit_tr_flat,
        y[refit_tr_idx],
        X_refit_val_flat,
        y[refit_val_idx],
        native_final,
        num_boost_round=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_res,
    )
    model = _xgbc_from_booster(bst_final)
    return model, study.best_trial.params, float(study.best_value), evals_res


def _shap_abs_mean_test(model: XGBClassifier, X_te: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(np.abs(sv).mean(axis=0), dtype=np.float64)


def main() -> None:
    t0 = time.perf_counter()
    run_dir = u.exp1_run_dir(
        COLAB_DIR,
        downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
        model_slug=MODEL_SLUG,
    )
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"

    X_3d, y, groups, sex, feat_names, slot_labels = u.load_tensor(
        CSV_PATH,
        EXP1_PATH,
        PAIR_ORDER,
        GROUP_KEY,
        require_sex=DOWNSAMPLE_GROUP_SEX,
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
    best_shap: tuple[XGBClassifier, np.ndarray, np.ndarray] | None = None

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

        model, best_params, best_val_auc, evals_res = _fit_xgb_optuna(
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
            f"Fold {fold_id + 1}/5 — Optuna (AUC val interna média em {len(inner_splits)} folds NCV="
            f"{best_val_auc:.4f}): {best_params}"
        )

        if fold_id == 0:
            booster = model.get_booster()
            dval = xgb.DMatrix(X_val_flat)
            evals = evals_res["val"]["logloss"]
            n_trees = len(evals)
            acc_curve: list[float] = []
            for i in range(1, n_trees + 1):
                proba = booster.predict(dval, iteration_range=(0, i))
                pred = (proba >= 0.5).astype(np.int32)
                acc_curve.append(accuracy_score(y[val_idx], pred))

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            ax1.plot(np.arange(1, n_trees + 1), evals)
            ax1.set_ylabel("logloss (validação)")
            ax2.plot(np.arange(1, n_trees + 1), acc_curve)
            ax2.set_ylabel("acurácia (validação)")
            ax2.set_xlabel("nº árvores (boosting)")
            fig2.suptitle(
                "Fold 1/5 — validação (holdout tr_fit|val após NCV interno na seleção de hiperparâmetros)."
            )
            fig2.tight_layout()
            u.save_pdf(fig2, fig_dir / "training_curves.pdf")
            u.save_training_curve_csv(
                tab_dir / "training_curves_fold0.csv",
                np.arange(1, n_trees + 1, dtype=np.int32),
                {
                    "logloss_val": np.asarray(evals, dtype=np.float64),
                    "accuracy_val": np.asarray(acc_curve, dtype=np.float64),
                },
            )

        proba_te = model.predict_proba(X_test_flat)[:, 1]
        pred_te = (proba_te >= 0.5).astype(np.int32)
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

        sm = _shap_abs_mean_test(model, X_test_flat)
        u.accumulate_flat_importance(
            shap_roi, shap_attr, sm, keep_final, feat_names, slot_labels
        )

        if len(test_idx) > best_shap_n:
            best_shap_n = len(test_idx)
            best_shap = (model, X_test_flat.copy(), keep_final.copy())

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
        title="XGBoost — matriz de confusão agregada (predições OOF, 5-fold SGK)",
    )
    u.plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        fig_dir / "roc_cv.pdf",
        fig_dir / "pr_cv.pdf",
        title_prefix="XGBoost",
        fpr_grid=FPR_GRID,
        rec_grid=REC_GRID,
    )
    u.plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        fig_dir / "metrics_box_cv.pdf",
        title="XGBoost — distribuição das métricas no teste (5 folds)",
    )

    u.plot_top_bars_pdf(
        shap_roi_m,
        fig_dir / "shap_top_roi.pdf",
        title="XGBoost — |SHAP| médio no teste, agregado por ROI (média dos 5 folds)",
        top_k=TOP_K_ROI,
        xlabel="Média da média |SHAP| por coluna (por fold)",
    )
    u.plot_top_bars_pdf(
        shap_attr_m,
        fig_dir / "shap_top_attr.pdf",
        title="XGBoost — |SHAP| médio no teste, agregado por nome de atributo",
        top_k=TOP_K_ATTR,
        xlabel="Média da média |SHAP| por coluna (por fold)",
    )

    if best_shap is not None:
        b_model, X_b, kf = best_shap
        explainer = shap.TreeExplainer(b_model)
        sv = explainer.shap_values(X_b)
        if isinstance(sv, list):
            sv = np.asarray(sv[1], dtype=np.float64)
        else:
            sv = np.asarray(sv, dtype=np.float64)
        nc = len(kf)
        flat_names: list[str] = []
        for fi in range(X_b.shape[1]):
            slot = fi // nc
            j = fi % nc
            roi = u.roi_from_slot_label(slot_labels[slot])
            fn = feat_names[int(kf[j])]
            flat_names.append(f"{roi}|{fn}"[:72])
        mean_abs = np.abs(sv).mean(axis=0)
        top_m = min(25, len(mean_abs))
        top_idx = np.argsort(-mean_abs)[:top_m]
        shap.summary_plot(
            sv[:, top_idx],
            X_b[:, top_idx],
            feature_names=np.array(flat_names, dtype=object)[top_idx],
            show=False,
            max_display=top_m,
        )
        fig_sh = plt.gcf()
        fig_sh.suptitle(
            "XGBoost — SHAP (fold com maior conjunto de teste; colunas mais relevantes)"
        )
        fig_sh.tight_layout()
        u.save_pdf(fig_sh, fig_dir / "shap_summary.pdf")

    elapsed = time.perf_counter() - t0
    print(f"Tempo total: {elapsed:.1f} s — artefactos em {run_dir}")
    u.write_run_meta_json(
        run_dir,
        model_slug=MODEL_SLUG,
        downsample_group_sex=DOWNSAMPLE_GROUP_SEX,
        duration_seconds=elapsed,
        extra={
            "inner_ncv_splits": INNER_NCV_SPLITS,
            "optuna_trials": OPTUNA_XGB_TRIALS,
            "csv_schema": "metrics_per_fold, oof_predictions, fold_test_scores, importance_shap_*",
        },
    )


if __name__ == "__main__":
    main()

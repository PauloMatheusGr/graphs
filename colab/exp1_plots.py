"""Regenera PDFs a partir dos CSVs em tables/ (exp1_xgboost / exp1_rocket / exp1_svm / exp1_lstm).

Edite apenas a secção CONFIG abaixo (pasta do run e textos dos gráficos), depois execute:

  python colab/exp1_plots.py

Se correr a partir da pasta colab/:

  python exp1_plots.py
"""

from __future__ import annotations

from pathlib import Path

import exp1_utils as u
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edite aqui
# ---------------------------------------------------------------------------

# Pasta do run: deve existir tables/ (e opcionalmente figures/) dentro dela.
# Por defeito: relativa à pasta colab/ onde está este ficheiro.
_COLAB = Path(__file__).resolve().parent
# Ex.: .../exp1/unbalanced/svm | .../balanced/lstm | .../unbalanced/xgboost
RUN_DIR = _COLAB / "exp1" / "unbalanced" / "lstm"

# Grelhas ROC/PR (altere se quiser mais ou menos pontos)
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)

# --- Contagens de atributos após filtragens (feature_counts_fold0.csv; fold externo 0) ---
TITLE_FEATURE_COUNTS = "Exp1 LSTM — Nº de atributos (fold 1, tr_fit)"
YLABEL_FEATURE_COUNTS = "Nº atributos"

# --- Curvas de treino Keras (training_curves_fold0.csv; só LSTM) ---
TITLE_TRAINING_CURVES = (
    "Exp1 LSTM — loss e AUC na validação (fold 1, holdout tr_fit|val)"
)

# --- Matriz de confusão (oof_predictions.csv) ---
TITLE_CONFUSION = "Exp1 LSTM — matriz de confusão (predições OOF, 5-fold)"
# Cmap: "Blues", "Greens", etc. (ver plot_confusion_oof_pdf em exp1_utils)
CMAP_CONFUSION = "Blues"

# --- ROC e PR (fold_test_scores.csv) ---
# O prefixo aparece no título; o scope aparece em "ROC (scope)" / "PR (scope)"
TITLE_PREFIX_ROC_PR = "Exp1 LSTM"
ROC_SCOPE_LABEL = "teste por fold"
PR_SCOPE_LABEL = "teste por fold"

# --- Boxplot métricas (metrics_per_fold.csv) ---
TITLE_METRICS_BOX = "Exp1 LSTM — distribuição das métricas no teste (5 folds)"
# Rótulos do eixo X do boxplot (Acc, AUC, F1)
XTICK_METRICS = ("Acc", "AUC", "F1")

# --- Barras importância SHAP (importance_shap_*_mean.csv), se existirem ---
TITLE_BARS_SHAP_ROI = "Exp1 LSTM — |SHAP| agregado por ROI (média dos folds)"
TITLE_BARS_SHAP_ATTR = "Exp1 LSTM — |SHAP| agregado por atributo"
XLABEL_BARS_SHAP = "Valor agregado"
TOP_K_SHAP = 20

# --- Barras |coef| SVM (importance_coef_*_mean.csv), se existirem ---
TITLE_BARS_COEF_ROI = "SVM linear — |coef.| agregado por ROI"
TITLE_BARS_COEF_ATTR = "SVM linear — |coef.| agregado por atributo"
XLABEL_BARS_COEF = "Valor agregado"
TOP_K_COEF = 20

# ---------------------------------------------------------------------------


def _importance_csv_to_dict(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(r["name"]): float(r["value"]) for _, r in df.iterrows()}


def main() -> None:
    run_dir = RUN_DIR.resolve()
    tab = run_dir / "tables"
    fig = run_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    fcounts = tab / "feature_counts_fold0.csv"
    if fcounts.is_file():
        u.plot_feature_counts_bar_pdf(
            fcounts,
            fig / "feature_counts.pdf",
            title=TITLE_FEATURE_COUNTS,
            ylabel=YLABEL_FEATURE_COUNTS,
        )

    tcurves = tab / "training_curves_fold0.csv"
    if tcurves.is_file():
        u.plot_training_curves_keras_pdf(
            tcurves,
            fig / "training_curves.pdf",
            title=TITLE_TRAINING_CURVES,
        )

    oof_path = tab / "oof_predictions.csv"
    if oof_path.is_file():
        yt, yp = u.load_oof_arrays(oof_path)
        u.plot_confusion_oof_pdf(
            yt,
            yp,
            fig / "confusion_oof.pdf",
            title=TITLE_CONFUSION,
            cmap=CMAP_CONFUSION,
        )

    fts = tab / "fold_test_scores.csv"
    if fts.is_file():
        y_splits, score_splits = u.load_fold_test_scores_for_plots(fts)
        u.plot_roc_pr_cv_pdf(
            y_splits,
            score_splits,
            fig / "roc_cv.pdf",
            fig / "pr_cv.pdf",
            title_prefix=TITLE_PREFIX_ROC_PR,
            fpr_grid=FPR_GRID,
            rec_grid=REC_GRID,
            roc_scope_label=ROC_SCOPE_LABEL,
            pr_scope_label=PR_SCOPE_LABEL,
        )

    mpath = tab / "metrics_per_fold.csv"
    if mpath.is_file():
        acc, auc_a, f1 = u.load_metrics_per_fold(mpath)
        u.plot_metrics_box_pdf(
            acc,
            auc_a,
            f1,
            fig / "metrics_box_cv.pdf",
            title=TITLE_METRICS_BOX,
            xtick_labels=XTICK_METRICS,
        )

    rp_shap = tab / "importance_shap_roi_mean.csv"
    ap_shap = tab / "importance_shap_attr_mean.csv"
    if rp_shap.is_file():
        d = _importance_csv_to_dict(rp_shap)
        u.plot_top_bars_pdf(
            d,
            fig / "shap_top_roi.pdf",
            title=TITLE_BARS_SHAP_ROI,
            top_k=min(TOP_K_SHAP, max(len(d), 1)),
            xlabel=XLABEL_BARS_SHAP,
        )
    if ap_shap.is_file():
        d = _importance_csv_to_dict(ap_shap)
        u.plot_top_bars_pdf(
            d,
            fig / "shap_top_attr.pdf",
            title=TITLE_BARS_SHAP_ATTR,
            top_k=min(TOP_K_SHAP, max(len(d), 1)),
            xlabel=XLABEL_BARS_SHAP,
        )

    rp_coef = tab / "importance_coef_roi_mean.csv"
    ap_coef = tab / "importance_coef_attr_mean.csv"
    if rp_coef.is_file():
        d = _importance_csv_to_dict(rp_coef)
        u.plot_top_bars_pdf(
            d,
            fig / "coef_top_roi.pdf",
            title=TITLE_BARS_COEF_ROI,
            top_k=min(TOP_K_COEF, max(len(d), 1)),
            xlabel=XLABEL_BARS_COEF,
        )
    if ap_coef.is_file():
        d = _importance_csv_to_dict(ap_coef)
        u.plot_top_bars_pdf(
            d,
            fig / "coef_top_attr.pdf",
            title=TITLE_BARS_COEF_ATTR,
            top_k=min(TOP_K_COEF, max(len(d), 1)),
            xlabel=XLABEL_BARS_COEF,
        )

    print(f"Figuras gravadas em {fig}")


if __name__ == "__main__":
    main()

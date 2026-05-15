"""Regenera PDFs a partir dos CSVs em tables/ (gerados por exp2_xgboost / exp2_rocket / exp2_svm).

Edite apenas a secção CONFIG abaixo (pasta do run e textos dos gráficos), depois execute:

  python colab/exp2_plots.py

Se correr a partir da pasta colab/:

  python exp2_plots.py
"""

from __future__ import annotations

from pathlib import Path

import exp1_utils as u
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edite aqui
# ---------------------------------------------------------------------------

_COLAB = Path(__file__).resolve().parent
RUN_DIR = _COLAB / "exp2" / "unbalanced" / "svm"

FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)

TITLE_FEATURE_COUNTS = (
    "Exp2 — Nº de atributos — Raw vs correlação vs variância (fold 1, tr_fit)"
)
YLABEL_FEATURE_COUNTS = "Nº atributos"

TITLE_CONFUSION = "Exp2 SVM linear — matriz de confusão (predições OOF, 5-fold)"
CMAP_CONFUSION = "Blues"

TITLE_PREFIX_ROC_PR = "Exp2 SVM linear"
ROC_SCOPE_LABEL = "teste por fold"
PR_SCOPE_LABEL = "teste por fold"

TITLE_METRICS_BOX = "Exp2 SVM linear — distribuição das métricas no teste (5 folds)"
XTICK_METRICS = ("Acc", "AUC", "F1")

TITLE_BARS_SHAP_ROI = "Exp2 XGBoost — |SHAP| agregado por ROI (média dos folds)"
TITLE_BARS_SHAP_ATTR = "Exp2 XGBoost — |SHAP| agregado por atributo"
XLABEL_BARS_SHAP = "Valor agregado"
TOP_K_SHAP = 20

TITLE_BARS_COEF_ROI = "Exp2 SVM linear — |coef.| agregado por ROI"
TITLE_BARS_COEF_ATTR = "Exp2 SVM linear — |coef.| agregado por atributo"
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

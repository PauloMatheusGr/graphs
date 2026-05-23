"""Regenera PDFs a partir dos CSVs em tables/ (runs exp1 ou exp2).

  python colab/exp_plots.py

Variável de ambiente RUN_DIR (pasta do run com tables/). Títulos inferem Exp1/Exp2
a partir do caminho (…/exp1/… ou …/exp2/…).
"""

from __future__ import annotations

import os as _os
from pathlib import Path

import exp_utils as u
import numpy as np
import pandas as pd

_COLAB = Path(__file__).resolve().parent
_DEFAULT_RUN = _COLAB / "exp2" / "unbalanced" / "lstm"
RUN_DIR = Path(_os.environ["RUN_DIR"]) if _os.environ.get("RUN_DIR") else _DEFAULT_RUN

FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)
CMAP_CONFUSION = "Blues"
ROC_SCOPE_LABEL = "teste por fold"
PR_SCOPE_LABEL = "teste por fold"
XLABEL_BARS = "Valor agregado"
TOP_K_SHAP = 20
TOP_K_COEF = 20


def _exp_label(run_dir: Path) -> str:
    parts = run_dir.parts
    if "exp1" in parts:
        return "Exp1"
    if "exp2" in parts:
        return "Exp2"
    return "Exp"


def _titles(run_dir: Path) -> dict[str, str]:
    e = _exp_label(run_dir)
    return {
        "feature_counts": (
            f"{e} LSTM — Nº de atributos — Raw vs correlação vs variância (fold 1, tr_fit)"
        ),
        "ylabel_feature_counts": "Nº atributos",
        "training_curves": f"{e} LSTM — loss e AUC na validação (fold 1, holdout tr_fit|val)",
        "training_curves_fold": (
            f"{e} — fold {{k}}/{{n}} — logloss e acurácia (treino vs validação, tr_fit|val)"
        ),
        "training_curves_mean": (
            f"{e} — média dos folds externos — logloss e acurácia (treino vs validação)"
        ),
        "xlabel_training": "passo",
        "confusion": f"{e} LSTM — matriz de confusão (predições OOF, 5-fold)",
        "roc_pr_prefix": f"{e} LSTM",
        "metrics_box": f"{e} LSTM — distribuição das métricas no teste (5 folds)",
        "shap_roi": f"{e} LSTM — |SHAP| agregado por ROI (média dos folds)",
        "shap_attr": f"{e} LSTM — |SHAP| agregado por atributo",
        "coef_roi": f"{e} SVM linear — |coef.| agregado por ROI",
        "coef_attr": f"{e} SVM linear — |coef.| agregado por atributo",
    }


def _importance_csv_to_dict(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(r["name"]): float(r["value"]) for _, r in df.iterrows()}


def main() -> None:
    run_dir = RUN_DIR.resolve()
    t = _titles(run_dir)
    tab = run_dir / "tables"
    fig = run_dir / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    fcounts = tab / "feature_counts_fold0.csv"
    if fcounts.is_file():
        u.plot_feature_counts_bar_pdf(
            fcounts,
            fig / "feature_counts.pdf",
            title=t["feature_counts"],
            ylabel=t["ylabel_feature_counts"],
        )

    has_curve = any(
        (tab / f"training_curves_fold{i}.csv").is_file() for i in range(5)
    ) or (tab / "training_curves_mean.csv").is_file()
    if has_curve:
        u.regenerate_supervised_training_curve_plots(
            tab,
            fig,
            title_fold_tpl=t["training_curves_fold"],
            title_mean=t["training_curves_mean"],
            xlabel=t["xlabel_training"],
        )

    oof_path = tab / "oof_predictions.csv"
    if oof_path.is_file():
        yt, yp = u.load_oof_arrays(oof_path)
        u.plot_confusion_oof_pdf(
            yt,
            yp,
            fig / "confusion_oof.pdf",
            title=t["confusion"],
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
            title_prefix=t["roc_pr_prefix"],
            fpr_grid=FPR_GRID,
            rec_grid=REC_GRID,
            roc_scope_label=ROC_SCOPE_LABEL,
            pr_scope_label=PR_SCOPE_LABEL,
        )

    mpath = tab / "metrics_per_fold.csv"
    if mpath.is_file():
        acc, auc_a, f1, ap_a = u.load_metrics_per_fold(mpath)
        has_ap = "ap" in pd.read_csv(mpath, nrows=0).columns
        u.plot_metrics_box_pdf(
            acc,
            auc_a,
            f1,
            fig / "metrics_box_cv.pdf",
            title=t["metrics_box"],
            xtick_labels=("Acc", "AUC", "F1", "AP") if has_ap else ("Acc", "AUC", "F1"),
            ap=ap_a if has_ap else None,
        )

    rp_shap = tab / "importance_shap_roi_mean.csv"
    ap_shap = tab / "importance_shap_attr_mean.csv"
    if rp_shap.is_file():
        d = _importance_csv_to_dict(rp_shap)
        u.plot_top_bars_pdf(
            d,
            fig / "shap_top_roi.pdf",
            title=t["shap_roi"],
            top_k=min(TOP_K_SHAP, max(len(d), 1)),
            xlabel=XLABEL_BARS,
        )
    if ap_shap.is_file():
        d = _importance_csv_to_dict(ap_shap)
        u.plot_top_bars_pdf(
            d,
            fig / "shap_top_attr.pdf",
            title=t["shap_attr"],
            top_k=min(TOP_K_SHAP, max(len(d), 1)),
            xlabel=XLABEL_BARS,
        )

    rp_coef = tab / "importance_coef_roi_mean.csv"
    ap_coef = tab / "importance_coef_attr_mean.csv"
    if rp_coef.is_file():
        d = _importance_csv_to_dict(rp_coef)
        u.plot_top_bars_pdf(
            d,
            fig / "coef_top_roi.pdf",
            title=t["coef_roi"],
            top_k=min(TOP_K_COEF, max(len(d), 1)),
            xlabel=XLABEL_BARS,
        )
    if ap_coef.is_file():
        d = _importance_csv_to_dict(ap_coef)
        u.plot_top_bars_pdf(
            d,
            fig / "coef_top_attr.pdf",
            title=t["coef_attr"],
            top_k=min(TOP_K_COEF, max(len(d), 1)),
            xlabel=XLABEL_BARS,
        )

    print(f"Figuras gravadas em {fig}")


if __name__ == "__main__":
    main()

"""Métricas OOF estratificadas por sexo (F/M). Não treina modelos.

Uso (a partir da raiz do repo):

  python colab/analyze_oof_demographics.py

Edite RUN_DIR abaixo para o run exp2 desejado.
"""

from __future__ import annotations

from pathlib import Path

import exp1_utils as u
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

_COLAB = Path(__file__).resolve().parent
ROOT = _COLAB.parent

# --- CONFIG ---
import os as _os

_DEFAULT_RUN = _COLAB / "exp2" / "balanced" / "xgboost"
RUN_DIR = Path(_os.environ["RUN_DIR"]) if _os.environ.get("RUN_DIR") else _DEFAULT_RUN
SOURCE_CSV = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"

SEX_LABELS = {0: "F", 1: "M"}


def _metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    out: dict[str, float] = {
        "n": float(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["ap"] = float(average_precision_score(y, score))
    except ValueError:
        out["ap"] = float("nan")
    return out


def _aggregate_patient(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for pid, g in df.groupby("group_id", sort=False):
        yv = g["y_true"].unique()
        if len(yv) != 1:
            continue
        rows.append(
            {
                "group_id": pid,
                "sex": int(g["sex"].iloc[0]),
                "y_true": int(yv[0]),
                "score": float(g["score"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["y_pred"] = (out["score"] >= 0.5).astype(int)
    return out


def _run_level(
    df: pd.DataFrame, level: str, out_tab: Path, out_fig: Path
) -> None:
    rows: list[dict] = []
    for sex_code, label in SEX_LABELS.items():
        sub = df[df["sex"] == sex_code]
        if sub.empty:
            continue
        m = _metrics(
            sub["y_true"].to_numpy(),
            sub["y_pred"].to_numpy(),
            sub["score"].to_numpy(),
        )
        m["sex"] = label
        m["level"] = level
        rows.append(m)
        u.plot_confusion_oof_pdf(
            sub["y_true"].to_numpy(),
            sub["y_pred"].to_numpy(),
            out_fig / f"confusion_oof_sex_{label}_{level}.pdf",
            title=f"OOF — sexo {label} ({level})",
        )
        cm = confusion_matrix(
            sub["y_true"], sub["y_pred"], labels=[0, 1]
        )
        pd.DataFrame(
            cm,
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(out_tab / f"confusion_counts_sex_{label}_{level}.csv")
    pd.DataFrame(rows).to_csv(out_tab / f"metrics_by_sex_{level}.csv", index=False)


def main() -> None:
    oof_path = RUN_DIR / "tables" / "oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)

    oof = pd.read_csv(oof_path)
    src = pd.read_csv(SOURCE_CSV, usecols=["ID_PT", "SEX"])
    src["ID_PT"] = src["ID_PT"].astype(str)
    sex_map = (
        src.drop_duplicates("ID_PT")
        .assign(sex=lambda d: u.encode_sex_column(d["SEX"]).astype(int))
        .set_index("ID_PT")["sex"]
    )
    oof["group_id"] = oof["group_id"].astype(str)
    oof["sex"] = oof["group_id"].map(sex_map)
    if oof["sex"].isna().any():
        miss = sorted(oof.loc[oof["sex"].isna(), "group_id"].unique().tolist())[:10]
        raise ValueError(f"SEX ausente para pacientes (ex.): {miss}")

    out_tab = RUN_DIR / "tables" / "demographics"
    out_fig = RUN_DIR / "figures" / "demographics"
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    _run_level(oof, "datapoint", out_tab, out_fig)
    _run_level(_aggregate_patient(oof), "patient", out_tab, out_fig)
    print(f"Demografia gravada em {out_tab} e {out_fig}")


if __name__ == "__main__":
    main()

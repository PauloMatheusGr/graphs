"""Smoke test: score do checkpoint XGB fold 0 vs OOF (tolerância numérica)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

import exp1_utils as u

_COLAB = Path(__file__).resolve().parent
RUN_DIR = _COLAB / "exp2" / "balanced" / "xgboost"
CSV_PATH = _COLAB.parent / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"
EXP2_PATH = _COLAB.parent / "exp2.md"
TOL = 1e-4


def main() -> None:
    ckpt = RUN_DIR / "checkpoints" / "fold_0"
    oof = pd.read_csv(RUN_DIR / "tables" / "oof_predictions.csv")
    row = oof[oof["outer_fold"] == 0].iloc[0]
    row_idx = int(row["row_idx"])
    expected = float(row["score"])

    bundle = u.load_preprocess_bundle(ckpt / "preprocess.joblib")
    keep_final = bundle["keep_final"]
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))

    X_3d, y, groups, sex, feat_names, slot_labels = u.load_tensor(
        CSV_PATH,
        EXP2_PATH,
        ["1", "2", "3"],
        ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"],
        require_sex=True,
        temporal_mode="baseline_rate",
        dt_epsilon=float(meta.get("dt_epsilon", 0.5)),
    )

    X = X_3d[row_idx : row_idx + 1][:, :, keep_final]
    s = X.shape
    X_scaled = bundle["scaler"].transform(X.reshape(-1, s[-1])).reshape(s)
    flat = X_scaled.reshape(1, -1)

    clf = XGBClassifier()
    clf.load_model(str(ckpt / "model.json"))
    got = float(clf.predict_proba(flat)[0, 1])
    print(f"expected OOF score: {expected:.8f}")
    print(f"checkpoint score:   {got:.8f}")
    print(f"abs diff:           {abs(got - expected):.2e}")
    if abs(got - expected) > TOL:
        raise SystemExit(f"FAIL: diff > {TOL}")
    print("OK")


if __name__ == "__main__":
    main()

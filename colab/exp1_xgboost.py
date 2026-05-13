"""exp1: XGBoost tabular a partir de exp1.md (pipeline minimalista)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "csvs/abordagem_teste/all_delta_features_neurocombat.csv"
EXP1_PATH = ROOT / "exp1.md"
OUT_DIR = Path(__file__).resolve().parent / "exp1_figs"
PAIR_ORDER = ["12", "13", "23"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
CORR_THR = 0.9
# VarianceThreshold(0.0) remove apenas colunas constantes no treino (exp1: baixa variância).
VAR_THR = 0.0
RANDOM_STATE = 42


def _parse_exp1_feature_columns() -> list[str]:
    text = EXP1_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("As colunas de atributos são"):
            return line.split("são", 1)[1].strip().split()
    raise RuntimeError("exp1.md: linha de atributos não encontrada.")


def load_tensor() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_names = _parse_exp1_feature_columns()
    df = pd.read_csv(CSV_PATH)
    df["ID_PT"] = df["ID_PT"].astype(str)
    df["pair"] = df["pair"].astype(str).str.strip()
    y_map = {"sMCI": 0, "pMCI": 1}
    df["y"] = df["GROUP"].astype(str).map(y_map)
    if df["y"].isna().any():
        raise ValueError("GROUP contém valores fora de sMCI/pMCI.")

    feat_names = [c for c in feat_names if c in df.columns]
    if not feat_names:
        raise ValueError("Nenhuma coluna de atributos do exp1 presente no CSV.")

    samples: list[np.ndarray] = []
    y_out: list[float] = []
    groups: list[str] = []

    for keys, g in df.groupby(GROUP_KEY, sort=False):
        rows = []
        for p in PAIR_ORDER:
            gp = g[g["pair"] == p]
            if gp.empty:
                rows = None
                break
            gp = gp.sort_values(["roi", "side", "label"], kind="mergesort")
            rows.append(gp)
        if rows is None:
            continue
        block = pd.concat(rows, axis=0)
        if len(block) != 60:
            print(f"Aviso: grupo {keys} tem {len(block)} linhas (esperado 60); ignorado.")
            continue
        samples.append(block[feat_names].to_numpy(dtype=np.float64, copy=True))
        y_out.append(float(block["y"].iloc[0]))
        groups.append(str(block["ID_PT"].iloc[0]))

    if not samples:
        raise RuntimeError("Nenhuma amostra válida (60 linhas por grupo).")

    X = np.stack(samples, axis=0)
    y = np.asarray(y_out, dtype=np.int32)
    groups = np.asarray(groups, dtype=object)
    return X, y, groups


def corr_keep_indices(X: np.ndarray, thr: float) -> np.ndarray:
    """X: (n_rows, n_features); mantém colunas com greedy |corr| <= thr vs já escolhidas."""
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if xf.shape[0] < 2:
        return np.arange(X.shape[1], dtype=int)
    c = np.corrcoef(xf.T)
    n = c.shape[0]
    keep: list[int] = []
    for j in range(n):
        ok = True
        for k in keep:
            v = c[j, k]
            if np.isfinite(v) and abs(v) > thr:
                ok = False
                break
        if ok:
            keep.append(j)
    return np.asarray(keep, dtype=int)


def inner_train_val(
    tr_idx: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide treino em ajuste/validação sem misturar ID_PT (StratifiedGroupKFold)."""
    y_tr = y[tr_idx]
    g_tr = groups[tr_idx]
    uniq = np.unique(g_tr)
    n_splits = min(5, len(uniq))
    n_splits = max(2, n_splits)
    dummy = np.zeros(len(tr_idx), dtype=np.int8)
    sgk = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 1
    )
    try:
        tr_rel, val_rel = next(sgk.split(dummy, y_tr, g_tr))
    except ValueError:
        # fallback: últimos 20% pacientes ordenados
        order = np.argsort(g_tr)
        cut = max(1, int(0.8 * len(order)))
        tr_rel, val_rel = order[:cut], order[cut:]
    return tr_idx[tr_rel], tr_idx[val_rel]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_3d, y, groups = load_tensor()
    n_raw = X_3d.shape[2]

    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    dummy = np.zeros(len(y), dtype=np.int8)
    train_idx, test_idx = next(sgk.split(dummy, y, groups))

    tr_fit_idx, val_idx = inner_train_val(train_idx, y, groups)

    X_trf = X_3d[tr_fit_idx].reshape(-1, X_3d.shape[2])
    keep_corr = corr_keep_indices(X_trf, CORR_THR)
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

    fig, ax = plt.subplots()
    ax.bar(["Raw", "Após correlação", "Após variância"], [n_raw, n_after_corr, n_after_var])
    ax.set_ylabel("Nº atributos")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp1_xgb_feature_counts.png", dpi=120)
    plt.close(fig)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(
        X_train_flat,
        y[tr_fit_idx],
        eval_set=[(X_val_flat, y[val_idx])],
        verbose=False,
    )

    booster = model.get_booster()
    dval = xgb.DMatrix(X_val_flat)
    evals = model.evals_result_["validation_0"]["logloss"]
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
    fig2.suptitle("Validação: subset do treino (StratifiedGroupKFold interno).")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "exp1_xgb_training_curves.png", dpi=120)
    plt.close(fig2)

    y_pred = model.predict(X_test_flat)
    acc = accuracy_score(y[test_idx], y_pred)
    print(f"Acurácia teste (1º fold SGK): {acc:.4f}")


if __name__ == "__main__":
    main()

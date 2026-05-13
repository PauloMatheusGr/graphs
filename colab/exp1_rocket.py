"""exp1: ROCKET + regressão logística L1 (saga) com Optuna e nested CV interno.

Nested CV: Optuna maximiza a média da AUC em K folds StratifiedGroupKFold dentro
do treino externo; ROCKET é reajustado uma vez por fold interno (sem vazamento).
Downsample opcional no treino externo por paciente (GROUP×SEX).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv"
EXP1_PATH = ROOT / "exp1.md"
OUT_DIR = Path(__file__).resolve().parent / "exp1_figs"
PAIR_ORDER = ["12", "13", "23"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]
CORR_THR = 0.9
VAR_THR = 0.0
RANDOM_STATE = 42
NUM_KERNELS = 10_000
# Optuna: média da AUC em INNER_NCV_SPLITS folds internos; C em escala log.
OPTUNA_ROCKET_TRIALS = 25
INNER_NCV_SPLITS = 3
# Grade só para figura diagnóstica (fold 1): acurácia vs log10(C).
C_DIAG_GRID = np.logspace(-4, 4, 17)
DOWNSAMPLE_GROUP_SEX = True
FPR_GRID = np.linspace(0.0, 1.0, 101)
REC_GRID = np.linspace(0.0, 1.0, 101)


def _parse_exp1_feature_columns() -> list[str]:
    text = EXP1_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("As colunas de atributos são"):
            return line.split("são", 1)[1].strip().split()
    raise RuntimeError("exp1.md: linha de atributos não encontrada.")


def _encode_sex_column(col: pd.Series) -> np.ndarray:
    """exp1.md: F=0, M=1 como float64 para a matriz de atributos."""
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or not np.isin(v, (0.0, 1.0)).all():
            raise ValueError("SEX numérico deve ser apenas 0 ou 1 (F=0, M=1).")
        return v
    upper = col.astype(str).str.strip().str.upper()
    mapped = upper.map({"F": 0.0, "M": 1.0})
    if mapped.isna().any():
        bad = sorted(upper[mapped.isna()].unique().tolist()[:20])
        raise ValueError(f"SEX inesperado: {bad} (esperado F ou M).")
    return mapped.to_numpy(dtype=np.float64, copy=True)


def block_to_feature_float_matrix(block: pd.DataFrame, feat_names: list[str]) -> np.ndarray:
    """(n_linhas, n_atributos) numérico; coluna SEX codificada F=0, M=1."""
    cols: list[np.ndarray] = []
    for name in feat_names:
        if name == "SEX":
            cols.append(_encode_sex_column(block[name]))
        else:
            arr = pd.to_numeric(block[name], errors="coerce")
            if arr.isna().any():
                raise ValueError(f"Valores não numéricos na coluna de atributo {name!r}.")
            cols.append(arr.to_numpy(dtype=np.float64, copy=True))
    return np.column_stack(cols)


def load_tensor(
    *, require_sex: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    feat_names = _parse_exp1_feature_columns()
    df = pd.read_csv(CSV_PATH)
    df["ID_PT"] = df["ID_PT"].astype(str)
    df["pair"] = df["pair"].astype(str).str.strip()
    y_map = {"sMCI": 0, "pMCI": 1}
    df["y"] = df["GROUP"].astype(str).map(y_map)
    if df["y"].isna().any():
        raise ValueError("GROUP contém valores fora de sMCI/pMCI.")
    has_sex = "SEX" in df.columns
    if require_sex and not has_sex:
        raise ValueError("Coluna SEX ausente no CSV (necessária com DOWNSAMPLE_GROUP_SEX=True).")

    feat_names = [c for c in feat_names if c in df.columns]
    if not feat_names:
        raise ValueError("Nenhuma coluna de atributos do exp1 presente no CSV.")

    samples: list[np.ndarray] = []
    y_out: list[float] = []
    groups: list[str] = []
    sex_out: list[int] = []
    slot_labels_ref: list[str] | None = None

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
        if slot_labels_ref is None:
            lab_row: list[str] = []
            for _, row in block.iterrows():
                pair = str(row["pair"])
                roi = str(row["roi"]) if "roi" in block.columns else "NA"
                side = str(row["side"]) if "side" in block.columns else "NA"
                label = str(row["label"]) if "label" in block.columns else "NA"
                lab_row.append(f"{pair}|{roi}|{side}|{label}")
            slot_labels_ref = lab_row
        samples.append(block_to_feature_float_matrix(block, feat_names))
        y_out.append(float(block["y"].iloc[0]))
        groups.append(str(block["ID_PT"].iloc[0]))
        if has_sex:
            sex_out.append(int(_encode_sex_column(block["SEX"])[0]))
        else:
            sex_out.append(0)

    if not samples:
        raise RuntimeError("Nenhuma amostra válida (60 linhas por grupo).")
    if slot_labels_ref is None or len(slot_labels_ref) != 60:
        raise RuntimeError("Não foi possível definir 60 labels de slot (roi/side/label).")

    X = np.stack(samples, axis=0)
    y = np.asarray(y_out, dtype=np.int32)
    groups = np.asarray(groups, dtype=object)
    sex = np.asarray(sex_out, dtype=np.int8)
    return X, y, groups, sex, feat_names, slot_labels_ref


def corr_keep_indices(X: np.ndarray, thr: float) -> np.ndarray:
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


def downsample_train_indices(
    train_idx: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    sex: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Por ID_PT: mantém todos os triplets dos pacientes escolhidos; equipara #pacientes por estrato y×sex."""
    train_idx = np.asarray(train_idx, dtype=int)
    id_pt = groups[train_idx].astype(str)
    yy = y[train_idx]
    ss = sex[train_idx]

    pt_strat: dict[str, str] = {}
    seen: set[str] = set()
    for i in range(len(train_idx)):
        pid = id_pt[i]
        if pid in seen:
            continue
        seen.add(pid)
        pt_strat[pid] = f"{int(yy[i])}_{int(ss[i])}"

    pts_by_strat: dict[str, list[str]] = {}
    for pid, st in pt_strat.items():
        pts_by_strat.setdefault(st, []).append(pid)

    active = {k: v for k, v in pts_by_strat.items() if v}
    if len(active) < 2:
        return train_idx

    min_n = min(len(v) for v in active.values())
    if min_n == 0:
        return train_idx

    rng = np.random.RandomState(int(seed))
    selected: set[str] = set()
    for pts in active.values():
        pts = list(pts)
        rng.shuffle(pts)
        selected.update(pts[:min_n])

    g_tr = groups[train_idx]
    keep = np.isin(g_tr, list(selected))
    return train_idx[keep]


def inner_train_val(
    tr_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    fold_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    y_tr = y[tr_idx]
    g_tr = groups[tr_idx]
    uniq = np.unique(g_tr)
    n_splits = min(5, len(uniq))
    n_splits = max(2, n_splits)
    dummy = np.zeros(len(tr_idx), dtype=np.int8)
    sgk = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE + 1 + int(fold_id),
    )
    try:
        tr_rel, val_rel = next(sgk.split(dummy, y_tr, g_tr))
    except ValueError:
        order = np.argsort(g_tr)
        cut = max(1, int(0.8 * len(order)))
        tr_rel, val_rel = order[:cut], order[cut:]
    return tr_idx[tr_rel], tr_idx[val_rel]


def inner_cv_splits(
    tr_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    fold_id: int = 0,
    n_splits_requested: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds internos (por paciente) sobre o treino externo, para média da AUC no Optuna."""
    train_idx = np.asarray(tr_idx, dtype=int)
    y_tr = y[train_idx]
    g_tr = groups[train_idx]
    uniq = np.unique(g_tr)
    n_sp = min(int(n_splits_requested), len(uniq))
    n_sp = max(2, n_sp)
    dummy = np.zeros(len(train_idx), dtype=np.int8)
    sgk = StratifiedGroupKFold(
        n_splits=n_sp,
        shuffle=True,
        random_state=RANDOM_STATE + 1 + int(fold_id),
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for tr_rel, va_rel in sgk.split(dummy, y_tr, g_tr):
            splits.append((train_idx[tr_rel], train_idx[va_rel]))
    except ValueError:
        order = np.argsort(g_tr)
        cut = max(1, int(0.8 * len(order)))
        tr_rel, va_rel = order[:cut], order[cut:]
        splits.append((train_idx[tr_rel], train_idx[va_rel]))
    return splits


def _prepare_scaled_rocket_inputs(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlação + variância + StandardScaler em idx_fit; painéis (n, c, t) para ROCKET."""
    idx_fit = np.asarray(idx_fit, dtype=int)
    idx_va = np.asarray(idx_va, dtype=int)
    X_trf = X_3d[idx_fit].reshape(-1, X_3d.shape[2])
    keep_corr = corr_keep_indices(X_trf, CORR_THR)
    X_trf_c = X_trf[:, keep_corr]
    vt = VarianceThreshold(threshold=VAR_THR)
    vt.fit(X_trf_c)
    keep_var = np.where(vt.get_support())[0]
    keep_final = keep_corr[keep_var]

    def apply_cols(A: np.ndarray) -> np.ndarray:
        return A[:, :, keep_final]

    X_train_fit = apply_cols(X_3d[idx_fit])
    X_val = apply_cols(X_3d[idx_va])
    n_row_fit = len(idx_fit) * 60
    scaler = StandardScaler()
    scaler.fit(X_train_fit.reshape(n_row_fit, -1))

    def scale(A: np.ndarray) -> np.ndarray:
        s = A.shape
        return scaler.transform(A.reshape(-1, s[-1])).reshape(s)

    X_train_fit = scale(X_train_fit)
    X_val = scale(X_val)
    return np.transpose(X_train_fit, (0, 2, 1)), np.transpose(X_val, (0, 2, 1))


def _test_metrics_scores(
    clf: LogisticRegression, Z_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, float, float]:
    y_pred = clf.predict(Z_te)
    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    scores = clf.predict_proba(Z_te)[:, 1]
    try:
        auc = float(roc_auc_score(y_te, scores))
    except ValueError:
        auc = float("nan")
    return acc, auc, f1


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


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_oof_pdf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        row = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm.astype(np.float64), np.maximum(row, 1))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    im0 = ax0.imshow(cm, interpolation="nearest", cmap="Greens")
    ax0.set_title("Contagens (OOF)")
    for (i, j), v in np.ndenumerate(cm):
        ax0.text(j, i, int(v), ha="center", va="center", color="black", fontsize=9)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["0", "1"])
    ax0.set_yticklabels(["0", "1"])
    ax0.set_xlabel("Predito")
    ax0.set_ylabel("Verdadeiro")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    im1 = ax1.imshow(cmn, vmin=0, vmax=1, cmap="Greens", interpolation="nearest")
    ax1.set_title("Normalizada por linha (recall por classe)")
    for (i, j), v in np.ndenumerate(cmn):
        ax1.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["0", "1"])
    ax1.set_yticklabels(["0", "1"])
    ax1.set_xlabel("Predito")
    ax1.set_ylabel("Verdadeiro")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.text(
        0.5,
        0.02,
        "Legenda classes: 0 = sMCI, 1 = pMCI",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _save_pdf(fig, path)


def _interp_tpr(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, fpr, tpr, left=0.0, right=1.0)


def _interp_pr(
    y: np.ndarray, p: np.ndarray, rec_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prec, rec, _ = precision_recall_curve(y, p)
    m = np.isfinite(prec) & np.isfinite(rec)
    prec, rec = prec[m], rec[m]
    ord_rec = np.argsort(rec)
    rec, prec = rec[ord_rec], prec[ord_rec]
    u_rec, u_idx = np.unique(rec, return_index=True)
    prec = prec[u_idx]
    rec = u_rec
    prec_i = np.interp(rec_grid, rec, prec, left=1.0, right=0.0)
    return prec_i, rec


def _plot_roc_pr_cv_pdf(
    y_splits: list[np.ndarray],
    score_splits: list[np.ndarray],
    out_roc: Path,
    out_pr: Path,
    *,
    title_prefix: str,
) -> None:
    tpr_rows: list[np.ndarray] = []
    prec_rows: list[np.ndarray] = []
    aucs: list[float] = []
    pr_aucs: list[float] = []
    for y_te, sc in zip(y_splits, score_splits):
        if len(np.unique(y_te)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_te, sc)
        tpr_rows.append(_interp_tpr(fpr, tpr, FPR_GRID))
        aucs.append(roc_auc_score(y_te, sc))
        pi, _ = _interp_pr(y_te, sc, REC_GRID)
        prec_rows.append(pi)
        pr_aucs.append(average_precision_score(y_te, sc))
    if not tpr_rows:
        return
    tpr_m = np.mean(tpr_rows, axis=0)
    tpr_s = np.std(tpr_rows, axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for row in tpr_rows:
        ax.plot(FPR_GRID, row, color="0.6", alpha=0.45, linewidth=0.9)
    ax.plot(FPR_GRID, tpr_m, color="C0", linewidth=2.2, label="média TPR")
    ax.fill_between(
        FPR_GRID,
        np.clip(tpr_m - tpr_s, 0, 1),
        np.clip(tpr_m + tpr_s, 0, 1),
        color="C0",
        alpha=0.25,
        label="±1 dp entre folds",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix} — ROC (teste)\nAUC médio={np.mean(aucs):.3f} ± {np.std(aucs, ddof=0):.3f}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_pdf(fig, out_roc)

    prec_m = np.mean(prec_rows, axis=0)
    prec_s = np.std(prec_rows, axis=0, ddof=0)
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    for row in prec_rows:
        ax2.plot(REC_GRID, row, color="0.6", alpha=0.45, linewidth=0.9)
    ax2.plot(REC_GRID, prec_m, color="C3", linewidth=2.2, label="média precision")
    ax2.fill_between(
        REC_GRID,
        np.clip(prec_m - prec_s, 0, 1),
        np.clip(prec_m + prec_s, 0, 1),
        color="C3",
        alpha=0.25,
        label="±1 dp entre folds",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(
        f"{title_prefix} — PR (teste)\nAP médio={np.mean(pr_aucs):.3f} ± {np.std(pr_aucs, ddof=0):.3f}"
    )
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    _save_pdf(fig2, out_pr)


def _plot_metrics_box_pdf(
    acc: np.ndarray,
    auc_a: np.ndarray,
    f1_a: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    positions = [1, 2, 3]
    auc_plot = np.asarray(auc_a, dtype=np.float64)
    if np.isnan(auc_plot).any():
        fill = float(np.nanmean(auc_plot))
        auc_plot = np.where(np.isnan(auc_plot), fill, auc_plot)
    bp = ax.boxplot(
        [acc, auc_plot, f1_a],
        positions=positions,
        widths=0.55,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.85")
    rng = np.random.default_rng(42)
    for i, arr in enumerate([acc, auc_a, f1_a]):
        arr_s = np.asarray(arr, dtype=np.float64)
        if i == 1 and np.isnan(arr_s).any():
            arr_s = np.where(np.isnan(arr_s), float(np.nanmean(arr_s)), arr_s)
        x = positions[i] + 0.08 * rng.standard_normal(len(arr_s))
        ax.scatter(x, arr_s, color="C0", s=28, zorder=3, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Acc", "AUC", "F1"])
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save_pdf(fig, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_3d, y, groups, sex, _feat_names, _slot_labels = load_tensor(
        require_sex=DOWNSAMPLE_GROUP_SEX
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

    if DOWNSAMPLE_GROUP_SEX:
        print("Downsample ativo: treino externo por paciente (estratos y×SEX).")

    for fold_id, (train_idx, test_idx) in enumerate(sgk.split(dummy, y, groups)):
        train_idx = np.asarray(train_idx, dtype=int)
        n_tr0 = len(train_idx)
        if DOWNSAMPLE_GROUP_SEX:
            train_idx = downsample_train_indices(
                train_idx, groups, y, sex, seed=RANDOM_STATE + 31 * fold_id
            )
        if fold_id == 0:
            print(
                f"Fold 1 — treino externo: {n_tr0} -> {len(train_idx)} amostras"
                + (" (após downsample)." if DOWNSAMPLE_GROUP_SEX else ".")
            )

        inner_splits = inner_cv_splits(
            train_idx,
            y,
            groups,
            fold_id=fold_id,
            n_splits_requested=INNER_NCV_SPLITS,
        )
        inner_z_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (in_tr, in_va) in enumerate(inner_splits):
            Xr_tr, Xr_va = _prepare_scaled_rocket_inputs(X_3d, in_tr, in_va)
            rk = Rocket(
                num_kernels=NUM_KERNELS,
                random_state=RANDOM_STATE + 17 * fold_id + i,
            )
            rk.fit(Xr_tr)
            inner_z_pairs.append((rk.transform(Xr_tr), rk.transform(Xr_va)))

        tr_fit_idx, val_idx = inner_train_val(train_idx, y, groups, fold_id=fold_id)

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
            _save_pdf(fig, OUT_DIR / "exp1_rocket_feature_counts.pdf")

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
            _save_pdf(fig2, OUT_DIR / "exp1_rocket_l1_C.pdf")

        pred_te = clf.predict(Z_te)
        sc_te = clf.predict_proba(Z_te)[:, 1]
        y_oof[test_idx] = y[test_idx]
        pred_oof[test_idx] = pred_te.astype(np.int32, copy=False)
        score_oof[test_idx] = sc_te
        y_splits.append(np.asarray(y[test_idx], dtype=np.int32))
        score_splits.append(np.asarray(sc_te, dtype=np.float64))

        acc, auc, f1 = _test_metrics_scores(clf, Z_te, y[test_idx])
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

    mask = y_oof >= 0
    _plot_confusion_oof_pdf(
        y_oof[mask],
        pred_oof[mask],
        OUT_DIR / "exp1_rocket_confusion_oof.pdf",
        title="ROCKET+L1 (Optuna) — matriz de confusão agregada (predições OOF, 5-fold SGK)",
    )
    _plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        OUT_DIR / "exp1_rocket_roc_cv.pdf",
        OUT_DIR / "exp1_rocket_pr_cv.pdf",
        title_prefix="ROCKET+L1 (Optuna)",
    )
    _plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        OUT_DIR / "exp1_rocket_metrics_box_cv.pdf",
        title="ROCKET+L1 (Optuna) — distribuição das métricas no teste (5 folds)",
    )


if __name__ == "__main__":
    main()

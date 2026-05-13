"""exp1: XGBoost tabular a partir de exp1.md (Optuna + nested CV interno + early stopping).

Nested CV: o Optuna maximiza a média da AUC em vários StratifiedGroupKFold internos
dentro do treino externo; o refit final usa tr_fit/val (holdout) como antes.
Downsample opcional no treino externo por paciente (GROUP×SEX).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
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
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_delta_features_neurocombat.csv"
EXP1_PATH = ROOT / "exp1.md"
OUT_DIR = Path(__file__).resolve().parent / "exp1_figs"
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
INNER_NCV_SPLITS = 3


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


def _flat_scaled_for_inner_split(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlação + variância + StandardScaler só em idx_fit; retorna flat para treino e validação."""
    idx_fit = np.asarray(idx_fit, dtype=int)
    idx_va = np.asarray(idx_va, dtype=int)
    X_fit_flat = X_3d[idx_fit].reshape(-1, X_3d.shape[2])
    keep_corr = corr_keep_indices(X_fit_flat, CORR_THR)
    X_trf_c = X_fit_flat[:, keep_corr]
    vt = VarianceThreshold(threshold=VAR_THR)
    vt.fit(X_trf_c)
    keep_var = np.where(vt.get_support())[0]
    keep_final = keep_corr[keep_var]

    def apply_cols(A: np.ndarray) -> np.ndarray:
        return A[:, :, keep_final]

    X_fit_3d = apply_cols(X_3d[idx_fit])
    X_va_3d = apply_cols(X_3d[idx_va])
    n_row_fit = len(idx_fit) * 60
    scaler = StandardScaler()
    scaler.fit(X_fit_3d.reshape(n_row_fit, -1))

    def scale(A: np.ndarray) -> np.ndarray:
        s = A.shape
        return scaler.transform(A.reshape(-1, s[-1])).reshape(s)

    X_tr_flat = scale(X_fit_3d).reshape(len(idx_fit), -1)
    X_va_flat = scale(X_va_3d).reshape(len(idx_va), -1)
    return X_tr_flat, X_va_flat


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
            X_tr_flat, X_va_flat = _flat_scaled_for_inner_split(X_3d, in_tr, in_va)
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
    """Subdivide treino em ajuste/validação sem misturar ID_PT (StratifiedGroupKFold)."""
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
        # fallback: últimos 20% pacientes ordenados
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
    """Lista de (treino interno, validação interna) por paciente; cobre todo o treino externo em K folds."""
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


def _test_metrics_xgb(
    model: XGBClassifier, X_test_flat: np.ndarray, y_te: np.ndarray
) -> tuple[float, float, float]:
    y_pred = model.predict(X_test_flat)
    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    proba = model.predict_proba(X_test_flat)[:, 1]
    try:
        auc = float(roc_auc_score(y_te, proba))
    except ValueError:
        auc = float("nan")
    return acc, auc, f1


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _roi_from_slot_label(slot_lab: str) -> str:
    parts = slot_lab.split("|")
    return parts[1] if len(parts) > 1 else slot_lab


def _accumulate_flat_importance(
    agg_roi: dict[str, float],
    agg_attr: dict[str, float],
    vec_flat: np.ndarray,
    keep_final: np.ndarray,
    feat_names: list[str],
    slot_labels: list[str],
) -> None:
    """Soma valores por ROI (2º campo do slot) e por nome de atributo."""
    ncols = len(keep_final)
    n_slots = len(slot_labels)
    for fi in range(min(len(vec_flat), n_slots * ncols)):
        slot = fi // ncols
        j = fi % ncols
        v = float(vec_flat[fi])
        roi = _roi_from_slot_label(slot_labels[slot])
        attr = feat_names[int(keep_final[j])]
        agg_roi[roi] = agg_roi.get(roi, 0.0) + v
        agg_attr[attr] = agg_attr.get(attr, 0.0) + v


def _gain_vector_from_xgb(model: XGBClassifier, n_features: int) -> np.ndarray:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    g = np.zeros(n_features, dtype=np.float64)
    for k, val in score.items():
        if k.startswith("f"):
            idx = int(k[1:])
            if 0 <= idx < n_features:
                g[idx] = float(val)
    s = float(g.sum())
    if s > 0:
        g /= s
    return g


def _shap_abs_mean_test(model: XGBClassifier, X_te: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(np.abs(sv).mean(axis=0), dtype=np.float64)


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
    im0 = ax0.imshow(cm, interpolation="nearest", cmap="Blues")
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
    im1 = ax1.imshow(cmn, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
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
    ax.set_title(f"{title_prefix} — ROC (teste por fold)\nAUC médio={np.mean(aucs):.3f} ± {np.std(aucs, ddof=0):.3f}")
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
        f"{title_prefix} — PR (teste por fold)\nAP médio={np.mean(pr_aucs):.3f} ± {np.std(pr_aucs, ddof=0):.3f}"
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
    labels = ["Acc", "AUC", "F1"]
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
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save_pdf(fig, path)


def _plot_top_bars_pdf(
    scores: dict[str, float],
    path: Path,
    *,
    title: str,
    top_k: int,
    xlabel: str,
) -> None:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not items:
        return
    labs, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(6, max(3.0, 0.28 * len(items))))
    y_pos = np.arange(len(items))
    ax.barh(y_pos, vals, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    _save_pdf(fig, path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_3d, y, groups, sex, feat_names, slot_labels = load_tensor(
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
    proba_oof = np.full(n_samples, np.nan, dtype=np.float64)
    y_splits: list[np.ndarray] = []
    score_splits: list[np.ndarray] = []

    gain_roi: dict[str, float] = defaultdict(float)
    gain_attr: dict[str, float] = defaultdict(float)
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
            _save_pdf(fig, OUT_DIR / "exp1_xgb_feature_counts.pdf")

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
            _save_pdf(fig2, OUT_DIR / "exp1_xgb_training_curves.pdf")

        proba_te = model.predict_proba(X_test_flat)[:, 1]
        pred_te = (proba_te >= 0.5).astype(np.int32)
        y_oof[test_idx] = y[test_idx]
        pred_oof[test_idx] = pred_te
        proba_oof[test_idx] = proba_te
        y_splits.append(np.asarray(y[test_idx], dtype=np.int32))
        score_splits.append(np.asarray(proba_te, dtype=np.float64))

        gvec = _gain_vector_from_xgb(model, X_test_flat.shape[1])
        _accumulate_flat_importance(
            gain_roi, gain_attr, gvec, keep_final, feat_names, slot_labels
        )
        sm = _shap_abs_mean_test(model, X_test_flat)
        _accumulate_flat_importance(
            shap_roi, shap_attr, sm, keep_final, feat_names, slot_labels
        )

        if len(test_idx) > best_shap_n:
            best_shap_n = len(test_idx)
            best_shap = (model, X_test_flat.copy(), keep_final.copy())

        acc, auc, f1 = _test_metrics_xgb(model, X_test_flat, y[test_idx])
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
        OUT_DIR / "exp1_xgb_confusion_oof.pdf",
        title="XGBoost — matriz de confusão agregada (predições OOF, 5-fold SGK)",
    )
    _plot_roc_pr_cv_pdf(
        y_splits,
        score_splits,
        OUT_DIR / "exp1_xgb_roc_cv.pdf",
        OUT_DIR / "exp1_xgb_pr_cv.pdf",
        title_prefix="XGBoost",
    )
    _plot_metrics_box_pdf(
        acc_a,
        auc_a,
        f1_a,
        OUT_DIR / "exp1_xgb_metrics_box_cv.pdf",
        title="XGBoost — distribuição das métricas no teste (5 folds)",
    )

    n_f = float(len(acc_folds))
    gain_roi_m = {k: v / n_f for k, v in gain_roi.items()}
    gain_attr_m = {k: v / n_f for k, v in gain_attr.items()}
    shap_roi_m = {k: v / n_f for k, v in shap_roi.items()}
    shap_attr_m = {k: v / n_f for k, v in shap_attr.items()}
    _plot_top_bars_pdf(
        gain_roi_m,
        OUT_DIR / "exp1_xgb_gain_top_roi.pdf",
        title="XGBoost — gain agregado por ROI (média dos 5 folds; gain normalizado por fold)",
        top_k=TOP_K_ROI,
        xlabel="Soma média do gain normalizado por fold",
    )
    _plot_top_bars_pdf(
        gain_attr_m,
        OUT_DIR / "exp1_xgb_gain_top_attr.pdf",
        title="XGBoost — gain agregado por atributo",
        top_k=TOP_K_ATTR,
        xlabel="Soma média do gain normalizado por fold",
    )
    _plot_top_bars_pdf(
        shap_roi_m,
        OUT_DIR / "exp1_xgb_shap_top_roi.pdf",
        title="XGBoost — |SHAP| médio no teste, agregado por ROI (média dos 5 folds)",
        top_k=TOP_K_ROI,
        xlabel="Média da média |SHAP| por coluna (por fold)",
    )
    _plot_top_bars_pdf(
        shap_attr_m,
        OUT_DIR / "exp1_xgb_shap_top_attr.pdf",
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
            roi = _roi_from_slot_label(slot_labels[slot])
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
        _save_pdf(fig_sh, OUT_DIR / "exp1_xgb_shap_summary.pdf")


if __name__ == "__main__":
    main()

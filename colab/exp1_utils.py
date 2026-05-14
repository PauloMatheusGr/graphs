"""Funções partilhadas entre exp1_xgboost.py, exp1_rocket.py, exp1_svm.py (dados, CV, plots)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
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


def parse_feature_columns(exp_md_path: Path) -> list[str]:
    text = exp_md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("As colunas de atributos são"):
            return line.split("são", 1)[1].strip().split()
    raise RuntimeError(f"{exp_md_path}: linha 'As colunas de atributos são' não encontrada.")


def encode_sex_column(col: pd.Series) -> np.ndarray:
    """F=0, M=1 como float64 para a matriz de atributos."""
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
    cols: list[np.ndarray] = []
    for name in feat_names:
        if name == "SEX":
            cols.append(encode_sex_column(block[name]))
        else:
            arr = pd.to_numeric(block[name], errors="coerce")
            if arr.isna().any():
                raise ValueError(f"Valores não numéricos na coluna de atributo {name!r}.")
            cols.append(arr.to_numpy(dtype=np.float64, copy=True))
    return np.column_stack(cols)


def load_tensor(
    csv_path: Path,
    exp_md_path: Path,
    pair_order: list[str],
    group_key: list[str],
    *,
    require_sex: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    feat_names = parse_feature_columns(exp_md_path)
    df = pd.read_csv(csv_path)
    df["ID_PT"] = df["ID_PT"].astype(str)
    df["pair"] = df["pair"].astype(str).str.strip()
    y_map = {"sMCI": 0, "pMCI": 1}
    df["y"] = df["GROUP"].astype(str).map(y_map)
    if df["y"].isna().any():
        raise ValueError("GROUP contém valores fora de sMCI/pMCI.")
    has_sex = "SEX" in df.columns
    if require_sex and not has_sex:
        raise ValueError("Coluna SEX ausente no CSV (necessária com require_sex=True).")

    feat_names = [c for c in feat_names if c in df.columns]
    if not feat_names:
        raise ValueError("Nenhuma coluna de atributos do .md presente no CSV.")

    samples: list[np.ndarray] = []
    y_out: list[float] = []
    groups: list[str] = []
    sex_out: list[int] = []
    slot_labels_ref: list[str] | None = None

    for keys, g in df.groupby(group_key, sort=False):
        rows = []
        for p in pair_order:
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
            sex_out.append(int(encode_sex_column(block["SEX"])[0]))
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
    random_state: int = 42,
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
        random_state=random_state + 1 + int(fold_id),
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
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
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
        random_state=random_state + 1 + int(fold_id),
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


def _corr_var_scale_panels(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
    *,
    corr_thr: float,
    var_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Painéis 3D (n, 60, n_cols) após correlação, variância e StandardScaler (fit só em idx_fit)."""
    idx_fit = np.asarray(idx_fit, dtype=int)
    idx_va = np.asarray(idx_va, dtype=int)
    X_fit_flat = X_3d[idx_fit].reshape(-1, X_3d.shape[2])
    keep_corr = corr_keep_indices(X_fit_flat, corr_thr)
    X_trf_c = X_fit_flat[:, keep_corr]
    vt = VarianceThreshold(threshold=var_thr)
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

    return scale(X_fit_3d), scale(X_va_3d)


def flat_scaled_tabular_train_val(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
    *,
    corr_thr: float,
    var_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_fit_3d, X_va_3d = _corr_var_scale_panels(
        X_3d, idx_fit, idx_va, corr_thr=corr_thr, var_thr=var_thr
    )
    idx_fit = np.asarray(idx_fit, dtype=int)
    idx_va = np.asarray(idx_va, dtype=int)
    return X_fit_3d.reshape(len(idx_fit), -1), X_va_3d.reshape(len(idx_va), -1)


def prepare_scaled_rocket_inputs(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
    *,
    corr_thr: float,
    var_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_fit, X_val = _corr_var_scale_panels(
        X_3d, idx_fit, idx_va, corr_thr=corr_thr, var_thr=var_thr
    )
    return np.transpose(X_train_fit, (0, 2, 1)), np.transpose(X_val, (0, 2, 1))


def binary_metrics_from_proba(
    y_te: np.ndarray, y_pred: np.ndarray, proba_pos: np.ndarray
) -> tuple[float, float, float]:
    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_te, proba_pos))
    except ValueError:
        auc = float("nan")
    return acc, auc, f1


def roi_from_slot_label(slot_lab: str) -> str:
    parts = slot_lab.split("|")
    return parts[1] if len(parts) > 1 else slot_lab


def accumulate_flat_importance(
    agg_roi: dict[str, float],
    agg_attr: dict[str, float],
    vec_flat: np.ndarray,
    keep_final: np.ndarray,
    feat_names: list[str],
    slot_labels: list[str],
) -> None:
    ncols = len(keep_final)
    n_slots = len(slot_labels)
    for fi in range(min(len(vec_flat), n_slots * ncols)):
        slot = fi // ncols
        j = fi % ncols
        v = float(vec_flat[fi])
        roi = roi_from_slot_label(slot_labels[slot])
        attr = feat_names[int(keep_final[j])]
        agg_roi[roi] = agg_roi.get(roi, 0.0) + v
        agg_attr[attr] = agg_attr.get(attr, 0.0) + v


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_oof_pdf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str = "Blues",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        row = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm.astype(np.float64), np.maximum(row, 1))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    im0 = ax0.imshow(cm, interpolation="nearest", cmap=cmap)
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
    im1 = ax1.imshow(cmn, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
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
    save_pdf(fig, path)


def interp_tpr(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, fpr, tpr, left=0.0, right=1.0)


def interp_pr(
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


def plot_roc_pr_cv_pdf(
    y_splits: list[np.ndarray],
    score_splits: list[np.ndarray],
    out_roc: Path,
    out_pr: Path,
    *,
    title_prefix: str,
    fpr_grid: np.ndarray,
    rec_grid: np.ndarray,
    roc_scope_label: str = "teste por fold",
    pr_scope_label: str = "teste por fold",
) -> None:
    tpr_rows: list[np.ndarray] = []
    prec_rows: list[np.ndarray] = []
    aucs: list[float] = []
    pr_aucs: list[float] = []
    for y_te, sc in zip(y_splits, score_splits):
        if len(np.unique(y_te)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_te, sc)
        tpr_rows.append(interp_tpr(fpr, tpr, fpr_grid))
        aucs.append(roc_auc_score(y_te, sc))
        pi, _ = interp_pr(y_te, sc, rec_grid)
        prec_rows.append(pi)
        pr_aucs.append(average_precision_score(y_te, sc))
    if not tpr_rows:
        return
    tpr_m = np.mean(tpr_rows, axis=0)
    tpr_s = np.std(tpr_rows, axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for row in tpr_rows:
        ax.plot(fpr_grid, row, color="0.6", alpha=0.45, linewidth=0.9)
    ax.plot(fpr_grid, tpr_m, color="C0", linewidth=2.2, label="média TPR")
    ax.fill_between(
        fpr_grid,
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
    ax.set_title(
        f"{title_prefix} — ROC ({roc_scope_label})\nAUC médio={np.mean(aucs):.3f} ± {np.std(aucs, ddof=0):.3f}"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_pdf(fig, out_roc)

    prec_m = np.mean(prec_rows, axis=0)
    prec_s = np.std(prec_rows, axis=0, ddof=0)
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    for row in prec_rows:
        ax2.plot(rec_grid, row, color="0.6", alpha=0.45, linewidth=0.9)
    ax2.plot(rec_grid, prec_m, color="C3", linewidth=2.2, label="média precision")
    ax2.fill_between(
        rec_grid,
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
        f"{title_prefix} — PR ({pr_scope_label})\nAP médio={np.mean(pr_aucs):.3f} ± {np.std(pr_aucs, ddof=0):.3f}"
    )
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    save_pdf(fig2, out_pr)


def plot_metrics_box_pdf(
    acc: np.ndarray,
    auc_a: np.ndarray,
    f1_a: np.ndarray,
    path: Path,
    *,
    title: str,
    xtick_labels: tuple[str, str, str] | None = None,
) -> None:
    labels = xtick_labels if xtick_labels is not None else ("Acc", "AUC", "F1")
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
    ax.set_xticklabels(list(labels))
    ax.set_ylabel("Valor")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_pdf(fig, path)


def plot_top_bars_pdf(
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
    save_pdf(fig, path)


def exp1_run_dir(
    colab_dir: Path, *, downsample_group_sex: bool, model_slug: str
) -> Path:
    """colab/exp1/{balanced|unbalanced}/{model_slug}/ com subpastas figures/ e tables/."""
    scenario = "balanced" if downsample_group_sex else "unbalanced"
    root = colab_dir / "exp1" / scenario / model_slug
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    return root


def write_run_meta_json(
    run_dir: Path,
    *,
    model_slug: str,
    downsample_group_sex: bool,
    duration_seconds: float,
    extra: dict | None = None,
) -> None:
    meta: dict = {
        "model_slug": model_slug,
        "downsample_group_sex": downsample_group_sex,
        "duration_seconds": round(duration_seconds, 3),
        "finished_unix": time.time(),
    }
    if extra:
        meta.update(extra)
    p = run_dir / "run_meta.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_metrics_per_fold_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_oof_predictions_csv(
    path: Path,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    outer_fold: np.ndarray,
    group_id: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "row_idx": row_idx,
            "y_true": y_true,
            "y_pred": y_pred,
            "score": score,
            "outer_fold": outer_fold,
            "group_id": group_id,
        }
    )
    df.to_csv(path, index=False)


def save_fold_test_scores_csv(
    path: Path, outer_fold: np.ndarray, y_true: np.ndarray, score: np.ndarray
) -> None:
    """Uma linha por amostra no conjunto de teste do fold externo (para ROC/PR por fold)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"outer_fold": outer_fold, "y_true": y_true, "score": score}
    ).to_csv(path, index=False)


def save_importance_long_csv(path: Path, names: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names, "value": values}).to_csv(path, index=False)


def save_training_curve_csv(
    path: Path, x: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {"x": np.asarray(x)}
    for k, v in columns.items():
        d[k] = np.asarray(v)
    pd.DataFrame(d).to_csv(path, index=False)


def load_fold_test_scores_for_plots(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Lê fold_test_scores.csv → listas y e score por valor distinto de outer_fold."""
    df = pd.read_csv(path)
    y_splits: list[np.ndarray] = []
    score_splits: list[np.ndarray] = []
    for f in sorted(df["outer_fold"].unique()):
        sub = df[df["outer_fold"] == f]
        y_splits.append(sub["y_true"].to_numpy(dtype=np.int32))
        score_splits.append(sub["score"].to_numpy(dtype=np.float64))
    return y_splits, score_splits


def load_metrics_per_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return (
        df["acc"].to_numpy(dtype=np.float64),
        df["auc"].to_numpy(dtype=np.float64),
        df["f1"].to_numpy(dtype=np.float64),
    )


def load_oof_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(dtype=np.int32), df["y_pred"].to_numpy(dtype=np.int32)

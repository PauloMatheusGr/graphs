"""Funções partilhadas entre exp1 e exp2 (xgboost, rocket, svm, lstm): dados, CV, plots."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def env_bool(key: str, default: bool) -> bool:
    """Lê variável de ambiente booleana (1/true/yes). Usado por run_exp2_all.py e scripts exp1/exp2."""
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes")


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


# pair (deltas i2-i1, i3-i1, i3-i2) -> coluna de intervalo em meses no CSV
PAIR_DT_COLUMN: dict[str, str] = {"12": "t12", "13": "t13", "23": "t23"}


def apply_temporal_rate_norm(
    X: np.ndarray,
    block: pd.DataFrame,
    feat_names: list[str],
    *,
    dt_epsilon: float = 0.5,
) -> np.ndarray:
    """Converte deltas em taxa por mês: x' = x / max(dt, dt_epsilon); SEX não é escalado."""
    if "pair" not in block.columns:
        raise ValueError("Coluna pair ausente (necessária para ponderação temporal).")
    missing_t = [c for c in PAIR_DT_COLUMN.values() if c not in block.columns]
    if missing_t:
        raise ValueError(f"Colunas de tempo ausentes: {missing_t}")

    X_out = np.array(X, dtype=np.float64, copy=True)
    pairs = block["pair"].astype(str).str.strip().to_numpy()
    scale_cols = [j for j, name in enumerate(feat_names) if name != "SEX"]
    if not scale_cols:
        return X_out

    eps = float(dt_epsilon)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError(f"dt_epsilon deve ser finito e > 0; recebido {dt_epsilon!r}.")

    for i, p in enumerate(pairs):
        dt_col = PAIR_DT_COLUMN.get(str(p))
        if dt_col is None:
            raise ValueError(
                f"pair={p!r} sem mapeamento temporal (esperado um de {list(PAIR_DT_COLUMN)})."
            )
        dt = float(pd.to_numeric(block[dt_col].iloc[i], errors="coerce"))
        if not np.isfinite(dt):
            raise ValueError(f"Valor não finito em {dt_col!r} na linha {i} (pair={p!r}).")
        denom = max(dt, eps)
        X_out[i, scale_cols] /= denom
    return X_out


def apply_temporal_baseline_rate(
    X: np.ndarray,
    block: pd.DataFrame,
    feat_names: list[str],
    pair_order: list[str],
    *,
    dt_epsilon: float = 0.5,
) -> np.ndarray:
    """Exp2: pair=1 mantém absolutos; pair=2/3 -> (x - x_baseline) / max(t12|t13, eps)."""
    if len(pair_order) != 3:
        raise ValueError(
            f"baseline_rate exige 3 passos temporais; pair_order={pair_order!r}."
        )
    for c in ("t12", "t13"):
        if c not in block.columns:
            raise ValueError(f"Coluna {c!r} ausente (necessária para baseline_rate).")

    n_rows = len(block)
    n_per = n_rows // 3
    if n_per * 3 != n_rows:
        raise ValueError(f"Bloco com {n_rows} linhas; esperado múltiplo de 3.")

    eps = float(dt_epsilon)
    if eps <= 0.0 or not np.isfinite(eps):
        raise ValueError(f"dt_epsilon deve ser finito e > 0; recebido {dt_epsilon!r}.")

    X_orig = np.asarray(X, dtype=np.float64)
    X_out = X_orig.copy()
    scale_cols = [j for j, name in enumerate(feat_names) if name != "SEX"]
    if not scale_cols:
        return X_out

    for slot in range(n_per):
        i_base = slot
        i2 = n_per + slot
        i3 = 2 * n_per + slot
        dt12 = float(pd.to_numeric(block["t12"].iloc[i2], errors="coerce"))
        dt13 = float(pd.to_numeric(block["t13"].iloc[i3], errors="coerce"))
        if not np.isfinite(dt12) or not np.isfinite(dt13):
            raise ValueError(f"t12/t13 não finitos no slot ROI {slot}.")
        d12 = max(dt12, eps)
        d13 = max(dt13, eps)
        base = X_orig[i_base, scale_cols]
        X_out[i2, scale_cols] = (X_orig[i2, scale_cols] - base) / d12
        X_out[i3, scale_cols] = (X_orig[i3, scale_cols] - base) / d13
    return X_out


def load_tensor(
    csv_path: Path,
    exp_md_path: Path,
    pair_order: list[str],
    group_key: list[str],
    *,
    require_sex: bool = False,
    temporal_mode: str = "none",
    temporal_rate_norm: bool | None = None,
    dt_epsilon: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """temporal_mode: none | delta_rate (exp1) | baseline_rate (exp2)."""
    df = pd.read_csv(csv_path)
    return load_tensor_from_dataframe(
        df,
        exp_md_path,
        pair_order,
        group_key,
        require_sex=require_sex,
        temporal_mode=temporal_mode,
        temporal_rate_norm=temporal_rate_norm,
        dt_epsilon=dt_epsilon,
    )


def load_tensor_from_dataframe(
    df: pd.DataFrame,
    exp_md_path: Path,
    pair_order: list[str],
    group_key: list[str],
    *,
    require_sex: bool = False,
    temporal_mode: str = "none",
    temporal_rate_norm: bool | None = None,
    dt_epsilon: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Como load_tensor, mas a partir de um DataFrame já em memória."""
    if temporal_rate_norm is not None:
        temporal_mode = "delta_rate" if temporal_rate_norm else "none"
    if temporal_mode not in ("none", "delta_rate", "baseline_rate"):
        raise ValueError(
            f"temporal_mode inválido: {temporal_mode!r} "
            "(esperado none, delta_rate ou baseline_rate)."
        )

    feat_names = parse_feature_columns(exp_md_path)
    df = df.copy()
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
        mat = block_to_feature_float_matrix(block, feat_names)
        if temporal_mode == "delta_rate":
            mat = apply_temporal_rate_norm(
                mat, block, feat_names, dt_epsilon=dt_epsilon
            )
        elif temporal_mode == "baseline_rate":
            mat = apply_temporal_baseline_rate(
                mat, block, feat_names, pair_order, dt_epsilon=dt_epsilon
            )
        samples.append(mat)
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


# 60 linhas do tensor = 3 passos temporais × 20 ROIs (ordem do load_tensor).
PANEL_SEQ_STEPS = 3


def panels_to_seq(X_3d: np.ndarray) -> np.ndarray:
    """(n, 60, f) → (n, 3, 20·f): concatena ROIs por passo temporal (pair)."""
    n, n_slots, f = X_3d.shape
    if n_slots % PANEL_SEQ_STEPS != 0:
        raise ValueError(
            f"Tensor com {n_slots} linhas; esperado múltiplo de {PANEL_SEQ_STEPS}."
        )
    n_per = n_slots // PANEL_SEQ_STEPS
    return X_3d.reshape(n, PANEL_SEQ_STEPS, n_per, f).reshape(n, PANEL_SEQ_STEPS, n_per * f)


def seq_scaled_train_val(
    X_3d: np.ndarray,
    idx_fit: np.ndarray,
    idx_va: np.ndarray,
    *,
    corr_thr: float,
    var_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NCV interno LSTM: correlação + variância + scaler no idx_fit; devolve sequências (n, 3, ·)."""
    X_fit_3d, X_va_3d = _corr_var_scale_panels(
        X_3d, idx_fit, idx_va, corr_thr=corr_thr, var_thr=var_thr
    )
    return panels_to_seq(X_fit_3d), panels_to_seq(X_va_3d)


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
) -> tuple[float, float, float, float]:
    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_te, proba_pos))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_te, proba_pos))
    except ValueError:
        ap = float("nan")
    return acc, auc, f1, ap


def roi_from_slot_label(slot_lab: str) -> str:
    parts = slot_lab.split("|")
    return parts[1] if len(parts) > 1 else slot_lab


def unique_rois_from_slot_labels(slot_labels: list[str]) -> list[str]:
    return sorted({roi_from_slot_label(s) for s in slot_labels})


def slot_indices_for_rois(
    slot_labels: list[str], rois_to_drop: set[str] | frozenset[str]
) -> list[int]:
    return [i for i, s in enumerate(slot_labels) if roi_from_slot_label(s) in rois_to_drop]


def mask_rois_in_X_3d(
    X_3d: np.ndarray,
    slot_labels: list[str],
    rois_to_drop: list[str] | set[str] | frozenset[str],
) -> np.ndarray:
    """Zera slots cujo roi está em rois_to_drop (3 pares × lados/labels dessa roi)."""
    drop = {str(r).strip() for r in rois_to_drop if str(r).strip()}
    if not drop:
        return X_3d
    idx = slot_indices_for_rois(slot_labels, drop)
    if not idx:
        return X_3d
    X = np.array(X_3d, copy=True)
    X[:, idx, :] = 0.0
    return X


def resolve_exp1_run_dir(
    colab_dir: Path,
    *,
    downsample_group_sex: bool,
    model_slug: str,
    run_dir_override: Path | str | None = None,
    create_checkpoints: bool = True,
    run_neurocombat: bool = False,
) -> Path:
    """RUN_DIR explícito ou colab/exp1/{scenario}/{model_slug}/."""
    if run_neurocombat:
        model_slug = f"{model_slug}_neurocombat"
    if run_dir_override is not None:
        root = Path(run_dir_override)
        (root / "figures").mkdir(parents=True, exist_ok=True)
        (root / "tables").mkdir(parents=True, exist_ok=True)
        if create_checkpoints:
            (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root
    return exp_run_dir(
        colab_dir,
        exp_name="exp1",
        downsample_group_sex=downsample_group_sex,
        model_slug=model_slug,
        create_checkpoints=create_checkpoints,
    )


def resolve_exp2_run_dir(
    colab_dir: Path,
    *,
    downsample_group_sex: bool,
    model_slug: str,
    run_dir_override: Path | str | None = None,
    create_checkpoints: bool = True,
    run_neurocombat: bool = False,
) -> Path:
    """RUN_DIR explícito ou colab/exp2/{scenario}/{model_slug}/."""
    if run_neurocombat:
        model_slug = f"{model_slug}_neurocombat"
    if run_dir_override is not None:
        root = Path(run_dir_override)
        (root / "figures").mkdir(parents=True, exist_ok=True)
        (root / "tables").mkdir(parents=True, exist_ok=True)
        if create_checkpoints:
            (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root
    return exp_run_dir(
        colab_dir,
        exp_name="exp2",
        downsample_group_sex=downsample_group_sex,
        model_slug=model_slug,
        create_checkpoints=create_checkpoints,
    )


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


def _cm_text_color(value: float, vmin: float, vmax: float, cmap_name: str) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    rgba = plt.get_cmap(cmap_name)(t)
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.45 else "black"


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
    vmax0 = max(int(cm.max()), 1)
    im0 = ax0.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax0)
    ax0.set_title("Contagens (OOF)")
    for (i, j), v in np.ndenumerate(cm):
        ax0.text(
            j,
            i,
            int(v),
            ha="center",
            va="center",
            color=_cm_text_color(float(v), 0.0, float(vmax0), cmap),
            fontsize=10,
            fontweight="bold",
        )
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
        ax1.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color=_cm_text_color(float(v), 0.0, 1.0, cmap),
            fontsize=10,
            fontweight="bold",
        )
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
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
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
    xtick_labels: tuple[str, ...] | None = None,
    ap: np.ndarray | None = None,
) -> None:
    series: list[np.ndarray] = [
        np.asarray(acc, dtype=np.float64),
        np.asarray(auc_a, dtype=np.float64),
        np.asarray(f1_a, dtype=np.float64),
    ]
    default_labels: list[str] = ["Acc", "AUC", "F1"]
    if ap is not None and len(ap) > 0:
        series.append(np.asarray(ap, dtype=np.float64))
        default_labels.append("AP")
    if xtick_labels is not None:
        labels = list(xtick_labels[: len(series)])
    else:
        labels = default_labels[: len(series)]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(series)), 4))
    positions = list(range(1, len(series) + 1))
    plot_series: list[np.ndarray] = []
    for i, arr in enumerate(series):
        arr_s = arr.copy()
        if i == 1 and np.isnan(arr_s).any():
            fill = float(np.nanmean(arr_s))
            arr_s = np.where(np.isnan(arr_s), fill, arr_s)
        plot_series.append(arr_s)
    bp = ax.boxplot(
        plot_series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.85")
    rng = np.random.default_rng(42)
    for i, (arr, arr_plot) in enumerate(zip(series, plot_series)):
        arr_s = np.asarray(arr, dtype=np.float64)
        if i == 1 and np.isnan(arr_s).any():
            arr_s = np.where(np.isnan(arr_s), float(np.nanmean(arr_s)), arr_s)
        x = positions[i] + 0.08 * rng.standard_normal(len(arr_s))
        ax.scatter(x, arr_plot, color="C0", s=28, zorder=3, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
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


def exp_run_dir(
    colab_dir: Path,
    *,
    exp_name: str,
    downsample_group_sex: bool,
    model_slug: str,
    create_checkpoints: bool = False,
) -> Path:
    """colab/{exp_name}/{balanced|unbalanced}/{model_slug}/ com figures/ e tables/."""
    scenario = "balanced" if downsample_group_sex else "unbalanced"
    root = colab_dir / exp_name / scenario / model_slug
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    if create_checkpoints:
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root


def exp1_run_dir(
    colab_dir: Path, *, downsample_group_sex: bool, model_slug: str
) -> Path:
    return exp_run_dir(
        colab_dir,
        exp_name="exp1",
        downsample_group_sex=downsample_group_sex,
        model_slug=model_slug,
    )


def exp2_run_dir(
    colab_dir: Path, *, downsample_group_sex: bool, model_slug: str
) -> Path:
    return exp_run_dir(
        colab_dir,
        exp_name="exp2",
        downsample_group_sex=downsample_group_sex,
        model_slug=model_slug,
        create_checkpoints=True,
    )


def fold_checkpoint_dir(run_dir: Path, fold_id: int) -> Path:
    p = run_dir / "checkpoints" / f"fold_{int(fold_id)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_checkpoint_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_preprocess_bundle(
    path: Path, *, scaler: StandardScaler, keep_final: np.ndarray
) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "keep_final": np.asarray(keep_final, dtype=int),
        },
        path,
    )


def load_preprocess_bundle(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def save_fold_best_params_json(path: Path, fold_params: list[dict]) -> None:
    """Persiste best_params por fold (fallback quando checkpoints LSTM ausentes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "folds": [
            {"fold": int(item["fold"]), "best_params": dict(item["best_params"])}
            for item in fold_params
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline_fold_params(baseline_run_dir: Path, fold_id: int) -> dict:
    """best_params do baseline: checkpoints/fold_k/meta.json ou tables/fold_best_params.json."""
    meta_path = baseline_run_dir / "checkpoints" / f"fold_{int(fold_id)}" / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bp = meta.get("best_params")
        if isinstance(bp, dict) and bp:
            return dict(bp)
        raise ValueError(f"best_params inválido em {meta_path}")

    table_path = baseline_run_dir / "tables" / "fold_best_params.json"
    if table_path.is_file():
        data = json.loads(table_path.read_text(encoding="utf-8"))
        for item in data.get("folds", []):
            if int(item.get("fold", -1)) == int(fold_id):
                bp = item.get("best_params")
                if isinstance(bp, dict) and bp:
                    return dict(bp)
        raise ValueError(f"fold {fold_id} ausente em {table_path}")

    raise FileNotFoundError(
        f"Sem hiperparâmetros para fold {fold_id} em {baseline_run_dir} "
        "(checkpoints/meta.json ou tables/fold_best_params.json)."
    )


def save_xgb_fold_checkpoint(
    run_dir: Path,
    fold_id: int,
    model: Any,
    *,
    scaler: StandardScaler,
    keep_final: np.ndarray,
    best_val_auc: float,
    best_params: dict,
    extra_meta: dict | None = None,
) -> Path:
    ckpt = fold_checkpoint_dir(run_dir, fold_id)
    model.get_booster().save_model(str(ckpt / "model.json"))
    save_preprocess_bundle(ckpt / "preprocess.joblib", scaler=scaler, keep_final=keep_final)
    meta: dict = {
        "model_type": "xgboost",
        "fold": int(fold_id),
        "selection_metric": "val_auc",
        "best_val_auc": float(best_val_auc),
        "best_params": best_params,
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_checkpoint_meta(ckpt / "meta.json", meta)
    return ckpt


def save_svm_fold_checkpoint(
    run_dir: Path,
    fold_id: int,
    model: Any,
    *,
    scaler: StandardScaler,
    keep_final: np.ndarray,
    best_val_auc: float,
    best_params: dict,
    extra_meta: dict | None = None,
) -> Path:
    import joblib

    ckpt = fold_checkpoint_dir(run_dir, fold_id)
    joblib.dump(model, ckpt / "model.joblib")
    save_preprocess_bundle(ckpt / "preprocess.joblib", scaler=scaler, keep_final=keep_final)
    meta: dict = {
        "model_type": "svm",
        "fold": int(fold_id),
        "selection_metric": "val_auc",
        "best_val_auc": float(best_val_auc),
        "best_params": best_params,
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_checkpoint_meta(ckpt / "meta.json", meta)
    return ckpt


def save_lstm_fold_checkpoint(
    run_dir: Path,
    fold_id: int,
    model: Any,
    *,
    scaler: StandardScaler,
    keep_final: np.ndarray,
    best_val_auc: float,
    best_params: dict,
    seq_len: int,
    n_feat: int,
    extra_meta: dict | None = None,
) -> Path:
    ckpt = fold_checkpoint_dir(run_dir, fold_id)
    model.save(ckpt / "model.keras")
    save_preprocess_bundle(ckpt / "preprocess.joblib", scaler=scaler, keep_final=keep_final)
    meta: dict = {
        "model_type": "lstm",
        "fold": int(fold_id),
        "selection_metric": "val_auc",
        "best_val_auc": float(best_val_auc),
        "best_params": best_params,
        "seq_len": int(seq_len),
        "n_feat": int(n_feat),
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_checkpoint_meta(ckpt / "meta.json", meta)
    return ckpt


def save_rocket_fold_checkpoint(
    run_dir: Path,
    fold_id: int,
    *,
    rocket: Any,
    clf: Any,
    scaler: StandardScaler,
    keep_final: np.ndarray,
    best_val_auc: float,
    best_params: dict,
    extra_meta: dict | None = None,
) -> Path:
    import joblib

    ckpt = fold_checkpoint_dir(run_dir, fold_id)
    joblib.dump({"rocket": rocket, "clf": clf}, ckpt / "pipeline.joblib")
    save_preprocess_bundle(ckpt / "preprocess.joblib", scaler=scaler, keep_final=keep_final)
    meta: dict = {
        "model_type": "rocket",
        "fold": int(fold_id),
        "selection_metric": "val_auc",
        "best_val_auc": float(best_val_auc),
        "best_params": best_params,
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_checkpoint_meta(ckpt / "meta.json", meta)
    return ckpt


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


def collect_xgb_training_curves(
    evals_result: dict[str, Any],
    booster: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, np.ndarray]:
    """Logloss (treino/val) do evals_result + acurácia por árvore em treino e validação."""
    import xgboost as xgb

    logloss_train = np.asarray(evals_result["train"]["logloss"], dtype=np.float64)
    logloss_val = np.asarray(evals_result["val"]["logloss"], dtype=np.float64)
    n_trees = int(len(logloss_val))
    if len(logloss_train) != n_trees:
        n_trees = min(len(logloss_train), len(logloss_val))
        logloss_train = logloss_train[:n_trees]
        logloss_val = logloss_val[:n_trees]

    y_tr = np.asarray(y_train).astype(int, copy=False)
    y_va = np.asarray(y_val).astype(int, copy=False)
    dtr = xgb.DMatrix(X_train, label=y_tr)
    dva = xgb.DMatrix(X_val, label=y_va)
    acc_train = np.empty(n_trees, dtype=np.float64)
    acc_val = np.empty(n_trees, dtype=np.float64)
    for i in range(1, n_trees + 1):
        it = (0, i)
        pred_tr = (booster.predict(dtr, iteration_range=it) >= 0.5).astype(np.int32)
        pred_va = (booster.predict(dva, iteration_range=it) >= 0.5).astype(np.int32)
        acc_train[i - 1] = accuracy_score(y_tr, pred_tr)
        acc_val[i - 1] = accuracy_score(y_va, pred_va)

    x = np.arange(1, n_trees + 1, dtype=np.int32)
    return {
        "x": x,
        "logloss_train": logloss_train,
        "logloss_val": logloss_val,
        "accuracy_train": acc_train,
        "accuracy_val": acc_val,
    }


TRAINING_CURVE_METRIC_KEYS = XGB_TRAINING_CURVE_METRIC_KEYS = (
    "logloss_train",
    "logloss_val",
    "accuracy_train",
    "accuracy_val",
)


def mean_training_curves_stack(
    curves_list: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Média (e dp) das curvas alinhadas ao menor nº de árvores entre folds."""
    if not curves_list:
        raise ValueError("curves_list vazio")
    n_min = min(int(len(c["logloss_val"])) for c in curves_list)
    x = np.arange(1, n_min + 1, dtype=np.int32)
    out: dict[str, np.ndarray] = {"x": x}
    for key in TRAINING_CURVE_METRIC_KEYS:
        stack = np.stack(
            [np.asarray(c[key], dtype=np.float64)[:n_min] for c in curves_list],
            axis=0,
        )
        out[key] = stack.mean(axis=0)
        out[f"{key}_std"] = stack.std(axis=0, ddof=0)
    return out


def keras_history_to_training_curves(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Converte History Keras para logloss/acurácia treino e validação (épocas)."""
    h = {k: list(v) for k, v in history.items()}
    n = len(h.get("loss", h.get("val_loss", [])))
    if n == 0:
        raise ValueError("History Keras vazio")
    out: dict[str, np.ndarray] = {
        "x": np.arange(1, n + 1, dtype=np.int32),
    }
    if "loss" in h:
        out["logloss_train"] = np.asarray(h["loss"], dtype=np.float64)
    elif "val_loss" in h:
        out["logloss_train"] = np.full(n, np.nan, dtype=np.float64)
    if "val_loss" in h:
        out["logloss_val"] = np.asarray(h["val_loss"], dtype=np.float64)
    if "accuracy" in h:
        out["accuracy_train"] = np.asarray(h["accuracy"], dtype=np.float64)
    elif "val_accuracy" in h:
        out["accuracy_train"] = np.full(n, np.nan, dtype=np.float64)
    if "val_accuracy" in h:
        out["accuracy_val"] = np.asarray(h["val_accuracy"], dtype=np.float64)
    missing = [k for k in TRAINING_CURVE_METRIC_KEYS if k not in out]
    if missing:
        raise ValueError(f"History Keras sem métricas para curvas: {missing}")
    for key in TRAINING_CURVE_METRIC_KEYS:
        if len(out[key]) != n:
            raise ValueError(f"Comprimento inconsistente em {key}")
    return out


def collect_logreg_saga_training_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    C: float,
    penalty: str = "l1",
    n_steps: int = 150,
    iterations_per_step: int = 5,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Curvas por passos SAGA (warm_start) — proxy alinhado a LogisticRegression L1/saga."""
    from sklearn.linear_model import LogisticRegression

    y_tr = np.asarray(y_train).astype(int, copy=False)
    y_va = np.asarray(y_val).astype(int, copy=False)
    clf = LogisticRegression(
        penalty=penalty,
        solver="saga",
        C=float(C),
        warm_start=True,
        max_iter=int(iterations_per_step),
        random_state=int(random_state),
        n_jobs=-1,
    )
    ll_tr: list[float] = []
    ll_va: list[float] = []
    acc_tr: list[float] = []
    acc_va: list[float] = []
    for _ in range(int(n_steps)):
        clf.fit(X_train, y_tr)
        p_tr = clf.predict_proba(X_train)[:, 1]
        p_va = clf.predict_proba(X_val)[:, 1]
        ll_tr.append(float(log_loss(y_tr, p_tr)))
        ll_va.append(float(log_loss(y_va, p_va)))
        acc_tr.append(float(accuracy_score(y_tr, (p_tr >= 0.5).astype(np.int32))))
        acc_va.append(float(accuracy_score(y_va, (p_va >= 0.5).astype(np.int32))))
    return {
        "x": np.arange(1, n_steps + 1, dtype=np.int32),
        "logloss_train": np.asarray(ll_tr, dtype=np.float64),
        "logloss_val": np.asarray(ll_va, dtype=np.float64),
        "accuracy_train": np.asarray(acc_tr, dtype=np.float64),
        "accuracy_val": np.asarray(acc_va, dtype=np.float64),
    }


def collect_linear_svc_sgd_training_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    C: float,
    n_epochs: int = 200,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Curvas SGD (hinge) com logloss via sigmoid(decision_function) — proxy de LinearSVC."""
    from sklearn.linear_model import SGDClassifier

    y_tr = np.asarray(y_train).astype(int, copy=False)
    y_va = np.asarray(y_val).astype(int, copy=False)
    n_tr = max(len(y_tr), 1)
    alpha = 1.0 / (float(C) * n_tr)
    clf = SGDClassifier(
        loss="hinge",
        alpha=alpha,
        max_iter=1,
        tol=None,
        random_state=int(random_state),
        learning_rate="optimal",
        early_stopping=False,
    )
    classes = np.array([0, 1], dtype=int)
    ll_tr: list[float] = []
    ll_va: list[float] = []
    acc_tr: list[float] = []
    acc_va: list[float] = []
    for _ in range(int(n_epochs)):
        clf.partial_fit(X_train, y_tr, classes=classes)
        sc_tr = clf.decision_function(X_train)
        sc_va = clf.decision_function(X_val)
        p_tr = 1.0 / (1.0 + np.exp(-np.clip(sc_tr.astype(np.float64), -50.0, 50.0)))
        p_va = 1.0 / (1.0 + np.exp(-np.clip(sc_va.astype(np.float64), -50.0, 50.0)))
        ll_tr.append(float(log_loss(y_tr, p_tr)))
        ll_va.append(float(log_loss(y_va, p_va)))
        acc_tr.append(float(accuracy_score(y_tr, clf.predict(X_train))))
        acc_va.append(float(accuracy_score(y_va, clf.predict(X_val))))
    return {
        "x": np.arange(1, n_epochs + 1, dtype=np.int32),
        "logloss_train": np.asarray(ll_tr, dtype=np.float64),
        "logloss_val": np.asarray(ll_va, dtype=np.float64),
        "accuracy_train": np.asarray(acc_tr, dtype=np.float64),
        "accuracy_val": np.asarray(acc_va, dtype=np.float64),
    }


def plot_xgb_training_curves_pdf(
    curves: dict[str, np.ndarray],
    out_pdf: Path,
    *,
    title: str,
    show_std: bool = False,
    xlabel: str = "passo",
) -> None:
    """Painéis: logloss (treino/val) e acurácia (treino/val)."""
    x = np.asarray(curves["x"], dtype=np.int32)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    for key, color, label in (
        ("logloss_train", "C0", "treino"),
        ("logloss_val", "C1", "validação"),
    ):
        y = np.asarray(curves[key], dtype=np.float64)
        axes[0].plot(x, y, color=color, label=label)
        if show_std and f"{key}_std" in curves:
            s = np.asarray(curves[f"{key}_std"], dtype=np.float64)
            axes[0].fill_between(x, y - s, y + s, color=color, alpha=0.2, linewidth=0)

    axes[0].set_ylabel("logloss")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for key, color, label in (
        ("accuracy_train", "C0", "treino"),
        ("accuracy_val", "C1", "validação"),
    ):
        y = np.asarray(curves[key], dtype=np.float64)
        axes[1].plot(x, y, color=color, label=label)
        if show_std and f"{key}_std" in curves:
            s = np.asarray(curves[f"{key}_std"], dtype=np.float64)
            axes[1].fill_between(x, y - s, y + s, color=color, alpha=0.2, linewidth=0)

    axes[1].set_ylabel("acurácia")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def save_and_plot_training_curves_fold(
    curves: dict[str, np.ndarray],
    *,
    csv_path: Path,
    pdf_path: Path,
    title: str,
    xlabel: str = "passo",
) -> None:
    cols = {k: curves[k] for k in TRAINING_CURVE_METRIC_KEYS}
    save_training_curve_csv(csv_path, curves["x"], cols)
    plot_xgb_training_curves_pdf(curves, pdf_path, title=title, xlabel=xlabel)


save_and_plot_xgb_training_curves_fold = save_and_plot_training_curves_fold


def finalize_supervised_training_curves(
    fold_training_curves: list[dict[str, np.ndarray]],
    tab_dir: Path,
    fig_dir: Path,
    *,
    title_mean: str,
    xlabel: str = "passo",
) -> None:
    """Grava CSV/PDF médios (± dp) a partir das curvas já coletadas por fold."""
    if not fold_training_curves:
        return
    mean_c = mean_training_curves_stack(fold_training_curves)
    mean_cols = {k: mean_c[k] for k in TRAINING_CURVE_METRIC_KEYS}
    for k in TRAINING_CURVE_METRIC_KEYS:
        mean_cols[f"{k}_std"] = mean_c[f"{k}_std"]
    save_training_curve_csv(tab_dir / "training_curves_mean.csv", mean_c["x"], mean_cols)
    plot_xgb_training_curves_pdf(
        mean_c,
        fig_dir / "training_curves_mean.pdf",
        title=title_mean,
        show_std=True,
        xlabel=xlabel,
    )
    plot_xgb_training_curves_pdf(
        mean_c,
        fig_dir / "training_curves.pdf",
        title=title_mean,
        show_std=True,
        xlabel=xlabel,
    )


def regenerate_supervised_training_curve_plots(
    tab_dir: Path,
    fig_dir: Path,
    *,
    n_folds: int = 5,
    title_fold_tpl: str = "Fold {k}/{n} — treino vs validação (holdout tr_fit|val)",
    title_mean: str = "Média dos folds externos — treino vs validação",
    xlabel: str = "passo",
) -> None:
    """Regenera PDFs por fold e média a partir de training_curves_fold*.csv."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[dict[str, np.ndarray]] = []
    for fold_id in range(n_folds):
        p = tab_dir / f"training_curves_fold{fold_id}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if not all(k in df.columns for k in TRAINING_CURVE_METRIC_KEYS):
            continue
        curves = {
            "x": df["x"].to_numpy(dtype=np.int32),
            **{
                k: df[k].to_numpy(dtype=np.float64)
                for k in TRAINING_CURVE_METRIC_KEYS
            },
        }
        plot_xgb_training_curves_pdf(
            curves,
            fig_dir / f"training_curves_fold{fold_id}.pdf",
            title=title_fold_tpl.format(k=fold_id + 1, n=n_folds),
            xlabel=xlabel,
        )
        loaded.append(curves)
    if not loaded:
        return
    finalize_supervised_training_curves(
        loaded,
        tab_dir,
        fig_dir,
        title_mean=title_mean,
        xlabel=xlabel,
    )


regenerate_xgb_training_curve_plots = regenerate_supervised_training_curve_plots


def save_feature_counts_fold0_csv(
    path: Path,
    *,
    n_raw: int,
    n_after_corr: int,
    n_after_variance: int,
    stage_labels: tuple[str, str, str] | None = None,
) -> None:
    """Contagens no fold externo 0 (tr_fit), alinhadas ao gráfico de barras dos scripts exp1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labs = stage_labels or ("Raw", "Após correlação", "Após variância")
    pd.DataFrame(
        {
            "stage": list(labs),
            "n_features": [int(n_raw), int(n_after_corr), int(n_after_variance)],
        }
    ).to_csv(path, index=False)


def plot_feature_counts_bar_pdf(
    csv_path: Path,
    out_pdf: Path,
    *,
    title: str,
    ylabel: str = "Nº atributos",
) -> None:
    """Lê CSV gravado por save_feature_counts_fold0_csv e gera PDF de barras."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    ax.bar(df["stage"].astype(str), df["n_features"].astype(int))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


def plot_training_curves_keras_pdf(
    csv_path: Path,
    out_pdf: Path,
    *,
    title: str,
) -> None:
    """Curvas de treino Keras a partir de training_curves_fold0.csv (colunas loss / val_*)."""
    df = pd.read_csv(csv_path)
    x = df["x"].to_numpy(dtype=np.int32)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    if "loss" in df.columns and "val_loss" in df.columns:
        axes[0].plot(x, df["loss"], label="treino")
        axes[0].plot(x, df["val_loss"], label="validação")
        axes[0].set_ylabel("loss")
        axes[0].legend(loc="best", fontsize=8)
    if "val_auc" in df.columns:
        axes[1].plot(x, df["val_auc"], label="val AUC")
        axes[1].set_ylabel("AUC (validação)")
    elif "accuracy" in df.columns and "val_accuracy" in df.columns:
        axes[1].plot(x, df["accuracy"], label="treino")
        axes[1].plot(x, df["val_accuracy"], label="validação")
        axes[1].set_ylabel("acurácia")
    axes[1].set_xlabel("época")
    fig.suptitle(title)
    fig.tight_layout()
    save_pdf(fig, out_pdf)


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


def load_metrics_per_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "ap" in df.columns:
        ap = df["ap"].to_numpy(dtype=np.float64)
    else:
        ap = np.full(len(df), np.nan, dtype=np.float64)
    return (
        df["acc"].to_numpy(dtype=np.float64),
        df["auc"].to_numpy(dtype=np.float64),
        df["f1"].to_numpy(dtype=np.float64),
        ap,
    )


def load_oof_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(dtype=np.int32), df["y_pred"].to_numpy(dtype=np.int32)

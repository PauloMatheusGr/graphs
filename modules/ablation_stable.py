"""Stable pool via bootstrap × L1 (corr/var uma vez no train → L1 em cada boot)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ablation_analysis import estimate_stable_pool_columns

STABLE_POOL_BOOTSTRAP = 50
STABLE_POOL_L1_C = 0.1
L1_COEF_TOL = 1e-9


def _scale_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas com estatísticas só desta amostra (boot ou train)."""
    if X.empty:
        return X.copy()
    arr = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(arr)
    return pd.DataFrame(scaled, columns=list(X.columns), index=X.index)


def _corr_var_names(
    X: pd.DataFrame,
    names: list[str],
    *,
    corr_threshold: float,
    var_threshold: float,
    y: np.ndarray | None = None,
) -> list[str]:
    from ablation_runner import corr_keep_mask, var_keep_mask

    if not names:
        return []
    X_arr = X[names].to_numpy(dtype=float)
    cmask = corr_keep_mask(X_arr, corr_threshold, feature_names=names, y=y)
    names = [n for n, k in zip(names, cmask) if k]
    if not names:
        return []
    X_arr = X_arr[:, cmask]
    vmask = var_keep_mask(X_arr, var_threshold)
    return [n for n, k in zip(names, vmask) if k]


def l1_selected_feature_names(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    C: float = STABLE_POOL_L1_C,
    coef_tol: float = L1_COEF_TOL,
    seed: int = 42,
) -> list[str]:
    names = list(X.columns)
    if not names:
        return []
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return []
    X_arr = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=float(C),
        max_iter=10000,
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(X_arr, y)
    coef = np.ravel(clf.coef_)
    if len(coef) != len(names):
        return []
    return [n for n, c in zip(names, coef) if abs(c) > coef_tol]


def inner_selections_l1_bootstrap(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    l1_c: float = STABLE_POOL_L1_C,
    corr_threshold: float = 0.85,
    var_threshold: float = 0.01,
    seed: int = 42,
) -> tuple[list[list[str]], list[str]]:
    """Corr/var uma vez no outer train; cada boot só L1 no espaço já podado."""
    y_train = np.asarray(y_train, dtype=int)
    n = len(y_train)
    if n < 2 or len(np.unique(y_train)) < 2:
        return [], []
    Xs = _scale_frame(X_train)
    filtered_names = _corr_var_names(
        Xs,
        list(Xs.columns),
        corr_threshold=corr_threshold,
        var_threshold=var_threshold,
        y=y_train,
    )
    if not filtered_names:
        return [], []
    X_filt = X_train[filtered_names]
    rng = np.random.default_rng(seed)
    inner_selected: list[list[str]] = []
    for b in range(int(n_bootstrap)):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_train[idx]
        if len(np.unique(yb)) < 2:
            continue
        Xb = _scale_frame(X_filt.iloc[idx])
        selected = l1_selected_feature_names(
            Xb,
            yb,
            C=l1_c,
            seed=seed + b,
        )
        if selected:
            inner_selected.append(selected)
    return inner_selected, filtered_names


def _max_var_column(X: pd.DataFrame) -> list[str]:
    if X.empty or X.shape[1] == 0:
        return []
    arr = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [str(X.columns[int(np.argmax(np.var(arr, axis=0)))])]


def stable_pool_for_outer_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    selection_mode: str,
    roi: str = "hippocampus",
    min_pct: int = 70,
    min_timepoints: int = 2,
    n_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    l1_c: float = STABLE_POOL_L1_C,
    seed: int = 42,
) -> tuple[list[str], list[str], str]:
    from ablation_runner import SELECTION_MODES

    cfg = SELECTION_MODES[selection_mode]
    cols = list(X_train.columns)
    if cfg.get("use_l1_stable_pool"):
        inner_selected, filtered_names = inner_selections_l1_bootstrap(
            X_train,
            y_train,
            n_bootstrap=n_bootstrap,
            l1_c=l1_c,
            seed=seed,
        )
    elif cfg.get("use_stable_pool"):
        raise ValueError("mrmr_stable removido; use l1_stable")
    else:
        return cols, [], "raw"

    kept, removed = estimate_stable_pool_columns(
        cols,
        inner_selected,
        min_pct=min_pct,
        min_timepoints=min_timepoints,
    )
    if kept:
        return kept, removed, "l1_stable"
    if filtered_names:
        removed = [c for c in cols if c not in set(filtered_names)]
        return list(filtered_names), removed, "corr_fallback"
    var_kept = _max_var_column(X_train)
    removed = [c for c in cols if c not in set(var_kept)]
    return var_kept, removed, "var_fallback"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 60, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"hippocampus_L_T1_f{i}" for i in range(p)])
    y = (rng.random(n) > 0.45).astype(int)
    X.iloc[:, 0] = y * 2.0 + rng.normal(scale=0.1, size=n)
    sel, filt = inner_selections_l1_bootstrap(X, y, n_bootstrap=10, seed=0)
    assert isinstance(sel, list) and isinstance(filt, list)
    kept, removed = estimate_stable_pool_columns(
        list(X.columns), sel, min_pct=50, min_timepoints=0,
    )
    empty_kept, empty_rem = estimate_stable_pool_columns(
        list(X.columns), [], min_pct=70, min_timepoints=0,
    )
    assert empty_kept == []
    assert empty_rem == list(X.columns)
    kept_s, rem_s, src_s = stable_pool_for_outer_train(
        X, y, selection_mode="l1_stable", min_pct=50, min_timepoints=0,
        n_bootstrap=10, seed=0,
    )
    assert kept_s and src_s in ("l1_stable", "corr_fallback", "var_fallback")
    a = rng.normal(size=n)
    Xf = pd.DataFrame(
        {
            "hippocampus_L_T1_original_glcm_Contrast": a,
            "hippocampus_R_T1_original_glcm_Dissimilarity": a + rng.normal(scale=0.01, size=n),
        }
    )
    yf = (rng.random(n) > 0.5).astype(int)
    kept_f, rem_f, src_f = stable_pool_for_outer_train(
        Xf, yf, selection_mode="l1_stable", min_pct=70, min_timepoints=0,
        n_bootstrap=8, l1_c=1e-12, seed=0,
    )
    assert src_f in ("corr_fallback", "var_fallback"), src_f
    assert kept_f
    assert set(kept_f) <= set(Xf.columns)
    if src_f == "corr_fallback":
        assert len(kept_f) < Xf.shape[1]
    print("ablation_stable self-check ok", "n_inner", len(sel), "src", src_s, src_f)

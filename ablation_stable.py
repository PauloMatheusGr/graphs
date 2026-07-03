"""Stable pool via bootstrap × L1 (corr/var antes de cada fit)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ablation_analysis import estimate_stable_pool_columns

if TYPE_CHECKING:
    from ablation_runner import CorrVarMRMRSelector

STABLE_POOL_BOOTSTRAP = 50
STABLE_POOL_L1_C = 0.1
L1_COEF_TOL = 1e-9


def _corr_var_names(
    X: pd.DataFrame,
    names: list[str],
    *,
    corr_threshold: float,
    var_threshold: float,
) -> list[str]:
    from ablation_runner import corr_keep_mask, var_keep_mask

    if not names:
        return []
    X_arr = X[names].to_numpy(dtype=float)
    cmask = corr_keep_mask(X_arr, corr_threshold, feature_names=names)
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
        return names
    selected = [n for n, c in zip(names, coef) if abs(c) > coef_tol]
    return selected or names


def inner_selections_l1_bootstrap(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    l1_c: float = STABLE_POOL_L1_C,
    corr_threshold: float = 0.90,
    var_threshold: float = 0.01,
    seed: int = 42,
) -> list[list[str]]:
    """corr/var → L1 em cada bootstrap do outer train."""
    y_train = np.asarray(y_train, dtype=int)
    n = len(y_train)
    if n < 2 or len(np.unique(y_train)) < 2:
        return []
    rng = np.random.default_rng(seed)
    inner_selected: list[list[str]] = []
    for b in range(int(n_bootstrap)):
        idx = rng.choice(n, size=n, replace=True)
        uniq = np.unique(idx)
        Xb = X_train.iloc[uniq]
        yb = y_train[uniq]
        if len(np.unique(yb)) < 2:
            continue
        names = _corr_var_names(
            Xb,
            list(Xb.columns),
            corr_threshold=corr_threshold,
            var_threshold=var_threshold,
        )
        if not names:
            continue
        selected = l1_selected_feature_names(
            Xb[names],
            yb,
            C=l1_c,
            seed=seed + b,
        )
        if selected:
            inner_selected.append(selected)
    if inner_selected:
        return inner_selected
    # ponytail: fallback 1× L1 no treino inteiro se bootstrap falhar
    names = _corr_var_names(
        X_train,
        list(X_train.columns),
        corr_threshold=corr_threshold,
        var_threshold=var_threshold,
    )
    if names:
        return [l1_selected_feature_names(X_train[names], y_train, C=l1_c, seed=seed)]
    return []


def inner_selections_mrmr_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    inner_cv: StratifiedKFold,
    roi: str,
    n_features: int,
) -> list[list[str]]:
    from ablation_runner import CorrVarMRMRSelector

    inner_selected: list[list[str]] = []
    for tr_idx, _ in inner_cv.split(X_train, y_train):
        pre = CorrVarMRMRSelector(
            use_mrmr=True,
            use_filters=True,
            n_features_total=n_features,
            roi=roi,
        )
        pre.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        inner_selected.append(list(pre.selected_names_))
    return inner_selected


def stable_pool_for_outer_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    selection_mode: str,
    inner_cv: StratifiedKFold | None = None,
    roi: str = "hippocampus",
    min_pct: int = 70,
    min_timepoints: int = 2,
    n_features: int = 50,
    n_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    l1_c: float = STABLE_POOL_L1_C,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    from ablation_runner import SELECTION_MODES

    cfg = SELECTION_MODES[selection_mode]
    if cfg.get("use_l1_stable_pool"):
        inner_selected = inner_selections_l1_bootstrap(
            X_train,
            y_train,
            n_bootstrap=n_bootstrap,
            l1_c=l1_c,
            seed=seed,
        )
    elif cfg.get("use_stable_pool"):
        if inner_cv is None:
            raise ValueError("inner_cv obrigatório para mrmr_stable")
        inner_selected = inner_selections_mrmr_cv(
            X_train,
            y_train,
            inner_cv=inner_cv,
            roi=roi,
            n_features=n_features,
        )
    else:
        return list(X_train.columns), []

    return estimate_stable_pool_columns(
        list(X_train.columns),
        inner_selected,
        min_pct=min_pct,
        min_timepoints=min_timepoints,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 60, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"hippocampus_L_T1_f{i}" for i in range(p)])
    y = (rng.random(n) > 0.45).astype(int)
    sel = inner_selections_l1_bootstrap(X, y, n_bootstrap=10, seed=0)
    assert len(sel) >= 1
    kept, _ = estimate_stable_pool_columns(list(X.columns), sel, min_pct=50, min_timepoints=0)
    assert kept
    print("ablation_stable self-check ok")

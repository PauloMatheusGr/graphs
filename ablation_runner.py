"""Runner de ablação: tarefas binárias entre grupos CN/sMCI/pMCI/AD, 4 modalidades, ComBat opcional por fold."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ablation_harmonize import harmonize_long_fold, image_ids_for_patients
from ablation_analysis import anatomical_key, summary_with_pooled
from ablation_stable import (
    STABLE_POOL_BOOTSTRAP,
    STABLE_POOL_L1_C,
    stable_pool_for_outer_train,
)
from ablation_prep import (
    ROI_FILTER_DEFAULT,
    modality_wide_columns,
    pivot_long_to_wide,
)
from ablation_representation import (
    apply_representation_wide,
    default_results_dir,
    feature_columns_for_representation,
    resolve_stable_pool_min_timepoints,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

try:
    from feature_engine.selection import MRMR

    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")

@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    groups: tuple[str, str]
    label_map: dict[str, int]
    positive: str
    negative: str

    @property
    def group_labels(self) -> dict[int, str]:
        return {v: k for k, v in self.label_map.items()}

    @property
    def label(self) -> str:
        return f"{self.negative} (0) vs {self.positive} (1)"


def make_binary_task(negative: str, positive: str) -> TaskConfig:
    """negative=0, positive=1; task_id ex.: cn_ad, smci_pmci."""
    task_id = f"{negative.lower()}_{positive.lower()}"
    return TaskConfig(
        task_id=task_id,
        groups=(negative, positive),
        label_map={negative: 0, positive: 1},
        positive=positive,
        negative=negative,
    )


# Pares pedidos: within-diagnosis (cn×ad, smci×pmci) + cross (cn×smci, …)
BINARY_TASK_PAIRS: tuple[tuple[str, str], ...] = (
    ("CN", "AD"),
    ("sMCI", "pMCI"),
    ("CN", "sMCI"),
    ("sMCI", "AD"),
    ("CN", "pMCI"),
    ("pMCI", "AD"),
)

TASKS: dict[str, TaskConfig] = {
    tc.task_id: tc for tc in (make_binary_task(n, p) for n, p in BINARY_TASK_PAIRS)
}

TASK_PRESETS: dict[str, tuple[str, ...]] = {
    "core": ("cn_ad", "smci_pmci"),
    "cross": ("cn_smci", "smci_ad", "cn_pmci", "pmci_ad"),
    "all": tuple(TASKS.keys()),
}

MODALITIES: dict[str, dict[str, str]] = {
    "vol": {"long": "vol_long.csv", "wide": "vol_wide.csv", "label": "volumétrico"},
    "shape": {"long": "shape_long.csv", "wide": "shape_wide.csv", "label": "shape"},
    "texture": {"long": "rad_long.csv", "wide": "texture_wide.csv", "label": "textura"},
    "disp": {"long": "disp_long.csv", "wide": "disp_wide.csv", "label": "deslocamento"},
    "all": {
        "long": "merge_long.csv",
        "wide": "all_wide.csv",
        "label": "merge (vol+shape+tex+disp)",
    },
}

SELECTION_MODES = {
    "mrmr": {"use_mrmr": True, "use_filters": True, "label": "corr + var + MRMR"},
    "mrmr_stable": {
        "use_mrmr": True,
        "use_filters": True,
        "use_stable_pool": True,
        "label": "stable pool (mRMR CV) + corr + var + MRMR",
    },
    "l1_stable": {
        "use_mrmr": False,
        "use_filters": True,
        "use_stable_pool": True,
        "use_l1_stable_pool": True,
        "label": "stable pool (L1 bootstrap) + corr + var",
    },
    "filters": {"use_mrmr": False, "use_filters": True, "label": "corr + var (sem MRMR)"},
    "raw": {"use_mrmr": False, "use_filters": False, "label": "sem seleção"},
}

EMBEDDED_MODEL_KEYS = frozenset({"logreg_l1", "elasticnet"})
EMBEDDED_COEF_TOL = 1e-9

# Estimativa do pool estável
STABLE_POOL_MIN_PCT = 70
STABLE_POOL_MIN_TIMEPOINTS = 2
STABLE_POOL_N_FEATURES = 50  # mRMR por inner fold (só mrmr_stable legado)

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "svm": {
        "preselect__n_features_total": [10, 15, 20, 30, 50],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale"],
        "clf__class_weight": [None, "balanced"],
    },
    "rf": {
        "preselect__n_features_total": [10, 15, 20, 30, 50],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__class_weight": [None, "balanced"],
    },
    "xgb": {
        "preselect__n_features_total": [10, 15, 20, 30, 50],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    },
    "mlp": {
        "preselect__n_features_total": [10, 15, 20, 30, 50],
        "clf__hidden_layer_sizes": [(64,), (128, 64)],
        "clf__alpha": [1e-4, 1e-3],
        "clf__max_iter": [2000],
    },
    "logreg_l1": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    },
    "elasticnet": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
        "clf__class_weight": [None, "balanced"],
    },
}


def fold_metrics(y_true, scores, pred, *, pos_label: int = 1) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "sens_pos": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "spec_neg": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_pos": float(f1_score(y_true, pred, pos_label=pos_label, zero_division=0)),
    }


def tune_youden_threshold(y_true, scores) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def corr_keep_mask(
    X: np.ndarray,
    threshold: float = 0.90,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Greedy corr prune. Pares com mesmo anatomical_key (T1/T2/T3) nunca se removem."""
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = xf.shape[1]
    if xf.shape[0] < 2:
        return np.ones(n, dtype=bool)
    if feature_names is not None and len(feature_names) != n:
        feature_names = None
    keys = [anatomical_key(name) for name in feature_names] if feature_names else None

    c = np.corrcoef(xf.T)
    keep_idx: list[int] = []
    for j in range(n):
        drop = False
        for k in keep_idx:
            if not np.isfinite(c[j, k]) or abs(c[j, k]) <= threshold:
                continue
            if keys is not None and keys[j] == keys[k]:
                continue
            drop = True
            break
        if not drop:
            keep_idx.append(j)
    mask = np.zeros(n, dtype=bool)
    mask[keep_idx] = True
    return mask


def var_keep_mask(X: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if xf.shape[0] < 2:
        return np.ones(xf.shape[1], dtype=bool)
    v = np.var(xf, axis=0)
    keep = v >= threshold
    # ponytail: vol ICV-normalizado costuma ter var < 0.01; relaxa só se filtro zera tudo
    if threshold > 0 and not keep.any():
        keep = v > 0
    return keep


def mrmr_global(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    max_features: int,
    random_state: int = 0,
) -> list[str]:
    """mRMR top-K global (relevância - redundância). Mantém ordem de entrada."""
    if not HAS_MRMR:
        raise RuntimeError("feature-engine ausente — mRMR exigido (pip install feature-engine)")
    # feature-engine exige max_features < n_colunas; senão nada a cortar
    if df.shape[1] <= max_features:
        return list(df.columns)
    selector = MRMR(
        method="MIQ",
        regression=False,
        max_features=max_features,
        random_state=random_state,
    )
    with np.errstate(divide="ignore", invalid="ignore"):  # MIQ: relevance/redundance=0 é benigno
        selector.fit(df, y)
    dropped = set(selector.features_to_drop_)  # selecionadas = variables_ - features_to_drop_
    return [c for c in selector.variables_ if c not in dropped]


class CorrVarMRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        use_mrmr: bool = True,
        use_filters: bool = True,
        n_features_total: int = 20,
        corr_threshold: float = 0.90,
        var_threshold: float = 0.01,
        roi: str = ROI_FILTER_DEFAULT,
        allowed_columns: list[str] | None = None,
    ):
        self.use_mrmr = use_mrmr
        self.use_filters = use_filters
        self.n_features_total = n_features_total
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold
        self.roi = roi
        self.allowed_columns = allowed_columns
        self.selected_names_: list[str] = []
        self.selected_indices_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: list[str] = []
        self.removed_by_stable_pool_: list[str] = []
        self.removed_by_filters_: list[str] = []
        self.removed_by_mrmr_: list[str] = []
        self.after_stable_pool_names_: list[str] = []
        self.after_filter_names_: list[str] = []

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y):
        df_in = self._as_frame(X)
        self.feature_names_in_ = list(df_in.columns)
        names = list(self.feature_names_in_)

        if self.allowed_columns is not None:
            allowed = set(self.allowed_columns)
            self.removed_by_stable_pool_ = [n for n in names if n not in allowed]
            names = [n for n in names if n in allowed]
        else:
            self.removed_by_stable_pool_ = []
        self.after_stable_pool_names_ = list(names)

        X_arr = df_in[names].to_numpy(dtype=float) if names else np.empty((len(df_in), 0))

        if self.use_filters and names:
            before = list(names)
            cmask = corr_keep_mask(X_arr, self.corr_threshold, feature_names=names)
            names = [n for n, k in zip(names, cmask) if k]
            X_arr = X_arr[:, cmask]
            vmask = var_keep_mask(X_arr, self.var_threshold)
            names = [n for n, k in zip(names, vmask) if k]
            X_arr = X_arr[:, vmask]
            self.removed_by_filters_ = [n for n in before if n not in set(names)]
        else:
            self.removed_by_filters_ = []
        self.after_filter_names_ = list(names)

        if self.use_mrmr and names:
            df_f = pd.DataFrame(X_arr, columns=names)
            before_mrmr = list(names)
            selected_names = mrmr_global(df_f, y, max_features=self.n_features_total)
            if not selected_names:
                selected_names = names[: min(self.n_features_total, len(names))]
            self.removed_by_mrmr_ = [n for n in before_mrmr if n not in set(selected_names)]
        else:
            selected_names = list(names)
            self.removed_by_mrmr_ = []

        if not selected_names:
            # ponytail: fallback se corr/var/MRMR zerarem (ex. vol após ICV norm)
            selected_names = list(self.feature_names_in_)
            self.removed_by_mrmr_ = []

        name_to_idx = {n: i for i, n in enumerate(self.feature_names_in_)}
        self.selected_names_ = selected_names
        self.selected_indices_ = np.array(
            [name_to_idx[n] for n in selected_names], dtype=int
        )
        return self

    def transform(self, X):
        df_in = self._as_frame(X)
        X_arr = df_in[self.feature_names_in_].to_numpy(dtype=float)
        if self.selected_indices_.size == 0:
            raise ValueError("Nenhuma feature selecionada após pré-filtros.")
        return X_arr[:, self.selected_indices_]


def embedded_selected_names(
    clf: LogisticRegression,
    feature_names: list[str],
    *,
    coef_tol: float = EMBEDDED_COEF_TOL,
) -> list[str]:
    coef = np.ravel(clf.coef_)
    if len(coef) != len(feature_names):
        return list(feature_names)
    return [n for n, c in zip(feature_names, coef) if abs(c) > coef_tol]


def make_classifier(model_key: str, *, seed: int):
    clf_map = {
        "svm": SVC(probability=True, random_state=seed),
        "rf": RandomForestClassifier(random_state=seed),
        "xgb": XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=seed),
        "mlp": MLPClassifier(
            activation="relu", alpha=1e-3, batch_size=32, random_state=seed, max_iter=500
        ),
        "logreg_l1": LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=seed,
        ),
        "elasticnet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=5000,
            random_state=seed,
        ),
    }
    if model_key not in clf_map:
        raise ValueError(f"modelo desconhecido: {model_key!r}")
    return clf_map[model_key]


def is_embedded_model(model_key: str) -> bool:
    return model_key in EMBEDDED_MODEL_KEYS


def gridsearch_n_jobs(model_key: str) -> int:
    return 1 if model_key == "xgb" else -1


def fold_selection_audit(
    best: ImbPipeline,
    X_train: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_key: str,
    selection_mode: str,
) -> dict[str, Any]:
    raw_names = list(feature_cols)
    if is_embedded_model(model_key):
        clf = best.named_steps["clf"]
        selected = embedded_selected_names(clf, raw_names)
        selected_set = set(selected)
        return {
            "after_stable_pool_names_": raw_names,
            "after_filter_names_": raw_names,
            "selected_names_": selected,
            "removed_by_stable_pool_": [],
            "removed_by_filters_": [],
            "removed_by_mrmr_": [n for n in raw_names if n not in selected_set],
            "n_features_selected": len(selected),
        }

    preselect = best.named_steps.get("preselect")
    if preselect is None or selection_mode == "raw":
        return {
            "after_stable_pool_names_": raw_names,
            "after_filter_names_": raw_names,
            "selected_names_": raw_names,
            "removed_by_stable_pool_": [],
            "removed_by_filters_": [],
            "removed_by_mrmr_": [],
            "n_features_selected": len(raw_names),
        }
    return {
        "after_stable_pool_names_": list(preselect.after_stable_pool_names_),
        "after_filter_names_": list(preselect.after_filter_names_),
        "selected_names_": list(preselect.selected_names_),
        "removed_by_stable_pool_": list(preselect.removed_by_stable_pool_),
        "removed_by_filters_": list(preselect.removed_by_filters_),
        "removed_by_mrmr_": list(preselect.removed_by_mrmr_),
        "n_features_selected": int(preselect.transform(X_train).shape[1]),
    }


def make_pipeline(selection_mode: str, model_key: str, *, roi: str, seed: int) -> ImbPipeline:
    cfg = SELECTION_MODES[selection_mode]
    steps: list[tuple[str, Any]] = []
    if not is_embedded_model(model_key):
        steps.append(
            (
                "preselect",
                CorrVarMRMRSelector(
                    use_mrmr=cfg["use_mrmr"],
                    use_filters=cfg["use_filters"],
                    roi=roi,
                ),
            )
        )
    steps.extend(
        [
            ("scaler", StandardScaler()),
            ("clf", make_classifier(model_key, seed=seed)),
        ]
    )
    return ImbPipeline(steps)


def param_grid_for(model_key: str, selection_mode: str) -> dict[str, Any]:
    grid = PARAM_GRIDS[model_key]
    if is_embedded_model(model_key):
        return dict(grid)
    return {
        k: v
        for k, v in grid.items()
        if k != "preselect__n_features_total"
        or selection_mode in ("mrmr", "mrmr_stable")
    }


def patient_labels_from_long(df_long: pd.DataFrame, task: TaskConfig) -> pd.DataFrame:
    sub = df_long[df_long["GROUP"].astype(str).isin(task.groups)].copy()
    pt = (
        sub.groupby("ID_PT", as_index=False)
        .agg(GROUP=("GROUP", "first"), SEX=("SEX", "first"))
        .astype({"ID_PT": str})
    )
    pt["y"] = pt["GROUP"].map(task.label_map)
    if pt["y"].isna().any():
        raise ValueError(f"GROUP inválido para task {task.task_id}")
    return pt


def wide_for_fold(
    df_long: pd.DataFrame,
    *,
    train_pts: set[str],
    test_pts: set[str],
    with_combat: bool,
    fold_id: int,
    combat_quiet: bool = True,
    representation: str = "wide",
    roi: str = ROI_FILTER_DEFAULT,
) -> pd.DataFrame:
    pts = train_pts | test_pts
    sub = df_long[df_long["ID_PT"].astype(str).isin(pts)].copy()
    if with_combat:
        train_imgs = image_ids_for_patients(sub, train_pts)
        transform_imgs = image_ids_for_patients(sub, pts)
        sub = harmonize_long_fold(
            sub,
            train_id_imgs=train_imgs,
            transform_id_imgs=transform_imgs,
            fold_id=fold_id,
            quiet=combat_quiet,
        )
    wide = pivot_long_to_wide(sub)
    return apply_representation_wide(wide, representation, roi=roi)


def repeat_ids(r_repeats: int) -> range:
    """R=0 → uma execução (repeat_id=0); R>0 → repeat_id em 0..R-1."""
    if r_repeats < 0:
        raise ValueError(f"r_repeats deve ser >= 0, recebido {r_repeats}")
    return range(1) if r_repeats == 0 else range(r_repeats)


def nested_cv_ablation(
    df_long: pd.DataFrame,
    *,
    task: TaskConfig,
    modality: str,
    model_key: str,
    selection_mode: str,
    with_combat: bool,
    roi: str,
    base_dir: Path,
    k_out: int = 5,
    k_in: int = 5,
    seed: int = 42,
    repeat_id: int = 0,
    combat_quiet: bool = True,
    stable_pool_min_pct: int = STABLE_POOL_MIN_PCT,
    stable_pool_min_timepoints: int = STABLE_POOL_MIN_TIMEPOINTS,
    stable_pool_n_features: int = STABLE_POOL_N_FEATURES,
    stable_pool_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    stable_pool_l1_c: float = STABLE_POOL_L1_C,
    tuner: str = "grid",
    optuna_trials: int = 30,
    verbose: bool = False,
    representation: str = "wide",
) -> pd.DataFrame:
    pt = patient_labels_from_long(df_long, task)
    y = pt["y"].to_numpy(dtype=int)
    pts = pt["ID_PT"].astype(str).to_numpy()

    class_counts = pd.Series(y).value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    if min_class < k_out:
        raise ValueError(
            f"Task {task.task_id}: classe minoritária tem {min_class} pacientes, "
            f"menos que k_out={k_out}. Reduza folds ou exclua a task."
        )

    outer_cv = StratifiedKFold(k_out, shuffle=True, random_state=seed)
    results: list[dict[str, Any]] = []

    for fold, (tr_rel, te_rel) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
        train_pts = set(pts[tr_rel])
        test_pts = set(pts[te_rel])

        wide = wide_for_fold(
            df_long,
            train_pts=train_pts,
            test_pts=test_pts,
            with_combat=with_combat,
            fold_id=fold,
            combat_quiet=combat_quiet,
            representation=representation,
            roi=roi,
        )
        wide = wide[wide["GROUP"].astype(str).isin(task.groups)].copy()
        wide["y"] = wide["GROUP"].map(task.label_map).astype(int)

        meta = {"ID_PT", "GROUP", "SEX", "y"}
        feature_cols = feature_columns_for_representation(
            wide.columns, modality, roi=roi, representation=representation,
        )
        if not feature_cols:
            raise ValueError(
                f"Nenhuma coluna de feature para modalidade={modality!r} roi={roi!r}"
            )
        X = wide[feature_cols].astype(float)
        y_wide = wide["y"].to_numpy(dtype=int)

        tr_mask = wide["ID_PT"].astype(str).isin(train_pts).to_numpy()
        te_mask = wide["ID_PT"].astype(str).isin(test_pts).to_numpy()
        train_idx = np.where(tr_mask)[0]
        test_idx = np.where(te_mask)[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_wide[train_idx], y_wide[test_idx]

        pipeline = make_pipeline(selection_mode, model_key, roi=roi, seed=seed)
        inner_cv = StratifiedKFold(k_in, shuffle=True, random_state=seed + fold)
        allowed_columns: list[str] | None = None
        if SELECTION_MODES[selection_mode].get("use_stable_pool"):
            allowed_columns, _ = stable_pool_for_outer_train(
                X_train,
                y_train,
                selection_mode=selection_mode,
                inner_cv=inner_cv,
                roi=roi,
                min_pct=stable_pool_min_pct,
                min_timepoints=stable_pool_min_timepoints,
                n_features=stable_pool_n_features,
                n_bootstrap=stable_pool_bootstrap,
                l1_c=stable_pool_l1_c,
                seed=seed + fold,
            )
            if is_embedded_model(model_key):
                X_train = X_train[allowed_columns].copy()
                X_test = X_test[allowed_columns].copy()
            else:
                pipeline.set_params(preselect__allowed_columns=allowed_columns)

        from ablation_optuna import tune_pipeline

        tune_res = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            model_key=model_key,
            selection_mode=selection_mode,
            tuner=tuner,  # type: ignore[arg-type]
            n_trials=optuna_trials,
            seed=seed + fold,
            n_jobs=gridsearch_n_jobs(model_key),
        )
        best = tune_res.estimator
        clf = best.named_steps["clf"]
        method = "predict_proba" if hasattr(clf, "predict_proba") else "decision_function"
        train_scores = cross_val_predict(best, X_train, y_train, cv=inner_cv, method=method)
        if method == "predict_proba":
            train_scores = train_scores[:, 1]
        threshold = tune_youden_threshold(y_train, train_scores)
        test_scores = (
            best.predict_proba(X_test)[:, 1]
            if hasattr(best, "predict_proba")
            else best.decision_function(X_test)
        )
        test_preds = (test_scores >= threshold).astype(int)
        test_id_pts = wide.iloc[test_idx]["ID_PT"].astype(str).tolist()

        audit = fold_selection_audit(
            best,
            X_train,
            list(X_train.columns),
            model_key=model_key,
            selection_mode=selection_mode,
        )
        row = {
            "representation": representation,
            "task": task.task_id,
            "with_combat": with_combat,
            "selection_mode": selection_mode,
            "modality": modality,
            "modality_label": MODALITIES[modality]["label"],
            "model_key": model_key,
            "tuner": tune_res.tuner,
            "repeat_id": repeat_id,
            "fold": fold,
            "best_model": clf.__class__.__name__,
            "best_inner_auc": tune_res.best_inner_auc,
            "best_params": json.dumps(tune_res.best_params, default=str),
            "threshold": threshold,
            "n_features_raw": X.shape[1],
            "n_features_after_stable_pool": len(audit["after_stable_pool_names_"]),
            "n_features_after_filters": len(audit["after_filter_names_"]),
            "n_features_selected": audit["n_features_selected"],
            "removed_by_stable_pool": json.dumps(audit["removed_by_stable_pool_"]),
            "removed_by_filters": json.dumps(audit["removed_by_filters_"]),
            "removed_by_mrmr": json.dumps(audit["removed_by_mrmr_"]),
            "selected_features": json.dumps(audit["selected_names_"]),
            "test_id_pts": json.dumps(test_id_pts),
            "test_y_true": json.dumps(y_test.tolist()),
            "test_scores": json.dumps(test_scores.tolist()),
            **fold_metrics(y_test, test_scores, test_preds),
        }
        results.append(row)
        if verbose:
            log.debug(
                "  fold %d/%d | inner_auc=%.3f | test_auc=%.3f | n_feat=%d",
                fold,
                k_out,
                row["best_inner_auc"],
                row.get("auc", float("nan")),
                row["n_features_selected"],
            )

    return pd.DataFrame(results)


def _results_dir_for_modality(
    base: Path,
    modality: str,
    results_dir: Path | str | None,
    representation: str = "wide",
) -> Path:
    return default_results_dir(
        base, modality, representation, protocol="abs", results_dir=results_dir,
    )


def run_full_ablation_suite(
    *,
    base_dir: Path | str = "csvs/longitudinal_4_groups/ablation/hippocampus",
    roi: str = ROI_FILTER_DEFAULT,
    tasks: tuple[str, ...] = ("cn_ad", "smci_pmci"),
    modalities: tuple[str, ...] = ("vol", "shape", "texture", "disp", "all"),
    models: tuple[str, ...] = ("svm", "rf", "xgb", "mlp"),
    selection_modes: tuple[str, ...] = ("mrmr",),
    with_combat_flags: tuple[bool, ...] = (False, True),
    results_dir: Path | str | None = None,
    seed: int = 42,
    r_repeats: int = 0,
    verbose: bool = False,
    combat_quiet: bool = True,
    stable_pool_min_pct: int = STABLE_POOL_MIN_PCT,
    stable_pool_min_timepoints: int = STABLE_POOL_MIN_TIMEPOINTS,
    stable_pool_n_features: int = STABLE_POOL_N_FEATURES,
    stable_pool_bootstrap: int = STABLE_POOL_BOOTSTRAP,
    stable_pool_l1_c: float = STABLE_POOL_L1_C,
    tuner: str = "grid",
    optuna_trials: int = 30,
    representation: str = "wide",
) -> pd.DataFrame:
    if not HAS_MRMR and {"mrmr", "mrmr_stable"} & set(selection_modes):
        raise ImportError("feature-engine necessário para modo mrmr: pip install feature-engine")

    stable_pool_min_timepoints = resolve_stable_pool_min_timepoints(
        representation, stable_pool_min_timepoints, log=log,
    )

    base = Path(base_dir)
    all_results: list[pd.DataFrame] = []
    n_reps = len(repeat_ids(r_repeats))
    total_jobs = (
        len(modalities)
        * len(tasks)
        * len(with_combat_flags)
        * len(selection_modes)
        * len(models)
        * n_reps
    )
    job_no = 0
    t0 = time.monotonic()

    for modality in modalities:
        out_dir = _results_dir_for_modality(
            base, modality, results_dir, representation=representation,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        modality_results: list[pd.DataFrame] = []

        long_path = base / MODALITIES[modality]["long"]
        if not long_path.is_file():
            raise FileNotFoundError(f"Long CSV ausente: {long_path}")
        log.info("modalidade %s | long=%s", modality, long_path)
        df_long = pd.read_csv(long_path)

        for task_id in tasks:
            task = TASKS[task_id]
            for with_combat in with_combat_flags:
                for selection_mode in selection_modes:
                    for model_key in models:
                        for repeat_id in repeat_ids(r_repeats):
                            job_no += 1
                            seed_rep = seed + repeat_id * 1000
                            rep_label = (
                                f"rep={repeat_id}"
                                if r_repeats > 0
                                else "rep=0 (única)"
                            )
                            elapsed = time.monotonic() - t0
                            eta_s = (elapsed / job_no) * (total_jobs - job_no) if job_no else 0
                            log.info(
                                "[%d/%d] %s | %s | combat=%s | %s | %s | %s | "
                                "elapsed=%s eta=%s",
                                job_no,
                                total_jobs,
                                task_id,
                                modality,
                                with_combat,
                                selection_mode,
                                model_key,
                                rep_label,
                                fmt_duration(elapsed),
                                fmt_duration(eta_s),
                            )
                            job_t0 = time.monotonic()
                            res = nested_cv_ablation(
                                df_long,
                                task=task,
                                modality=modality,
                                model_key=model_key,
                                selection_mode=selection_mode,
                                with_combat=with_combat,
                                roi=roi,
                                base_dir=base,
                                seed=seed_rep,
                                repeat_id=repeat_id,
                                combat_quiet=combat_quiet,
                                stable_pool_min_pct=stable_pool_min_pct,
                                stable_pool_min_timepoints=stable_pool_min_timepoints,
                                stable_pool_n_features=stable_pool_n_features,
                                stable_pool_bootstrap=stable_pool_bootstrap,
                                stable_pool_l1_c=stable_pool_l1_c,
                                tuner=tuner,
                                optuna_trials=optuna_trials,
                                verbose=verbose,
                                representation=representation,
                            )
                            auc_mean = float(res["auc"].mean()) if "auc" in res.columns else float("nan")
                            log.info(
                                "[%d/%d] ok | auc_mean=%.3f | %d folds | job=%s",
                                job_no,
                                total_jobs,
                                auc_mean,
                                len(res),
                                fmt_duration(time.monotonic() - job_t0),
                            )
                            modality_results.append(res)

        combined_mod = pd.concat(modality_results, ignore_index=True)
        combined_mod.to_csv(out_dir / "ablation_results_all.csv", index=False)
        summary = summary_with_pooled(combined_mod)
        summary.to_csv(out_dir / "ablation_summary.csv", index=False)
        log.info(
            "salvo %s (%d linhas, %d configs)",
            out_dir / "ablation_results_all.csv",
            len(combined_mod),
            len(summary),
        )
        all_results.append(combined_mod)

    log.info("concluído | %d jobs | %s", job_no, fmt_duration(time.monotonic() - t0))
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


if __name__ == "__main__":
    # ponytail: T1/T2/T3 mesmo biomarcador não caem por corr entre si
    rng = np.random.default_rng(0)
    n = 80
    base = rng.normal(size=n)
    X = np.column_stack(
        [
            base,
            base + rng.normal(scale=0.01, size=n),
            base + rng.normal(scale=0.01, size=n),
            base * 0.98 + rng.normal(scale=0.02, size=n),
        ]
    )
    names = [
        "hippocampus_L_T1_gm_norm",
        "hippocampus_L_T2_gm_norm",
        "hippocampus_L_T3_gm_norm",
        "hippocampus_L_T1_wm_norm",
    ]
    mask = corr_keep_mask(X, 0.90, feature_names=names)
    assert mask[:3].all(), "visitas T1/T2/T3 devem sobreviver"
    assert mask.sum() == 3, "wm_norm redundante com gm_norm deve cair"
    print("ablation_runner self-check ok")

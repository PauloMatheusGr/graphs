"""Runner de ablação: tarefas binárias entre grupos CN/sMCI/pMCI/AD, 4 modalidades, ComBat opcional por fold."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ablation_harmonize import harmonize_long_fold, image_ids_for_patients
from ablation_analysis import summary_with_pooled
from ablation_prep import (
    ROI_FILTER_DEFAULT,
    block_key_regex,
    modality_wide_columns,
    pivot_long_to_wide,
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
    "filters": {"use_mrmr": False, "use_filters": True, "label": "corr + var (sem MRMR)"},
    "raw": {"use_mrmr": False, "use_filters": False, "label": "sem seleção"},
}

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "svm": {
        "preselect__k_per_block": [1, 2, 3, 4],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale"],
        "clf__class_weight": [None, "balanced"],
    },
    "rf": {
        "preselect__k_per_block": [1, 2, 3, 4],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__class_weight": [None, "balanced"],
    },
    "xgb": {
        "preselect__k_per_block": [1, 2, 3, 4],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    },
    "mlp": {
        "preselect__k_per_block": [1, 2, 3, 4],
        "clf__hidden_layer_sizes": [(64,), (128, 64)],
        "clf__alpha": [1e-4, 1e-3],
        "clf__max_iter": [2000],
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


def corr_keep_mask(X: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = xf.shape[1]
    if xf.shape[0] < 2:
        return np.ones(n, dtype=bool)
    c = np.corrcoef(xf.T)
    keep_idx: list[int] = []
    for j in range(n):
        if all(not (np.isfinite(c[j, k]) and abs(c[j, k]) > threshold) for k in keep_idx):
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


def mrmr_by_block(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    block_re: re.Pattern[str],
    max_per_block: int = 4,
) -> list[str]:
    if not HAS_MRMR:
        return list(df.columns)
    selected: list[str] = []
    blocks: dict[tuple[str, str], list[str]] = {}
    for col in df.columns:
        m = block_re.search(col)
        if m:
            blocks.setdefault((m.group(1), m.group(2)), []).append(col)
    for cols in blocks.values():
        if not cols:
            continue
        # ponytail: feature-engine exige max_features < len(variables)
        if len(cols) <= max_per_block:
            selected.extend(cols)
            continue
        selector = MRMR(
            variables=cols,
            regression=False,
            max_features=max_per_block,
        )
        selector.fit(df[cols], y)
        selected.extend(selector.variables_)
    return selected


class CorrVarMRMRByBlock(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        use_mrmr: bool = True,
        use_filters: bool = True,
        k_per_block: int = 2, #4,
        corr_threshold: float = 0.90,
        var_threshold: float = 0.01,
        roi: str = ROI_FILTER_DEFAULT,
    ):
        self.use_mrmr = use_mrmr
        self.use_filters = use_filters
        self.k_per_block = k_per_block
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold
        self.roi = roi
        self.block_re = block_key_regex(roi)
        self.selected_names_: list[str] = []
        self.selected_indices_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: list[str] = []

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y):
        df_in = self._as_frame(X)
        self.feature_names_in_ = list(df_in.columns)
        names = list(df_in.columns)
        X_arr = df_in.to_numpy(dtype=float)

        if self.use_filters:
            cmask = corr_keep_mask(X_arr, self.corr_threshold)
            names = [n for n, k in zip(names, cmask) if k]
            X_arr = X_arr[:, cmask]
            vmask = var_keep_mask(X_arr, self.var_threshold)
            names = [n for n, k in zip(names, vmask) if k]
            X_arr = X_arr[:, vmask]

        if self.use_mrmr and names:
            df_f = pd.DataFrame(X_arr, columns=names)
            selected_names = mrmr_by_block(
                df_f, y, block_re=self.block_re, max_per_block=self.k_per_block
            )
            if not selected_names:
                selected_names = names[: min(10, len(names))]
        else:
            selected_names = names

        if not selected_names:
            # ponytail: fallback se corr/var/MRMR zerarem (ex. vol após ICV norm)
            selected_names = list(self.feature_names_in_)

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


def make_pipeline(selection_mode: str, model_key: str, *, roi: str, seed: int) -> ImbPipeline:
    cfg = SELECTION_MODES[selection_mode]
    clf_map = {
        "svm": SVC(probability=True, random_state=seed),
        "rf": RandomForestClassifier(random_state=seed),
        "xgb": XGBClassifier(eval_metric="logloss", random_state=seed),
        "mlp": MLPClassifier(
            activation="relu", alpha=1e-3, batch_size=32, random_state=seed, max_iter=500
        ),
    }
    return ImbPipeline(
        [
            (
                "preselect",
                CorrVarMRMRByBlock(
                    use_mrmr=cfg["use_mrmr"],
                    use_filters=cfg["use_filters"],
                    roi=roi,
                ),
            ),
            ("scaler", StandardScaler()),
            ("clf", clf_map[model_key]),
        ]
    )


def param_grid_for(model_key: str, selection_mode: str) -> dict[str, Any]:
    return {
        k: v
        for k, v in PARAM_GRIDS[model_key].items()
        if k != "preselect__k_per_block" or selection_mode == "mrmr"
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
    return pivot_long_to_wide(sub)


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
        )
        wide = wide[wide["GROUP"].astype(str).isin(task.groups)].copy()
        wide["y"] = wide["GROUP"].map(task.label_map).astype(int)

        meta = {"ID_PT", "GROUP", "SEX", "y"}
        feature_cols = modality_wide_columns(wide.columns, modality, roi=roi)
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
        grid = GridSearchCV(
            pipeline,
            param_grid_for(model_key, selection_mode),
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
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

        preselect = best.named_steps["preselect"]
        row = {
            "task": task.task_id,
            "with_combat": with_combat,
            "selection_mode": selection_mode,
            "modality": modality,
            "modality_label": MODALITIES[modality]["label"],
            "model_key": model_key,
            "repeat_id": repeat_id,
            "fold": fold,
            "best_model": clf.__class__.__name__,
            "best_inner_auc": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, default=str),
            "threshold": threshold,
            "n_features_raw": X.shape[1],
            "n_features_selected": int(best.named_steps["preselect"].transform(X_train).shape[1]),
            "selected_features": json.dumps(list(preselect.selected_names_)),
            "test_id_pts": json.dumps(test_id_pts),
            "test_y_true": json.dumps(y_test.tolist()),
            "test_scores": json.dumps(test_scores.tolist()),
            **fold_metrics(y_test, test_scores, test_preds),
        }
        results.append(row)

    return pd.DataFrame(results)


def _results_dir_for_modality(
    base: Path,
    modality: str,
    results_dir: Path | str | None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    return base.parent.parent / "ablation_results" / modality


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
) -> pd.DataFrame:
    if not HAS_MRMR and "mrmr" in selection_modes:
        raise ImportError("feature-engine necessário para modo mrmr: pip install feature-engine")

    base = Path(base_dir)
    all_results: list[pd.DataFrame] = []

    for modality in modalities:
        out_dir = _results_dir_for_modality(base, modality, results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        modality_results: list[pd.DataFrame] = []

        long_path = base / MODALITIES[modality]["long"]
        if not long_path.is_file():
            raise FileNotFoundError(f"Long CSV ausente: {long_path}")
        df_long = pd.read_csv(long_path)

        for task_id in tasks:
            task = TASKS[task_id]
            for with_combat in with_combat_flags:
                for selection_mode in selection_modes:
                    for model_key in models:
                        for repeat_id in repeat_ids(r_repeats):
                            seed_rep = seed + repeat_id * 1000
                            rep_label = (
                                f"rep={repeat_id}"
                                if r_repeats > 0
                                else "rep=0 (única)"
                            )
                            if verbose:
                                print(
                                    f"\n=== {task_id} | {modality} | combat={with_combat} | "
                                    f"{selection_mode} | {model_key} | {rep_label} ==="
                                )
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
                            )
                            modality_results.append(res)

        combined_mod = pd.concat(modality_results, ignore_index=True)
        combined_mod.to_csv(out_dir / "ablation_results_all.csv", index=False)
        summary = summary_with_pooled(combined_mod)
        summary.to_csv(out_dir / "ablation_summary.csv", index=False)
        if verbose:
            print(f"\nSalvo: {out_dir / 'ablation_results_all.csv'} ({len(combined_mod)} linhas)")
        all_results.append(combined_mod)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

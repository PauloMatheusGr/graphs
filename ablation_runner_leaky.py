"""Runner ablação protocolo LEAKY (literatura): pré-processamento no dataset inteiro antes do CV.

AVISO: z-score global, ComBat global, seleção de atributos com rótulos de todo o conjunto.
Resultados inflados de propósito — só para comparar com estudos que não evitam leakage.
Avaliação principal: ablation_runner.py + 2_run_ablation.py.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

import re

from ablation_analysis import estimate_stable_pool_columns, summary_with_pooled
from ablation_harmonize import harmonize_long_fold, image_ids_for_patients
from ablation_prep import (
    DISP_PREFIX_DROP,
    DISP_STAT_DROP,
    META_COLS_WIDE,
    ROI_FILTER_DEFAULT,
    SHAPE_RE,
    SLOT_ORDER,
    TEXTURE_RE,
    VOL_FEATURE_SUFFIXES,
    modality_wide_columns,
    pivot_long_to_wide,
)
from ablation_runner import (
    MODALITIES,
    PARAM_GRIDS,
    SELECTION_MODES,
    STABLE_POOL_MIN_PCT,
    STABLE_POOL_MIN_TIMEPOINTS,
    STABLE_POOL_N_FEATURES,
    TASKS,
    CorrVarMRMRSelector,
    TaskConfig,
    fold_metrics,
    fold_selection_audit,
    fmt_duration,
    gridsearch_n_jobs,
    is_embedded_model,
    make_classifier,
    param_grid_for,
    patient_labels_from_long,
    repeat_ids,
    tune_youden_threshold,
)

log = logging.getLogger(__name__)

LEAKY_INNER_FOLDS = 5  # ponytail: só para contagem do stable pool; cada fit usa dataset inteiro
UNIVARIATE_K_GRID = [10, 20, 50, 100, 200, 500]
UNIVARIATE_SCORE_FUNCS: dict[str, Any] = {
    "f_classif": f_classif,
    "mutual_info": mutual_info_classif,
}


def global_zscore(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score coluna a coluna usando média/desvio de todas as linhas (leakage)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan).fillna(1.0)
    return ((X - mu) / sd).astype(float)


def _collapsed_col_pat(roi: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(roi)}_([LR])_(.+)$")


def modality_collapsed_columns(
    columns: list[str], modality: str, *, roi: str = ROI_FILTER_DEFAULT
) -> list[str]:
    """Seleção de modalidade em colunas sem token temporal: {roi}_{L|R}_{feat}."""
    pat = _collapsed_col_pat(roi)

    def feat_of(col: str) -> str | None:
        m = pat.match(col)
        return m.group(2) if m else None

    def keep_vol(f: str) -> bool:
        return f in VOL_FEATURE_SUFFIXES

    def keep_shape(f: str) -> bool:
        return bool(SHAPE_RE.match(f))

    def keep_texture(f: str) -> bool:
        return bool(TEXTURE_RE.search(f))

    def keep_disp(f: str) -> bool:
        if any(f.startswith(p) for p in DISP_PREFIX_DROP):
            return False
        if any(f.endswith(s) for s in DISP_STAT_DROP):
            return False
        return True

    keepers = {
        "vol": keep_vol,
        "shape": keep_shape,
        "texture": keep_texture,
        "disp": keep_disp,
    }
    if modality == "all":
        out: list[str] = []
        for mod in ("vol", "shape", "texture", "disp"):
            out += modality_collapsed_columns(columns, mod, roi=roi)
        return list(dict.fromkeys(out))
    if modality not in keepers:
        raise ValueError(f"modalidade desconhecida: {modality}")
    keep = keepers[modality]
    out = []
    for col in columns:
        f = feat_of(col)
        if f is not None and keep(f):
            out.append(col)
    return out


def wide_per_visit_task(
    df_long: pd.DataFrame,
    *,
    task: TaskConfig,
    with_combat: bool,
    combat_quiet: bool = True,
) -> pd.DataFrame:
    """Pseudo-replicação: 1 linha por (paciente, visita). Lados L/R viram colunas.

    LEAK FORTE: rótulo é por paciente, mas cada visita é tratada como amostra
    independente. Sem GroupKFold, visitas do mesmo paciente caem em treino e teste.
    """
    sub = df_long[df_long["GROUP"].astype(str).isin(task.groups)].copy()
    if with_combat:
        all_pts = set(sub["ID_PT"].astype(str))
        all_imgs = image_ids_for_patients(sub, all_pts)
        sub = harmonize_long_fold(
            sub,
            train_id_imgs=all_imgs,
            transform_id_imgs=all_imgs,
            fold_id=0,
            quiet=combat_quiet,
        )

    sub["ID_PT"] = sub["ID_PT"].astype(str)
    sub["_visit"] = sub["slot"].map(SLOT_ORDER).fillna(99).astype(int)
    roi_name = str(sub["roi"].iloc[0]) if "roi" in sub.columns else ROI_FILTER_DEFAULT

    feature_cols = [c for c in sub.columns if c not in META_COLS_WIDE and c != "_visit"]
    wide = sub.pivot_table(
        index=["ID_PT", "GROUP", "SEX", "_visit"],
        columns="side",
        values=feature_cols,
    )
    wide.columns = [f"{roi_name}_{side}_{feat}" for feat, side in wide.columns]
    wide = wide.reset_index()
    wide["SEX"] = wide["SEX"].map({"M": 0, "F": 1, 0: 0, 1: 1})
    wide["y"] = wide["GROUP"].map(task.label_map).astype(int)
    return wide


def wide_leaky_task(
    df_long: pd.DataFrame,
    *,
    task: TaskConfig,
    with_combat: bool,
    combat_quiet: bool = True,
) -> pd.DataFrame:
    """Wide de todos os pacientes da task; ComBat fit+transform no dataset inteiro."""
    sub = df_long[df_long["GROUP"].astype(str).isin(task.groups)].copy()
    if with_combat:
        all_pts = set(sub["ID_PT"].astype(str))
        all_imgs = image_ids_for_patients(sub, all_pts)
        sub = harmonize_long_fold(
            sub,
            train_id_imgs=all_imgs,
            transform_id_imgs=all_imgs,
            fold_id=0,
            quiet=combat_quiet,
        )
    wide = pivot_long_to_wide(sub)
    wide["y"] = wide["GROUP"].map(task.label_map).astype(int)
    return wide


def leaky_stable_pool_columns(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    *,
    roi: str,
    n_features: int,
    min_pct: int,
    min_timepoints: int,
) -> list[str]:
    """Stable pool com 'inner folds' todos ajustados no dataset inteiro (leakage)."""
    inner_selected: list[list[str]] = []
    for _ in range(LEAKY_INNER_FOLDS):
        pre = CorrVarMRMRSelector(
            use_mrmr=True,
            use_filters=True,
            n_features_total=n_features,
            roi=roi,
        )
        pre.fit(X_full, y_full)
        inner_selected.append(list(pre.selected_names_))
    allowed, _ = estimate_stable_pool_columns(
        list(X_full.columns),
        inner_selected,
        min_pct=min_pct,
        min_timepoints=min_timepoints,
    )
    return allowed


class LeakyGlobalPreselector(BaseEstimator, TransformerMixin):
    """Seleção corr/var/mRMR sempre no dataset global (ignora X/y do fold de treino)."""

    def __init__(
        self,
        *,
        selection_mode: str = "mrmr_stable",
        roi: str = ROI_FILTER_DEFAULT,
        n_features_total: int = 20,
        stable_pool_min_pct: int = STABLE_POOL_MIN_PCT,
        stable_pool_min_timepoints: int = STABLE_POOL_MIN_TIMEPOINTS,
        stable_pool_n_features: int = STABLE_POOL_N_FEATURES,
        X_global: pd.DataFrame | None = None,
        y_global: np.ndarray | None = None,
    ):
        self.selection_mode = selection_mode
        self.roi = roi
        self.n_features_total = n_features_total
        self.stable_pool_min_pct = stable_pool_min_pct
        self.stable_pool_min_timepoints = stable_pool_min_timepoints
        self.stable_pool_n_features = stable_pool_n_features
        self.X_global = X_global
        self.y_global = y_global
        self._pre = CorrVarMRMRSelector(use_mrmr=False, use_filters=False, roi=roi)
        self.selected_names_: list[str] = []
        self.selected_indices_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: list[str] = []
        self.removed_by_stable_pool_: list[str] = []
        self.removed_by_filters_: list[str] = []
        self.removed_by_mrmr_: list[str] = []
        self.after_stable_pool_names_: list[str] = []
        self.after_filter_names_: list[str] = []

    def _copy_attrs(self, pre: CorrVarMRMRSelector) -> None:
        self.selected_names_ = list(pre.selected_names_)
        self.selected_indices_ = pre.selected_indices_
        self.feature_names_in_ = list(pre.feature_names_in_)
        self.removed_by_stable_pool_ = list(pre.removed_by_stable_pool_)
        self.removed_by_filters_ = list(pre.removed_by_filters_)
        self.removed_by_mrmr_ = list(pre.removed_by_mrmr_)
        self.after_stable_pool_names_ = list(pre.after_stable_pool_names_)
        self.after_filter_names_ = list(pre.after_filter_names_)

    def fit(self, X, y):
        if self.X_global is None or self.y_global is None:
            raise ValueError("LeakyGlobalPreselector exige X_global e y_global.")
        cfg = SELECTION_MODES[self.selection_mode]
        pre = CorrVarMRMRSelector(
            use_mrmr=cfg["use_mrmr"],
            use_filters=cfg["use_filters"],
            n_features_total=self.n_features_total,
            roi=self.roi,
        )
        if cfg.get("use_stable_pool"):
            allowed = leaky_stable_pool_columns(
                self.X_global,
                self.y_global,
                roi=self.roi,
                n_features=self.stable_pool_n_features,
                min_pct=self.stable_pool_min_pct,
                min_timepoints=self.stable_pool_min_timepoints,
            )
            pre.allowed_columns = allowed
        pre.fit(self.X_global, self.y_global)
        self._pre = pre
        self._copy_attrs(pre)
        return self

    def transform(self, X):
        return self._pre.transform(X)


class LeakyUnivariatePreselector(BaseEstimator, TransformerMixin):
    """SelectKBest (ANOVA F ou MI) no dataset global — ignora X/y do fold de treino."""

    def __init__(
        self,
        *,
        k: int = 50,
        score_func: str = "f_classif",
        X_global: pd.DataFrame | None = None,
        y_global: np.ndarray | None = None,
    ):
        self.k = k
        self.score_func = score_func
        self.X_global = X_global
        self.y_global = y_global
        self.selected_names_: list[str] = []
        self.selected_indices_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: list[str] = []
        self.removed_by_stable_pool_: list[str] = []
        self.removed_by_filters_: list[str] = []
        self.removed_by_mrmr_: list[str] = []
        self.after_stable_pool_names_: list[str] = []
        self.after_filter_names_: list[str] = []

    def fit(self, X, y):
        if self.X_global is None or self.y_global is None:
            raise ValueError("LeakyUnivariatePreselector exige X_global e y_global.")
        score = UNIVARIATE_SCORE_FUNCS.get(self.score_func, f_classif)
        names = list(self.X_global.columns)
        self.feature_names_in_ = names
        self.after_stable_pool_names_ = list(names)
        self.after_filter_names_ = list(names)
        self.removed_by_stable_pool_ = []
        self.removed_by_filters_ = []

        k = min(int(self.k), len(names))
        if k < 1:
            k = 1
        selector = SelectKBest(score_func=score, k=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            selector.fit(self.X_global.to_numpy(dtype=float), self.y_global)
        mask = selector.get_support()
        self.selected_names_ = [n for n, keep in zip(names, mask) if keep]
        self.removed_by_mrmr_ = [n for n in names if n not in set(self.selected_names_)]
        name_to_idx = {n: i for i, n in enumerate(names)}
        self.selected_indices_ = np.array(
            [name_to_idx[n] for n in self.selected_names_], dtype=int
        )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names_in_].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        if self.selected_indices_.size == 0:
            raise ValueError("Nenhuma feature selecionada por SelectKBest.")
        return X_arr[:, self.selected_indices_]


def param_grid_leaky_univariate(model_key: str) -> dict[str, list[Any]]:
    """Grid: top-K univariado + hiperparâmetros do classificador (sem mRMR)."""
    if is_embedded_model(model_key):
        return param_grid_for(model_key, "raw")
    grid = {
        k: v
        for k, v in PARAM_GRIDS[model_key].items()
        if not k.startswith("preselect__")
    }
    grid["preselect__k"] = UNIVARIATE_K_GRID
    grid["preselect__score_func"] = list(UNIVARIATE_SCORE_FUNCS.keys())
    return grid


def make_leaky_pipeline(
    selection_mode: str,
    model_key: str,
    *,
    roi: str,
    seed: int,
    X_global: pd.DataFrame,
    y_global: np.ndarray,
    stable_pool_min_pct: int,
    stable_pool_min_timepoints: int,
    stable_pool_n_features: int,
    univariate_global: bool = False,
) -> ImbPipeline:
    steps: list[tuple[str, Any]] = []
    embedded = is_embedded_model(model_key)
    if univariate_global and not embedded:
        steps.append(
            (
                "preselect",
                LeakyUnivariatePreselector(
                    k=50,
                    score_func="f_classif",
                    X_global=X_global,
                    y_global=y_global,
                ),
            )
        )
    elif selection_mode != "raw" and not embedded:
        steps.append(
            (
                "preselect",
                LeakyGlobalPreselector(
                    selection_mode=selection_mode,
                    roi=roi,
                    X_global=X_global,
                    y_global=y_global,
                    stable_pool_min_pct=stable_pool_min_pct,
                    stable_pool_min_timepoints=stable_pool_min_timepoints,
                    stable_pool_n_features=stable_pool_n_features,
                ),
            )
        )
    steps.append(("clf", make_classifier(model_key, seed=seed)))
    return ImbPipeline(steps)


def nested_cv_ablation_leaky(
    df_long: pd.DataFrame,
    *,
    task: TaskConfig,
    modality: str,
    model_key: str,
    selection_mode: str,
    with_combat: bool,
    roi: str,
    k_out: int = 5,
    k_in: int = 5,
    seed: int = 42,
    repeat_id: int = 0,
    combat_quiet: bool = True,
    stable_pool_min_pct: int = STABLE_POOL_MIN_PCT,
    stable_pool_min_timepoints: int = STABLE_POOL_MIN_TIMEPOINTS,
    stable_pool_n_features: int = STABLE_POOL_N_FEATURES,
    pseudo_replication: bool = False,
    tune_on_full: bool = False,
    threshold_on_test: bool = False,
    univariate_global: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    if pseudo_replication:
        wide = wide_per_visit_task(
            df_long, task=task, with_combat=with_combat, combat_quiet=combat_quiet
        )
        feature_cols = modality_collapsed_columns(list(wide.columns), modality, roi=roi)
    else:
        wide = wide_leaky_task(
            df_long, task=task, with_combat=with_combat, combat_quiet=combat_quiet
        )
        feature_cols = modality_wide_columns(wide.columns, modality, roi=roi)
    if not feature_cols:
        raise ValueError(f"Nenhuma coluna de feature para modalidade={modality!r} roi={roi!r}")

    X_all = global_zscore(wide[feature_cols].astype(float))
    y_all = wide["y"].to_numpy(dtype=int)
    id_pts = wide["ID_PT"].astype(str).to_numpy()

    # LEAK pseudo_replication: split por LINHA (visita), não por paciente.
    class_counts = pd.Series(y_all).value_counts()
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    if min_class < k_out:
        raise ValueError(
            f"Task {task.task_id}: classe minoritária tem {min_class} amostras, "
            f"menos que k_out={k_out}."
        )

    outer_cv = StratifiedKFold(k_out, shuffle=True, random_state=seed)
    results: list[dict[str, Any]] = []

    for fold, (tr_idx, te_idx) in enumerate(
        outer_cv.split(np.zeros(len(y_all)), y_all), start=1
    ):
        X_train, X_test = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_train, y_test = y_all[tr_idx], y_all[te_idx]

        pipeline = make_leaky_pipeline(
            selection_mode,
            model_key,
            roi=roi,
            seed=seed,
            X_global=X_all,
            y_global=y_all,
            stable_pool_min_pct=stable_pool_min_pct,
            stable_pool_min_timepoints=stable_pool_min_timepoints,
            stable_pool_n_features=stable_pool_n_features,
            univariate_global=univariate_global,
        )
        inner_cv = StratifiedKFold(k_in, shuffle=True, random_state=seed + fold)
        pgrid = (
            param_grid_leaky_univariate(model_key)
            if univariate_global
            else param_grid_for(model_key, selection_mode)
        )
        grid = GridSearchCV(
            pipeline,
            pgrid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=gridsearch_n_jobs(model_key),
            refit=True,
            verbose=0,
        )
        # LEAK tune_on_full: hiperparâmetros escolhidos vendo o dataset inteiro.
        grid.fit(X_all if tune_on_full else X_train, y_all if tune_on_full else y_train)
        best = grid.best_estimator_
        clf = best.named_steps["clf"]
        method = "predict_proba" if hasattr(clf, "predict_proba") else "decision_function"
        test_scores = (
            best.predict_proba(X_test)[:, 1]
            if hasattr(best, "predict_proba")
            else best.decision_function(X_test)
        )
        if threshold_on_test:
            # LEAK threshold_on_test: corte de decisão otimizado nos rótulos do teste.
            threshold = tune_youden_threshold(y_test, test_scores)
        else:
            train_scores = cross_val_predict(best, X_train, y_train, cv=inner_cv, method=method)
            if method == "predict_proba":
                train_scores = train_scores[:, 1]
            threshold = tune_youden_threshold(y_train, train_scores)
        test_preds = (test_scores >= threshold).astype(int)
        test_id_pts = id_pts[te_idx].tolist()

        n_raw = X_all.shape[1]
        if is_embedded_model(model_key):
            audit = fold_selection_audit(
                best,
                X_train,
                list(X_all.columns),
                model_key=model_key,
                selection_mode=selection_mode,
            )
            n_sel = audit["n_features_selected"]
            pre_attrs = SimpleNamespace(
                after_stable_pool_names_=audit["after_stable_pool_names_"],
                after_filter_names_=audit["after_filter_names_"],
                selected_names_=audit["selected_names_"],
                removed_by_stable_pool_=audit["removed_by_stable_pool_"],
                removed_by_filters_=audit["removed_by_filters_"],
                removed_by_mrmr_=audit["removed_by_mrmr_"],
            )
        elif selection_mode == "raw" and not univariate_global:
            n_sel = n_raw
            pre_attrs = SimpleNamespace(
                after_stable_pool_names_=list(X_all.columns),
                after_filter_names_=list(X_all.columns),
                selected_names_=list(X_all.columns),
                removed_by_stable_pool_=[],
                removed_by_filters_=[],
                removed_by_mrmr_=[],
            )
        else:
            pre = best.named_steps["preselect"]
            n_sel = int(pre.transform(X_train).shape[1])
            pre_attrs = pre

        protocol_tags = ["leaky_global"]
        if univariate_global:
            protocol_tags.append("univariate")
        if pseudo_replication:
            protocol_tags.append("pseudo")
        if tune_on_full:
            protocol_tags.append("fulltune")
        if threshold_on_test:
            protocol_tags.append("testthr")

        row = {
            "task": task.task_id,
            "with_combat": with_combat,
            "selection_mode": selection_mode,
            "modality": modality,
            "modality_label": MODALITIES[modality]["label"],
            "model_key": model_key,
            "repeat_id": repeat_id,
            "fold": fold,
            "protocol": "+".join(protocol_tags),
            "best_model": clf.__class__.__name__,
            "best_inner_auc": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, default=str),
            "threshold": threshold,
            "n_features_raw": n_raw,
            "n_features_after_stable_pool": len(pre_attrs.after_stable_pool_names_),
            "n_features_after_filters": len(pre_attrs.after_filter_names_),
            "n_features_selected": n_sel,
            "removed_by_stable_pool": json.dumps(pre_attrs.removed_by_stable_pool_),
            "removed_by_filters": json.dumps(pre_attrs.removed_by_filters_),
            "removed_by_mrmr": json.dumps(pre_attrs.removed_by_mrmr_),
            "selected_features": json.dumps(list(pre_attrs.selected_names_)),
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
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    return base.parent.parent / "ablation_results_leaky" / modality


def run_full_ablation_suite_leaky(
    *,
    base_dir: Path | str = "csvs/longitudinal_4_groups/ablation/hippocampus",
    roi: str = ROI_FILTER_DEFAULT,
    tasks: tuple[str, ...] = ("cn_ad", "smci_pmci"),
    modalities: tuple[str, ...] = ("vol", "shape", "texture", "disp", "all"),
    models: tuple[str, ...] = ("svm", "rf", "mlp"),
    selection_modes: tuple[str, ...] = ("mrmr_stable",),
    with_combat_flags: tuple[bool, ...] = (False,),
    results_dir: Path | str | None = None,
    seed: int = 42,
    r_repeats: int = 0,
    verbose: bool = False,
    combat_quiet: bool = True,
    stable_pool_min_pct: int = STABLE_POOL_MIN_PCT,
    stable_pool_min_timepoints: int = STABLE_POOL_MIN_TIMEPOINTS,
    stable_pool_n_features: int = STABLE_POOL_N_FEATURES,
    pseudo_replication: bool = False,
    tune_on_full: bool = False,
    threshold_on_test: bool = False,
    univariate_global: bool = False,
) -> pd.DataFrame:
    from ablation_runner import HAS_MRMR

    if (
        not HAS_MRMR
        and not univariate_global
        and {"mrmr", "mrmr_stable"} & set(selection_modes)
    ):
        raise ImportError("feature-engine necessário para modo mrmr: pip install feature-engine")

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

    log.warning(
        "PROTOCOLO LEAKY: z-score/ComBat/seleção no dataset inteiro antes do CV. "
        "Resultados inflados — comparar com literatura, não usar como avaliação principal."
    )

    for modality in modalities:
        out_dir = _results_dir_for_modality(base, modality, results_dir)
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
                            rep_label = f"rep={repeat_id}" if r_repeats > 0 else "rep=0 (única)"
                            elapsed = time.monotonic() - t0
                            eta_s = (elapsed / job_no) * (total_jobs - job_no) if job_no else 0
                            log.info(
                                "[%d/%d] %s | %s | combat=%s | %s | %s | %s | "
                                "elapsed=%s eta=%s | LEAKY",
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
                            res = nested_cv_ablation_leaky(
                                df_long,
                                task=task,
                                modality=modality,
                                model_key=model_key,
                                selection_mode=selection_mode,
                                with_combat=with_combat,
                                roi=roi,
                                seed=seed_rep,
                                repeat_id=repeat_id,
                                combat_quiet=combat_quiet,
                                stable_pool_min_pct=stable_pool_min_pct,
                                stable_pool_min_timepoints=stable_pool_min_timepoints,
                                stable_pool_n_features=stable_pool_n_features,
                                pseudo_replication=pseudo_replication,
                                tune_on_full=tune_on_full,
                                threshold_on_test=threshold_on_test,
                                univariate_global=univariate_global,
                                verbose=verbose,
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

    log.info("concluído LEAKY | %d jobs | %s", job_no, fmt_duration(time.monotonic() - t0))
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40
    cols = [f"hippocampus_L_T1_f{i}" for i in range(8)]
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=cols)
    y = (rng.random(n) > 0.5).astype(int)
    Xz = global_zscore(X)
    assert abs(float(Xz.mean().mean())) < 0.2
    sel = LeakyGlobalPreselector(
        selection_mode="mrmr",
        roi="hippocampus",
        n_features_total=3,
        X_global=Xz,
        y_global=y,
    )
    sel.fit(Xz.iloc[:20], y[:20])
    assert len(sel.selected_names_) <= 3
    uni = LeakyUnivariatePreselector(k=3, score_func="f_classif", X_global=Xz, y_global=y)
    uni.fit(Xz.iloc[:10], y[:10])
    assert 1 <= len(uni.selected_names_) <= 3
    assert len(param_grid_leaky_univariate("svm")["preselect__k"]) == len(UNIVARIATE_K_GRID)
    emb_pipe = make_leaky_pipeline(
        "mrmr_stable",
        "logreg_l1",
        roi="hippocampus",
        seed=0,
        X_global=Xz,
        y_global=y,
        stable_pool_min_pct=70,
        stable_pool_min_timepoints=2,
        stable_pool_n_features=50,
    )
    assert "preselect" not in emb_pipe.named_steps
    assert emb_pipe.named_steps["clf"].__class__.__name__ == "LogisticRegression"
    print("OK leaky self-check")

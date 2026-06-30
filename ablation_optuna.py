"""Tuning do inner CV: GridSearchCV ou Optuna TPE (sem pruning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

TunerKind = Literal["grid", "optuna"]
OPTUNA_TRIALS_DEFAULT = 30


@dataclass(frozen=True)
class TuneResult:
    estimator: Any
    best_inner_auc: float
    best_params: dict[str, Any]
    tuner: str


def _fold_auc(estimator, X, y, tr: np.ndarray, va: np.ndarray) -> float | None:
    est = clone(estimator)
    X_tr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
    X_va = X.iloc[va] if isinstance(X, pd.DataFrame) else X[va]
    est.fit(X_tr, y[tr])
    if hasattr(est, "predict_proba"):
        scores = est.predict_proba(X_va)[:, 1]
    elif hasattr(est, "decision_function"):
        scores = est.decision_function(X_va)
    else:
        return None
    y_va = y[va]
    if len(np.unique(y_va)) < 2:
        return None
    return float(roc_auc_score(y_va, scores))


def inner_cv_mean_auc(estimator, X, y, inner_cv: BaseCrossValidator) -> float:
    aucs: list[float] = []
    for tr, va in inner_cv.split(X, y):
        auc = _fold_auc(estimator, X, y, tr, va)
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("-inf")


def suggest_sklearn_params(
    trial: Any,
    model_key: str,
    selection_mode: str,
    *,
    leaky_univariate: bool = False,
) -> dict[str, Any]:
    from ablation_runner import is_embedded_model

    params: dict[str, Any] = {}
    if leaky_univariate and not is_embedded_model(model_key):
        from ablation_runner_leaky import UNIVARIATE_K_GRID, UNIVARIATE_SCORE_FUNCS

        params["preselect__k"] = trial.suggest_categorical(
            "preselect__k", list(UNIVARIATE_K_GRID)
        )
        params["preselect__score_func"] = trial.suggest_categorical(
            "preselect__score_func", list(UNIVARIATE_SCORE_FUNCS.keys())
        )
    elif not is_embedded_model(model_key) and selection_mode in ("mrmr", "mrmr_stable"):
        params["preselect__n_features_total"] = trial.suggest_int(
            "preselect__n_features_total", 10, 50, step=5
        )

    if model_key == "logreg_l1":
        params["clf__C"] = trial.suggest_float("clf__C", 1e-4, 1e4, log=True)
        params["clf__class_weight"] = trial.suggest_categorical(
            "clf__class_weight", [None, "balanced"]
        )
    elif model_key == "elasticnet":
        params["clf__C"] = trial.suggest_float("clf__C", 1e-4, 1e4, log=True)
        params["clf__l1_ratio"] = trial.suggest_float("clf__l1_ratio", 0.05, 0.95)
        params["clf__class_weight"] = trial.suggest_categorical(
            "clf__class_weight", [None, "balanced"]
        )
    elif model_key == "svm":
        params["clf__C"] = trial.suggest_float("clf__C", 1e-4, 1e4, log=True)
        params["clf__kernel"] = trial.suggest_categorical("clf__kernel", ["linear", "rbf"])
        if params["clf__kernel"] == "rbf":
            params["clf__gamma"] = trial.suggest_float("clf__gamma", 1e-4, 1.0, log=True)
        params["clf__class_weight"] = trial.suggest_categorical(
            "clf__class_weight", [None, "balanced"]
        )
    elif model_key == "rf":
        params["clf__n_estimators"] = trial.suggest_int(
            "clf__n_estimators", 50, 500, step=50
        )
        params["clf__max_depth"] = trial.suggest_categorical(
            "clf__max_depth", [None, 5, 10, 20]
        )
        params["clf__class_weight"] = trial.suggest_categorical(
            "clf__class_weight", [None, "balanced"]
        )
    elif model_key == "xgb":
        params["clf__n_estimators"] = trial.suggest_int(
            "clf__n_estimators", 50, 500, step=50
        )
        params["clf__max_depth"] = trial.suggest_int("clf__max_depth", 3, 10)
        params["clf__learning_rate"] = trial.suggest_float(
            "clf__learning_rate", 0.01, 0.3, log=True
        )
        params["clf__subsample"] = trial.suggest_float("clf__subsample", 0.6, 1.0)
    elif model_key == "mlp":
        params["clf__hidden_layer_sizes"] = trial.suggest_categorical(
            "clf__hidden_layer_sizes", [(64,), (128, 64), (128, 64, 32)]
        )
        params["clf__alpha"] = trial.suggest_float("clf__alpha", 1e-5, 1e-2, log=True)
        params["clf__max_iter"] = 2000
    else:
        raise ValueError(f"modelo sem espaço Optuna: {model_key!r}")
    return params


def tune_pipeline_optuna(
    pipeline,
    X,
    y,
    inner_cv: BaseCrossValidator,
    *,
    model_key: str,
    selection_mode: str,
    n_trials: int = OPTUNA_TRIALS_DEFAULT,
    seed: int = 42,
    leaky_univariate: bool = False,
) -> TuneResult:
    if not HAS_OPTUNA:
        raise ImportError("Optuna não instalado — pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_sklearn_params(
            trial,
            model_key,
            selection_mode,
            leaky_univariate=leaky_univariate,
        )
        est = clone(pipeline)
        est.set_params(**params)
        return inner_cv_mean_auc(est, X, y, inner_cv)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best = clone(pipeline)
    best.set_params(**best_params)
    best.fit(X, y)
    return TuneResult(best, float(study.best_value), best_params, "optuna")


def tune_pipeline_grid(
    pipeline,
    X,
    y,
    inner_cv: BaseCrossValidator,
    *,
    model_key: str,
    selection_mode: str,
    n_jobs: int = -1,
    param_grid: dict[str, list[Any]] | None = None,
    leaky_univariate: bool = False,
) -> TuneResult:
    from ablation_runner import param_grid_for

    if param_grid is not None:
        pgrid = param_grid
    elif leaky_univariate:
        from ablation_runner_leaky import param_grid_leaky_univariate

        pgrid = param_grid_leaky_univariate(model_key)
    else:
        pgrid = param_grid_for(model_key, selection_mode)

    grid = GridSearchCV(
        pipeline,
        pgrid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    grid.fit(X, y)
    return TuneResult(
        grid.best_estimator_,
        float(grid.best_score_),
        dict(grid.best_params_),
        "grid",
    )


def tune_pipeline(
    pipeline,
    X,
    y,
    inner_cv: BaseCrossValidator,
    *,
    model_key: str,
    selection_mode: str = "raw",
    tuner: TunerKind = "grid",
    n_trials: int = OPTUNA_TRIALS_DEFAULT,
    seed: int = 42,
    n_jobs: int = -1,
    param_grid: dict[str, list[Any]] | None = None,
    leaky_univariate: bool = False,
) -> TuneResult:
    if tuner == "optuna":
        return tune_pipeline_optuna(
            pipeline,
            X,
            y,
            inner_cv,
            model_key=model_key,
            selection_mode=selection_mode,
            n_trials=n_trials,
            seed=seed,
            leaky_univariate=leaky_univariate,
        )
    return tune_pipeline_grid(
        pipeline,
        X,
        y,
        inner_cv,
        model_key=model_key,
        selection_mode=selection_mode,
        n_jobs=n_jobs,
        param_grid=param_grid,
        leaky_univariate=leaky_univariate,
    )


def parse_tuner(value: str) -> TunerKind:
    v = value.strip().lower()
    if v not in ("grid", "optuna"):
        raise ValueError(f"tuner inválido: {value!r} (use grid | optuna)")
    return v  # type: ignore[return-value]


if __name__ == "__main__":
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from ablation_runner import make_classifier

    rng = np.random.default_rng(0)
    n, p = 50, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = (rng.random(n) > 0.45).astype(int)
    pipe = ImbPipeline([("scaler", StandardScaler()), ("clf", make_classifier("logreg_l1", seed=0))])
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    res_g = tune_pipeline(pipe, X, y, cv, model_key="logreg_l1", tuner="grid")
    assert res_g.tuner == "grid" and res_g.best_inner_auc > 0
    if HAS_OPTUNA:
        res_o = tune_pipeline(
            pipe, X, y, cv, model_key="logreg_l1", tuner="optuna", n_trials=5, seed=0
        )
        assert res_o.tuner == "optuna" and "clf__C" in res_o.best_params
    print("ablation_optuna self-check ok")

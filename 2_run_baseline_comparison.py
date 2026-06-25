#!/usr/bin/env python3
"""Comparação nested CV: clínico baseline vs fusão (imagem + clínico).

Imagem only: use 2_run_ablation.py.

Fusão: hiperparâmetros do classificador são tunados só na imagem (mesmo
pipeline da ablação); depois o clf é reajustado em [clínico + imagem].
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ablation_analysis import prepare_ablation_df, summary_with_pooled
from ablation_prep import ROI_FILTER_DEFAULT, modality_wide_columns
from ablation_runner import (
    MODALITIES,
    PARAM_GRIDS,
    SELECTION_MODES,
    TASKS,
    TASK_PRESETS,
    fold_metrics,
    fmt_duration,
    make_pipeline,
    param_grid_for,
    patient_labels_from_long,
    repeat_ids,
    stable_pool_for_outer_train,
    tune_youden_threshold,
    wide_for_fold,
)

log = logging.getLogger("baseline_comparison")

CLINICAL_COLS = ("SEX", "AGE", "MMSE_SCORE", "ADAS_SCORE", "FAQ_SCORE", "CDR_GLOBAL")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _parse_tasks(value: str) -> tuple[str, ...]:
    if value in TASK_PRESETS:
        return TASK_PRESETS[value]
    tasks = _split_csv(value)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Tasks desconhecidas: {sorted(unknown)}")
    return tasks


def baseline_clinical_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """1 linha/paciente, só visita baseline."""
    sub = df_long.copy()
    if "slot" in sub.columns:
        sub = sub[sub["slot"].astype(str) == "baseline"]
    sub = sub.sort_values(["ID_PT", "ID_IMG"]).groupby("ID_PT", as_index=False).first()

    out = pd.DataFrame({"ID_PT": sub["ID_PT"].astype(str)})
    sex_map = {"M": 0, "F": 1, 0: 0, 1: 1}
    for col in CLINICAL_COLS:
        if col == "SEX":
            out[col] = sub["SEX"].map(sex_map)
        else:
            out[col] = pd.to_numeric(sub[col], errors="coerce")
    return out


def attach_clinical(wide: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    wide["ID_PT"] = wide["ID_PT"].astype(str)
    clin = clinical.set_index("ID_PT")
    for col in CLINICAL_COLS:
        wide[col] = wide["ID_PT"].map(clin[col])
    return wide


CLINICAL_MODELS = ("svm", "rf", "xgb", "mlp")


def _clinical_pipeline(model_key: str, seed: int) -> Pipeline:
    clf_map = {
        "svm": SVC(probability=True, random_state=seed),
        "rf": RandomForestClassifier(random_state=seed),
        "xgb": XGBClassifier(eval_metric="logloss", random_state=seed),
        "mlp": MLPClassifier(
            activation="relu", alpha=1e-3, batch_size=32, random_state=seed, max_iter=500,
        ),
    }
    if model_key not in clf_map:
        raise ValueError(f"modelo clínico suportado: {CLINICAL_MODELS}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf_map[model_key])])


def _clinical_param_grid(model_key: str) -> dict[str, list[Any]]:
    if model_key not in PARAM_GRIDS:
        raise ValueError(f"modelo desconhecido: {model_key}")
    return {k: v for k, v in PARAM_GRIDS[model_key].items() if k.startswith("clf__")}


def nested_cv_clinical(
    df_long: pd.DataFrame,
    *,
    task_id: str,
    model_key: str,
    seed: int = 42,
    repeat_id: int = 0,
    k_out: int = 5,
    k_in: int = 5,
) -> pd.DataFrame:
    task = TASKS[task_id]
    clinical = baseline_clinical_table(df_long)
    pt = patient_labels_from_long(df_long, task)
    y = pt["y"].to_numpy(dtype=int)
    pts = pt["ID_PT"].astype(str).to_numpy()

    outer_cv = StratifiedKFold(k_out, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []

    for fold, (tr_rel, te_rel) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
        train_pts = set(pts[tr_rel])
        test_pts = set(pts[te_rel])

        wide = patient_labels_from_long(df_long, task)[["ID_PT", "GROUP"]].merge(
            clinical, on="ID_PT", how="left",
        )
        wide["y"] = wide["GROUP"].map(task.label_map).astype(int)
        if wide[list(CLINICAL_COLS)].isna().any().any():
            raise ValueError(
                f"fold {fold}: NaN em variáveis clínicas baseline — "
                f"verifique MMSE/ADAS/FAQ/CDR em merge_long"
            )
        feature_cols = list(CLINICAL_COLS)

        X = wide[feature_cols].astype(float)
        y_wide = wide["y"].to_numpy(dtype=int)
        tr_mask = wide["ID_PT"].astype(str).isin(train_pts).to_numpy()
        te_mask = wide["ID_PT"].astype(str).isin(test_pts).to_numpy()
        train_idx = np.where(tr_mask)[0]
        test_idx = np.where(te_mask)[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_wide[train_idx], y_wide[test_idx]

        inner_cv = StratifiedKFold(k_in, shuffle=True, random_state=seed + fold)
        grid = GridSearchCV(
            _clinical_pipeline(model_key, seed),
            _clinical_param_grid(model_key),
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        train_scores = cross_val_predict(best, X_train, y_train, cv=inner_cv, method="predict_proba")[:, 1]
        threshold = tune_youden_threshold(y_train, train_scores)
        test_scores = best.predict_proba(X_test)[:, 1]
        test_preds = (test_scores >= threshold).astype(int)

        rows.append({
            "feature_set": "clinical",
            "task": task.task_id,
            "modality": "clinical",
            "modality_label": "clínico baseline",
            "model_key": model_key,
            "with_combat": False,
            "selection_mode": "none",
            "repeat_id": repeat_id,
            "fold": fold,
            "best_model": best.named_steps["clf"].__class__.__name__,
            "best_inner_auc": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, default=str),
            "threshold": threshold,
            "n_features_raw": len(feature_cols),
            "n_features_selected": len(feature_cols),
            "selected_features": json.dumps(feature_cols),
            "test_id_pts": json.dumps(wide.iloc[test_idx]["ID_PT"].astype(str).tolist()),
            "test_y_true": json.dumps(y_test.tolist()),
            "test_scores": json.dumps(test_scores.tolist()),
            **fold_metrics(y_test, test_scores, test_preds),
        })
    return pd.DataFrame(rows)


def nested_cv_fusion(
    df_long: pd.DataFrame,
    *,
    task_id: str,
    modality: str,
    model_key: str,
    selection_mode: str,
    with_combat: bool,
    roi: str,
    seed: int = 42,
    repeat_id: int = 0,
    k_out: int = 5,
    k_in: int = 5,
) -> pd.DataFrame:
    """Imagem (com seleção) + clínico baseline concatenados."""
    task = TASKS[task_id]
    clinical = baseline_clinical_table(df_long)
    pt = patient_labels_from_long(df_long, task)
    y = pt["y"].to_numpy(dtype=int)
    pts = pt["ID_PT"].astype(str).to_numpy()

    outer_cv = StratifiedKFold(k_out, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []

    for fold, (tr_rel, te_rel) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
        train_pts = set(pts[tr_rel])
        test_pts = set(pts[te_rel])

        wide = wide_for_fold(
            df_long,
            train_pts=train_pts,
            test_pts=test_pts,
            with_combat=with_combat,
            fold_id=fold,
            combat_quiet=True,
        )
        wide = wide[wide["GROUP"].astype(str).isin(task.groups)].copy()
        wide = attach_clinical(wide, clinical)
        wide["y"] = wide["GROUP"].map(task.label_map).astype(int)
        if wide[list(CLINICAL_COLS)].isna().any().any():
            raise ValueError(
                f"fold {fold}: NaN em variáveis clínicas baseline — "
                f"verifique MMSE/ADAS/FAQ/CDR em merge_long"
            )

        image_cols = modality_wide_columns(wide.columns, modality, roi=roi)
        feature_cols = list(CLINICAL_COLS) + image_cols
        X = wide[feature_cols].astype(float)
        y_wide = wide["y"].to_numpy(dtype=int)

        tr_mask = wide["ID_PT"].astype(str).isin(train_pts).to_numpy()
        te_mask = wide["ID_PT"].astype(str).isin(test_pts).to_numpy()
        train_idx = np.where(tr_mask)[0]
        test_idx = np.where(te_mask)[0]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train, y_test = y_wide[train_idx], y_wide[test_idx]

        # ponytail: clínico passa direto; imagem passa por preselect+scaler do pipeline existente
        img_train = X_train[image_cols]
        img_test = X_test[image_cols]
        clin_train = X_train[list(CLINICAL_COLS)].to_numpy(dtype=float)
        clin_test = X_test[list(CLINICAL_COLS)].to_numpy(dtype=float)

        pipeline = make_pipeline(selection_mode, model_key, roi=roi, seed=seed)
        inner_cv = StratifiedKFold(k_in, shuffle=True, random_state=seed + fold)

        if SELECTION_MODES[selection_mode].get("use_stable_pool"):
            allowed, _ = stable_pool_for_outer_train(
                img_train, y_train, inner_cv=inner_cv, roi=roi,
            )
            pipeline.set_params(preselect__allowed_columns=allowed)

        grid = GridSearchCV(
            pipeline,
            param_grid_for(model_key, selection_mode),
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(img_train, y_train)
        best = grid.best_estimator_

        scaler = StandardScaler()
        clin_train_s = scaler.fit_transform(clin_train)
        clin_test_s = scaler.transform(clin_test)

        img_train_t = best.named_steps["preselect"].transform(img_train)
        img_test_t = best.named_steps["preselect"].transform(img_test)
        img_train_s = best.named_steps["scaler"].transform(img_train_t)
        img_test_s = best.named_steps["scaler"].transform(img_test_t)

        X_train_f = np.hstack([clin_train_s, img_train_s])
        X_test_f = np.hstack([clin_test_s, img_test_s])

        clf = best.named_steps["clf"]
        clf.fit(X_train_f, y_train)

        train_scores = cross_val_predict(clf, X_train_f, y_train, cv=inner_cv, method="predict_proba")[:, 1]
        threshold = tune_youden_threshold(y_train, train_scores)
        test_scores = clf.predict_proba(X_test_f)[:, 1]
        test_preds = (test_scores >= threshold).astype(int)

        preselect = best.named_steps["preselect"]
        selected = list(CLINICAL_COLS) + list(preselect.selected_names_)
        rows.append({
            "feature_set": "fusion",
            "task": task.task_id,
            "modality": modality,
            "modality_label": f"fusion ({MODALITIES[modality]['label']} + clínico)",
            "model_key": model_key,
            "with_combat": with_combat,
            "selection_mode": selection_mode,
            "repeat_id": repeat_id,
            "fold": fold,
            "best_model": clf.__class__.__name__,
            "best_inner_auc": float(grid.best_score_),
            "best_params": json.dumps(grid.best_params_, default=str),
            "threshold": threshold,
            "n_features_raw": len(feature_cols),
            "n_features_selected": len(selected),
            "selected_features": json.dumps(selected),
            "test_id_pts": json.dumps(wide.iloc[test_idx]["ID_PT"].astype(str).tolist()),
            "test_y_true": json.dumps(y_test.tolist()),
            "test_scores": json.dumps(test_scores.tolist()),
            **fold_metrics(y_test, test_scores, test_preds),
        })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline clínico vs fusão (imagem+clínico).")
    p.add_argument("--feature-set", choices=["clinical", "fusion"], required=True)
    p.add_argument("--tasks", default="smci_pmci")
    p.add_argument("--modality", default="disp", help="Só para fusion: disp|vol|all|...")
    p.add_argument("--models", default="svm,rf,xgb,mlp")
    p.add_argument("--selection", default="mrmr_stable")
    p.add_argument("--combat", choices=["false", "true"], default="false")
    p.add_argument("--repeats", "-r", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", default=ROI_FILTER_DEFAULT)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("csvs/longitudinal_4_groups/ablation_results/baseline_comparison"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    tasks = _parse_tasks(args.tasks)
    models = _split_csv(args.models)
    with_combat = args.combat == "true"
    base = Path(f"csvs/longitudinal_4_groups/ablation/{args.roi}")
    out_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "clinical":
        df_long = pd.read_csv(base / "merge_long.csv")
    else:
        long_path = base / MODALITIES[args.modality]["long"]
        df_long = pd.read_csv(long_path)

    all_rows: list[pd.DataFrame] = []
    t0 = time.monotonic()

    for task_id in tasks:
        for model_key in models:
            for repeat_id in repeat_ids(args.repeats):
                seed_rep = args.seed + repeat_id * 1000
                log.info(
                    "%s | task=%s | model=%s | rep=%s",
                    args.feature_set, task_id, model_key, repeat_id,
                )
                if args.feature_set == "clinical":
                    res = nested_cv_clinical(
                        df_long,
                        task_id=task_id,
                        model_key=model_key,
                        seed=seed_rep,
                        repeat_id=repeat_id,
                    )
                else:
                    res = nested_cv_fusion(
                        df_long,
                        task_id=task_id,
                        modality=args.modality,
                        model_key=model_key,
                        selection_mode=args.selection,
                        with_combat=with_combat,
                        roi=args.roi,
                        seed=seed_rep,
                        repeat_id=repeat_id,
                    )
                all_rows.append(res)

    df = prepare_ablation_df(pd.concat(all_rows, ignore_index=True))
    summary = summary_with_pooled(df)

    tag = f"{args.feature_set}"
    if args.feature_set == "fusion":
        tag += f"_{args.modality}_{args.selection}_{'combat' if with_combat else 'nocombat'}"

    raw_path = out_dir / f"{tag}_results_all.csv"
    sum_path = out_dir / f"{tag}_summary.csv"
    df.to_csv(raw_path, index=False)
    summary.to_csv(sum_path, index=False)

    log.info("tempo: %s", fmt_duration(time.monotonic() - t0))
    log.info("salvo: %s", raw_path)
    log.info("salvo: %s", sum_path)
    cols = [
        "task", "modality", "model_key", "with_combat", "selection_mode",
        "auc_mean", "auc_std", "auc_pooled", "n_features_mean",
    ]
    cols = [c for c in cols if c in summary.columns]
    log.info("\n%s", summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
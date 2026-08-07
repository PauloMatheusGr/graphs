"""Late fusion: combina scores OOF de branches mono-mod (mesmo fold/repeat)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from ablation_analysis import summary_with_pooled
from ablation_representation import (
    FusionSlot,
    default_late_fusion_results_dir,
    fusion_fingerprint,
    fusion_label,
    mono_results_dir_for_slot,
    resolve_stable_pool_min_timepoints,
)
from ablation_runner import (
    MODALITIES,
    TASKS,
    fmt_duration,
    nested_cv_ablation,
    repeat_ids,
)

log = logging.getLogger(__name__)

COMBINE_MODES = ("mean", "weighted")


def _filter_branch(
    df: pd.DataFrame,
    *,
    task: str,
    model_key: str,
    with_combat: bool,
    selection_mode: str,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["task"].astype(str) == task]
    out = out[out["model_key"].astype(str) == model_key]
    out = out[out["with_combat"] == with_combat]
    if "selection_mode" in out.columns:
        out = out[out["selection_mode"].astype(str) == selection_mode]
    return out.reset_index(drop=True)


def _explode_fold(row: pd.Series) -> pd.DataFrame:
    ids = [str(x) for x in json.loads(row["test_id_pts"])]
    y = [int(x) for x in json.loads(row["test_y_true"])]
    s = [float(x) for x in json.loads(row["test_scores"])]
    if not (len(ids) == len(y) == len(s)):
        raise ValueError("test_id_pts / y_true / scores length mismatch")
    return pd.DataFrame({"ID_PT": ids, "y": y, "score": s})


def _fold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    out: dict[str, float] = {
        "auc": float("nan"),
        "auc_pr": float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "bal_acc": float(balanced_accuracy_score(y, pred)) if len(y) else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "sens_pos": float("nan"),
        "spec_neg": float("nan"),
        "f1_pos": float("nan"),
    }
    if len(np.unique(y)) >= 2:
        out["auc"] = float(roc_auc_score(y, scores))
        out["auc_pr"] = float(average_precision_score(y, scores))
        out["f1_pos"] = float(f1_score(y, pred, pos_label=1, zero_division=0))
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        out["sens_pos"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out["spec_neg"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return out


def combine_branch_frames(
    branches: list[pd.DataFrame],
    *,
    weights: list[float] | None,
    combine: str,
    modality_label: str,
    fingerprint: str,
) -> pd.DataFrame:
    """Une branches mono-mod fold a fold; score = mean ou weighted mean."""
    if combine not in COMBINE_MODES:
        raise ValueError(f"combine inválido: {combine!r} (use {'|'.join(COMBINE_MODES)})")
    if len(branches) < 2:
        raise ValueError("late fusion exige ≥2 branches")
    n = len(branches)
    if weights is None:
        w = np.ones(n, dtype=float) / n
    else:
        if len(weights) != n:
            raise ValueError(f"weights len={len(weights)} != n_branches={n}")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("weights devem ser ≥0 e somar >0")
        w = w / w.sum()

    keys = ["task", "model_key", "with_combat", "selection_mode", "repeat_id", "fold"]
    for i, b in enumerate(branches):
        miss = [c for c in keys + ["test_id_pts", "test_y_true", "test_scores"] if c not in b.columns]
        if miss:
            raise ValueError(f"branch[{i}] sem colunas: {miss}")

    # interseção de folds presentes em todos
    key_sets = []
    for b in branches:
        key_sets.append({tuple(r[c] for c in keys) for _, r in b.iterrows()})
    common = set.intersection(*key_sets)
    if not common:
        raise ValueError("nenhum (repeat,fold) comum entre branches — confira seeds/repeats")

    rows: list[dict[str, Any]] = []
    for key in sorted(common, key=lambda t: (str(t[0]), str(t[1]), bool(t[2]), str(t[3]), int(t[4]), int(t[5]))):
        fold_parts = []
        meta0 = None
        n_feat = 0
        sel_all: list[str] = []
        for b, wi in zip(branches, w):
            mask = np.ones(len(b), dtype=bool)
            for c, v in zip(keys, key):
                mask &= b[c].to_numpy() == v
            sub = b.loc[mask]
            if len(sub) != 1:
                raise ValueError(f"fold ambíguo {key}: {len(sub)} rows")
            r = sub.iloc[0]
            if meta0 is None:
                meta0 = r
            part = _explode_fold(r).rename(columns={"score": f"s_{len(fold_parts)}"})
            fold_parts.append(part)
            n_feat += int(r.get("n_features_selected", 0) or 0)
            if "selected_features" in r and pd.notna(r["selected_features"]):
                try:
                    sel_all.extend(json.loads(r["selected_features"]))
                except (TypeError, json.JSONDecodeError):
                    pass

        merged = fold_parts[0][["ID_PT", "y"]]
        for i, part in enumerate(fold_parts):
            merged = merged.merge(part[["ID_PT", f"s_{i}"]], on="ID_PT", how="inner")
        if merged.empty:
            continue
        S = np.column_stack([merged[f"s_{i}"].to_numpy(dtype=float) for i in range(n)])
        scores = (S * w.reshape(1, -1)).sum(axis=1)
        y = merged["y"].to_numpy(dtype=int)
        thr = float(meta0["threshold"]) if meta0 is not None and "threshold" in meta0.index else 0.5
        metrics = _fold_metrics(y, scores, threshold=thr)
        assert meta0 is not None
        rows.append(
            {
                "representation": "late_fusion",
                "task": meta0["task"],
                "with_combat": bool(meta0["with_combat"]),
                "selection_mode": meta0["selection_mode"],
                "modality": f"late__{fingerprint}",
                "modality_label": modality_label,
                "model_key": meta0["model_key"],
                "tuner": meta0["tuner"] if "tuner" in meta0.index else "reuse",
                "repeat_id": int(meta0["repeat_id"]),
                "fold": int(meta0["fold"]),
                "best_model": meta0["best_model"] if "best_model" in meta0.index else meta0["model_key"],
                "best_inner_auc": float("nan"),
                "best_params": json.dumps({"late_combine": combine, "weights": w.tolist()}),
                "threshold": thr,
                "n_features_raw": n_feat,
                "n_features_after_stable_pool": n_feat,
                "n_features_after_filters": n_feat,
                "n_features_selected": n_feat,
                "removed_by_stable_pool": "[]",
                "removed_by_filters": "[]",
                "removed_by_mrmr": "[]",
                "selected_features": json.dumps(sel_all),
                "test_id_pts": json.dumps(merged["ID_PT"].tolist()),
                "test_y_true": json.dumps(y.tolist()),
                "test_scores": json.dumps(scores.tolist()),
                **metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _load_or_run_branch(
    *,
    slot: FusionSlot,
    base_dir: Path,
    task_id: str,
    model_key: str,
    selection_mode: str,
    with_combat: bool,
    reuse_disk: bool,
    run_missing: bool,
    roi: str,
    seed: int,
    r_repeats: int,
    combat_quiet: bool,
    stable_pool_min_pct: int,
    stable_pool_min_timepoints: int,
    stable_pool_bootstrap: int,
    stable_pool_l1_c: float,
    tuner: str,
    optuna_trials: int,
    verbose: bool,
) -> pd.DataFrame:
    mod, rep = slot
    out_dir = mono_results_dir_for_slot(base_dir, slot, protocol="abs")
    csv_path = out_dir / "ablation_results_all.csv"
    if reuse_disk and csv_path.is_file():
        raw = pd.read_csv(csv_path)
        filt = _filter_branch(
            raw,
            task=task_id,
            model_key=model_key,
            with_combat=with_combat,
            selection_mode=selection_mode,
        )
        if not filt.empty:
            log.info("reuse %s@%s ← %s (%d folds)", mod, rep, csv_path, len(filt))
            return filt
        log.warning("reuse miss filter %s — %s", csv_path, slot)

    if not run_missing:
        raise FileNotFoundError(
            f"branch ausente/filtrado: {mod}@{rep} ({csv_path}). "
            f"Rode mono-mod ou passe --run-missing"
        )

    log.info("run missing branch %s@%s → %s", mod, rep, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = base_dir / MODALITIES[mod]["long"]
    if not long_path.is_file():
        # all-merge long works for any modality columns
        long_path = base_dir / MODALITIES["all"]["long"]
    df_long = pd.read_csv(long_path)
    stable_tp = resolve_stable_pool_min_timepoints(rep, stable_pool_min_timepoints, log=log)
    task = TASKS[task_id]
    reps = list(repeat_ids(r_repeats))
    n_reps = len(reps)
    parts: list[pd.DataFrame] = []
    t0 = time.monotonic()
    for job_no, repeat_id in enumerate(reps, start=1):
        seed_rep = seed + repeat_id * 1000
        elapsed = time.monotonic() - t0
        eta_s = (elapsed / job_no) * (n_reps - job_no) if job_no else 0.0
        log.info(
            "[%d/%d] %s | %s@%s | combat=%s | %s | %s | rep=%d | elapsed=%s eta=%s",
            job_no,
            n_reps,
            task_id,
            mod,
            rep,
            with_combat,
            selection_mode,
            model_key,
            repeat_id,
            fmt_duration(elapsed),
            fmt_duration(eta_s),
        )
        job_t0 = time.monotonic()
        res = nested_cv_ablation(
            df_long,
            task=task,
            modality=mod,
            model_key=model_key,
            selection_mode=selection_mode,
            with_combat=with_combat,
            roi=roi,
            base_dir=base_dir,
            seed=seed_rep,
            repeat_id=repeat_id,
            combat_quiet=combat_quiet,
            stable_pool_min_pct=stable_pool_min_pct,
            stable_pool_min_timepoints=stable_tp,
            stable_pool_bootstrap=stable_pool_bootstrap,
            stable_pool_l1_c=stable_pool_l1_c,
            tuner=tuner,
            optuna_trials=optuna_trials,
            verbose=verbose,
            representation=rep,
        )
        parts.append(res)
        auc_mean = float(res["auc"].mean()) if "auc" in res.columns and len(res) else float("nan")
        log.info(
            "[%d/%d] ok | %s@%s | auc_mean=%.3f | %d folds | job=%s",
            job_no,
            n_reps,
            mod,
            rep,
            auc_mean,
            len(res),
            fmt_duration(time.monotonic() - job_t0),
        )
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    summary_with_pooled(combined).to_csv(out_dir / "ablation_summary.csv", index=False)
    log.info(
        "branch done %s@%s | %d folds | %s → %s",
        mod,
        rep,
        len(combined),
        fmt_duration(time.monotonic() - t0),
        csv_path,
    )
    return _filter_branch(
        combined,
        task=task_id,
        model_key=model_key,
        with_combat=with_combat,
        selection_mode=selection_mode,
    )


def run_late_fusion_ablation_suite(
    *,
    fusion_slots: tuple[FusionSlot, ...],
    base_dir: Path | str = "csvs/cohorts/48m_12m/ablation/hippocampus",
    roi: str = "hippocampus",
    tasks: tuple[str, ...] = ("smci_pmci",),
    models: tuple[str, ...] = ("svm",),
    selection_modes: tuple[str, ...] = ("l1_stable",),
    with_combat_flags: tuple[bool, ...] = (False,),
    results_dir: Path | str | None = None,
    seed: int = 42,
    r_repeats: int = 10,
    verbose: bool = False,
    combat_quiet: bool = True,
    stable_pool_min_pct: int = 70,
    stable_pool_min_timepoints: int = 0,
    stable_pool_bootstrap: int = 50,
    stable_pool_l1_c: float = 0.1,
    tuner: str = "optuna",
    optuna_trials: int = 30,
    reuse_disk: bool = True,
    run_missing: bool = False,
    combine: str = "mean",
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Late fusion → ablation_results_late_fusion/{fingerprint}/; modality=late__{fp}."""
    base = Path(base_dir)
    out_dir = default_late_fusion_results_dir(base, fusion_slots, results_dir=results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = fusion_fingerprint(fusion_slots)
    label = fusion_label(fusion_slots)
    n_branches = len(fusion_slots)
    log.info(
        "late fusion %s | protocol=late__%s | branches=%d | out=%s",
        label, fp, n_branches, out_dir,
    )

    t0 = time.monotonic()
    all_rows: list[pd.DataFrame] = []
    for task_id in tasks:
        for with_combat in with_combat_flags:
            for selection_mode in selection_modes:
                for model_key in models:
                    branches: list[pd.DataFrame] = []
                    for b_i, slot in enumerate(fusion_slots, start=1):
                        log.info(
                            "branch %d/%d | %s@%s | reuse_disk=%s run_missing=%s",
                            b_i, n_branches, slot[0], slot[1], reuse_disk, run_missing,
                        )
                        branches.append(
                            _load_or_run_branch(
                                slot=slot,
                                base_dir=base,
                                task_id=task_id,
                                model_key=model_key,
                                selection_mode=selection_mode,
                                with_combat=with_combat,
                                reuse_disk=reuse_disk,
                                run_missing=run_missing,
                                roi=roi,
                                seed=seed,
                                r_repeats=r_repeats,
                                combat_quiet=combat_quiet,
                                stable_pool_min_pct=stable_pool_min_pct,
                                stable_pool_min_timepoints=stable_pool_min_timepoints,
                                stable_pool_bootstrap=stable_pool_bootstrap,
                                stable_pool_l1_c=stable_pool_l1_c,
                                tuner=tuner,
                                optuna_trials=optuna_trials,
                                verbose=verbose,
                            )
                        )
                    log.info("combinando scores (%s) | %d branches…", combine, n_branches)
                    combined = combine_branch_frames(
                        branches,
                        weights=weights,
                        combine=combine,
                        modality_label=f"late({label})",
                        fingerprint=fp,
                    )
                    all_rows.append(combined)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        out.to_csv(out_dir / "ablation_results_all.csv", index=False)
        summary = summary_with_pooled(out)
        summary.to_csv(out_dir / "ablation_summary.csv", index=False)
        log.info(
            "salvo %s (%d folds) | %s",
            out_dir / "ablation_results_all.csv",
            len(out),
            fmt_duration(time.monotonic() - t0),
        )
        cols = [c for c in ("modality", "auc_patient_mean", "auc_pooled", "n_features_mean") if c in summary.columns]
        log.info("\n%s", summary[cols].to_string(index=False))
    else:
        log.error("late fusion sem linhas")
    return out


if __name__ == "__main__":
    # ponytail: self-check com scores sintéticos alinhados
    def _fake(mod: str, scores: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "task": "smci_pmci",
                    "model_key": "svm",
                    "with_combat": False,
                    "selection_mode": "l1_stable",
                    "repeat_id": 0,
                    "fold": 0,
                    "modality": mod,
                    "tuner": "optuna",
                    "best_model": "svm",
                    "threshold": 0.5,
                    "n_features_selected": 1,
                    "selected_features": json.dumps([f"{mod}_f"]),
                    "test_id_pts": json.dumps(["a", "b", "c", "d"]),
                    "test_y_true": json.dumps([0, 0, 1, 1]),
                    "test_scores": json.dumps(scores),
                    "auc": 0.5,
                }
            ]
        )

    c = combine_branch_frames(
        [_fake("shape", [0.1, 0.2, 0.8, 0.9]), _fake("tex", [0.2, 0.3, 0.7, 0.85])],
        weights=None,
        combine="mean",
        modality_label="late(test)",
        fingerprint="t1_shape__t1_deltas_texture",
    )
    assert len(c) == 1
    assert c.iloc[0]["modality"] == "late__t1_shape__t1_deltas_texture"
    s = json.loads(c.iloc[0]["test_scores"])
    assert abs(s[0] - 0.15) < 1e-9
    print("ablation_late_fusion self-check ok")

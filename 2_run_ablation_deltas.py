#!/usr/bin/env python3

# python 2_run_ablation_deltas.py --modality vol,shape,texture,all --tasks smci_pmci \
#   --selection raw --models svm,logreg_l1,elasticnet --combat false --repeats 10
#
# Seleção embarcada (L1/ElasticNet): --selection raw --models logreg_l1 ou elasticnet
# (preselect ignorado; coef ≠ 0 = features selecionadas)



"""Nested CV ablation com T1 + deltas relativos (D21, D31, SLOPE)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from ablation_analysis import prepare_ablation_df, summary_with_pooled
from ablation_deltas import PROTOCOL_T1_DELTAS
from ablation_prep import ROI_FILTER_DEFAULT
from ablation_runner import (
    MODALITIES,
    SELECTION_MODES,
    STABLE_POOL_MIN_PCT,
    STABLE_POOL_MIN_TIMEPOINTS,
    STABLE_POOL_N_FEATURES,
    STABLE_POOL_BOOTSTRAP,
    STABLE_POOL_L1_C,
    TASKS,
    TASK_PRESETS,
    fmt_duration,
    run_full_ablation_suite,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("ablation_deltas")


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


def _parse_selection(value: str) -> tuple[str, ...]:
    modes = _split_csv(value)
    unknown = set(modes) - set(SELECTION_MODES)
    if unknown:
        raise argparse.ArgumentTypeError(f"Modos desconhecidos: {sorted(unknown)}")
    return modes


def _parse_combat(value: str) -> tuple[bool, ...]:
    v = value.strip().lower()
    if v == "both":
        return (False, True)
    if v in ("true", "1", "yes"):
        return (True,)
    if v in ("false", "0", "no"):
        return (False,)
    raise argparse.ArgumentTypeError("combat: use false | true | both")


def _parse_modalities(value: str) -> tuple[str, ...]:
    mods = _split_csv(value)
    unknown = set(mods) - set(MODALITIES)
    if unknown:
        raise argparse.ArgumentTypeError(f"Modalidades desconhecidas: {sorted(unknown)}")
    return mods


def setup_logging(*, log_file: Path | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logging.getLogger("ablation_deltas").setLevel(level)
    logging.getLogger("ablation_runner").setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablação nested CV com T1 + deltas relativos (D21, D31, SLOPE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modality", default="all")
    p.add_argument("--tasks", default="all")
    p.add_argument("--selection", default="l1_stable")
    p.add_argument("--models", default="svm,logreg_l1,elasticnet", help="svm,rf,mlp,logreg_l1,elasticnet,...")
    p.add_argument("--combat", default="false", type=_parse_combat)
    p.add_argument("--repeats", "-r", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", default=ROI_FILTER_DEFAULT)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override pasta de saída (default: ablation_results_deltas/{modality})",
    )
    p.add_argument("--stable-pool-min-pct", type=int, default=STABLE_POOL_MIN_PCT)
    p.add_argument("--stable-pool-min-timepoints", type=int, default=STABLE_POOL_MIN_TIMEPOINTS,
                   help="0 = sem filtro temporal no pool")
    p.add_argument("--stable-pool-n", type=int, default=STABLE_POOL_N_FEATURES)
    p.add_argument("--stable-bootstrap", type=int, default=STABLE_POOL_BOOTSTRAP)
    p.add_argument("--stable-l1-c", type=float, default=STABLE_POOL_L1_C)
    p.add_argument("--tuner", choices=["grid", "optuna"], default="grid")
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modalities = _parse_modalities(args.modality)
    tasks = _parse_tasks(args.tasks)
    selection_modes = _parse_selection(args.selection)
    models = _split_csv(args.models)

    base_dir = Path(f"csvs/longitudinal_4_groups/ablation/{args.roi}")

    if len(modalities) == 1 and args.results_dir is None:
        results_dir = Path(f"csvs/longitudinal_4_groups/ablation_results_deltas/{modalities[0]}")
    else:
        results_dir = args.results_dir

    log_path = None
    if not args.no_log_file:
        log_path = args.log_file or Path(
            f"logs/ablation_deltas_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
    setup_logging(log_file=log_path, verbose=args.verbose)

    n_reps = 1 if args.repeats == 0 else args.repeats
    total_jobs = (
        len(modalities)
        * len(tasks)
        * len(args.combat)
        * len(selection_modes)
        * len(models)
        * n_reps
    )

    log.info("=== ablação nested CV (%s) ===", PROTOCOL_T1_DELTAS)
    log.info("modalidades:  %s", modalities)
    log.info("tasks:        %s", tasks)
    log.info("seleção:      %s", selection_modes)
    log.info("modelos:      %s", models)
    log.info("combat:       %s", args.combat)
    log.info("repetições:   %s", args.repeats)
    log.info("jobs totais:  %d", total_jobs)
    log.info("base:         %s", base_dir)
    log.info("resultados:   %s", results_dir or "ablation_results_deltas/{modality}/")
    if log_path:
        log.info("log file:     %s", log_path.resolve())

    t0 = time.monotonic()
    try:
        df = run_full_ablation_suite(
            base_dir=base_dir,
            roi=args.roi,
            tasks=tasks,
            modalities=modalities,
            models=models,
            selection_modes=selection_modes,
            with_combat_flags=args.combat,
            results_dir=results_dir,
            seed=args.seed,
            r_repeats=args.repeats,
            verbose=args.verbose,
            combat_quiet=True,
            stable_pool_min_pct=args.stable_pool_min_pct,
            stable_pool_min_timepoints=args.stable_pool_min_timepoints,
            stable_pool_n_features=args.stable_pool_n,
            stable_pool_bootstrap=args.stable_bootstrap,
            stable_pool_l1_c=args.stable_l1_c,
            tuner=args.tuner,
            optuna_trials=args.optuna_trials,
        )
    except Exception:
        elapsed = time.monotonic() - t0
        log.exception("ablacao deltas falhou apos %s", fmt_duration(elapsed))
        return 1

    elapsed = time.monotonic() - t0
    df = prepare_ablation_df(df)
    summary = summary_with_pooled(df)

    out_dirs = {results_dir} if results_dir else {
        Path(f"csvs/longitudinal_4_groups/ablation_results_deltas/{m}") for m in modalities
    }
    for d in sorted(out_dirs, key=str):
        log.info("csv: %s", d / "ablation_results_all.csv")
        log.info("csv: %s", d / "ablation_summary.csv")

    cols = [
        "selection_mode", "task", "modality", "model_key", "with_combat",
        "n_features_mean", "auc_mean", "auc_std", "auc_pooled",
    ]
    cols = [c for c in cols if c in summary.columns]
    avg_job = elapsed / total_jobs if total_jobs else 0
    log.info(
        "=== tempo: %s (%d jobs, média %s/job) ===",
        fmt_duration(elapsed),
        total_jobs,
        fmt_duration(avg_job),
    )
    log.info("%d linhas | %d configs", len(df), len(summary))
    log.info("\n%s", summary[cols].to_string(index=False))
    log.info("análise: 3_results.ipynb (RESULTS_ROOT=ablation_results_deltas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

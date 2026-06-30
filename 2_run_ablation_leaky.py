#!/usr/bin/env python3
"""Ablação protocolo LEAKY (literatura): pré-processamento global antes do CV.

AVISO: resultados inflados de propósito — comparar com estudos sem anti-leakage.
Saída: csvs/longitudinal_4_groups/ablation_results_leaky/{modality}/
Análise: 3_results.ipynb com RESULTS_ROOT apontando para ablation_results_leaky.
"""

# # só leaky_global (mais defensável pra comparar com literatura)
# python 2_run_ablation_leaky.py --inflate ""

# # pseudo-replicação isolada
# python 2_run_ablation_leaky.py --inflate pseudo

# # combina infladores
# python 2_run_ablation_leaky.py --inflate pseudo,fulltune

# # pior caso absurdo (o que gerou AUC 1.000)
# python 2_run_ablation_leaky.py --inflate max

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
)
from ablation_runner_leaky import run_full_ablation_suite_leaky

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("ablation_leaky")


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

    logging.getLogger("ablation_leaky").setLevel(level)
    logging.getLogger("ablation_runner_leaky").setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablação LEAKY — z-score/ComBat/seleção no dataset inteiro (comparar com literatura).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--modality", default="all", help="vol|shape|texture|disp|all")
    p.add_argument("--tasks", default="all", help="Preset core|cross|all ou lista")
    p.add_argument("--selection", default="l1_stable", help=f"Modos: {','.join(SELECTION_MODES)}")
    p.add_argument("--models", default="svm,rf,mlp", help="Modelos separados por vírgula")
    p.add_argument("--combat", default="false", type=_parse_combat, help="false | true | both")
    p.add_argument("--repeats", "-r", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", default=ROI_FILTER_DEFAULT)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override saída (default: ablation_results_leaky/{modality}/)",
    )
    p.add_argument("--stable-pool-min-pct", type=int, default=STABLE_POOL_MIN_PCT)
    p.add_argument("--stable-pool-min-timepoints", type=int, default=STABLE_POOL_MIN_TIMEPOINTS)
    p.add_argument("--stable-pool-n", type=int, default=STABLE_POOL_N_FEATURES)
    p.add_argument("--stable-bootstrap", type=int, default=STABLE_POOL_BOOTSTRAP)
    p.add_argument("--stable-l1-c", type=float, default=STABLE_POOL_L1_C)
    p.add_argument("--tuner", choices=["grid", "optuna"], default="grid")
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument(
        "--inflate",
        default="",
        help=(
            "Infladores extra (vírgula): pseudo (1 amostra/visita, split por linha) | "
            "fulltune (GridSearch no dataset inteiro) | testthr (threshold no teste) | "
            "univariate (SelectKBest ANOVA/MI global, grid K) | "
            "max (todos). Vazio = só leaky_global."
        ),
    )
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


INFLATE_CHOICES = {"pseudo", "fulltune", "testthr", "univariate", "max"}


def _parse_inflate(value: str) -> dict[str, bool]:
    tokens = _split_csv(value)
    unknown = set(tokens) - INFLATE_CHOICES
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--inflate desconhecido: {sorted(unknown)} (use {sorted(INFLATE_CHOICES)})"
        )
    use_max = "max" in tokens
    return {
        "pseudo_replication": use_max or "pseudo" in tokens,
        "tune_on_full": use_max or "fulltune" in tokens,
        "threshold_on_test": use_max or "testthr" in tokens,
        "univariate_global": use_max or "univariate" in tokens,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modalities = _parse_modalities(args.modality)
    tasks = _parse_tasks(args.tasks)
    selection_modes = _parse_selection(args.selection)
    models = _split_csv(args.models)
    inflate = _parse_inflate(args.inflate)

    base_dir = Path(f"csvs/longitudinal_4_groups/ablation/{args.roi}")

    if len(modalities) == 1 and args.results_dir is None:
        results_dir = Path(f"csvs/longitudinal_4_groups/ablation_results_leaky/{modalities[0]}")
    else:
        results_dir = args.results_dir

    log_path = None
    if not args.no_log_file:
        log_path = args.log_file or Path(f"logs/ablation_leaky_{datetime.now():%Y%m%d_%H%M%S}.log")
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

    active_inflate = [k for k, v in inflate.items() if v] or ["nenhum (só leaky_global)"]
    log.warning("=== ablação LEAKY (pré-processamento global antes do CV) ===")
    log.warning("infladores:   %s", ", ".join(active_inflate))
    log.info("modalidades:  %s", modalities)
    log.info("tasks:        %s", tasks)
    log.info("seleção:      %s", selection_modes)
    log.info("modelos:      %s", models)
    log.info("combat:       %s", args.combat)
    log.info("repetições:   %s", args.repeats)
    log.info("jobs totais:  %d", total_jobs)
    log.info("base:         %s", base_dir)
    log.info("resultados:   %s", results_dir or "ablation_results_leaky/{modality}/")
    if log_path:
        log.info("log file:     %s", log_path.resolve())

    t0 = time.monotonic()
    try:
        df = run_full_ablation_suite_leaky(
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
            **inflate,
        )
    except Exception:
        log.exception("ablacao leaky falhou apos %s", fmt_duration(time.monotonic() - t0))
        return 1

    elapsed = time.monotonic() - t0
    df = prepare_ablation_df(df)
    summary = summary_with_pooled(df)

    out_dirs = {results_dir} if results_dir else {
        Path(f"csvs/longitudinal_4_groups/ablation_results_leaky/{m}") for m in modalities
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
    log.info("análise: 3_results.ipynb → RESULTS_ROOT = ablation_results_leaky")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

# python 5_ablation_late_fusion.py --cohort 48m_12m \
#   --fusion shape:t1_only,vol:t1_d21_d32,texture:t1_d21_d32,disp:t1_d21_d32,firstorder:t1_d21_d32 \
#   --tasks smci_pmci --selection l1_stable --models svm --combat false \
#   --combine mean
#
# Late fusion = média (ou weighted) dos scores OOF de cada mod:rep.
# Default: reusa CSVs mono-mod já rodados (--reuse-disk). Use --run-missing
# se alguma branch não existir.
# Early fusion (concat feats) = 5_ablation_early_fusion.py
# Saída: ablation_results_late_fusion/{fingerprint}/  modality=late__{fingerprint}

"""Late fusion cross-mod: combina scores de branches mono-mod."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

_MOD = Path(__file__).resolve().parent / "modules"
if str(_MOD) not in sys.path:
    sys.path.insert(0, str(_MOD))

from ablation_analysis import prepare_ablation_df, summary_with_pooled
from ablation_late_fusion import COMBINE_MODES, run_late_fusion_ablation_suite
from ablation_prep import ROI_FILTER_DEFAULT, param_soft_pmci_of
from ablation_representation import (
    DEFAULT_LATE_FUSION_SPEC,
    FUSION_MODALITIES,
    default_late_fusion_results_dir,
    fusion_fingerprint,
    fusion_label,
    parse_fusion_spec,
)
from ablation_runner import (
    SELECTION_MODES,
    STABLE_POOL_BOOTSTRAP,
    STABLE_POOL_L1_C,
    STABLE_POOL_MIN_PCT,
    TASKS,
    TASK_PRESETS,
    fmt_duration,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

COHORT = "48m_12m"
log = logging.getLogger("ablation_late_fusion")


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


def _parse_weights(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"weights inválidos: {value!r}") from e


def _resolve_fusion_spec(args: argparse.Namespace) -> str:
    if args.fusion:
        return args.fusion.strip()
    if args.partner_modality:
        partners = _split_csv(args.partner_modality)
        unknown = set(partners) - set(FUSION_MODALITIES)
        if unknown:
            raise SystemExit(f"partner-modality desconhecida: {sorted(unknown)}")
        anchor = f"{args.anchor_modality}:{args.anchor_rep}"
        return ",".join([anchor] + [f"{p}:{args.partner_rep}" for p in partners])
    return DEFAULT_LATE_FUSION_SPEC


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
    logging.getLogger("ablation_late_fusion").setLevel(level)
    logging.getLogger("ablation_runner").setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Late fusion: combina scores OOF de mod:rep,... (sem concat de feats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fusion", default=None, help=f"default: {DEFAULT_LATE_FUSION_SPEC}")
    p.add_argument("--partner-modality", default=None)
    p.add_argument("--anchor-modality", default="shape")
    p.add_argument("--anchor-rep", default="t1_only")
    p.add_argument("--partner-rep", default="t1_d21_d32")
    p.add_argument("--combine", choices=COMBINE_MODES, default="mean")
    p.add_argument(
        "--weights",
        default=None,
        type=_parse_weights,
        help="pesos por slot (ex. 0.7,0.3); default = uniforme",
    )
    p.add_argument(
        "--reuse-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reusa ablation_results_* mono-mod se existirem",
    )
    p.add_argument(
        "--run-missing",
        action="store_true",
        help="roda nested CV nas branches ausentes",
    )
    p.add_argument("--tasks", default="smci_pmci", type=_parse_tasks)
    p.add_argument("--selection", default="l1_stable", type=_parse_selection)
    p.add_argument("--models", default="svm")
    p.add_argument("--combat", default="false", type=_parse_combat)
    p.add_argument("--repeats", "-r", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", default=ROI_FILTER_DEFAULT)
    p.add_argument("--cohort", default=COHORT)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--stable-pool-min-pct", type=int, default=STABLE_POOL_MIN_PCT)
    p.add_argument("--stable-pool-min-timepoints", type=int, default=0)
    p.add_argument("--stable-bootstrap", type=int, default=STABLE_POOL_BOOTSTRAP)
    p.add_argument("--stable-l1-c", type=float, default=STABLE_POOL_L1_C)
    p.add_argument("--tuner", choices=["grid", "optuna"], default="optuna")
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        slots = parse_fusion_spec(_resolve_fusion_spec(args))
    except ValueError as e:
        print(f"erro fusion: {e}", file=sys.stderr)
        return 2

    if args.weights is not None and len(args.weights) != len(slots):
        print(
            f"erro: --weights tem {len(args.weights)} valores, fusion tem {len(slots)} slots",
            file=sys.stderr,
        )
        return 2
    if args.combine == "weighted" and args.weights is None:
        print("erro: --combine weighted exige --weights", file=sys.stderr)
        return 2

    models = _split_csv(args.models)
    base_dir = Path(f"csvs/cohorts/{args.cohort}/ablation/{args.roi}")
    results_dir = default_late_fusion_results_dir(
        base_dir, slots, results_dir=args.results_dir,
    )
    log_path = None
    if not args.no_log_file:
        log_path = args.log_file or Path(
            f"logs/ablation_late_fusion_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
    setup_logging(log_file=log_path, verbose=args.verbose)

    fp = fusion_fingerprint(slots)
    log.info("=== ablação late fusion ===")
    log.info(
        "cohort: %s | fusion: %s | protocol: late__%s | combine=%s",
        args.cohort, fusion_label(slots), fp, args.combine,
    )
    _long, _soft = param_soft_pmci_of(args.cohort)
    log.info("PARAM_SOFT_PMCI=%s | csv=%s", _soft, _long)
    log.info("reuse_disk=%s run_missing=%s | out: %s", args.reuse_disk, args.run_missing, results_dir)

    t0 = time.monotonic()
    try:
        df = run_late_fusion_ablation_suite(
            fusion_slots=slots,
            base_dir=base_dir,
            roi=args.roi,
            tasks=args.tasks,
            models=models,
            selection_modes=args.selection,
            with_combat_flags=args.combat,
            results_dir=results_dir,
            seed=args.seed,
            r_repeats=args.repeats,
            verbose=args.verbose,
            combat_quiet=True,
            stable_pool_min_pct=args.stable_pool_min_pct,
            stable_pool_min_timepoints=args.stable_pool_min_timepoints,
            stable_pool_bootstrap=args.stable_bootstrap,
            stable_pool_l1_c=args.stable_l1_c,
            tuner=args.tuner,
            optuna_trials=args.optuna_trials,
            reuse_disk=args.reuse_disk,
            run_missing=args.run_missing,
            combine=args.combine,
            weights=args.weights,
        )
    except Exception:
        log.exception("late fusion falhou apos %s", fmt_duration(time.monotonic() - t0))
        return 1

    if df.empty:
        log.error("sem resultados")
        return 1
    df = prepare_ablation_df(df)
    summary = summary_with_pooled(df)
    log.info(
        "csv: %s | %s",
        results_dir / "ablation_results_all.csv",
        results_dir / "ablation_summary.csv",
    )
    cols = [c for c in ("modality", "auc_patient_mean", "auc_pooled", "n_features_mean") if c in summary.columns]
    log.info("\n%s", summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

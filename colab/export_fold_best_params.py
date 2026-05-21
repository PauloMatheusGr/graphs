"""Exporta tables/fold_best_params.json a partir de checkpoints (útil para LSTM sem re-treino completo).

Uso (raiz do repo):

  python colab/export_fold_best_params.py

  RUN_DIR=colab/exp2/balanced/lstm python colab/export_fold_best_params.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import exp1_utils as u

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent
DEFAULT_RUN = COLAB / "exp2" / "balanced" / "lstm"


def main() -> None:
    run_env = os.environ.get("RUN_DIR", "").strip()
    run_dir = Path(run_env) if run_env else DEFAULT_RUN
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    if not (run_dir / "tables" / "metrics_per_fold.csv").is_file():
        print(f"Baseline incompleto: {run_dir}", file=sys.stderr)
        sys.exit(1)

    folds: list[dict] = []
    for fold_id in range(5):
        bp = u.load_baseline_fold_params(run_dir, fold_id)
        folds.append({"fold": fold_id, "best_params": bp})

    out = run_dir / "tables" / "fold_best_params.json"
    u.save_fold_best_params_json(out, folds)
    print(f"OK: {out} ({len(folds)} folds)")


if __name__ == "__main__":
    main()

"""Regenera PDFs e demografia para todos os runs exp2 com OOF.

Uso (a partir da raiz do repo):

  python colab/postprocess_exp2_runs.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent

SCENARIOS = ("balanced", "unbalanced")
MODELS = ("xgboost", "svm", "rocket", "lstm")


def _resolve_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _iter_run_dirs() -> list[Path]:
    runs: list[Path] = []
    for scenario in SCENARIOS:
        for model in MODELS:
            p = COLAB / "exp2" / scenario / model
            if p.is_dir():
                runs.append(p)
    return runs


def _run_script(script: str, run_dir: Path, python: str) -> int:
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir.resolve())
    return subprocess.run(
        [python, str(COLAB / script)],
        cwd=ROOT,
        env=env,
        check=False,
    ).returncode


def main() -> None:
    python = _resolve_python()
    os.chdir(ROOT)

    for run_dir in _iter_run_dirs():
        oof = run_dir / "tables" / "oof_predictions.csv"
        if not oof.is_file():
            print(f"skip (sem OOF): {run_dir}")
            continue
        print(f"postprocess: {run_dir}")
        for script in ("exp2_plots.py", "analyze_oof_demographics.py"):
            code = _run_script(script, run_dir, python)
            if code != 0:
                print(f"Falhou ({code}): {script} em {run_dir}", file=sys.stderr)
                sys.exit(code)

    print("Postprocess concluído.")


if __name__ == "__main__":
    main()

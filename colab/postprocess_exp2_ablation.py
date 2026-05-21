"""PDFs, demografia por sexo e resumo agregado para runs de ablação exp2 (balanced).

Uso (raiz do repo):

  python colab/postprocess_exp2_ablation.py

  ABLATION_MODEL=xgboost python colab/postprocess_exp2_ablation.py
  ABLATION_MODELS=xgboost,svm,lstm python colab/postprocess_exp2_ablation.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent
SCENARIO = "balanced"

MODEL_ABLATION_DIRS = {
    "xgboost": "xgboost_ablation",
    "svm": "svm_ablation",
    "lstm": "lstm_ablation",
}


def _resolve_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _models_from_env() -> list[str]:
    multi = os.environ.get("ABLATION_MODELS", "").strip()
    if multi:
        return [m.strip() for m in multi.split(",") if m.strip()]
    single = os.environ.get("ABLATION_MODEL", "").strip()
    if single:
        return [single]
    return list(MODEL_ABLATION_DIRS.keys())


def _iter_ablation_run_dirs(model: str) -> list[Path]:
    root_name = MODEL_ABLATION_DIRS.get(model)
    if root_name is None:
        raise ValueError(f"Modelo desconhecido: {model!r}")
    root = COLAB / "exp2" / SCENARIO / root_name
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("drop_"):
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


def _write_demographics_summary(model: str, ablation_root: Path) -> None:
    rows: list[dict] = []
    for run_dir in _iter_ablation_run_dirs(model):
        roi = run_dir.name.removeprefix("drop_").replace("_", " ")
        for level in ("patient", "datapoint"):
            mpath = run_dir / "tables" / "demographics" / f"metrics_by_sex_{level}.csv"
            if not mpath.is_file():
                continue
            df = pd.read_csv(mpath)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "model": model,
                        "roi_dropped": roi,
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        "level": level,
                        "sex": str(r["sex"]),
                        "n": int(r["n"]),
                        "acc": float(r["acc"]),
                        "auc": float(r["auc"]),
                        "f1": float(r["f1"]),
                        "ap": float(r["ap"]),
                    }
                )
    if not rows:
        print(f"Sem demografia para resumir: {ablation_root}")
        return
    out = ablation_root / "ablation_demographics_by_sex.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Resumo demografia ({model}): {out}")


def main() -> None:
    python = _resolve_python()
    os.chdir(ROOT)
    models = _models_from_env()

    for model in models:
        if model not in MODEL_ABLATION_DIRS:
            print(f"Ignorado modelo desconhecido: {model}", file=sys.stderr)
            continue
        ablation_root = COLAB / "exp2" / SCENARIO / MODEL_ABLATION_DIRS[model]
        if not ablation_root.is_dir():
            print(f"skip (sem pasta): {ablation_root}")
            continue

        for run_dir in _iter_ablation_run_dirs(model):
            oof = run_dir / "tables" / "oof_predictions.csv"
            if not oof.is_file():
                print(f"skip (sem OOF): {run_dir}")
                continue
            print(f"postprocess ablation: {run_dir}")
            for script in ("exp2_plots.py", "analyze_oof_demographics.py"):
                code = _run_script(script, run_dir, python)
                if code != 0:
                    print(f"Falhou ({code}): {script} em {run_dir}", file=sys.stderr)
                    sys.exit(code)

        _write_demographics_summary(model, ablation_root)

    print("Postprocess ablação concluído.")


if __name__ == "__main__":
    main()

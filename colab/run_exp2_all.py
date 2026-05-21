"""Re-treina todos os modelos exp2 (balanced).

Uso (a partir da raiz do repo):

  python colab/run_exp2_all.py

Logs em colab/logs_exp2_retrain/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent
LOG_DIR = COLAB / "logs_exp2_retrain"

TRAIN_SCRIPTS = (
    "exp2_xgboost.py",
    "exp2_svm.py",
    "exp2_rocket.py",
    "exp2_lstm.py",
)


def _resolve_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _run_one(script: str, *, balanced: bool, python: str) -> int:
    tag = "balanced" if balanced else "unbalanced"
    name = f"{Path(script).stem}_{tag}"
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env["DOWNSAMPLE_GROUP_SEX"] = "1" if balanced else "0"
    cmd = [python, str(COLAB / script)]

    print(f"=== {name} ===", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("stdout não disponível no subprocesso.")
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        code = proc.wait()
    if code != 0:
        print(f"Falhou com código {code}: {' '.join(cmd)}", file=sys.stderr)
    return code


def main() -> None:
    python = _resolve_python()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    for script in TRAIN_SCRIPTS:
        code = _run_one(script, balanced=True, python=python)
        if code != 0:
            sys.exit(code)

    print(f"Treinos concluídos. Logs em {LOG_DIR}")


if __name__ == "__main__":
    main()

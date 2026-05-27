"""Re-treina em série os quatro modelos de um experimento.

Editar LOG_DIR, TRAIN_SCRIPTS, BALANCED e HARMONIZATION antes de correr.

Uso (a partir da raiz do repo):

  python colab/run_exp_all.py

Saídas:
  - sem harmonização: colab/exp{1,2}/{balanced|unbalanced}/{modelo}/
  - com harmonização: colab/exp{1,2}/{balanced|unbalanced}/{modelo}_neurocombat/
Logs em LOG_DIR (ex.: logs_exp1_retrain/).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent

# --- CONFIG — editar antes de correr ---

LOG_DIR = COLAB / "logs_exp1_no_combat_balanced"
TRAIN_SCRIPTS = (
    "exp1_xgboost.py",
    "exp1_svm.py",
    "exp1_rocket.py",
    "exp1_lstm.py",
)

# LOG_DIR = COLAB / "logs_exp2_no_combat_balanced"
# TRAIN_SCRIPTS = (
#     "exp2_xgboost.py",
#     "exp2_svm.py",
#     "exp2_rocket.py",
#     "exp2_lstm.py",
# )

BALANCED = True
HARMONIZATION = False


def _resolve_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _run_one(
    script: str,
    *,
    balanced: bool,
    harmonization: bool,
    python: str,
) -> int:
    stem = Path(script).stem
    balance_tag = "balanced" if balanced else "unbalanced"
    harmon_tag = "harmonized" if harmonization else "no_harmon"
    name = f"{stem}_{balance_tag}_{harmon_tag}"
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env["DOWNSAMPLE_GROUP_SEX"] = "1" if balanced else "0"
    env["HARMONIZATION"] = "1" if harmonization else "0"
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

    print(
        f"Config: balanced={BALANCED}, harmonization={HARMONIZATION}, "
        f"scripts={len(TRAIN_SCRIPTS)}",
        flush=True,
    )

    for script in TRAIN_SCRIPTS:
        code = _run_one(
            script,
            balanced=BALANCED,
            harmonization=HARMONIZATION,
            python=python,
        )
        if code != 0:
            sys.exit(code)

    print(f"Treinos concluídos. Logs em {LOG_DIR}")


if __name__ == "__main__":
    main()

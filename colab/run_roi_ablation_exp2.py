"""Leave-one-ROI-out no XGBoost exp2 (balanced).

Requer baseline treinado em colab/exp2/balanced/xgboost/ com checkpoints
fold_0..4 e meta.json (best_params).

Uso (raiz do repo):

  python colab/run_roi_ablation_exp2.py

Uma ROI removida por run (máscara a zero). Saídas em
colab/exp2/balanced/xgboost_ablation/drop_<roi>/.

Variáveis de ambiente:
  ABLATION_FORCE_OPTUNA=1   — ignora params do baseline (Optuna em cada ROI)
  ABLATION_SKIP_EXISTING=1  — não re-treina se metrics_per_fold.csv existir
  ABLATION_ROIS=a,b         — só estas ROIs (vírgula)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import exp1_utils as u

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path(__file__).resolve().parent
CSV_PATH = ROOT / "csvs/abordagem_4_sMCI_pMCI/all_unitary_features_neurocombat.csv"
EXP2_PATH = ROOT / "exp2.md"
PAIR_ORDER = ["1", "2", "3"]
GROUP_KEY = ["ID_PT", "COMBINATION_NUMBER", "TRIPLET_IDX"]

BASELINE_RUN_DIR = COLAB / "exp2" / "balanced" / "xgboost"
ABLATION_ROOT = COLAB / "exp2" / "balanced" / "xgboost_ablation"
LOG_DIR = COLAB / "logs_exp2_ablation"
TRAIN_SCRIPT = COLAB / "exp2_xgboost.py"


def _resolve_python() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes")


def _roi_dir_name(roi: str) -> str:
    s = roi.strip().lower().replace(" ", "_")
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return f"drop_{s}"


def _list_rois() -> list[str]:
    _, _, _, _, _, slot_labels = u.load_tensor(
        CSV_PATH,
        EXP2_PATH,
        PAIR_ORDER,
        GROUP_KEY,
        require_sex=True,
        temporal_mode="baseline_rate",
        dt_epsilon=0.5,
    )
    return u.unique_rois_from_slot_labels(slot_labels)


def _baseline_ready() -> bool:
    for fold_id in range(5):
        meta = BASELINE_RUN_DIR / "checkpoints" / f"fold_{fold_id}" / "meta.json"
        if not meta.is_file():
            return False
    return (BASELINE_RUN_DIR / "tables" / "metrics_per_fold.csv").is_file()


def _run_one_ablation(roi: str, *, python: str, force_optuna: bool) -> int:
    run_dir = ABLATION_ROOT / _roi_dir_name(roi)
    metrics_path = run_dir / "tables" / "metrics_per_fold.csv"
    if _env_bool("ABLATION_SKIP_EXISTING", False) and metrics_path.is_file():
        print(f"Pulando (já existe): {run_dir}", flush=True)
        return 0

    env = os.environ.copy()
    env["DOWNSAMPLE_GROUP_SEX"] = "1"
    env["ABLATION_DROP_ROIS"] = roi
    env["RUN_DIR"] = str(run_dir.relative_to(ROOT))
    if force_optuna:
        env.pop("ABLATION_SKIP_OPTUNA", None)
        env.pop("ABLATION_BASELINE_RUN_DIR", None)
    else:
        env["ABLATION_SKIP_OPTUNA"] = "1"
        env["ABLATION_BASELINE_RUN_DIR"] = str(BASELINE_RUN_DIR.relative_to(ROOT))

    tag = _roi_dir_name(roi)
    log_path = LOG_DIR / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(TRAIN_SCRIPT)]

    print(f"=== ablation {roi} -> {run_dir} ===", flush=True)
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
            raise RuntimeError("stdout indisponível no subprocesso.")
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        code = proc.wait()
    if code != 0:
        print(f"Falhou ({code}): {roi}", file=sys.stderr)
    return code


def _fold_means(metrics_path: Path) -> dict[str, float]:
    acc, auc_a, f1_a, ap = u.load_metrics_per_fold(metrics_path)
    out = {
        "acc_mean": float(np.nanmean(acc)),
        "auc_mean": float(np.nanmean(auc_a)),
        "f1_mean": float(np.nanmean(f1_a)),
        "ap_mean": float(np.nanmean(ap)),
    }
    return out


def _oof_metrics(oof_path: Path) -> dict[str, float]:
    df = pd.read_csv(oof_path)
    y = df["y_true"].to_numpy(dtype=np.int32)
    if "score" in df.columns:
        proba = df["score"].to_numpy(dtype=np.float64)
    else:
        return {"auc_oof": float("nan"), "ap_oof": float("nan")}
    if len(np.unique(y)) < 2:
        return {"auc_oof": float("nan"), "ap_oof": float("nan")}
    return {
        "auc_oof": float(roc_auc_score(y, proba)),
        "ap_oof": float(average_precision_score(y, proba)),
    }


def _write_summary(rois: list[str]) -> None:
    baseline_metrics = BASELINE_RUN_DIR / "tables" / "metrics_per_fold.csv"
    baseline_oof = BASELINE_RUN_DIR / "tables" / "oof_predictions.csv"
    if not baseline_metrics.is_file():
        print("Aviso: baseline sem metrics_per_fold.csv; sem deltas.", file=sys.stderr)
        base = {"auc_mean": np.nan, "ap_mean": np.nan, "auc_oof": np.nan, "ap_oof": np.nan}
    else:
        base = {**_fold_means(baseline_metrics)}
        if baseline_oof.is_file():
            base.update(_oof_metrics(baseline_oof))

    rows: list[dict[str, object]] = []
    for roi in rois:
        run_dir = ABLATION_ROOT / _roi_dir_name(roi)
        mpath = run_dir / "tables" / "metrics_per_fold.csv"
        opath = run_dir / "tables" / "oof_predictions.csv"
        row: dict[str, object] = {
            "roi_dropped": roi,
            "run_dir": str(run_dir.relative_to(ROOT)),
        }
        if not mpath.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        row["status"] = "ok"
        fm = _fold_means(mpath)
        row.update(fm)
        if opath.is_file():
            row.update(_oof_metrics(opath))
        for key in ("auc_mean", "ap_mean", "auc_oof", "ap_oof"):
            b = base.get(key, np.nan)
            v = row.get(key, np.nan)
            if b is not None and not (isinstance(b, float) and np.isnan(b)):
                row[f"delta_{key}"] = float(v) - float(b) if v == v else np.nan
            else:
                row[f"delta_{key}"] = np.nan
        rows.append(row)

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = ABLATION_ROOT / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Resumo: {summary_path}")

    ok_rows = [r for r in rows if r.get("status") == "ok" and "delta_auc_oof" in r]
    if not ok_rows:
        return
    names = [str(r["roi_dropped"]) for r in ok_rows]
    deltas = [float(r["delta_auc_oof"]) for r in ok_rows]
    order = np.argsort(deltas)
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.28 * len(names))))
    y_pos = np.arange(len(names))
    colors = ["#c44e52" if d < 0 else "#55a868" for d in deltas]
    ax.barh(y_pos, deltas, color=colors)
    ax.axvline(0.0, color="0.3", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Δ AUC OOF vs baseline")
    ax.set_title("Ablação LOO — XGBoost balanced (ROI removida)")
    fig.tight_layout()
    fig_path = ABLATION_ROOT / "ablation_delta_auc_oof.pdf"
    u.save_pdf(fig, fig_path)
    print(f"Gráfico: {fig_path}")


def main() -> None:
    force_optuna = _env_bool("ABLATION_FORCE_OPTUNA", False)
    if not force_optuna and not _baseline_ready():
        print(
            f"Baseline incompleto em {BASELINE_RUN_DIR}.\n"
            "Treine antes: DOWNSAMPLE_GROUP_SEX=1 python colab/exp2_xgboost.py\n"
            "Ou use ABLATION_FORCE_OPTUNA=1 (Optuna em cada ROI; muito mais lento).",
            file=sys.stderr,
        )
        sys.exit(1)

    rois = _list_rois()
    filter_env = os.environ.get("ABLATION_ROIS", "").strip()
    if filter_env:
        want = {p.strip() for p in filter_env.split(",") if p.strip()}
        rois = [r for r in rois if r in want]
        unknown = want - set(rois)
        if unknown:
            print(f"Aviso: ROIs não encontradas no tensor: {sorted(unknown)}")

    print(f"ROIs ({len(rois)}): {rois}")
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    python = _resolve_python()
    os.chdir(ROOT)

    for roi in rois:
        code = _run_one_ablation(roi, python=python, force_optuna=force_optuna)
        if code != 0:
            sys.exit(code)

    _write_summary(rois)
    print(f"Ablação concluída. Raiz: {ABLATION_ROOT}")


if __name__ == "__main__":
    main()

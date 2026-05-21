"""Leave-one-ROI-out exp2 balanced (XGBoost, SVM, LSTM).

Requer baseline em colab/exp2/balanced/<model>/ com métricas OOF e hiperparâmetros
(checkpoints/fold_*/meta.json ou tables/fold_best_params.json).

Uso (raiz do repo):

  ABLATION_ROIS=hippocampus,amygdala python colab/run_roi_ablation_exp2.py
  ABLATION_MODEL=svm ABLATION_ROIS=... python colab/run_roi_ablation_exp2.py
  ABLATION_MODELS=xgboost,svm python colab/run_roi_ablation_exp2.py

Saídas: colab/exp2/balanced/<model>_ablation/drop_<roi>/

Variáveis de ambiente:
  ABLATION_MODEL        — xgboost (default), svm ou lstm
  ABLATION_MODELS       — lista vírgula (ex.: xgboost,svm); sobrepõe ABLATION_MODEL
  ABLATION_ROIS         — só estas ROIs
  ABLATION_SKIP_EXISTING=1 — não re-treina se metrics_per_fold.csv existir
  ABLATION_FORCE_OPTUNA=1  — Optuna em cada ROI (ignora baseline params)
  ABLATION_SKIP_POSTPROCESS=1 — não corre postprocess/demografia após cada ROI
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
SCENARIO = "balanced"

MODEL_CONFIG: dict[str, dict[str, str]] = {
    "xgboost": {
        "train_script": "exp2_xgboost.py",
        "ablation_dir": "xgboost_ablation",
        "title": "XGBoost",
    },
    "svm": {
        "train_script": "exp2_svm.py",
        "ablation_dir": "svm_ablation",
        "title": "SVM linear",
    },
    "lstm": {
        "train_script": "exp2_lstm.py",
        "ablation_dir": "lstm_ablation",
        "title": "LSTM",
    },
}


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


def _models_from_env() -> list[str]:
    multi = os.environ.get("ABLATION_MODELS", "").strip()
    if multi:
        return [m.strip() for m in multi.split(",") if m.strip()]
    single = os.environ.get("ABLATION_MODEL", "xgboost").strip() or "xgboost"
    return [single]


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


def _baseline_run_dir(model: str) -> Path:
    return COLAB / "exp2" / SCENARIO / model


def _ablation_root(model: str) -> Path:
    return COLAB / "exp2" / SCENARIO / MODEL_CONFIG[model]["ablation_dir"]


def _baseline_ready(baseline_dir: Path) -> bool:
    if not (baseline_dir / "tables" / "metrics_per_fold.csv").is_file():
        return False
    try:
        for fold_id in range(5):
            u.load_baseline_fold_params(baseline_dir, fold_id)
        return True
    except (FileNotFoundError, ValueError):
        return False


def _run_postprocess_one(run_dir: Path, *, python: str) -> int:
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir.resolve())
    for script in ("exp2_plots.py", "analyze_oof_demographics.py"):
        code = subprocess.run(
            [python, str(COLAB / script)],
            cwd=ROOT,
            env=env,
            check=False,
        ).returncode
        if code != 0:
            return code
    return 0


def _run_one_ablation(
    model: str,
    roi: str,
    *,
    python: str,
    force_optuna: bool,
    skip_post: bool,
) -> int:
    cfg = MODEL_CONFIG[model]
    baseline_dir = _baseline_run_dir(model)
    ablation_root = _ablation_root(model)
    run_dir = ablation_root / _roi_dir_name(roi)
    metrics_path = run_dir / "tables" / "metrics_per_fold.csv"
    if _env_bool("ABLATION_SKIP_EXISTING", False) and metrics_path.is_file():
        print(f"Pulando (já existe): {run_dir}", flush=True)
        if not skip_post:
            _run_postprocess_one(run_dir, python=python)
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
        env["ABLATION_BASELINE_RUN_DIR"] = str(baseline_dir.relative_to(ROOT))

    log_dir = COLAB / "logs_exp2_ablation" / model
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_roi_dir_name(roi)}.log"
    cmd = [python, str(COLAB / cfg["train_script"])]

    print(f"=== [{model}] ablation {roi} -> {run_dir} ===", flush=True)
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
        print(f"Falhou ({code}): {model} {roi}", file=sys.stderr)
        return code

    if not skip_post:
        pcode = _run_postprocess_one(run_dir, python=python)
        if pcode != 0:
            return pcode
    return 0


def _fold_means(metrics_path: Path) -> dict[str, float]:
    acc, auc_a, f1_a, ap = u.load_metrics_per_fold(metrics_path)
    return {
        "acc_mean": float(np.nanmean(acc)),
        "auc_mean": float(np.nanmean(auc_a)),
        "f1_mean": float(np.nanmean(f1_a)),
        "ap_mean": float(np.nanmean(ap)),
    }


def _oof_metrics(oof_path: Path) -> dict[str, float]:
    df = pd.read_csv(oof_path)
    y = df["y_true"].to_numpy(dtype=np.int32)
    if "score" not in df.columns:
        return {"auc_oof": float("nan"), "ap_oof": float("nan")}
    proba = df["score"].to_numpy(dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"auc_oof": float("nan"), "ap_oof": float("nan")}
    return {
        "auc_oof": float(roc_auc_score(y, proba)),
        "ap_oof": float(average_precision_score(y, proba)),
    }


def _write_summary(model: str, rois: list[str]) -> None:
    cfg = MODEL_CONFIG[model]
    baseline_dir = _baseline_run_dir(model)
    ablation_root = _ablation_root(model)
    baseline_metrics = baseline_dir / "tables" / "metrics_per_fold.csv"
    baseline_oof = baseline_dir / "tables" / "oof_predictions.csv"
    if not baseline_metrics.is_file():
        print(f"Aviso: baseline sem metrics ({model}); sem deltas.", file=sys.stderr)
        base = {"auc_mean": np.nan, "ap_mean": np.nan, "auc_oof": np.nan, "ap_oof": np.nan}
    else:
        base = {**_fold_means(baseline_metrics)}
        if baseline_oof.is_file():
            base.update(_oof_metrics(baseline_oof))

    rows: list[dict[str, object]] = []
    for roi in rois:
        run_dir = ablation_root / _roi_dir_name(roi)
        mpath = run_dir / "tables" / "metrics_per_fold.csv"
        opath = run_dir / "tables" / "oof_predictions.csv"
        row: dict[str, object] = {
            "model": model,
            "roi_dropped": roi,
            "run_dir": str(run_dir.relative_to(ROOT)),
        }
        if not mpath.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        row["status"] = "ok"
        row.update(_fold_means(mpath))
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

    ablation_root.mkdir(parents=True, exist_ok=True)
    summary_path = ablation_root / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Resumo ({model}): {summary_path}")

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
    ax.set_title(f"Ablação LOO — {cfg['title']} balanced (ROI removida)")
    fig.tight_layout()
    fig_path = ablation_root / "ablation_delta_auc_oof.pdf"
    u.save_pdf(fig, fig_path)
    print(f"Gráfico ({model}): {fig_path}")


def _write_demographics_summary(model: str) -> None:
    ablation_root = _ablation_root(model)
    rows: list[dict] = []
    for run_dir in sorted(ablation_root.iterdir()) if ablation_root.is_dir() else []:
        if not run_dir.is_dir() or not run_dir.name.startswith("drop_"):
            continue
        roi = run_dir.name.removeprefix("drop_")
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
        return
    out = ablation_root / "ablation_demographics_by_sex.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Resumo demografia ({model}): {out}")


def _run_model(model: str, rois: list[str], *, python: str) -> None:
    if model not in MODEL_CONFIG:
        raise ValueError(f"ABLATION_MODEL inválido: {model!r}")

    force_optuna = _env_bool("ABLATION_FORCE_OPTUNA", False)
    skip_post = _env_bool("ABLATION_SKIP_POSTPROCESS", False)
    baseline_dir = _baseline_run_dir(model)

    if not force_optuna and not _baseline_ready(baseline_dir):
        print(
            f"Baseline incompleto para {model} em {baseline_dir}.\n"
            "Treine antes ou exporte fold_best_params:\n"
            f"  RUN_DIR={baseline_dir.relative_to(ROOT)} python colab/export_fold_best_params.py\n"
            "Para LSTM sem checkpoints, re-treine exp2_lstm ou use ABLATION_FORCE_OPTUNA=1.",
            file=sys.stderr,
        )
        sys.exit(1)

    ablation_root = _ablation_root(model)
    ablation_root.mkdir(parents=True, exist_ok=True)
    print(f"Modelo: {model} — ROIs ({len(rois)}): {rois}")

    for roi in rois:
        code = _run_one_ablation(
            model,
            roi,
            python=python,
            force_optuna=force_optuna,
            skip_post=skip_post,
        )
        if code != 0:
            sys.exit(code)

    _write_summary(model, rois)
    _write_demographics_summary(model)
    print(f"Ablação [{model}] concluída: {ablation_root}")


def main() -> None:
    rois = _list_rois()
    filter_env = os.environ.get("ABLATION_ROIS", "").strip()
    if filter_env:
        want = {p.strip() for p in filter_env.split(",") if p.strip()}
        rois = [r for r in rois if r in want]
        unknown = want - set(rois)
        if unknown:
            print(f"Aviso: ROIs não encontradas no tensor: {sorted(unknown)}")

    python = _resolve_python()
    os.chdir(ROOT)

    for model in _models_from_env():
        if model not in MODEL_CONFIG:
            print(f"Ignorado modelo desconhecido: {model}", file=sys.stderr)
            continue
        _run_model(model, rois, python=python)

    print("Ablação exp2 concluída.")


if __name__ == "__main__":
    main()

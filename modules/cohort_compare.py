"""Consolida resultados e atributos selecionados across cohorts → 2 CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ablation_analysis import (
    feature_freq_table_grouped,
    patient_mean_predictions,
    prepare_ablation_df,
)

COHORT_RE = re.compile(r"^(\d+)m_(\d+)m$")
SELECTION = "l1_stable"

PROTOCOL_DIR = {
    "abs": "ablation_results",
    "t1_only": "ablation_results_t1_only",
    "deltas": "ablation_results_deltas",
    "deltas_only": "ablation_results_deltas_only",
    "deltas_rel": "ablation_results_deltas_rel",
    "global": "ablation_results_leaky",
    "t1_only_global": "ablation_results_leaky_t1_only",
    "clinica": "ablation_results_clinic",
    "clinica+img": "ablation_results_clinic_img",
    "clinica+img_t1": "ablation_results_clinic_img_t1_only",
}

# métricas do ablation_summary (mean ± sd quando existir)
METRIC_PAIRS = [
    ("auc_mean", "auc_std"),
    ("auc_pr_mean", "auc_pr_std"),
    ("accuracy_mean", "accuracy_std"),
    ("bal_acc_mean", "bal_acc_std"),
    ("mcc_mean", "mcc_std"),
    ("sens_pos_mean", "sens_pos_std"),
    ("spec_neg_mean", "spec_neg_std"),
    ("f1_pos_mean", "f1_pos_std"),
]


def parse_cohort_name(cohort: str) -> tuple[int | None, int | None]:
    m = COHORT_RE.match(cohort.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def cohort_group_counts(long_csv: Path) -> tuple[int, int]:
    """n_smci, n_pmci (1 paciente = 1 conjunto)."""
    if not long_csv.is_file():
        return 0, 0
    df = pd.read_csv(long_csv, usecols=["ID_PT", "GROUP"])
    one = df.drop_duplicates("ID_PT")
    vc = one["GROUP"].value_counts()
    return int(vc.get("sMCI", 0)), int(vc.get("pMCI", 0))


def bootstrap_auc_std(
    y,
    scores,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))
    if not aucs:
        return float("nan")
    return float(np.std(aucs, ddof=0))


def _stable_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % (2**31)


def _summary_files(root: Path) -> list[Path]:
    files = list(root.glob("*/ablation_summary.csv"))
    files += [f for f in root.glob("*summary*.csv") if not f.name.startswith("all_")]
    return files


def _results_all_for_summary(summary_path: Path) -> Path | None:
    """Parea ablation_summary.csv → ablation_results_all.csv (mesmo diretório)."""
    stem = summary_path.name
    if stem == "ablation_summary.csv":
        cand = summary_path.with_name("ablation_results_all.csv")
    elif stem.endswith("_summary.csv"):
        cand = summary_path.with_name(stem.replace("_summary.csv", "_results_all.csv"))
    else:
        cand = summary_path.parent / "ablation_results_all.csv"
    return cand if cand.is_file() else None


def _filter_l1(df: pd.DataFrame) -> pd.DataFrame:
    if "selection_mode" not in df.columns:
        return df
    return df.loc[df["selection_mode"].astype(str) == SELECTION].copy()


def _patient_auc_boot(raw: pd.DataFrame, row: pd.Series, *, n_boot: int) -> tuple[float, float]:
    """auc_patient_mean + bootstrap SD para a config da linha do summary."""
    need = ["task", "modality", "model_key", "with_combat"]
    if any(c not in raw.columns for c in need) or "selected_features" not in raw.columns:
        return float("nan"), float("nan")
    if "test_y_true" not in raw.columns or "test_scores" not in raw.columns:
        return float("nan"), float("nan")

    sub = prepare_ablation_df(raw)
    mask = (
        (sub["task"].astype(str) == str(row["task"]))
        & (sub["modality"].astype(str) == str(row["modality"]))
        & (sub["model_key"].astype(str) == str(row["model_key"]))
        & (sub["with_combat"] == bool(row["with_combat"]))
    )
    if "selection_mode" in sub.columns:
        mask &= sub["selection_mode"].astype(str) == SELECTION
    sub = sub.loc[mask]
    if sub.empty:
        return float("nan"), float("nan")
    try:
        y, s = patient_mean_predictions(sub)
        auc = float(roc_auc_score(y, s))
        seed = _stable_seed(
            row.get("task"), row.get("modality"), row.get("model_key"), row.get("with_combat"),
        )
        sd = bootstrap_auc_std(y, s, n_boot=n_boot, seed=seed)
        return auc, sd
    except (ValueError, KeyError, json.JSONDecodeError):
        return float("nan"), float("nan")


def build_cohort_results(
    cohorts: list[str],
    *,
    cohorts_root: Path = Path("csvs/cohorts"),
    n_boot: int = 2000,
    protocols: dict[str, str] | None = None,
) -> pd.DataFrame:
    protocols = protocols or PROTOCOL_DIR
    rows: list[dict] = []

    for cohort in cohorts:
        base = cohorts_root / cohort
        if not base.is_dir():
            continue
        t_janela, t_imagens = parse_cohort_name(cohort)
        n_smci, n_pmci = cohort_group_counts(base / "adnimerged_longitudinal.csv")

        for proto, dirname in protocols.items():
            root = base / dirname
            if not root.is_dir():
                continue
            for fsum in _summary_files(root):
                summary = _filter_l1(pd.read_csv(fsum))
                if summary.empty:
                    continue
                raw_path = _results_all_for_summary(fsum)
                raw = pd.read_csv(raw_path) if raw_path is not None else None
                if raw is not None:
                    raw = _filter_l1(raw)

                for _, r in summary.iterrows():
                    auc_p, auc_p_sd = float("nan"), float("nan")
                    if raw is not None:
                        auc_p, auc_p_sd = _patient_auc_boot(raw, r, n_boot=n_boot)
                    # fallback: summary já tem auc_patient_mean
                    if not np.isfinite(auc_p) and "auc_patient_mean" in summary.columns:
                        auc_p = float(r["auc_patient_mean"]) if pd.notna(r["auc_patient_mean"]) else float("nan")

                    out = {
                        "cohort": cohort,
                        "t_janela": t_janela,
                        "t_imagens": t_imagens,
                        "n_smci": n_smci,
                        "n_pmci": n_pmci,
                        "protocol": proto,
                        "task": r.get("task"),
                        "modality": r.get("modality"),
                        "model_key": r.get("model_key"),
                        "with_combat": bool(r["with_combat"]) if "with_combat" in r.index and pd.notna(r["with_combat"]) else r.get("with_combat"),
                        "n_outer_evals": r.get("n_outer_evals"),
                        "n_repeats": r.get("n_repeats"),
                        "n_features_mean": r.get("n_features_mean"),
                        "auc_patient_mean": auc_p,
                        "auc_patient_std": auc_p_sd,
                        "auc_pooled": r.get("auc_pooled"),
                        "source_file": str(fsum.relative_to(cohorts_root)),
                    }
                    for mean_c, std_c in METRIC_PAIRS:
                        out[mean_c] = r.get(mean_c)
                        out[std_c] = r.get(std_c)
                    rows.append(out)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # ordenação estável
    sort_cols = [c for c in ("cohort", "protocol", "task", "modality", "model_key", "with_combat") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_cohort_features_long(
    cohorts: list[str],
    *,
    cohorts_root: Path = Path("csvs/cohorts"),
    protocols: dict[str, str] | None = None,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """1 linha = anatomical_key × config (freq por visita T1/T2/T3)."""
    protocols = protocols or PROTOCOL_DIR
    frames: list[pd.DataFrame] = []

    for cohort in cohorts:
        base = cohorts_root / cohort
        if not base.is_dir():
            continue
        t_janela, t_imagens = parse_cohort_name(cohort)

        for proto, dirname in protocols.items():
            root = base / dirname
            if not root.is_dir():
                continue
            # results_all por modalidade (imagem)
            for raw_path in root.glob("*/ablation_results_all.csv"):
                raw = _filter_l1(pd.read_csv(raw_path))
                if raw.empty or "selected_features" not in raw.columns:
                    continue
                raw = prepare_ablation_df(raw)
                keys = ["task", "modality", "model_key", "with_combat"]
                if any(c not in raw.columns for c in keys):
                    continue
                for key_vals, sub in raw.groupby(keys, dropna=False):
                    task, modality, model_key, with_combat = key_vals
                    freq = feature_freq_table_grouped(sub, min_coverage=min_coverage)
                    if freq.empty:
                        continue
                    freq = freq.copy()
                    # meta do helper pode repetir — sobrescreve com chaves do groupby
                    for col in ("task", "modality", "model_key", "with_combat", "selection_mode", "combat_label"):
                        if col in freq.columns:
                            freq = freq.drop(columns=col)
                    freq.insert(0, "cohort", cohort)
                    freq.insert(1, "t_janela", t_janela)
                    freq.insert(2, "t_imagens", t_imagens)
                    freq.insert(3, "protocol", proto)
                    freq.insert(4, "task", task)
                    freq.insert(5, "modality", modality)
                    freq.insert(6, "model_key", model_key)
                    freq.insert(7, "with_combat", bool(with_combat))
                    freq = freq.rename(columns={
                        "feature_group": "anatomical_key",
                        "coverage_pct": "pct",
                    })
                    frames.append(freq)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # colunas preferidas à frente
    front = [
        "cohort", "t_janela", "t_imagens", "protocol", "task", "modality",
        "model_key", "with_combat", "anatomical_key", "feature_short",
        "pct", "freq_T1", "freq_T2", "freq_T3", "pct_T1", "pct_T2", "pct_T3",
        "amplitude", "n_folds_any", "total_selections", "n_runs",
    ]
    front = [c for c in front if c in out.columns]
    out = out[front + [c for c in out.columns if c not in front]]
    return out.sort_values(
        ["cohort", "protocol", "task", "modality", "model_key", "with_combat", "pct"],
        ascending=[True, True, True, True, True, True, False],
    ).reset_index(drop=True)


def save_cohort_comparison(
    cohorts: list[str],
    out_dir: Path,
    *,
    cohorts_root: Path = Path("csvs/cohorts"),
    n_boot: int = 2000,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = build_cohort_results(cohorts, cohorts_root=cohorts_root, n_boot=n_boot)
    features = build_cohort_features_long(cohorts, cohorts_root=cohorts_root)
    p_res = out_dir / "cohort_results.csv"
    p_feat = out_dir / "cohort_features_long.csv"
    results.to_csv(p_res, index=False)
    features.to_csv(p_feat, index=False)
    return p_res, p_feat, results, features


if __name__ == "__main__":
    import sys
    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    root = Path("csvs/cohorts")
    cohorts = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and COHORT_RE.match(p.name) and (p / "ablation_results").is_dir()
    )
    assert cohorts, "nenhum cohort com ablation_results"
    res = build_cohort_results(cohorts[:1], n_boot=50)
    assert not res.empty, "cohort_results vazio"
    assert "auc_patient_mean" in res.columns and "auc_patient_std" in res.columns
    assert "selection_mode" not in res.columns
    assert "n_soft_pmci" not in res.columns
    print(f"ok: results={len(res)} rows cohorts={cohorts[:1]}")
    feat = build_cohort_features_long(cohorts[:1])
    print(f"ok: features={len(feat)} rows")
    if not feat.empty:
        assert "anatomical_key" in feat.columns and "pct" in feat.columns

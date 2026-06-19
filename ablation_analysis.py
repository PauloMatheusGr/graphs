"""Análise pós-ablação: AUC pooled, frequência de features, gráficos de estabilidade."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONFIG_COLS = ("task", "modality", "model_key", "with_combat", "selection_mode")
FEAT_RE = re.compile(r"^hippocampus_([LR])_(T[123])_(.+)$")
METRIC_COLS = (
    "accuracy",
    "auc",
    "auc_pr",
    "bal_acc",
    "mcc",
    "sens_pos",
    "spec_neg",
    "f1_pos",
)


def coerce_with_combat(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def short_feature(name: str) -> str:
    m = FEAT_RE.match(name)
    if m:
        feat = m.group(3)
        if feat.startswith("original_shape_"):
            feat = feat.removeprefix("original_shape_")
        return f"{m.group(1)} {m.group(2)} | {feat}"
    return name


def prepare_ablation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "with_combat" in out.columns:
        out["with_combat"] = coerce_with_combat(out["with_combat"])
        out["combat_label"] = out["with_combat"].map({True: "ComBat", False: "sem ComBat"})
    if "repeat_id" not in out.columns:
        out["repeat_id"] = 0
    return out


def n_repeats_in_df(df: pd.DataFrame) -> int:
    return int(df["repeat_id"].nunique()) if "repeat_id" in df.columns else 1


def n_outer_evals(df: pd.DataFrame) -> int:
    """Total de avaliações externas (folds × repetições) no subconjunto."""
    return len(df)


def filter_ablation_config(
    df: pd.DataFrame,
    *,
    task: str,
    modality: str,
    model_key: str,
    with_combat: bool,
    selection_mode: str = "mrmr",
    expected_folds: int = 5,
) -> pd.DataFrame:
    sub = prepare_ablation_df(df)
    mask = (
        (sub["task"].astype(str) == task)
        & (sub["modality"].astype(str) == modality)
        & (sub["model_key"].astype(str) == model_key)
        & (sub["with_combat"] == with_combat)
    )
    if "selection_mode" in sub.columns:
        mask &= sub["selection_mode"].astype(str) == selection_mode
    out = sub.loc[mask].copy()
    if out.empty:
        raise ValueError(
            f"Nenhuma linha para task={task!r} modality={modality!r} "
            f"model={model_key!r} combat={with_combat!r} selection={selection_mode!r}"
        )
    n_rep = n_repeats_in_df(out)
    expected = expected_folds * n_rep
    if len(out) != expected:
        raise ValueError(
            f"Esperado {expected} linhas ({expected_folds} folds × {n_rep} repetições), "
            f"obtido {len(out)}"
        )
    return out.sort_values(["repeat_id", "fold"]).reset_index(drop=True)


def pooled_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "test_y_true" not in df.columns or "test_scores" not in df.columns:
        raise KeyError("Colunas test_y_true / test_scores ausentes — rode ablation_runner atualizado.")
    y_parts: list[int] = []
    s_parts: list[float] = []
    for _, row in df.iterrows():
        y_parts.extend(json.loads(row["test_y_true"]))
        s_parts.extend(json.loads(row["test_scores"]))
    y = np.asarray(y_parts, dtype=int)
    s = np.asarray(s_parts, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("AUC pooled requer ambas as classes nas predições externas.")
    return y, s


def pooled_auc(df: pd.DataFrame) -> float:
    y, s = pooled_predictions(df)
    return float(roc_auc_score(y, s))


def feature_freq_table(
    df: pd.DataFrame,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Frequência por atributo. `df` deve ser uma única configuração (mesmo modelo/combat/etc.)."""
    df = prepare_ablation_df(df)
    n_runs = n_outer_evals(df)
    if n_runs == 0:
        return pd.DataFrame(
            columns=["feature", "feature_short", "count", "freq", "pct", "n_runs"]
        )

    meta = df.iloc[0]
    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        for feat in json.loads(row["selected_features"]):
            counts[feat] = counts.get(feat, 0) + 1

    rows = [
        {
            "task": meta.get("task"),
            "modality": meta.get("modality"),
            "combat_label": meta.get("combat_label"),
            "model_key": meta.get("model_key"),
            "selection_mode": meta.get("selection_mode"),
            "feature": feat,
            "feature_short": short_feature(feat),
            "count": cnt,
            "freq": cnt / n_runs,
            "pct": int(round(100 * cnt / n_runs)),
            "n_runs": n_runs,
        }
        for feat, cnt in counts.items()
    ]
    out = pd.DataFrame(rows).sort_values(["freq", "feature"], ascending=[False, True])
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def summary_with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por configuração com AUC média dos folds e AUC pooled."""
    df = prepare_ablation_df(df)
    group_cols = [c for c in CONFIG_COLS if c in df.columns]
    if "modality_label" in df.columns:
        group_cols = group_cols + ["modality_label"]
    if "best_model" in df.columns:
        group_cols = group_cols + ["best_model"]

    rows: list[dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_outer_evals"] = len(grp)
        row["n_repeats"] = n_repeats_in_df(grp)
        row["auc_mean"] = float(grp["auc"].mean())
        row["auc_std"] = float(grp["auc"].std(ddof=0))
        row["auc_pooled"] = pooled_auc(grp)
        row["n_features_mean"] = float(grp["n_features_selected"].mean())
        for col in METRIC_COLS:
            if col in grp.columns and col != "auc":
                row[f"{col}_mean"] = float(grp[col].mean())
                row[f"{col}_std"] = float(grp[col].std(ddof=0))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("auc_pooled", ascending=False).reset_index(drop=True)


def plot_feature_stability(
    df_config: pd.DataFrame,
    *,
    title: str | None = None,
    out_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Barras horizontais: todos os atributos selecionados em ≥1 fold, eixo 0–100%."""
    freq = feature_freq_table(df_config, top_n=None)
    if freq.empty:
        raise ValueError("Nenhum atributo selecionado para plotar.")

    freq = freq.sort_values(["pct", "feature_short"], ascending=[True, True])
    n = len(freq)
    if figsize is None:
        figsize = (8, max(4, 0.28 * n + 1))

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n)
    ax.barh(y_pos, freq["pct"], height=0.75, color="#4477AA", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(freq["feature_short"], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frequência de seleção (%)")
    for x in (20, 40, 60, 80):
        ax.axvline(x, color="gray", ls="--", lw=0.5, alpha=0.5)

    meta = freq.iloc[0]
    if title is None:
        title = (
            f"{meta['task']} | {meta['modality']} | {meta['model_key']} | "
            f"{meta['combat_label']} | n={meta['n_runs']} avaliações externas"
        )
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def default_results_dir(modality: str, base: Path | None = None) -> Path:
    root = base if base is not None else Path("csvs/longitudinal_4_groups/ablation_results")
    return root / modality


if __name__ == "__main__":
    # ponytail: smoke test da lógica de frequência e pooled AUC
    demo_rows = []
    feats_ab = ["hippocampus_L_T1_gm_norm", "hippocampus_R_T2_gm_norm"]
    feats_cd = ["hippocampus_L_T1_gm_norm", "hippocampus_L_T3_gm_norm"]
    for rep in (0, 1):
        for fold, feats, y_true, scores in (
            (1, feats_ab, [0, 1], [0.2, 0.8]),
            (2, feats_cd, [1, 0], [0.7, 0.3]),
        ):
            demo_rows.append(
                {
                    "repeat_id": rep,
                    "fold": fold,
                    "task": "smci_pmci",
                    "modality": "vol",
                    "model_key": "svm",
                    "with_combat": True,
                    "selection_mode": "mrmr",
                    "combat_label": "ComBat",
                    "selected_features": json.dumps(feats),
                    "test_y_true": json.dumps(y_true),
                    "test_scores": json.dumps(scores),
                    "auc": 0.5,
                    "n_features_selected": len(feats),
                }
            )
    demo = pd.DataFrame(demo_rows)
    freq = feature_freq_table(demo)
    assert freq.loc[freq["feature"] == "hippocampus_L_T1_gm_norm", "pct"].iloc[0] == 100
    assert pooled_auc(demo) == 1.0
    print("ablation_analysis self-check ok")

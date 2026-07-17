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
FEAT_RE = re.compile(r"^hippocampus_([LR])_(T[123]|D21|D31|D32|SLOPE)_(.+)$")
TIME_ORDER = ("T1", "T2", "T3")
DELTA_TIME_ORDER = ("T1", "D21", "D31", "D32")
DELTA_TIME_ORDER_LEGACY = ("T1", "D21", "D31", "SLOPE")
LINE_PALETTE = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#88CCEE",
    "#44AA99",
    "#DDCC77",
)
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


def parse_feature(name: str) -> tuple[str, str, str] | None:
    """(lado, tempo, sufixo) ou None se não casar FEAT_RE."""
    m = FEAT_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def anatomical_key(name: str) -> str:
    """Colapsa T1/T2/T3/D21/D31/D32/SLOPE: hippocampus_L_T2_gm_norm → hippocampus_L_gm_norm."""
    parsed = parse_feature(name)
    if parsed is None:
        return name
    side, _, feat = parsed
    roi = name.split(f"_{side}_", 1)[0]
    return f"{roi}_{side}_{feat}"


def short_feature(name: str) -> str:
    m = FEAT_RE.match(name)
    if m:
        feat = m.group(3)
        if feat.startswith("original_shape_"):
            feat = feat.removeprefix("original_shape_")
        return f"{m.group(1)} {m.group(2)} | {feat}"
    return name


ANATOMICAL_RE = re.compile(r"^hippocampus_([LR])_(.+)$")


def short_anatomical_key(key: str) -> str:
    m = ANATOMICAL_RE.match(key)
    if m:
        feat = m.group(2)
        if feat.startswith("original_shape_"):
            feat = feat.removeprefix("original_shape_")
        return f"{m.group(1)} | {feat}"
    return key


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


def patient_mean_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Uma predição por paciente (média entre avaliações externas); alinha com 4_stats."""
    pat = explode_patient_predictions(df).groupby("ID_PT", as_index=False).agg(
        y=("y", "first"), score=("score", "mean")
    )
    return pat["y"].to_numpy(dtype=int), pat["score"].to_numpy(dtype=float)


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


def patient_mean_auc(df: pd.DataFrame) -> float:
    """AUC após média de escore OOF por paciente (1 linha / ID_PT)."""
    y, s = patient_mean_predictions(df)
    if len(np.unique(y)) < 2:
        raise ValueError("AUC patient-mean requer ambas as classes.")
    return float(roc_auc_score(y, s))


def explode_patient_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (paciente, fold, repeat) com score de teste externo."""
    required = {"test_id_pts", "test_y_true", "test_scores"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes: {sorted(missing)}")
    df = prepare_ablation_df(df)
    meta_cols = [
        "task",
        "modality",
        "model_key",
        "with_combat",
        "selection_mode",
        "fold",
        "repeat_id",
    ]
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        meta = {c: r[c] for c in meta_cols if c in df.columns}
        for pid, y, s in zip(
            json.loads(r["test_id_pts"]),
            json.loads(r["test_y_true"]),
            json.loads(r["test_scores"]),
        ):
            rows.append(
                {
                    "ID_PT": str(pid),
                    "y": int(y),
                    "score": float(s),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def rank_patients_by_discordance(pat: pd.DataFrame) -> pd.DataFrame:
    """Agrega scores por paciente; discordance = distância ao rótulo (0/1)."""
    empty = pd.DataFrame(
        columns=["ID_PT", "y", "score_mean", "score_std", "n_test", "discordance"]
    )
    if pat.empty:
        return empty
    rank = pat.groupby("ID_PT", as_index=False).agg(
        y=("y", "first"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        n_test=("score", "count"),
    )
    rank["score_std"] = rank["score_std"].fillna(0.0)
    rank["discordance"] = np.where(
        rank["y"] == 1,
        1.0 - rank["score_mean"],
        rank["score_mean"],
    )
    return rank.sort_values("discordance", ascending=False).reset_index(drop=True)


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


def feature_freq_table_grouped(
    df: pd.DataFrame,
    *,
    top_n: int | None = None,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Estabilidade por biomarcador anatômico (T1/T2/T3 colapsados).

  - coverage: % folds com ≥1 tempo selecionado
  - freq_T1/T2/T3: % folds em que cada visita entrou
  - amplitude: média de (#tempos no fold / 3) nos folds em que entrou
    """
    df = prepare_ablation_df(df)
    n_runs = n_outer_evals(df)
    empty_cols = [
        "feature_group",
        "feature_short",
        "coverage",
        "coverage_pct",
        "freq_T1",
        "freq_T2",
        "freq_T3",
        "pct_T1",
        "pct_T2",
        "pct_T3",
        "amplitude",
        "n_folds_any",
        "total_selections",
        "n_runs",
    ]
    if n_runs == 0:
        return pd.DataFrame(columns=empty_cols)

    all_feats: list[str] = []
    for _, row in df.iterrows():
        all_feats.extend(json.loads(row["selected_features"]))
    time_order = _time_order_for_names(all_feats)
    n_slots = len(time_order)

    meta = df.iloc[0]
    fold_any: dict[str, int] = {}
    fold_time: dict[str, dict[str, int]] = {}
    fold_n_times: dict[str, list[int]] = {}
    total_sel: dict[str, int] = {}

    for _, row in df.iterrows():
        by_group: dict[str, set[str]] = {}
        for feat in json.loads(row["selected_features"]):
            grp = anatomical_key(feat)
            parsed = parse_feature(feat)
            if parsed is None:
                continue
            _, time_pt, _ = parsed
            by_group.setdefault(grp, set()).add(time_pt)
            total_sel[grp] = total_sel.get(grp, 0) + 1

        for grp, times in by_group.items():
            fold_any[grp] = fold_any.get(grp, 0) + 1
            fold_n_times.setdefault(grp, []).append(len(times))
            ft = fold_time.setdefault(grp, _empty_fold_time(time_order))
            for t in times:
                if t in ft:
                    ft[t] += 1

    rows: list[dict[str, Any]] = []
    for grp in sorted(fold_any.keys(), key=lambda g: (-fold_any[g], g)):
        cov = fold_any[grp] / n_runs
        if cov < min_coverage:
            continue
        ft = fold_time.get(grp, _empty_fold_time(time_order))
        amp_vals = fold_n_times.get(grp, [])
        amplitude = float(np.mean([n / n_slots for n in amp_vals])) if amp_vals else 0.0
        row_dict: dict[str, Any] = {
            "task": meta.get("task"),
            "modality": meta.get("modality"),
            "combat_label": meta.get("combat_label"),
            "model_key": meta.get("model_key"),
            "selection_mode": meta.get("selection_mode"),
            "feature_group": grp,
            "feature_short": short_anatomical_key(grp),
            "coverage": cov,
            "coverage_pct": int(round(100 * cov)),
            "amplitude": amplitude,
            "n_folds_any": fold_any[grp],
            "total_selections": total_sel.get(grp, 0),
            "n_runs": n_runs,
        }
        for t in time_order:
            row_dict[f"freq_{t}"] = ft[t] / n_runs
            row_dict[f"pct_{t}"] = int(round(100 * ft[t] / n_runs))
        rows.append(row_dict)

    out = pd.DataFrame(rows).sort_values(
        ["coverage", "amplitude", "feature_group"], ascending=[False, False, True]
    )
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def _time_order_for_names(names: list[str]) -> tuple[str, ...]:
    tokens = {parse_feature(n)[1] for n in names if parse_feature(n) is not None}
    if tokens & (set(DELTA_TIME_ORDER[1:]) | set(DELTA_TIME_ORDER_LEGACY[1:])):
        return DELTA_TIME_ORDER if "D32" in tokens else DELTA_TIME_ORDER_LEGACY
    return TIME_ORDER


def _empty_fold_time(time_order: tuple[str, ...]) -> dict[str, int]:
    return {t: 0 for t in time_order}


def count_stable_timepoints(
    grp: pd.DataFrame,
    *,
    min_pct: int = 70,
    time_order: tuple[str, ...] = TIME_ORDER,
) -> pd.Series:
    """Quantas visitas/representações têm incidência >= min_pct."""
    cols = [f"pct_{t}" for t in time_order if f"pct_{t}" in grp.columns]
    if not cols:
        cols = [c for c in grp.columns if c.startswith("pct_")]
    pcts = grp[cols]
    return (pcts >= min_pct).sum(axis=1)


def filter_temporally_stable(
    grp: pd.DataFrame,
    *,
    min_pct: int = 70,
    min_timepoints: int = 2,
    time_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Pós-hoc: mantém biomarcadores com >= min_timepoints visitas em >= min_pct%."""
    if grp.empty:
        return grp.copy()
    if time_order is None:
        time_order = DELTA_TIME_ORDER if "pct_D32" in grp.columns else (
            DELTA_TIME_ORDER_LEGACY if "pct_SLOPE" in grp.columns else TIME_ORDER
        )
    n = count_stable_timepoints(grp, min_pct=min_pct, time_order=time_order)
    return grp.loc[n >= min_timepoints].reset_index(drop=True)


def estimate_stable_pool_columns(
    feature_names: list[str],
    inner_selected: list[list[str]],
    *,
    min_pct: int = 70,
    min_timepoints: int = 2,
) -> tuple[list[str], list[str]]:
    """Pool restrito a partir de seleções em inner folds (sem leakage do teste externo)."""
    if not feature_names:
        return [], []
    n_inner = len(inner_selected)
    if n_inner == 0:
        return list(feature_names), []

    time_order = _time_order_for_names(feature_names)
    fold_any: dict[str, int] = {}
    fold_time: dict[str, dict[str, int]] = {}
    for selected in inner_selected:
        by_group: dict[str, set[str]] = {}
        for feat in selected:
            grp = anatomical_key(feat)
            parsed = parse_feature(feat)
            if parsed is None:
                continue
            _, time_pt, _ = parsed
            by_group.setdefault(grp, set()).add(time_pt)
        for grp, times in by_group.items():
            fold_any[grp] = fold_any.get(grp, 0) + 1
            ft = fold_time.setdefault(grp, _empty_fold_time(time_order))
            for t in times:
                if t in ft:
                    ft[t] += 1

    passing: set[str] = set()
    for grp in fold_any:
        if min_timepoints < 1:
            if 100 * fold_any[grp] / n_inner >= min_pct:
                passing.add(grp)
            continue
        ft = fold_time[grp]
        n_pass = sum(1 for t in time_order if 100 * ft[t] / n_inner >= min_pct)
        if n_pass >= min_timepoints:
            passing.add(grp)

    kept: list[str] = []
    removed: list[str] = []
    for col in feature_names:
        parsed = parse_feature(col)
        if parsed is None:
            kept.append(col)
            continue
        if anatomical_key(col) in passing:
            kept.append(col)
        else:
            removed.append(col)

    if not kept:
        return list(feature_names), []  # ponytail: evita pool vazio
    return kept, removed


def summarize_selection_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por fold: contagem e listas removidas em cada estágio."""
    cols = [
        "task",
        "modality",
        "model_key",
        "with_combat",
        "selection_mode",
        "repeat_id",
        "fold",
        "n_features_raw",
        "n_features_after_stable_pool",
        "n_features_after_filters",
        "n_features_selected",
        "removed_by_stable_pool",
        "removed_by_filters",
        "removed_by_mrmr",
        "selected_features",
    ]
    present = [c for c in cols if c in df.columns]
    return df[present].copy()


def union_removed(df: pd.DataFrame, col: str) -> list[str]:
    """União de features removidas numa coluna JSON ao longo dos folds."""
    out: set[str] = set()
    if col not in df.columns:
        return []
    for val in df[col].dropna():
        out.update(json.loads(val) if isinstance(val, str) else val)
    return sorted(out)


def feature_short_name(name: str) -> str:
    s = short_anatomical_key(name)
    return s if "|" in s else short_feature(name)


def _col_or(frame: pd.DataFrame, name: str, fallback: pd.Series) -> pd.Series:
    return frame[name] if name in frame.columns else fallback


def selection_audit_report(
    df: pd.DataFrame,
    *,
    task: str,
    model: str,
    with_combat: bool,
    modality: str | None = None,
    selection_modes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Contagem wide→final e remoções por estágio (para 3_results.ipynb)."""
    df = prepare_ablation_df(df)
    mask = (
        (df["task"].astype(str) == task)
        & (df["model_key"].astype(str) == model)
        & (df["with_combat"] == with_combat)
    )
    if modality is not None:
        mask &= df["modality"].astype(str) == modality
    sub = df.loc[mask]
    if sub.empty:
        print(f"Auditoria vazia: {task}/{model}/combat={with_combat}")
        return pd.DataFrame()

    present = sorted(sub["selection_mode"].astype(str).unique())
    modes = selection_modes or tuple(present)

    for mode in modes:
        if mode not in present:
            print(f"\n[{mode}] sem dados")
            continue
        part = sub[sub["selection_mode"].astype(str) == mode]
        mod_label = modality or str(part["modality"].iloc[0])
        n_inicial = int(part["n_features_raw"].iloc[0])
        n_final = float(part["n_features_selected"].mean())

        print(f"\n{'=' * 60}")
        print(f"[{mode}] {task} | {mod_label} | {model} | combat={with_combat}")
        print(f"  wide (inicial):     {n_inicial} colunas")
        if mode in ("mrmr_stable", "l1_stable") and "n_features_after_stable_pool" in part.columns:
            n_pool = float(_col_or(part, "n_features_after_stable_pool", part["n_features_raw"]).mean())
            print(f"  após pool estável:  {n_pool:.1f} (média folds)")
        if mode in ("mrmr", "mrmr_stable", "l1_stable", "filters") and "n_features_after_filters" in part.columns:
            n_filt = float(_col_or(part, "n_features_after_filters", part["n_features_selected"]).mean())
            print(f"  após corr/var:      {n_filt:.1f} (média folds)")
        print(
            f"  selecionados final: {n_final:.1f} (média) | "
            f"min={part['n_features_selected'].min()} max={part['n_features_selected'].max()}"
        )

        if mode in ("mrmr_stable", "l1_stable") and "removed_by_stable_pool" in part.columns:
            stable_rm = union_removed(part, "removed_by_stable_pool")
            print(f"  removidos pool estável ({len(stable_rm)}): {[feature_short_name(x) for x in stable_rm[:12]]}")

        if "removed_by_filters" in part.columns:
            filt_rm = union_removed(part, "removed_by_filters")
            if filt_rm:
                print(f"  removidos corr/var ({len(filt_rm)}): {[feature_short_name(x) for x in filt_rm[:12]]}")
        if mode in ("mrmr", "mrmr_stable") and "removed_by_mrmr" in part.columns:
            mrmr_rm = union_removed(part, "removed_by_mrmr")
            if mrmr_rm:
                print(f"  removidos mRMR ({len(mrmr_rm)}): {[feature_short_name(x) for x in mrmr_rm[:12]]}")

    rows = []
    for mode in modes:
        part = sub[sub["selection_mode"].astype(str) == mode]
        if part.empty:
            continue
        rows.append(
            {
                "selection_mode": mode,
                "n_wide": int(part["n_features_raw"].iloc[0]),
                "n_final_mean": round(float(part["n_features_selected"].mean()), 1),
                "n_removidos_mean": round(
                    float(part["n_features_raw"].iloc[0] - part["n_features_selected"].mean()), 1
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        print("\nResumo wide → final:")
        print(summary.to_string(index=False))
    return summary


def _time_order_from_freq(freq: pd.DataFrame) -> tuple[str, ...]:
    tokens = [c.removeprefix("pct_") for c in freq.columns if c.startswith("pct_")]
    if set(tokens) & (set(DELTA_TIME_ORDER[1:]) | set(DELTA_TIME_ORDER_LEGACY[1:])):
        return DELTA_TIME_ORDER if "D32" in tokens else DELTA_TIME_ORDER_LEGACY
    return TIME_ORDER


def plot_temporal_lines_on_ax(
    ax: plt.Axes, freq: pd.DataFrame, *, show_legend: bool = True
) -> None:
    """Uma linha por biomarcador; eixo X = T1/T2/T3; eixo Y = incidência (%)."""
    _plot_temporal_lines_on_ax(ax, freq, show_legend=show_legend)


def _plot_temporal_lines_on_ax(
    ax: plt.Axes, freq: pd.DataFrame, *, show_legend: bool = True
) -> None:
    """Uma linha por biomarcador; eixo X = visitas ou representações; Y = incidência (%)."""
    time_order = _time_order_from_freq(freq)
    x = np.arange(len(time_order))
    for i, row in enumerate(freq.itertuples()):
        ys = [getattr(row, f"pct_{t}", 0) for t in time_order]
        ax.plot(
            x,
            ys,
            color=LINE_PALETTE[i % len(LINE_PALETTE)],
            marker="o",
            markersize=7,
            linewidth=1.8,
            label=row.feature_short,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(time_order)
    ax.set_xlabel("Visita" if time_order == TIME_ORDER else "Representação")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Incidência (% das avaliações externas)")
    for y in (20, 40, 60, 80):
        ax.axhline(y, color="gray", ls="--", lw=0.4, alpha=0.45)
    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            framealpha=0.9,
        )


def plot_temporal_stability_lines(
    df_config: pd.DataFrame,
    *,
    freq: pd.DataFrame | None = None,
    title: str | None = None,
    out_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    min_coverage: float = 0.0,
    show_coverage: bool = False,  # ponytail: cobertura não mapeia a um único T
    sort_by: str = "feature_short",
) -> plt.Figure:
    """Uma linha por biomarcador; eixo X = T1/T2/T3; eixo Y = % de incidência."""
    if freq is None:
        freq = feature_freq_table_grouped(df_config, min_coverage=min_coverage)
    if freq.empty:
        raise ValueError("Nenhum biomarcador agrupado para plotar.")

    if sort_by == "coverage":
        freq = freq.sort_values(
            ["coverage_pct", "feature_short"], ascending=[False, True]
        )
    else:
        freq = freq.sort_values("feature_short")

    n = len(freq)
    if figsize is None:
        figsize = (7, max(4.5, 0.35 * n + 3.5))

    fig, ax = plt.subplots(figsize=figsize)
    _plot_temporal_lines_on_ax(ax, freq)

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


def plot_feature_stability_grouped(
    df_config: pd.DataFrame,
    *,
    title: str | None = None,
    out_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
    min_coverage: float = 0.0,
    show_coverage: bool = True,
) -> plt.Figure:
    """Alias: gráfico de linhas T1/T2/T3 (substitui barras de cobertura)."""
    return plot_temporal_stability_lines(
        df_config,
        title=title,
        out_path=out_path,
        figsize=figsize,
        min_coverage=min_coverage,
        show_coverage=show_coverage,
    )


def plot_compare_stability(
    df_config: pd.DataFrame,
    *,
    title: str | None = None,
    out_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Empilhado: barras por coluna (cima) + linhas T1/T2/T3 agrupadas (baixo)."""
    ind = feature_freq_table(df_config, top_n=None)
    grp = feature_freq_table_grouped(df_config)
    if ind.empty and grp.empty:
        raise ValueError("Nenhum atributo selecionado para comparar.")

    ind = ind.sort_values(["pct", "feature_short"], ascending=[True, True])
    grp = grp.sort_values("feature_short")
    n_ind, n_grp = len(ind), len(grp)
    if figsize is None:
        figsize = (9, max(8, 0.22 * n_ind + 5))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [max(n_ind, 3), 3]}
    )
    if n_ind:
        y = np.arange(n_ind)
        ax_top.barh(y, ind["pct"], height=0.75, color="#4477AA", alpha=0.9)
        ax_top.set_yticks(y)
        ax_top.set_yticklabels(ind["feature_short"], fontsize=7)
    ax_top.set_xlim(0, 100)
    ax_top.set_xlabel("Frequência (%)")
    ax_top.set_title("Por coluna (T1/T2/T3 separados)", fontsize=9)
    for x in (20, 40, 60, 80):
        ax_top.axvline(x, color="gray", ls="--", lw=0.5, alpha=0.5)

    if n_grp:
        _plot_temporal_lines_on_ax(ax_bot, grp)
    ax_bot.set_title("Agrupado — incidência por visita (linha = biomarcador)", fontsize=9)

    meta = ind.iloc[0] if n_ind else grp.iloc[0]
    if title is None:
        title = (
            f"{meta['task']} | {meta['modality']} | {meta['model_key']} | "
            f"{meta['combat_label']} | n={meta['n_runs']}"
        )
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def summary_with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por configuração: AUC patient-mean (primário), pooled e média por fold."""
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
        row["auc_patient_mean"] = patient_mean_auc(grp)
        row["n_features_mean"] = float(grp["n_features_selected"].mean())
        for col in METRIC_COLS:
            if col in grp.columns and col != "auc":
                row[f"{col}_mean"] = float(grp[col].mean())
                row[f"{col}_std"] = float(grp[col].std(ddof=0))
        rows.append(row)

    out = pd.DataFrame(rows)
    # primário alinhado a ROC / 4_stats; pooled permanece como coluna de sensibilidade
    sort_col = "auc_patient_mean" if "auc_patient_mean" in out.columns else "auc_pooled"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


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
                    "test_id_pts": json.dumps([f"P{fold}{i}" for i in range(len(y_true))]),
                    "auc": 0.5,
                    "n_features_selected": len(feats),
                }
            )
    demo = pd.DataFrame(demo_rows)
    freq = feature_freq_table(demo)
    grp = feature_freq_table_grouped(demo)
    assert freq.loc[freq["feature"] == "hippocampus_L_T1_gm_norm", "pct"].iloc[0] == 100
    assert grp.loc[grp["feature_group"] == "hippocampus_L_gm_norm", "coverage_pct"].iloc[0] == 100
    assert grp.loc[grp["feature_group"] == "hippocampus_L_gm_norm", "pct_T1"].iloc[0] == 100
    assert grp.loc[grp["feature_group"] == "hippocampus_L_gm_norm", "pct_T3"].iloc[0] == 50
    assert pooled_auc(demo) == 1.0
    assert patient_mean_auc(demo) == 1.0
    demo_grp = pd.DataFrame(
        [
            {"feature_short": "R | wm_norm", "pct_T1": 12, "pct_T2": 100, "pct_T3": 24},
            {"feature_short": "L | wm_norm", "pct_T1": 98, "pct_T2": 100, "pct_T3": 92},
        ]
    )
    stable = filter_temporally_stable(demo_grp, min_pct=70, min_timepoints=2)
    assert len(stable) == 1 and stable.iloc[0]["feature_short"] == "L | wm_norm"
    kept, removed = estimate_stable_pool_columns(
        [
            "hippocampus_R_T1_wm_norm",
            "hippocampus_R_T2_wm_norm",
            "hippocampus_L_T1_gm_norm",
            "hippocampus_L_T2_gm_norm",
        ],
        [
            [
                "hippocampus_R_T2_wm_norm",
                "hippocampus_L_T1_gm_norm",
                "hippocampus_L_T2_gm_norm",
                "hippocampus_L_T3_gm_norm",
            ],
            [
                "hippocampus_R_T2_wm_norm",
                "hippocampus_L_T1_gm_norm",
                "hippocampus_L_T2_gm_norm",
                "hippocampus_L_T3_gm_norm",
            ],
            [
                "hippocampus_R_T2_wm_norm",
                "hippocampus_L_T1_gm_norm",
                "hippocampus_L_T2_gm_norm",
                "hippocampus_L_T3_gm_norm",
            ],
        ],
        min_pct=70,
        min_timepoints=2,
    )
    assert "hippocampus_L_T1_gm_norm" in kept and "hippocampus_R_T1_wm_norm" in removed
    pat = explode_patient_predictions(demo)
    rank = rank_patients_by_discordance(pat)
    assert len(pat) == 8 and len(rank) == 4
    assert rank.iloc[0]["discordance"] >= rank.iloc[-1]["discordance"]
    print("ablation_analysis self-check ok")

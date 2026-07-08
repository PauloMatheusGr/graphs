"""Comparações pareadas de AUC para 4_stats.ipynb (bootstrap + FDR)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def combat_suffix(with_combat: bool) -> str:
    return "combat" if with_combat else "nocombat"


def image_ablation_path(base: Path, protocol: str, modality: str) -> Path:
    roots = {
        "abs": base / "ablation_results",
        "t1_only": base / "ablation_results_t1_only",
        "global": base / "ablation_results_leaky",
    }
    if protocol not in roots:
        raise KeyError(f"protocolo imagem desconhecido: {protocol}")
    return roots[protocol] / modality / "ablation_results_all.csv"


def fusion_results_path(
    base: Path,
    modality: str,
    *,
    selection_mode: str = "l1_stable",
    with_combat: bool = False,
    representation: str = "wide",
) -> Path:
    rep_roots = {
        "wide": base / "ablation_results_clinic_img",
        "t1_only": base / "ablation_results_clinic_img_t1_only",
    }
    root = rep_roots.get(representation)
    if root is None:
        raise KeyError(f"representation fusion desconhecida: {representation}")
    tag = combat_suffix(with_combat)
    if representation == "t1_only":
        fname = f"fusion_{modality}_{selection_mode}_{tag}_{representation}_results_all.csv"
    else:
        fname = f"fusion_{modality}_{selection_mode}_{tag}_results_all.csv"
    return root / fname


def clinical_results_path(base: Path) -> Path:
    return base / "ablation_results_clinic" / "clinical_results_all.csv"


def apply_bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR; NaN permanece NaN."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    pv = p[mask]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    filled = np.empty(n)
    filled[order] = q
    out[mask] = filled
    return out


def bootstrap_auc_diff_test(
    y,
    scores_a,
    scores_b,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    y = np.asarray(y, dtype=int)
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    obs = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    prng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = prng.integers(0, len(y), size=len(y))
        yb, ab, bb = y[idx], a[idx], b[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(float(roc_auc_score(yb, ab) - roc_auc_score(yb, bb)))
    diffs_arr = np.asarray(diffs)
    ci_lo, ci_hi = np.percentile(diffs_arr, [2.5, 97.5])
    p_one = float((np.sum(diffs_arr <= 0) + 1) / (len(diffs_arr) + 1))
    p_two = float(2 * min(p_one, 1 - p_one))
    return obs, float(ci_lo), float(ci_hi), p_one, p_two


def paired_comparison_row(
    paired: pd.DataFrame,
    *,
    score_a: str,
    score_b: str,
    label_a: str,
    label_b: str,
    n_boot: int,
    seed: int,
    permutation_auc_p: Callable[..., tuple[float, float]],
    n_perm: int,
    alpha: float = 0.05,
) -> dict:
    y = paired["y"].to_numpy()
    sa = paired[score_a].to_numpy()
    sb = paired[score_b].to_numpy()
    auc_a, p_perm_a = permutation_auc_p(y, sa, n_perm=n_perm, seed=seed)
    auc_b, p_perm_b = permutation_auc_p(y, sb, n_perm=n_perm, seed=seed + 1)
    d, lo, hi, p_one, p_two = bootstrap_auc_diff_test(
        y, sa, sb, n_boot=n_boot, seed=seed + 2,
    )
    return {
        "n_pacientes": len(paired),
        f"auc_{label_a}": auc_a,
        f"auc_{label_b}": auc_b,
        "delta_auc": d,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "p_bootstrap_one_sided": p_one,
        "p_bootstrap_two_sided": p_two,
        f"{label_a}_superior": bool(lo > 0),
        f"p_perm_{label_a}": p_perm_a,
        f"p_perm_{label_b}": p_perm_b,
        "significant_fdr": False,  # preenchido depois
        "significant_raw": p_one < alpha and lo > 0,
    }


def compare_modalities(
    modalities: tuple[str, ...],
    *,
    path_a: Callable[[str], Path | None],
    path_b: Callable[[str], Path | None],
    cfg_for_mod: Callable[[str], object],
    load_patients: Callable[[Path, object], pd.DataFrame],
    permutation_auc_p: Callable[..., tuple[float, float]],
    n_perm: int,
    n_bootstrap: int,
    seed: int,
    label_a: str,
    label_b: str,
    comparison: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, mod in enumerate(modalities):
        pa, pb = path_a(mod), path_b(mod)
        if pa is None or pb is None or not pa.exists() or not pb.exists():
            continue
        cfg = cfg_for_mod(mod)
        pat_a = load_patients(pa, cfg)
        pat_b = load_patients(pb, cfg)
        paired = pat_a.merge(pat_b, on=["ID_PT", "y"], suffixes=(f"_{label_a}", f"_{label_b}"))
        if paired.empty:
            continue
        row = paired_comparison_row(
            paired,
            score_a=f"score_{label_a}",
            score_b=f"score_{label_b}",
            label_a=label_a,
            label_b=label_b,
            n_boot=n_bootstrap,
            seed=seed + i * 17,
            permutation_auc_p=permutation_auc_p,
            n_perm=n_perm,
            alpha=alpha,
        )
        row["modality"] = mod
        row["comparison"] = comparison
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "p_bootstrap_one_sided" in out.columns:
        out["p_fdr_bh"] = apply_bh_fdr(out["p_bootstrap_one_sided"].to_numpy())
        out["significant_fdr"] = (out["p_fdr_bh"] < alpha) & (out["ci95_lo"] > 0)
    return out


def print_comparison_summary(df: pd.DataFrame, *, label_a: str, label_b: str) -> None:
    if df.empty:
        print("  (sem dados)")
        return
    for _, r in df.iterrows():
        sig = "FDR sig." if r.get("significant_fdr") else (
            "raw sig." if r.get("significant_raw") else "sem evidência"
        )
        print(
            f"  {r['modality']}: ΔAUC={r['delta_auc']:.3f} "
            f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]  "
            f"p={r['p_bootstrap_one_sided']:.4f}  q={r.get('p_fdr_bh', float('nan')):.4f}  → {sig}"
        )


if __name__ == "__main__":
    # ponytail: sanity check — q(FDR) ∈ [0,1], ordenação BH
    p = np.array([0.01, 0.04, 0.2, np.nan])
    q = apply_bh_fdr(p)
    assert np.all((q[np.isfinite(q)] >= 0) & (q[np.isfinite(q)] <= 1))
    assert np.isnan(q[3])
    print("stats_compare ok")

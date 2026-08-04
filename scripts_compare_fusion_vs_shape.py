#!/usr/bin/env python3
"""Bootstrap pareado patient-level: fusion (early ou late) vs t1_only/shape."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_MOD = Path(__file__).resolve().parent / "modules"
if str(_MOD) not in sys.path:
    sys.path.insert(0, str(_MOD))


def _patient_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        ids = json.loads(r["test_id_pts"])
        y = json.loads(r["test_y_true"])
        s = json.loads(r["test_scores"])
        for i, yi, si in zip(ids, y, s):
            rows.append(
                {
                    "ID_PT": str(i),
                    "y": int(yi),
                    "score": float(si),
                    "repeat_id": int(r["repeat_id"]),
                    "fold": int(r["fold"]),
                }
            )
    long = pd.DataFrame(rows)
    return long.groupby("ID_PT", as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def bootstrap_delta(y, a, b, *, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    base = roc_auc_score(y, a) - roc_auc_score(y, b)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_one = float(np.mean(diffs <= 0))
    p_two = float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return base, lo, hi, p_one, min(p_two, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="48m_12m")
    ap.add_argument("--fingerprint", default="t1_shape__t1_deltas_texture")
    ap.add_argument(
        "--mode",
        choices=["early", "late"],
        default="early",
        help="early=ablation_results_fusion | late=ablation_results_late_fusion",
    )
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base = Path(f"csvs/cohorts/{args.cohort}")
    root = (
        "ablation_results_late_fusion"
        if args.mode == "late"
        else "ablation_results_fusion"
    )
    fusion = base / root / args.fingerprint / "ablation_results_all.csv"
    shape = base / "ablation_results_t1_only" / "shape" / "ablation_results_all.csv"
    if not fusion.is_file():
        print(f"missing {fusion}", file=sys.stderr)
        return 1
    if not shape.is_file():
        print(f"missing {shape}", file=sys.stderr)
        return 1
    df_f = pd.read_csv(fusion)
    df_s = pd.read_csv(shape)
    df_f = df_f[
        (df_f["task"] == "smci_pmci")
        & (df_f["model_key"] == "svm")
        & (df_f["with_combat"] == False)
    ]
    df_s = df_s[
        (df_s["task"] == "smci_pmci")
        & (df_s["model_key"] == "svm")
        & (df_s["with_combat"] == False)
    ]
    pf = _patient_scores(df_f).rename(columns={"score": "score_fusion"})
    ps = _patient_scores(df_s).rename(columns={"score": "score_shape"})
    m = pf.merge(ps[["ID_PT", "score_shape"]], on="ID_PT", how="inner")
    sum_f = pd.read_csv(fusion.parent / "ablation_summary.csv")
    sum_s = pd.read_csv(shape.parent / "ablation_summary.csv")
    af = float(
        sum_f.loc[
            (sum_f["task"] == "smci_pmci") & (sum_f["model_key"] == "svm"),
            "auc_patient_mean",
        ].iloc[0]
    )
    as_ = float(
        sum_s.loc[
            (sum_s["task"] == "smci_pmci") & (sum_s["model_key"] == "svm"),
            "auc_patient_mean",
        ].iloc[0]
    )
    delta, lo, hi, p1, p2 = bootstrap_delta(
        m["y"].to_numpy(),
        m["score_fusion"].to_numpy(),
        m["score_shape"].to_numpy(),
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(
        f"cohort={args.cohort} mode={args.mode} fingerprint={args.fingerprint} "
        f"n_patients={len(m)}"
    )
    print(f"auc_patient fusion={af:.4f}  shape_t1={as_:.4f}  point_delta={af - as_:+.4f}")
    print(
        f"bootstrap ΔAUC={delta:+.4f}  IC95=[{lo:+.4f}, {hi:+.4f}]  "
        f"p_one(fusion>shape)={p1:.4f}  p_two={p2:.4f}"
    )
    ok = delta > 0 and lo > 0
    print(f"gate_pass={ok}  (Δ>0 e IC95 lo>0)")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

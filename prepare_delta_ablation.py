#!/usr/bin/env python3
"""Exporta CSV wide T1 + deltas relativos para inspeção (ablation_deltas/{roi}/)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ablation_deltas import PROTOCOL_T1_DELTAS, add_delta_columns, modality_wide_columns
from ablation_prep import (
    ROI_FILTER_DEFAULT,
    pivot_long_to_wide,
)

MODALITIES = {
    "vol": "vol_long.csv",
    "shape": "shape_long.csv",
    "texture": "rad_long.csv",
    "disp": "disp_long.csv",
    "all": "merge_long.csv",
}


def export_delta_wide(
    base_dir: Path | str,
    *,
    roi: str = ROI_FILTER_DEFAULT,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    base = Path(base_dir)
    ablation_dir = base / "ablation" / roi
    dest = Path(out_dir) if out_dir is not None else base / "ablation_deltas" / roi
    dest.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    meta = ["ID_PT", "GROUP", "SEX"]

    for mod, long_name in MODALITIES.items():
        long_path = ablation_dir / long_name
        if not long_path.is_file():
            raise FileNotFoundError(f"Long CSV ausente: {long_path}")
        df_long = pd.read_csv(long_path)
        wide = pivot_long_to_wide(df_long)
        wide = add_delta_columns(wide, roi)
        cols = modality_wide_columns(wide.columns, mod, roi=roi, use_deltas=True)
        out = wide[meta + cols].copy()
        p = dest / f"{mod}_wide.csv"
        out.to_csv(p, index=False)
        paths[mod] = p
        print(f"[{PROTOCOL_T1_DELTAS}] {p.name}: pacientes={len(out)} features={len(cols)}")

    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exporta wide T1+deltas para ablation_deltas/")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("csvs/longitudinal_4_groups"),
        help="Raiz com ablation/{roi}/",
    )
    p.add_argument("--roi", default=ROI_FILTER_DEFAULT)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override saída (default: {base}/ablation_deltas/{roi})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_delta_wide(args.base_dir, roi=args.roi, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

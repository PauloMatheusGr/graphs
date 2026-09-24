#!/usr/bin/env python3
"""
DVF v4 âncora AD: mesmos mapas D/V/S da v4 CN, warps em displacement_field_v3_ad.

Pré-requisito: 3.1_feat_gen_dvf_ad.py → images/displacement_field_v3_ad/

Saídas:
  csvs/cohorts/all_population/features_displacement_v4_ad.csv
  images/displacement_field_v3_ad/features_v4/all_population/
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent
V4_SCRIPT = ROOT / "3.2_feat_dvf_v4.py"
ANCHOR_DIAG = "AD"


def _load_v4() -> ModuleType:
    spec = importlib.util.spec_from_file_location("feat_dvf_v4_cn", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {V4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Avoid running CN __main__; loader still executes module body (defines helpers).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v4 = _load_v4()
base = v4.base

base.WARPS_DIR = "./images/displacement_field_v3_ad"
base.OUT_CSV = f"{base.COHORT_DIR}/features_displacement_v4_ad.csv"
base.RUN_DIR = os.path.join(base.WARPS_DIR, "features_v4", base.COHORT)


def _ref_tag(sex: str, age_range: str) -> str:
    return f"{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}"


def template_path_for(sex: str, age_range: str) -> str:
    return os.path.join(
        base.GROUPWISE_DIR,
        f"groupwise_DIAG-{ANCHOR_DIAG}_SEX-{sex}_AGE-{age_range}_N-20_template.nii.gz",
    )


base._ref_tag = _ref_tag
base.template_path_for = template_path_for


def persist_run_metadata(
    *,
    run_meta_path: str,
    out_csv: str,
    done_keys_path: str,
    images_csv: str,
) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": "dvf_v4_mbec_maps_ad_only",
        "inputs": {
            "images_csv": str(images_csv),
            "warps_dir": str(base.WARPS_DIR),
            "warp_pattern": "*_1Warp.nii.gz",
            "anchor": ANCHOR_DIAG,
        },
        "outputs": {
            "out_csv": str(out_csv),
            "done_keys_path": str(done_keys_path),
        },
        "paper_alignment": {
            "reference": "MBEC-D-26-01909 (Andrade et al.) §§3.5–3.6",
            "maps": {
                "D": "mag = ||u||_2",
                "V": "jac_det = det(I + grad(u))",
                "S": "strain_fro = ||eps||_F, eps = 0.5 (H + H^T)",
            },
            "roi_stats": list(v4.PAPER_MOMENT_KEYS),
        },
        "conventions": {
            "domain": "AD template (fixed)",
            "displacement": "forward SyN nonlinear warp only; affine excluded",
            "labels_to_template": "[forward warp, affine], nearest-neighbor",
            "n_rois": len(v4.ROI_TABLE),
            "paper_prefixes": list(v4.PAPER_MAP_PREFIXES),
        },
    }
    Path(run_meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(run_meta_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    v4.persist_run_metadata = persist_run_metadata
    # Re-bind v4 hooks onto base (same as v4.main), then run.
    base.inv_list_for = v4.nonlinear_warp_list_for
    base.compute_unitary_scalar_arrays = v4.compute_unitary_scalar_arrays
    base._build_roi_feature_row = v4.build_roi_feature_row
    base.persist_run_metadata = persist_run_metadata
    base.main()


def self_check() -> None:
    assert _ref_tag("F", "60-69") == "AD_SEX-F_AGE-60-69"
    assert "DIAG-AD" in template_path_for("F", "60-69")
    assert base.WARPS_DIR.endswith("displacement_field_v3_ad")
    assert base.OUT_CSV.endswith("features_displacement_v4_ad.csv")
    paths = v4.nonlinear_warp_list_for(base.WARPS_DIR, "img", "F", "60-69")
    assert len(paths) == 1 and "AD_SEX-F_AGE-60-69_1Warp" in paths[0]
    print("ok: DVF v4 AD = same maps, warps/csv AD-isolated")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()

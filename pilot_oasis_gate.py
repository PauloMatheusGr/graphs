#!/usr/bin/env python3
"""Piloto DVF OASIS: listas de imagens e critérios de continuidade (gates).

  python pilot_oasis_gate.py ids-a    # 40 CN + 40 AD baseline 48m_6m (20 F + 20 M cada)
  python pilot_oasis_gate.py ids-b    # baselines de 48m_6m ∪ 48m_6m_soft_False
  python pilot_oasis_gate.py gate-a   # tempo, |rho| jac × volume/ICV, AUC univariada CN×AD
  python pilot_oasis_gate.py gate-b   # ablação t1_only pareada: disp_oasis* × disp*
  python pilot_oasis_gate.py --self-check
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "modules"))
from ablation_analysis import explode_patient_predictions  # noqa: E402
from stats_compare import bootstrap_auc_diff_test  # noqa: E402

SEED = 42
PILOT = Path("csvs/pilot")
FEAT = Path("csvs/cohorts/all_population")
COHORTS = ("48m_6m", "48m_6m_soft_False")
ROIS = ("hippocampus_d4", "hippocampus", "hippocampus_d2", "hippocampus_d8", "hippocampus_shell4")
PRIMARY_ROI = "hippocampus_d4"
PAIRS = (("disp_oasis", "disp"), ("disp_oasis_ad", "disp_ad"), ("disp_oasis_cnad", "disp_cnad"))
OLD_FEAT = {"cn": "features_displacement_v4.csv", "ad": "features_displacement_v4_ad.csv"}
NEW_FEAT = {"cn": "features_displacement_oasis_cn.csv", "ad": "features_displacement_oasis_ad.csv"}


def baselines(cohort: str) -> pd.DataFrame:
    long = pd.read_csv(f"csvs/cohorts/{cohort}/adnimerged_longitudinal.csv")
    b = long[long["slot"] == "t0"]
    assert b["ID_PT"].is_unique, f"{cohort}: mais de um t0 por paciente"
    return b


def ids_a() -> Path:
    b = baselines("48m_6m")
    b = b[b["GROUP"].isin(["CN", "AD"])]
    out = b.groupby(["GROUP", "SEX"], group_keys=False).sample(20, random_state=SEED)
    assert len(out) == 80
    PILOT.mkdir(parents=True, exist_ok=True)
    p = PILOT / "oasis_gateA_ids.csv"
    out[["ID_IMG", "ID_PT", "GROUP", "SEX", "AGE"]].to_csv(p, index=False)
    print(pd.crosstab(out["GROUP"], out["SEX"]), f"\n→ {p}")
    return p


def ids_b() -> Path:
    b = pd.concat([baselines(c) for c in COHORTS]).drop_duplicates("ID_IMG")
    PILOT.mkdir(parents=True, exist_ok=True)
    p = PILOT / "oasis_gateB_ids.csv"
    b[["ID_IMG", "ID_PT", "GROUP", "SEX", "AGE"]].to_csv(p, index=False)
    print(f"{len(b)} baselines → {p}")
    return p


def hippo_vol_icv() -> pd.DataFrame:
    v = pd.read_csv(FEAT / "features_volumetric.csv", usecols=["ID_IMG", "roi", "side", "mask_mm3"])
    icv = v[v["roi"] == "__global__"].drop_duplicates("ID_IMG", keep="last").set_index("ID_IMG")["mask_mm3"]
    h = v[v["roi"] == "hippocampus"].drop_duplicates(["ID_IMG", "side"], keep="last").copy()
    h["vol_icv"] = h["mask_mm3"] / h["ID_IMG"].map(icv)
    return h[["ID_IMG", "side", "vol_icv"]]


def univariate(feat: pd.DataFrame, ids: pd.DataFrame, vol: pd.DataFrame, roi: str) -> dict:
    f = feat[(feat["roi"] == roi) & feat["ID_IMG"].isin(ids["ID_IMG"])]
    f = f[["ID_IMG", "side", "jac_det_mean"]].merge(vol, on=["ID_IMG", "side"], validate="one_to_one")
    rho = float(np.mean([abs(spearmanr(g["jac_det_mean"], g["vol_icv"])[0]) for _, g in f.groupby("side")]))
    per_img = f.groupby("ID_IMG")["jac_det_mean"].mean().rename("jac").reset_index().merge(ids, on="ID_IMG")
    auc = roc_auc_score(per_img["GROUP"] == "AD", per_img["jac"])
    return {"roi": roi, "n_img": per_img["ID_IMG"].nunique(), "abs_rho_vol": rho, "auc_cn_ad": max(auc, 1 - auc)}


def gate_a() -> bool:
    ids = pd.read_csv(PILOT / "oasis_gateA_ids.csv")
    vol = hippo_vol_icv()
    rows = []
    for anchor in ("cn", "ad"):
        times = pd.concat([pd.read_csv(p) for p in glob.glob(f"images/displacement_field_oasis_{anchor}/reg_times*.csv")])
        times = times[times["ID_IMG"].isin(ids["ID_IMG"])]
        new = pd.read_csv(FEAT / NEW_FEAT[anchor])
        old = pd.read_csv(FEAT / OLD_FEAT[anchor], usecols=["ID_IMG", "roi", "side", "jac_det_mean"])
        o = univariate(old, ids, vol, "hippocampus")
        for roi in ROIS:
            r = univariate(new, ids, vol, roi)
            rows.append({"anchor": anchor, **r, "old_abs_rho_vol": o["abs_rho_vol"], "old_auc_cn_ad": o["auc_cn_ad"],
                         "old_n_img": o["n_img"], "reg_min_mean": times["seconds"].mean() / 60, "n_reg": len(times)})
    t = pd.DataFrame(rows)
    t.to_csv(PILOT / "gateA_summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    prim = t[t["roi"] == PRIMARY_ROI]
    assert (prim["n_img"] == len(ids)).all(), "Gate A incompleto: nem todas as 80 imagens têm atributos"
    ok = bool(((prim["abs_rho_vol"] > prim["old_abs_rho_vol"]) & (prim["auc_cn_ad"] >= 0.75)).any())
    print(f"GATE A ({PRIMARY_ROI}): {'PASSA' if ok else 'FALHA'}")
    return ok


def folds(df: pd.DataFrame) -> dict:
    return {(int(r.repeat_id), int(r.fold)): frozenset(json.loads(r.test_id_pts)) for r in df.itertuples()}


def load_results(path: Path, task: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    m = (d["task"] == task) & (d["model_key"] == "svm") & (d["selection_mode"] == "l1_stable")
    m &= d["with_combat"].astype(str).str.lower().isin(["false", "0"])
    d = d[m]
    assert len(d) == 50, f"{path} {task}: esperado 50 folds externos, achou {len(d)}"
    return d


def patient_scores(d: pd.DataFrame) -> pd.DataFrame:
    return explode_patient_predictions(d).groupby("ID_PT", as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def new_results_path(cohort: str, roi: str, mod: str) -> Path:
    return Path(f"csvs/cohorts/{cohort}/ablation_results_oasis/{roi}/t1_only/{mod}/ablation_results_all.csv")


def gate_b(n_boot: int = 5000) -> str:
    rows = []
    for cohort in COHORTS:
        for roi in ROIS:
            for new_mod, old_mod in PAIRS:
                p_new = new_results_path(cohort, roi, new_mod)
                p_old = Path(f"csvs/cohorts/{cohort}/ablation_results_t1_only/{old_mod}/ablation_results_all.csv")
                if not p_new.is_file() or not p_old.is_file():
                    print(f"[skip] {cohort} {roi} {new_mod}: falta {p_new if not p_new.is_file() else p_old}")
                    continue
                a, b = load_results(p_new, "smci_pmci"), load_results(p_old, "smci_pmci")
                assert folds(a) == folds(b), f"folds diferentes: {p_new} × {p_old}"
                sa, sb = patient_scores(a), patient_scores(b)
                paired = sa.merge(sb, on=["ID_PT", "y"], suffixes=("_new", "_old"), validate="one_to_one")
                assert len(paired) == len(sa) == len(sb), (len(paired), len(sa), len(sb))
                d, lo, hi, p1, _ = bootstrap_auc_diff_test(
                    paired["y"], paired["score_new"], paired["score_old"], n_boot=n_boot, seed=SEED)
                cn_ad = np.nan
                try:
                    cn_ad = roc_auc_score(*patient_scores(load_results(p_new, "cn_ad"))[["y", "score"]].T.values)
                except AssertionError:
                    pass
                rows.append({"cohort": cohort, "roi": roi, "new": new_mod, "old": old_mod, "n_pts": len(paired),
                             "auc_new": roc_auc_score(paired["y"], paired["score_new"]),
                             "auc_old": roc_auc_score(paired["y"], paired["score_old"]),
                             "delta": d, "ci95_lo": lo, "ci95_hi": hi, "p_one": p1, "auc_cn_ad_new": cn_ad})
    t = pd.DataFrame(rows)
    t.to_csv(PILOT / "gateB_summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    prim = t[t["roi"] == PRIMARY_ROI]
    cn_ad_ok = bool((prim.loc[prim["new"] == "disp_oasis", "auc_cn_ad_new"] >= 0.75).any())
    if cn_ad_ok and (prim["ci95_lo"] > 0).any():
        verdict = "PASSA"
    elif (prim["delta"] >= 0.03).any():
        verdict = "INCONCLUSIVO (Δ ≥ 0.03 mas IC cruza 0): decisão do usuário sobre F7"
    else:
        verdict = "FALHA (reportar como resultado negativo)"
    print(f"GATE B ({PRIMARY_ROI}, cn_ad disp_oasis ok={cn_ad_ok}): {verdict}")
    return verdict


def self_check() -> None:
    r = pd.DataFrame({"repeat_id": [0, 0], "fold": [1, 2], "test_id_pts": ['["a","b"]', '["c"]']})
    r2 = r.copy(); r2["test_id_pts"] = ['["b","a"]', '["c"]']
    assert folds(r) == folds(r2)
    r2.loc[1, "test_id_pts"] = '["d"]'
    assert folds(r) != folds(r2)
    print("ok: pilot_oasis_gate")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", choices=("ids-a", "ids-b", "gate-a", "gate-b"))
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check or a.cmd is None:
        self_check()
    else:
        {"ids-a": ids_a, "ids-b": ids_b, "gate-a": gate_a, "gate-b": gate_b}[a.cmd]()

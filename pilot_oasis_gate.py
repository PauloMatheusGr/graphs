#!/usr/bin/env python3
"""Piloto DVF OASIS: listas de imagens e critérios de continuidade (gates).

  python pilot_oasis_gate.py ids-a    # 40 CN + 40 AD baseline 48m_6m (20 F + 20 M cada)
  python pilot_oasis_gate.py ids-b    # baselines de 48m_6m ∪ 48m_6m_soft_False
  python pilot_oasis_gate.py gate-a   # tempo, |rho| jac_det × volume/ICV, AUC univariada CN×AD
  python pilot_oasis_gate.py gate-b   # sMCI×pMCI pareado: × disp ADNI, × vol, d2 × núcleo
  python pilot_oasis_gate.py ids-full                  # todas as visitas das coortes
  python pilot_oasis_gate.py compare --rep t1_r10      # mesmo pareamento do gate-b; t1_r10 | t1_ols
  python pilot_oasis_gate.py --self-check

Pré-especificado: ROI principal = hippocampus_d2 (esfera de 2 voxels, literatura); núcleo
OASIS = controle (separa efeito do template do efeito da dilatação).
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
from ablation_representation import RESULTS_ROOT_BY_PROTOCOL  # noqa: E402
from stats_compare import bootstrap_auc_diff_test  # noqa: E402

SEED = 42
PILOT = Path("csvs/pilot")
FEAT = Path("csvs/cohorts/all_population")
COHORTS = ("48m_6m", "48m_6m_soft_False")
PRIMARY_ROI = "hippocampus_d2"
CORE_ROI = "hippocampus"
ROIS = (PRIMARY_ROI, CORE_ROI)
LONG_REPS = ("t1_r10", "t1_ols")  # S0,R10 (2 visitas) e D (OLS, 3 visitas)
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


def ids_full() -> Path:
    """Todas as visitas (t0/t1/t2) das coortes: base para R10 (2 visitas) e OLS (3 visitas)."""
    long = pd.concat([pd.read_csv(f"csvs/cohorts/{c}/adnimerged_longitudinal.csv") for c in COHORTS])
    b = long.drop_duplicates("ID_IMG").sort_values(["slot", "ID_PT"])  # t1 antes de t2: R10 fica pronto antes
    PILOT.mkdir(parents=True, exist_ok=True)
    p = PILOT / "oasis_full_ids.csv"
    b[["ID_IMG", "ID_PT", "GROUP", "SEX", "AGE", "slot"]].to_csv(p, index=False)
    print(b["slot"].value_counts().sort_index().to_string(), f"\n{len(b)} imagens → {p}")
    return p


def hippo_vol_icv() -> pd.DataFrame:
    v = pd.read_csv(FEAT / "features_volumetric.csv", usecols=["ID_IMG", "roi", "side", "mask_mm3"])
    icv = v[v["roi"] == "__global__"].drop_duplicates("ID_IMG", keep="last").set_index("ID_IMG")["mask_mm3"]
    h = v[v["roi"] == "hippocampus"].drop_duplicates(["ID_IMG", "side"], keep="last").copy()
    h["vol_icv"] = h["mask_mm3"] / h["ID_IMG"].map(icv)
    return h[["ID_IMG", "side", "vol_icv"]]


def univariate(feat: pd.DataFrame, ids: pd.DataFrame, vol: pd.DataFrame, roi: str, col: str) -> dict:
    f = feat[(feat["roi"] == roi) & feat["ID_IMG"].isin(ids["ID_IMG"])]
    f = f[["ID_IMG", "side", col]].merge(vol, on=["ID_IMG", "side"], validate="one_to_one")
    rho = float(np.mean([abs(spearmanr(g[col], g["vol_icv"])[0]) for _, g in f.groupby("side")]))
    per_img = f.groupby("ID_IMG")[col].mean().rename("x").reset_index().merge(ids, on="ID_IMG")
    auc = roc_auc_score(per_img["GROUP"] == "AD", per_img["x"])
    return {"roi": roi, "feat": col, "n_img": per_img["ID_IMG"].nunique(), "abs_rho_vol": rho,
            "auc_cn_ad": max(auc, 1 - auc)}


def gate_a() -> bool:
    ids = pd.read_csv(PILOT / "oasis_gateA_ids.csv")
    vol = hippo_vol_icv()
    rows = []
    for anchor in ("cn", "ad"):
        times = pd.concat([pd.read_csv(p) for p in glob.glob(f"images/displacement_field_oasis_{anchor}/reg_times*.csv")])
        times = times[times["ID_IMG"].isin(ids["ID_IMG"])]
        new = pd.read_csv(FEAT / NEW_FEAT[anchor])
        old = pd.read_csv(FEAT / OLD_FEAT[anchor], usecols=["ID_IMG", "roi", "side", "jac_det_mean"])
        o = univariate(old, ids, vol, CORE_ROI, "jac_det_mean")
        for roi in ROIS:
            r = univariate(new, ids, vol, roi, "jac_det_mean")
            rows.append({"anchor": anchor, **r, "old_abs_rho_vol": o["abs_rho_vol"], "old_auc_cn_ad": o["auc_cn_ad"],
                         "old_n_img": o["n_img"], "reg_min_mean": times["seconds"].mean() / 60, "n_reg": len(times)})
    t = pd.DataFrame(rows)
    t.to_csv(PILOT / "gateA_summary.csv", index=False)
    print(t.round(3).to_string(index=False))
    assert (t["n_img"] == len(ids)).all(), "Gate A incompleto: nem todas as 80 imagens têm atributos"
    prim = t[t["roi"] == PRIMARY_ROI]
    ok = bool(((prim["abs_rho_vol"] > prim["old_abs_rho_vol"]) & (prim["auc_cn_ad"] >= 0.75)).any())
    print(f"GATE A ({PRIMARY_ROI} jac_det_mean): {'PASSA' if ok else 'FALHA'}")
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


def new_results_path(cohort: str, roi: str, mod: str, rep: str = "t1_only") -> Path:
    return Path(f"csvs/cohorts/{cohort}/ablation_results_oasis/{roi}/{rep}/{mod}/ablation_results_all.csv")


def ref_results_path(cohort: str, mod: str, rep: str = "t1_only") -> Path:
    return Path(f"csvs/cohorts/{cohort}/{RESULTS_ROOT_BY_PROTOCOL['abs'][rep]}/{mod}/ablation_results_all.csv")


def cn_ad_auc(path: Path) -> float:
    if not path.is_file():
        return np.nan
    try:
        s = patient_scores(load_results(path, "cn_ad"))
    except AssertionError:
        return np.nan
    return float(roc_auc_score(s["y"], s["score"]))


def paired_auc(p_new: Path, p_ref: Path, n_boot: int) -> dict:
    a, b = load_results(p_new, "smci_pmci"), load_results(p_ref, "smci_pmci")
    assert folds(a) == folds(b), f"folds diferentes: {p_new} × {p_ref}"
    sa, sb = patient_scores(a), patient_scores(b)
    paired = sa.merge(sb, on=["ID_PT", "y"], suffixes=("_new", "_ref"), validate="one_to_one")
    assert len(paired) == len(sa) == len(sb), (len(paired), len(sa), len(sb))
    d, lo, hi, p1, _ = bootstrap_auc_diff_test(
        paired["y"], paired["score_new"], paired["score_ref"], n_boot=n_boot, seed=SEED)
    return {"n_pts": len(paired), "auc_new": roc_auc_score(paired["y"], paired["score_new"]),
            "auc_ref": roc_auc_score(paired["y"], paired["score_ref"]),
            "delta": d, "ci95_lo": lo, "ci95_hi": hi, "p_one": p1}


def compare(rep: str, out: Path, n_boot: int = 5000) -> pd.DataFrame:
    """sMCI×pMCI pareado na representação rep: OASIS × disp ADNI, × vol, d2 × núcleo OASIS."""
    rows = []
    for cohort in COHORTS:
        for roi in ROIS:
            for new_mod, old_mod in PAIRS:
                p_new = new_results_path(cohort, roi, new_mod, rep)
                refs = [(old_mod, ref_results_path(cohort, old_mod, rep)), ("vol", ref_results_path(cohort, "vol", rep))]
                if roi == PRIMARY_ROI:
                    refs.append((f"{new_mod}@{CORE_ROI}", new_results_path(cohort, CORE_ROI, new_mod, rep)))
                for ref, p_ref in refs:
                    if not p_new.is_file() or not p_ref.is_file():
                        print(f"[skip] {cohort} {roi} {new_mod} × {ref}: falta {p_new if not p_new.is_file() else p_ref}")
                        continue
                    rows.append({"rep": rep, "cohort": cohort, "roi": roi, "new": new_mod, "ref": ref,
                                 **paired_auc(p_new, p_ref, n_boot), "auc_cn_ad_new": cn_ad_auc(p_new)})
    t = pd.DataFrame(rows)
    assert not t.empty, f"nenhum par com resultados para {rep}"
    t.to_csv(out, index=False)
    print(t.round(3).to_string(index=False), f"\n→ {out}")
    for r in t.itertuples():
        side = "> ref" if r.ci95_lo > 0 else "< ref" if r.ci95_hi < 0 else "≈ ref (IC cruza 0)"
        print(f"  {r.cohort} {r.roi} {r.new} × {r.ref}: Δ={r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] → {side}")
    return t


def gate_b(n_boot: int = 5000) -> str:
    t = compare("t1_only", PILOT / "gateB_summary.csv", n_boot)
    prim = t[t["roi"] == PRIMARY_ROI]
    vs_disp = prim[prim["ref"].isin([o for _, o in PAIRS])]
    cn_ad = prim.loc[(prim["new"] == "disp_oasis") & (prim["cohort"] == COHORTS[0]), "auc_cn_ad_new"].max()
    if cn_ad >= 0.75 and (vs_disp["ci95_lo"] > 0).any():
        verdict = "PASSA"
    elif (vs_disp["delta"] >= 0.03).any():
        verdict = "INCONCLUSIVO (Δ ≥ 0.03 mas IC cruza 0): decisão do usuário sobre a rodada completa"
    else:
        verdict = "FALHA (reportar como resultado negativo)"
    print(f"GATE B ({PRIMARY_ROI}, CN×AD disp_oasis={cn_ad:.3f}) × disp ADNI: {verdict}")
    return verdict


def self_check() -> None:
    r = pd.DataFrame({"repeat_id": [0, 0], "fold": [1, 2], "test_id_pts": ['["a","b"]', '["c"]']})
    r2 = r.copy(); r2["test_id_pts"] = ['["b","a"]', '["c"]']
    assert folds(r) == folds(r2)
    r2.loc[1, "test_id_pts"] = '["d"]'
    assert folds(r) != folds(r2)
    assert "/ablation_results_ols/disp_ad/" in str(ref_results_path("c", "disp_ad", "t1_ols"))
    assert "/ablation_results_r10/vol/" in str(ref_results_path("c", "vol", "t1_r10"))
    assert "/hippocampus_d2/t1_r10/disp_oasis/" in str(new_results_path("c", PRIMARY_ROI, "disp_oasis", "t1_r10"))
    print("ok: pilot_oasis_gate")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", choices=("ids-a", "ids-b", "ids-full", "gate-a", "gate-b", "compare"))
    p.add_argument("--rep", default="t1_only", choices=("t1_only", *LONG_REPS), help="compare: representação")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check or a.cmd is None:
        self_check()
    elif a.cmd == "compare":
        compare(a.rep, PILOT / f"compare_{a.rep}.csv")
    else:
        {"ids-a": ids_a, "ids-b": ids_b, "ids-full": ids_full, "gate-a": gate_a, "gate-b": gate_b}[a.cmd]()

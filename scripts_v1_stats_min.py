#!/usr/bin/env python3
"""Stats mínimas para artigo v1 — números só de disco + bootstrap pareado."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "modules")
from ablation_analysis import prepare_ablation_df, explode_patient_predictions
from stats_compare import apply_bh_fdr, paired_comparison_row, image_ablation_path, clinical_results_path, fusion_results_path

TAB = Path("Artigo 1 pgirardi/tables")
TAB.mkdir(parents=True, exist_ok=True)
CLAIM = "48m_6m"
BASE = Path(f"csvs/cohorts/{CLAIM}")
N_BOOT = 5000
N_PERM = 5000
SEED = 42
ALPHA = 0.05
MODS = ("vol", "shape", "texture", "disp", "firstorder")


def patient_scores(path: Path, *, task="smci_pmci", model="svm", combat=False, selection="l1_stable") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = prepare_ablation_df(df)
    # filter
    m = (df["task"] == task) & (df["model_key"] == model)
    if "with_combat" in df.columns:
        m &= df["with_combat"].astype(bool) == combat
    if "selection_mode" in df.columns:
        m &= df["selection_mode"] == selection
    df = df.loc[m]
    if df.empty:
        raise RuntimeError(f"vazio: {path}")
    # explode patient preds
    pat = explode_patient_predictions(df)
    # expect columns ID_PT, y, score
    cols = {c.lower(): c for c in pat.columns}
    # normalize
    if "score" not in pat.columns:
        for cand in ("y_score", "proba", "y_pred_proba"):
            if cand in pat.columns:
                pat = pat.rename(columns={cand: "score"})
                break
    keep = [c for c in ("ID_PT", "y", "score") if c in pat.columns]
    out = pat[keep].groupby("ID_PT", as_index=False).agg(y=("y", "first"), score=("score", "mean"))
    return out


def boot_delta(y, a, b, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y, int); a=np.asarray(a,float); b=np.asarray(b,float)
    rng = np.random.default_rng(seed)
    deltas=[]
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, ab, bb = y[idx], a[idx], b[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, ab) - roc_auc_score(yb, bb))
    d = np.asarray(deltas)
    auc_a = roc_auc_score(y, a); auc_b = roc_auc_score(y, b)
    delta = auc_a - auc_b
    lo, hi = np.quantile(d, [0.025, 0.975])
    # one-sided H1: a > b
    p1 = (np.sum(d <= 0) + 1) / (len(d) + 1)
    p2 = 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))
    return dict(auc_a=auc_a, auc_b=auc_b, delta_auc=delta, ci95_lo=lo, ci95_hi=hi,
                p_bootstrap_one_sided=p1, p_bootstrap_two_sided=p2)


def uni_path(base: Path, protocol: str, mod: str) -> Path:
    folder = {
        "t1_only": "ablation_results_t1_only",
        "t1_d21": "ablation_results_d21",
        "t1_d21_d32": "ablation_results_d21d32",
        "t1_d21_d32_longcombat": "ablation_results_d21d32_longcombat",
        "t1_d21_d32_leaky": "ablation_results_leaky_d21d32",
    }[protocol]
    return base / folder / mod / "ablation_results_all.csv"


def late_path(base: Path, proto: str, combat=False) -> Path:
    root = "ablation_results_late_fusion_longcombat" if combat else "ablation_results_late_fusion"
    # protocol name without late__
    name = proto.replace("late__", "")
    return base / root / name / "ablation_results_all.csv"


print("=== DEMO ===", flush=True)
long = pd.read_csv(BASE / "adnimerged_longitudinal.csv")
# one row per patient at baseline-ish
pt = long.drop_duplicates("ID_PT").copy()
pt = pt[pt["GROUP"].isin(["sMCI", "pMCI"])].copy()
age_s = pt.loc[pt.GROUP=="sMCI", "AGE"].dropna()
age_p = pt.loc[pt.GROUP=="pMCI", "AGE"].dropna()
u = stats.mannwhitneyu(age_s, age_p, alternative="two-sided")
# sex
ct = pd.crosstab(pt["GROUP"], pt["SEX"]) if "SEX" in pt.columns else None
if ct is None and "PTGENDER" in pt.columns:
    ct = pd.crosstab(pt["GROUP"], pt["PTGENDER"])
chi2, p_sex, _, _ = stats.chi2_contingency(ct)
demo = pd.DataFrame([
    {"test":"age_mannwhitney", "n_smci":len(age_s), "n_pmci":len(age_p),
     "mean_smci":float(age_s.mean()), "mean_pmci":float(age_p.mean()), "p":float(u.pvalue)},
    {"test":"sex_chi2", "n_smci":int((pt.GROUP=="sMCI").sum()), "n_pmci":int((pt.GROUP=="pMCI").sum()),
     "stat":float(chi2), "p":float(p_sex)},
])
demo.to_csv(TAB/"stats_demo_48m6m.csv", index=False)
print(demo.to_string(index=False), flush=True)

print("=== Q4 vs T1 claim ===", flush=True)
rows=[]
for i, mod in enumerate(MODS):
    a = patient_scores(uni_path(BASE, "t1_d21_d32", mod))
    b = patient_scores(uni_path(BASE, "t1_only", mod))
    paired = a.merge(b, on=["ID_PT","y"], suffixes=("_q4","_t1"))
    r = boot_delta(paired.y, paired.score_q4, paired.score_t1, seed=SEED+i)
    r.update({"modality":mod, "comparison":"q4_vs_t1", "cohort":CLAIM})
    rows.append(r)
    print(mod, f"Δ={r['delta_auc']:+.4f} p1={r['p_bootstrap_one_sided']:.4f}", flush=True)
cmp = pd.DataFrame(rows)
cmp["p_fdr_bh"] = apply_bh_fdr(cmp["p_bootstrap_one_sided"].to_numpy())
cmp["significant_fdr"] = (cmp["p_fdr_bh"] < ALPHA) & (cmp["ci95_lo"] > 0)
cmp.to_csv(TAB/"stats_q4_vs_t1_48m6m.csv", index=False)

print("=== D21 vs T1 claim ===", flush=True)
rows=[]
for i, mod in enumerate(MODS):
    a = patient_scores(uni_path(BASE, "t1_d21", mod))
    b = patient_scores(uni_path(BASE, "t1_only", mod))
    paired = a.merge(b, on=["ID_PT","y"], suffixes=("_d21","_t1"))
    r = boot_delta(paired.y, paired.score_d21, paired.score_t1, seed=SEED+100+i)
    r.update({"modality":mod, "comparison":"d21_vs_t1", "cohort":CLAIM})
    rows.append(r)
    print(mod, f"Δ={r['delta_auc']:+.4f} p1={r['p_bootstrap_one_sided']:.4f}", flush=True)
cmp_d21 = pd.DataFrame(rows)
cmp_d21["p_fdr_bh"] = apply_bh_fdr(cmp_d21["p_bootstrap_one_sided"].to_numpy())
cmp_d21["significant_fdr"] = (cmp_d21["p_fdr_bh"] < ALPHA) & (cmp_d21["ci95_lo"] > 0)
cmp_d21.to_csv(TAB/"stats_d21_vs_t1_48m6m.csv", index=False)

print("=== gradient Q4 vs T1 ===", flush=True)
grow=[]
for cname in ("36m_6m","36m_12m","48m_6m","48m_12m"):
    base = Path(f"csvs/cohorts/{cname}")
    for i, mod in enumerate(MODS):
        try:
            a = patient_scores(uni_path(base, "t1_d21_d32", mod))
            b = patient_scores(uni_path(base, "t1_only", mod))
        except Exception as e:
            print("skip", cname, mod, e); continue
        paired = a.merge(b, on=["ID_PT","y"], suffixes=("_q4","_t1"))
        r = boot_delta(paired.y, paired.score_q4, paired.score_t1, seed=SEED+200+hash(cname+mod)%1000)
        r.update({"modality":mod, "cohort":cname})
        grow.append(r)
        print(cname, mod, f"Δ={r['delta_auc']:+.4f}", flush=True)
gdf = pd.DataFrame(grow)
# FDR within cohort
parts=[]
for cname, g in gdf.groupby("cohort"):
    g = g.copy()
    g["p_fdr_bh"] = apply_bh_fdr(g["p_bootstrap_one_sided"].to_numpy())
    g["significant_fdr"] = (g["p_fdr_bh"] < ALPHA) & (g["ci95_lo"] > 0)
    parts.append(g)
gdf = pd.concat(parts, ignore_index=True)
gdf.to_csv(TAB/"stats_q4_vs_t1_gradient.csv", index=False)

print("=== late vs vol T1 ===", flush=True)
vol = patient_scores(uni_path(BASE, "t1_only", "vol"))
specs = [
    ("late T1", "late__t1_vol__t1_shape__t1_texture__t1_disp__t1_firstorder"),
    ("late Q4", "late__t1_d21d32_vol__t1_d21d32_shape__t1_d21d32_texture__t1_d21d32_disp__t1_d21d32_firstorder"),
    ("late ancora", "late__t1_vol__t1_d21d32_shape"),
    ("late grelha max", "late__t1_shape__t1_d21d32_vol__t1_d21d32_texture__t1_d21d32_firstorder"),
]
lrows=[]
for i,(lab,proto) in enumerate(specs):
    path = late_path(BASE, proto)
    if not path.exists():
        print("MISSING", path); continue
    # late files may not have modality filter same way
    df = pd.read_csv(path)
    df = prepare_ablation_df(df)
    m = (df["task"]=="smci_pmci") & (df["model_key"]=="svm")
    df = df.loc[m]
    pat = explode_patient_predictions(df)
    late = pat.groupby("ID_PT", as_index=False).agg(y=("y","first"), score=("score","mean"))
    paired = late.merge(vol, on=["ID_PT","y"], suffixes=("_late","_vol"))
    r = boot_delta(paired.y, paired.score_late, paired.score_vol, seed=SEED+400+i)
    r.update({"spec":lab, "protocol":proto})
    lrows.append(r)
    print(lab, f"AUC_late={r['auc_a']:.4f} Δ={r['delta_auc']:+.4f} p1={r['p_bootstrap_one_sided']:.4f}", flush=True)
ldf = pd.DataFrame(lrows)
if len(ldf):
    ldf["p_fdr_bh"] = apply_bh_fdr(ldf["p_bootstrap_one_sided"].to_numpy())
    ldf.to_csv(TAB/"stats_late_vs_vol_48m6m.csv", index=False)

print("=== clinic ===", flush=True)
# clinical results
clin_path = BASE/"ablation_results_clinic/clinical_results_all.csv"
fus_path = BASE/"ablation_results_clinic_img_t1_only/fusion_vol_l1_stable_nocombat_t1_only_results_all.csv"
clin = patient_scores(clin_path, selection="none") if "selection_mode" in pd.read_csv(clin_path, nrows=1).columns else None
# robust loaders
def load_pat(path, selection=None):
    df = pd.read_csv(path)
    df = prepare_ablation_df(df)
    m = (df["task"]=="smci_pmci") & (df["model_key"]=="svm")
    if selection is not None and "selection_mode" in df.columns:
        m &= df["selection_mode"]==selection
    df = df.loc[m]
    pat = explode_patient_predictions(df)
    return pat.groupby("ID_PT", as_index=False).agg(y=("y","first"), score=("score","mean"))

clin = load_pat(clin_path)
fus = load_pat(fus_path)
vol = patient_scores(uni_path(BASE, "t1_only", "vol"))
crows=[]
for lab, a, b, sa, sb in [
    ("clinic_vs_vol", clin, vol, "_clinic", "_vol"),
    ("fusion_vs_clinic", fus, clin, "_fus", "_clinic"),
    ("fusion_vs_vol", fus, vol, "_fus", "_vol"),
]:
    paired = a.merge(b, on=["ID_PT","y"], suffixes=(sa, sb))
    # columns score_x score_y after merge - rename carefully
    scols = [c for c in paired.columns if c.startswith("score")]
    r = boot_delta(paired.y, paired[scols[0]], paired[scols[1]], seed=SEED+500+len(crows))
    r.update({"comparison":lab})
    crows.append(r)
    print(lab, f"Δ={r['delta_auc']:+.4f} [{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]", flush=True)
pd.DataFrame(crows).to_csv(TAB/"stats_clinic_48m6m.csv", index=False)

print("=== leaky ===", flush=True)
a = patient_scores(uni_path(BASE, "t1_d21_d32", "vol"))
b = patient_scores(uni_path(BASE, "t1_d21_d32_leaky", "vol"))
paired = a.merge(b, on=["ID_PT","y"], suffixes=("_q4","_leaky"))
r = boot_delta(paired.y, paired.score_q4, paired.score_leaky, seed=SEED+600)
# note: delta = strict - leaky; negative means leaky higher
pd.DataFrame([{**r, "comparison":"vol_q4_vs_leaky"}]).to_csv(TAB/"stats_leaky_48m6m.csv", index=False)
print("leaky", r, flush=True)

print("=== combat descriptive ===", flush=True)
brows=[]
for mod in MODS:
    a = patient_scores(uni_path(BASE, "t1_d21_d32", mod))  # nocombat
    bpath = uni_path(BASE, "t1_d21_d32_longcombat", mod)
    b = patient_scores(bpath, combat=True)
    paired = a.merge(b, on=["ID_PT","y"], suffixes=("_no","_lc"))
    r = boot_delta(paired.y, paired.score_lc, paired.score_no, seed=SEED+700)
    r.update({"modality":mod, "comparison":"longcombat_minus_nocombat"})
    brows.append(r)
    print(mod, f"Δ(lc-no)={r['delta_auc']:+.4f}", flush=True)
# late
a = load_pat(late_path(BASE, "late__t1_d21d32_vol__t1_d21d32_shape__t1_d21d32_texture__t1_d21d32_disp__t1_d21d32_firstorder"))
b = load_pat(late_path(BASE, "late__t1_d21d32_vol__t1_d21d32_shape__t1_d21d32_texture__t1_d21d32_disp__t1_d21d32_firstorder", combat=True))
# combat late may have with_combat True
paired = a.merge(b, on=["ID_PT","y"], suffixes=("_no","_lc"))
r = boot_delta(paired.y, paired.score_lc, paired.score_no, seed=SEED+800)
r.update({"modality":"late_all_q4", "comparison":"longcombat_minus_nocombat"})
brows.append(r)
pd.DataFrame(brows).to_csv(TAB/"stats_combat_vs_nocombat.csv", index=False)

print("=== confound leve idade/sexo vs vol T1 ===", flush=True)
# univariate age/sex CV
pt = long.drop_duplicates("ID_PT")
pt = pt[pt.GROUP.isin(["sMCI","pMCI"])].copy()
pt["y"] = (pt["GROUP"]=="pMCI").astype(int)
sex_col = "SEX" if "SEX" in pt.columns else "PTGENDER"
pt["SEX_bin"] = (pt[sex_col].astype(str).str.lower().isin(["male","m","1","1.0"])).astype(float)
# map F/M
if pt["SEX_bin"].nunique()<2:
    pt["SEX_bin"] = pt[sex_col].astype("category").cat.codes.astype(float)

def uni_auc(x, y, seed=0):
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = cross_val_predict(pipe, x.reshape(-1,1), y, cv=cv, method="predict_proba")[:,1]
    return float(roc_auc_score(y, scores)), scores

pt2 = pt.dropna(subset=["AGE","SEX_bin"]).copy()
auc_age, s_age = uni_auc(pt2["AGE"].to_numpy(float), pt2["y"].to_numpy(), 0)
auc_sex, s_sex = uni_auc(pt2["SEX_bin"].to_numpy(float), pt2["y"].to_numpy(), 1)
vol = patient_scores(uni_path(BASE, "t1_only", "vol"))
img = vol.merge(pt2[["ID_PT","y"]], on=["ID_PT","y"])
# align age/sex scores
tmp = pt2[["ID_PT","y"]].copy(); tmp["score_age"]=s_age; tmp["score_sex"]=s_sex
img = img.merge(tmp, on=["ID_PT","y"])
d_age = boot_delta(img.y, img.score, img.score_age, seed=SEED+900)
d_sex = boot_delta(img.y, img.score, img.score_sex, seed=SEED+901)
conf = pd.DataFrame([
    {"modelo":"idade", "auc":auc_age},
    {"modelo":"sexo", "auc":auc_sex},
    {"modelo":"vol T1", "auc":float(roc_auc_score(img.y, img.score))},
    {"modelo":"vol T1 − idade", "delta_auc":d_age["delta_auc"], "ci95_lo":d_age["ci95_lo"], "ci95_hi":d_age["ci95_hi"]},
    {"modelo":"vol T1 − sexo", "delta_auc":d_sex["delta_auc"], "ci95_lo":d_sex["ci95_lo"], "ci95_hi":d_sex["ci95_hi"]},
])
conf.to_csv(TAB/"stats_confound_48m6m.csv", index=False)
print(conf.to_string(index=False), flush=True)
print("DONE", flush=True)

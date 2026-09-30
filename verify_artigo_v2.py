"""Confere os números do artigo_v2.tex contra os experimentos.

Três camadas:
  1. tabelas do .tex == CSVs de estatística (Artigo 1 pgirardi/tables), no arredondamento exibido;
  2. AUCs desses CSVs == AUCs recomputadas dos resultados brutos (ablation_results_all.csv);
  3. q-valores BH, contagens do protocolo em dois níveis e demografia recomputados.
Por fim, lista números do texto corrido sem correspondência em nenhuma fonte (revisão manual).

Uso: .venv/bin/python verify_artigo_v2.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules"))
from ablation_analysis import explode_patient_predictions, prepare_ablation_df  # noqa: E402
from ablation_deltas import ols_slope_three, visit_times_months  # noqa: E402
from cohort_compare import _patient_auc_boot, _stable_seed  # noqa: E402
from stats_compare import apply_bh_fdr  # noqa: E402

ART = ROOT / "Artigo 1 pgirardi"
TEX = ART / "artigo_v2.tex"
T = ART / "tables"
C = ROOT / "csvs" / "cohorts"
COH = ("36m_6m", "36m_12m", "48m_6m", "48m_12m")
MODS = ("vol", "shape", "texture", "firstorder", "disp")
DIRS = {
    "t1_only": "ablation_results_t1_only",
    "r10": "ablation_results_r10",
    "r10r21": "ablation_results_r10r21",
    "ols": "ablation_results_ols",
}
FAILS: list[str] = []
N_OK = 0


# ---------------------------------------------------------------- tex parsing
BODY = re.sub(r"(?<!\\)%.*", "", TEX.read_text(encoding="utf-8"))
NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d+")


def nums(s: str) -> list[tuple[float, int]]:
    return [(float(m), len(m.split(".")[1])) for m in NUM.findall(s.replace("−", "-"))]


def table_chunk(label: str) -> str:
    i = BODY.index(f"\\label{{{label}}}")
    j = BODY.index("\\end{tabular}", i)
    seg = BODY[i:j]
    k = seg.find("\\toprule")
    return seg[k:] if k >= 0 else seg


def table_rows(label: str) -> list[str]:
    return [r for r in table_chunk(label).split("\\\\") if NUM.search(r)]


def close(tex: float, ndec: int, exp: float) -> bool:
    return abs(tex - exp) <= 0.5 * 10 ** -ndec + 1e-9


def fail(msg: str) -> None:
    FAILS.append(msg)


def ok() -> None:
    global N_OK
    N_OK += 1


def check_rows(label: str, expected: list[list[float]]) -> None:
    rows = table_rows(label)
    if len(rows) != len(expected):
        fail(f"{label}: {len(rows)} linhas numéricas no tex, esperado {len(expected)}")
    for i, (row, exp) in enumerate(zip(rows, expected)):
        got = nums(row)
        if len(got) != len(exp):
            fail(f"{label} linha {i + 1}: {len(got)} números no tex, esperado {len(exp)} | {row.strip()[:90]}")
            continue
        for k, ((t, d), e) in enumerate(zip(got, exp)):
            if e is None:
                continue
            if close(t, d, e):
                ok()
            else:
                fail(f"{label} linha {i + 1} pos {k + 1}: tex {t:.{d}f} ≠ esperado {e:.{d}f} | {row.strip()[:70]}")


def check_val(name: str, got: float, exp: float, tol: float = 1e-6) -> None:
    if np.isfinite(got) and np.isfinite(exp) and abs(got - exp) <= tol:
        ok()
    else:
        fail(f"{name}: {got} ≠ {exp}")


def check_eq(name: str, got, exp) -> None:
    (ok if got == exp else lambda: fail(f"{name}: {got} ≠ {exp}"))()


# ---------------------------------------------------------------- raw loaders
_RAW: dict[str, pd.DataFrame] = {}


def raw(path: Path) -> pd.DataFrame:
    k = str(path)
    if k not in _RAW:
        _RAW[k] = pd.read_csv(path)
    return _RAW[k]


_PAT: dict[tuple, pd.DataFrame] = {}


def pat(path: Path, *, task="smci_pmci", model="svm", combat=False, sel="l1_stable", mod=None) -> pd.DataFrame:
    key = (str(path), task, model, combat, sel, mod)
    if key not in _PAT:
        _PAT[key] = _pat(path, task=task, model=model, combat=combat, sel=sel, mod=mod)
    return _PAT[key]


def _pat(path: Path, *, task, model, combat, sel, mod) -> pd.DataFrame:
    df = prepare_ablation_df(raw(path))
    m = (df["task"] == task) & (df["model_key"] == model) & (df["with_combat"] == combat)
    if sel is not None and "selection_mode" in df:
        m &= df["selection_mode"] == sel
    if mod is not None:
        m &= df["modality"] == mod
    sub = df.loc[m]
    assert not sub.empty, f"sem linhas: {path} {task} {model} {combat} {mod}"
    assert sub["modality"].nunique() == 1, f"modalidade ambígua em {path}"
    p = explode_patient_predictions(sub)
    return p.groupby("ID_PT", as_index=False).agg(y=("y", "first"), score=("score", "mean"))


def auc(p: pd.DataFrame) -> float:
    return float(roc_auc_score(p["y"], p["score"]))


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = stats.rankdata(s)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def auc_sd(path: Path, mod: str, *, task="smci_pmci", model="svm", combat=False) -> tuple[float, float]:
    # mesma sequência aleatória de cohort_compare.bootstrap_auc_std (SD das tabelas), AUC por postos
    p = pat(path, task=task, model=model, combat=combat, mod=mod)
    y, s = p["y"].to_numpy(int), p["score"].to_numpy(float)
    rng = np.random.default_rng(_stable_seed(task, mod, model, combat))
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(fast_auc(y[idx], s[idx]))
    return fast_auc(y, s), float(np.std(aucs))


def late(parts: list[pd.DataFrame]) -> pd.DataFrame:
    out = parts[0].rename(columns={"score": "s0"})
    for i, p in enumerate(parts[1:], 1):
        out = out.merge(p.rename(columns={"score": f"s{i}"}), on=["ID_PT", "y"])
    assert len(out) == len(parts[0]), "fusão perdeu pacientes"
    out["score"] = out[[f"s{i}" for i in range(len(parts))]].mean(axis=1)
    return out[["ID_PT", "y", "score"]]


def summ(path: Path, *, task="smci_pmci", model="svm") -> pd.Series:
    s = pd.read_csv(path)
    s = s[(s["task"] == task) & (s["model_key"] == model) & (~s["with_combat"].astype(bool))]
    if "selection_mode" in s:
        s = s[s["selection_mode"] == "l1_stable"]
    assert len(s) == 1, f"summary ambíguo {path}"
    return s.iloc[0]


def img_path(cohort: str, proto: str, mod: str) -> Path:
    return C / cohort / DIRS[proto] / mod / "ablation_results_all.csv"


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


# ================================================================ 1. AUCs brutas
print("recomputando AUCs dos resultados brutos…")
_ref = _patient_auc_boot(raw(img_path("48m_6m", "t1_only", "vol")),
                         pd.Series({"task": "smci_pmci", "modality": "vol", "model_key": "svm", "with_combat": False}), n_boot=2000)
_new = auc_sd(img_path("48m_6m", "t1_only", "vol"), "vol")
assert abs(_ref[0] - _new[0]) < 1e-12 and abs(_ref[1] - _new[1]) < 1e-12, (_ref, _new)
AUC: dict[tuple, float] = {}
SD: dict[tuple, float] = {}
PAT: dict[tuple, pd.DataFrame] = {}
for c in COH + ("48m_6m_soft_False",):
    for pr in DIRS:
        for m in MODS:
            path = img_path(c, pr, m)
            a, sd = auc_sd(path, m)
            p = pat(path, mod=m)
            check_val(f"AUC paciente {c}/{pr}/{m} (postos vs sklearn)", auc(p), a, 1e-12)
            AUC[c, pr, m], SD[c, pr, m], PAT[c, pr, m] = a, sd, p

# ---- tab:uni_4 (AUC ± SD bootstrap, 4 coortes × baseline/2v/D)
exp = []
for m in MODS:
    exp.append([AUC[c, pr, m] for c in COH for pr in ("t1_only", "r10", "ols")])
    exp.append([SD[c, pr, m] for c in COH for pr in ("t1_only", "r10", "ols")])
check_rows("tab:uni_4", exp)

# ================================================================ 2. contrastes uniclasse
GRAD = {"r10": "stats_r10_vs_t1_gradient", "r10r21": "stats_r10r21_vs_t1_gradient", "ols": "stats_ols_vs_t1_gradient"}
CNAME = {"r10": "t1_r10_vs_t1_only", "r10r21": "t1_r10_r21_vs_t1_only", "ols": "t1_ols_vs_t1_only"}
sens = tab("stats_sensitivity_alpha01")
uni = sens[sens["scope"] == "uni_5fam_x_4coortes"]
U: dict[tuple, pd.Series] = {}
for pr, f in GRAD.items():
    g = tab(f)
    for _, r in g.iterrows():
        c, m = r["cohort"], r["modality"]
        check_val(f"{f} auc_long {c}/{m}", r["auc_long"], AUC[c, pr, m])
        check_val(f"{f} auc_t1 {c}/{m}", r["auc_t1_only"], AUC[c, "t1_only", m])
        check_val(f"{f} delta {c}/{m}", r["delta_auc"], AUC[c, pr, m] - AUC[c, "t1_only", m])
        s = uni[(uni["contrast"] == CNAME[pr]) & (uni["cohort"] == c) & (uni["modality"] == m)]
        assert len(s) == 1
        s = s.iloc[0]
        for col in ("delta_auc", "ci95_lo", "ci95_hi", "p_bootstrap_one_sided", "p_fdr_bh"):
            check_val(f"sensitivity vs {f} {col} {c}/{m}", s[col], r[col], 1e-9)
        U[pr, c, m] = s
    # BH recomputado
    sub = uni[uni["contrast"] == CNAME[pr]]
    for c in COH:
        cc = sub[sub["cohort"] == c]
        for q, qq in zip(apply_bh_fdr(cc["p_bootstrap_one_sided"].to_numpy()), cc["p_fdr_bh"]):
            check_val(f"BH coorte {pr}/{c}", q, qq, 1e-9)
    for q, qq in zip(apply_bh_fdr(sub["p_bootstrap_one_sided"].to_numpy()), sub["p_fdr_global"]):
        check_val(f"BH global {pr}", q, qq, 1e-9)

# ---- tab:uni_all
exp = []
for c in COH:
    for m in MODS:
        row = []
        for pr in ("r10", "r10r21", "ols"):
            s = U[pr, c, m]
            row += [s["delta_auc"], s["ci95_lo"], s["ci95_hi"], s["p_bootstrap_one_sided"], s["p_fdr_bh"], s["p_fdr_global"]]
        exp.append(row)
check_rows("tab:uni_all", exp)


# ---- tab:gradient
def grow(pr, c, m):
    s = U[pr, c, m]
    return [s["delta_auc"], s["ci95_lo"], s["ci95_hi"], s["ci99_lo"], s["ci99_hi"],
            s["p_bootstrap_one_sided"], s["p_fdr_bh"], s["p_fdr_global"]]


check_rows("tab:gradient", [
    grow("ols", "48m_12m", "vol"), grow("ols", "36m_6m", "texture"), grow("ols", "48m_6m", "disp"),
    grow("r10", "36m_6m", "texture"), grow("r10r21", "48m_12m", "vol"), grow("r10r21", "36m_6m", "texture"),
])
# todo contraste com p<0.05 deve estar em tab:gradient (6 esperados)
check_eq("n contrastes uniclasse com p<0.05", int((uni["p_bootstrap_one_sided"] < 0.05).sum()), 6)

# ================================================================ 3. união multiclasse
un = tab("stats_union_4cohorts")
for c in COH:
    allT1 = late([PAT[c, "t1_only", m] for m in MODS])
    allD = late([PAT[c, "ols", m] for m in MODS])
    uT1 = max(MODS, key=lambda m: AUC[c, "t1_only", m])
    uD = max(MODS, key=lambda m: AUC[c, "ols", m])
    arms = {"all_T1": auc(allT1), "all_D": auc(allD), "uni_T1": AUC[c, "t1_only", uT1], "uni_D": AUC[c, "ols", uD]}
    for _, r in un[un["cohort"] == c].iterrows():
        check_val(f"união {c} {r['contrast']} auc_a", r["auc_a"], arms[r["arm_a"]])
        check_val(f"união {c} {r['contrast']} auc_b", r["auc_b"], arms[r["arm_b"]])
        check_eq(f"união {c} melhor uni T1", r["best_uni_T1"], uT1)
        check_eq(f"união {c} melhor uni D", r["best_uni_D"], uD)
for ct in un["contrast"].unique():
    cc = un[un["contrast"] == ct]
    for q, qq in zip(apply_bh_fdr(cc["p_bootstrap_one_sided"].to_numpy()), cc["p_fdr_bh"]):
        check_val(f"BH 4 coortes {ct}", q, qq, 1e-9)
for q, qq in zip(apply_bh_fdr(un["p_bootstrap_one_sided"].to_numpy()), un["p_fdr_global"]):
    check_val("BH 12 união", q, qq, 1e-9)
exp = []
for c in COH:
    for ct in ("all_D_minus_all_T1", "all_T1_minus_uni_T1", "all_D_minus_uni_D"):
        r = un[(un["cohort"] == c) & (un["contrast"] == ct)].iloc[0]
        row = [r["auc_a"], r["auc_b"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"]]
        if np.isfinite(r["ci99_lo"]):
            row += [r["ci99_lo"], r["ci99_hi"]]
        exp.append(row + [r["p_bootstrap_one_sided"], r["p_fdr_bh"], r["p_fdr_global"]])
check_rows("tab:union4", exp)

# ================================================================ 4. tetos (48m_6m)
LF = C / "48m_6m" / "ablation_results_late_fusion"
fc = tab("stats_four_ceilings_48m6m")
b = "48m_6m"
bestT1 = fc["proto_best_late_T1"].iloc[0].removeprefix("late__")
bestD = fc["proto_best_late_D"].iloc[0].removeprefix("late__")


def late_from_name(fp: str) -> pd.DataFrame:
    parts = []
    for tok in fp.split("__"):
        pr, m = ("ols", tok[len("t1_ols_"):]) if tok.startswith("t1_ols_") else ("t1_only", tok[len("t1_"):])
        parts.append(PAT[b, pr, m])
    return late(parts)


arms = {
    "uni_T1": AUC[b, "t1_only", "vol"], "uni_D": AUC[b, "ols", "vol"],
    "all_T1": auc(late([PAT[b, "t1_only", m] for m in MODS])),
    "all_D": auc(late([PAT[b, "ols", m] for m in MODS])),
    "best_late_T1": auc(late_from_name(bestT1)), "best_late_D": auc(late_from_name(bestD)),
    "volT1_shapeD": auc(late([PAT[b, "t1_only", "vol"], PAT[b, "ols", "shape"]])),
}
for _, r in fc.iterrows():
    check_val(f"tetos {r['contrast']} auc_a", r["auc_a"], arms[r["arm_a"]])
    check_val(f"tetos {r['contrast']} auc_b", r["auc_b"], arms[r["arm_b"]])
for q, qq in zip(apply_bh_fdr(fc["p_bootstrap_one_sided"].to_numpy()), fc["p_fdr_bh"]):
    check_val("BH 7 tetos", q, qq, 1e-9)
# grade de fusão tardia: tamanho e argmax, e AUC da grade == média dos escores uniclasse atuais
grid = {}
for d in LF.iterdir():
    toks = d.name.split("__")
    if all(re.fullmatch(r"t1_(ols_)?(vol|shape|texture|firstorder|disp)", t) for t in toks):
        grid[d.name] = summ(d / "ablation_summary.csv")["auc_patient_mean"]
gT1 = {k: v for k, v in grid.items() if "ols" not in k}
gD = {k: v for k, v in grid.items() if "ols" in k}
check_eq("grade só-baseline (n uniões)", len(gT1), 26)
check_eq("grade com D (n uniões)", len(gD), 206)
check_eq("argmax grade baseline", max(gT1, key=gT1.get), bestT1)
check_eq("argmax grade D", max(gD, key=gD.get), bestD)
for fp in (bestT1, bestD, "t1_vol__t1_ols_shape", "__".join(f"t1_{m}" for m in ("vol", "shape", "texture", "disp", "firstorder")),
           "__".join(f"t1_ols_{m}" for m in ("vol", "shape", "texture", "disp", "firstorder"))):
    check_val(f"grade {fp}: AUC salva vs média dos escores atuais", grid[fp], auc(late_from_name(fp)), 1e-9)
order = ["uni_D_minus_uni_T1", "all_D_minus_all_T1", "best_late_D_minus_best_late_T1", "all_T1_minus_uni_T1",
         "all_D_minus_uni_D", "best_late_D_minus_uni_D", "volT1_shapeD_minus_uni_T1"]
check_rows("tab:late_claim", [
    [r["auc_a"], r["auc_b"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"], r["p_bootstrap_one_sided"], r["p_fdr_bh"]]
    for r in (fc.set_index("contrast").loc[o] for o in order)
])

# ================================================================ 5. demografia e clínico
lg = pd.read_csv(C / b / "adnimerged_longitudinal.csv")
base = lg[lg["slot"] == "t0"].sort_values(["ID_PT", "ID_IMG"]).groupby("ID_PT").first()
base = base[base["GROUP"].isin(["sMCI", "pMCI"])]
ag = [base.loc[base["GROUP"] == g, "AGE"] for g in ("sMCI", "pMCI")]
p_age = stats.mannwhitneyu(*ag, alternative="two-sided").pvalue
sexF = base["SEX"].eq("F").astype(int)
p_sex = stats.chi2_contingency(pd.crosstab(base["GROUP"], sexF))[1]
dm = tab("stats_demo_48m6m").set_index("variavel")
check_val("demo p idade", dm.loc["idade", "p"], p_age, 1e-9)
check_val("demo p sexo", dm.loc["sexo", "p"], p_sex, 1e-9)
CL = C / b / "ablation_results_clinic"
cf = tab("stats_confound_48m6m").set_index("demo")
DEMO_FILE = {"idade": "clinical_age", "sexo": "clinical_sex", "idade+sexo": "clinical_sex_age"}
for k, f in DEMO_FILE.items():
    check_val(f"AUC modelo {k}", cf.loc[k, "auc_demo"], auc(pat(CL / f"{f}_results_all.csv", sel="none")))
    check_val(f"AUC vol T1 em confound {k}", cf.loc[k, "auc_vol_t1"], AUC[b, "t1_only", "vol"])
a01 = tab("stats_complement_alpha01").set_index("analise")
rows = [[dm.loc["idade", "media_sMCI"], dm.loc["idade", "media_pMCI"], p_age],
        [100 * dm.loc["sexo", "prop_F_sMCI"], 100 * dm.loc["sexo", "prop_F_pMCI"], p_sex],
        [AUC[b, "t1_only", "vol"]]]
for k in ("idade", "sexo", "idade+sexo"):
    r, q = cf.loc[k], a01.loc[f"vol T1 − {k}"]
    rows.append([r["auc_demo"], r["p_perm_demo"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"], q["ci99_lo"], q["ci99_hi"]])
check_rows("tab:demo", rows)

cl = tab("stats_clinic_48m6m").set_index("comparison")
ex = tab("stats_complement_extra_48m6m")
FU = C / b / "ablation_results_clinic_img_t1_only"
a_clin = auc(pat(CL / "clinical_results_all.csv", sel="none"))
a_fus = auc(pat(FU / "fusion_vol_l1_stable_nocombat_t1_only_results_all.csv"))
a_fdemo = auc(pat(FU / "fusion_vol_l1_stable_nocombat_t1_only_sex_age_results_all.csv"))
check_val("AUC clínico", cl.loc["vol_t1_vs_clinical", "auc_clin"], a_clin)
check_val("AUC clínico ∪ vol", cl.loc["fusion_vs_vol_t1", "auc_fusion"], a_fus)
AR = {"vol T1": AUC[b, "t1_only", "vol"], "vol ∪ idade+sexo": a_fdemo, "vol ∪ clínico": a_fus,
      "idade+sexo": cf.loc["idade+sexo", "auc_demo"],
      "disp CN": AUC[b, "t1_only", "disp"],
      "disp AD": auc(pat(C / b / DIRS["t1_only"] / "disp_ad" / "ablation_results_all.csv")),
      "disp CN+AD": auc(pat(C / b / DIRS["t1_only"] / "disp_cnad" / "ablation_results_all.csv"))}
for _, r in ex.iterrows():
    check_val(f"extra {r['arm_a']} auc", r["auc_a"], AR[r["arm_a"]])
    check_val(f"extra {r['arm_b']} auc", r["auc_b"], AR[r["arm_b"]])
for fam in ex["familia"].unique():
    cc = ex[ex["familia"] == fam]
    for q, qq in zip(apply_bh_fdr(cc["p_bootstrap_one_sided"].to_numpy()), cc["p_fdr_bh"]):
        check_val(f"BH extra {fam}", q, qq, 1e-9)
check_val("BH bloco clínico", apply_bh_fdr(cl["p_bootstrap_one_sided"].to_numpy())[2], cl["p_fdr_bh"].iloc[2], 1e-9)
e = ex.set_index(["arm_a", "arm_b"])
e1, e2, e3 = e.loc[("vol ∪ idade+sexo", "vol T1")], e.loc[("vol ∪ idade+sexo", "idade+sexo")], e.loc[("vol ∪ clínico", "vol ∪ idade+sexo")]
v, fv, fc_ = cl.loc["vol_t1_vs_clinical"], cl.loc["fusion_vs_vol_t1"], cl.loc["fusion_vs_clinical"]
fq = a01.loc["(clínico + vol T1) − vol T1"]
check_rows("tab:clinic", [
    [cf.loc["idade+sexo", "auc_demo"], AUC[b, "t1_only", "vol"], a_fdemo, a_clin, a_fus],
    [e1["delta_auc"], e1["ci95_lo"], e1["ci95_hi"], e1["p_bootstrap_one_sided"], e1["p_fdr_bh"]],
    [e2["delta_auc"], e2["ci95_lo"], e2["ci95_hi"], e2["ci99_lo"], e2["ci99_hi"], e2["p_bootstrap_one_sided"], e2["p_fdr_bh"]],
    [e3["delta_auc"], e3["ci95_lo"], e3["ci95_hi"], e3["ci99_lo"], e3["ci99_hi"], e3["p_bootstrap_one_sided"], e3["p_fdr_bh"]],
    [-v["delta_auc"], -v["ci95_hi"], -v["ci95_lo"], v["p_bootstrap_two_sided"]],
    [fc_["delta_auc"], fc_["ci95_lo"], fc_["ci95_hi"], fc_["p_bootstrap_one_sided"], fc_["p_fdr_bh"]],
    [fv["delta_auc"], fv["ci95_lo"], fv["ci95_hi"], fq["ci99_lo"], fq["ci99_hi"], fv["p_bootstrap_one_sided"], fv["p_fdr_bh"]],
])

# ================================================================ 6. elegibilidade restrita
S = "48m_6m_soft_False"
SOFT = {pr: tab(f"stats_{pr}_vs_t1_soft_False").set_index("modality") for pr in ("r10", "r10r21", "ols")}
for pr, d in SOFT.items():
    for m in MODS:
        check_val(f"restrita {pr} auc_long {m}", d.loc[m, f"auc_{pr}"], AUC[S, pr, m])
        check_val(f"restrita {pr} auc_t1 {m}", d.loc[m, "auc_t1_only"], AUC[S, "t1_only", m])
    for q, qq in zip(apply_bh_fdr(d["p_bootstrap_one_sided"].to_numpy()), d["p_fdr_bh"]):
        check_val(f"BH restrita {pr}", q, qq, 1e-9)
check_rows("tab:soft", [
    [AUC[S, pr, m] for pr in ("t1_only", "r10", "r10r21", "ols")]
    + sum(([SOFT[pr].loc[m, "delta_auc"], SOFT[pr].loc[m, "ci95_lo"], SOFT[pr].loc[m, "ci95_hi"]] for pr in ("r10", "r10r21", "ols")), [])
    for m in MODS
])
st = tab("stats_soft_true_vs_false")
for _, r in st.iterrows():
    pr = {"t1_only": "t1_only", "t1_ols": "ols"}[r["protocol"]]
    check_val(f"ampliada−restrita {pr}/{r['modality']}", r["delta_true_minus_false"], AUC[b, pr, r["modality"]] - AUC[S, pr, r["modality"]])

# ================================================================ 7. ComBat
cb = tab("stats_combat_vs_nocombat")
A_CB = {}
for m in MODS:
    A_CB["t1_neurocombat", m] = auc(pat(C / b / "ablation_results_combat_t1_only" / m / "ablation_results_all.csv", combat=True))
    A_CB["ols_longcombat", m] = auc(pat(C / b / "ablation_results_ols_longcombat" / m / "ablation_results_all.csv", combat=True))
    A_CB["t1", m], A_CB["ols", m] = AUC[b, "t1_only", m], AUC[b, "ols", m]
for _, r in cb.iterrows():
    arm_a, arm_b = [x.strip() for x in r["protocol"].split("−")]
    check_val(f"ComBat {r['protocol']} {r['modality']} auc_a", r["auc_a"], A_CB[arm_a, r["modality"]])
    check_val(f"ComBat {r['protocol']} {r['modality']} auc_b", r["auc_b"], A_CB[arm_b, r["modality"]])
for pro in cb["protocol"].unique():
    cc = cb[cb["protocol"] == pro]
    for q, qq in zip(apply_bh_fdr(cc["p_bootstrap_one_sided"].to_numpy()), cc["p_fdr_bh"]):
        check_val(f"BH ComBat {pro}", q, qq, 1e-9)
cbi = cb.set_index(["protocol", "modality"])
rows = []
for m in ("vol", "shape", "texture", "disp", "firstorder"):
    row = []
    for pro in ("t1_neurocombat − t1", "ols_longcombat − ols", "ols_longcombat − t1_neurocombat"):
        r = cbi.loc[(pro, m)]
        row += [r["delta_auc"], r["ci95_lo"], r["ci95_hi"]]
        if pro == "ols_longcombat − ols" and m == "texture":
            row.append(r["p_fdr_bh"])
    rows.append(row)
check_rows("tab:combat", rows)

# ================================================================ 8. métricas secundárias
rows = []
for m in MODS:
    for pr in ("t1_only", "r10", "ols"):
        s = summ(C / b / DIRS[pr] / m / "ablation_summary.csv")
        rows.append([AUC[b, pr, m], s["auc_mean"], s["auc_std"], s["auc_pr_mean"], s["auc_pr_std"], s["bal_acc_mean"],
                     s["bal_acc_std"], s["sens_pos_mean"], s["sens_pos_std"], s["spec_neg_mean"], s["spec_neg_std"],
                     s["mcc_mean"], s["mcc_std"], s["n_features_mean"]])
for fp in ("t1_vol__t1_shape__t1_texture__t1_disp__t1_firstorder",
           "t1_ols_vol__t1_ols_shape__t1_ols_texture__t1_ols_disp__t1_ols_firstorder",
           "t1_vol__t1_ols_shape", bestT1, bestD):
    s = summ(LF / fp / "ablation_summary.csv")
    rows.append([s["auc_patient_mean"], s["auc_mean"], s["auc_std"], s["auc_pr_mean"], s["auc_pr_std"], s["bal_acc_mean"],
                 s["bal_acc_std"], s["sens_pos_mean"], s["sens_pos_std"], s["spec_neg_mean"], s["spec_neg_std"],
                 s["mcc_mean"], s["mcc_std"], s["n_features_mean"]])
check_rows("tab:secondary", rows)

# ================================================================ 9. quatro algoritmos e CN vs AD
MODELS = ("svm", "elasticnet", "rf", "xgb")
MOD4: dict[tuple, tuple[float, float]] = {}
for task in ("smci_pmci", "cn_ad"):
    for pr in ("t1_only", "ols"):
        for m in MODS:
            if task == "cn_ad" and pr == "ols" and m != "vol":
                continue
            for mk in MODELS:
                MOD4[task, pr, m, mk] = auc_sd(img_path(b, pr, m), m, task=task, model=mk)
                s = summ(C / b / DIRS[pr] / m / "ablation_summary.csv", task=task, model=mk)
                check_val(f"summary auc_patient_mean {task}/{pr}/{m}/{mk}", s["auc_patient_mean"], MOD4[task, pr, m, mk][0])
rows = [[MOD4["cn_ad", "t1_only", m, mk][0] for mk in MODELS] for m in MODS]
rows.append([MOD4["cn_ad", "ols", "vol", mk][0] for mk in MODELS])
check_rows("tab:cnad", rows)
check_val("SD bootstrap CN-AD vol D SVM (legenda 0.024)", round(MOD4["cn_ad", "ols", "vol", "svm"][1], 3), 0.024, 1e-9)

# ================================================================ 10. tab:sig5
g = {pr: tab(f).set_index(["cohort", "modality"]) for pr, f in GRAD.items()}


def gsig(pr, c, m, two=False):
    r, s = g[pr].loc[(c, m)], U[pr, c, m]
    base_ = [r["auc_long"], r["auc_t1_only"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"]]
    return base_ + ([s["p_bootstrap_two_sided"]] if two else [r["p_bootstrap_one_sided"], r["p_fdr_bh"]])


def srow(pr, m, two=False):
    r = SOFT[pr].loc[m]
    base_ = [r[f"auc_{pr}"], r["auc_t1_only"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"]]
    return base_ + ([r["p_bootstrap_two_sided"]] if two else [r["p_bootstrap_one_sided"], r["p_fdr_bh"]])


def urow(c):
    r = un[(un["cohort"] == c) & (un["contrast"] == "all_D_minus_all_T1")].iloc[0]
    return [r["auc_a"], r["auc_b"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"], r["p_bootstrap_one_sided"], r["p_fdr_bh"]]


def crow(m, two=False):
    r = cbi.loc[("ols_longcombat − ols", m)]
    base_ = [r["auc_a"], r["auc_b"], r["delta_auc"], r["ci95_lo"], r["ci95_hi"]]
    return base_ + ([r["p_bootstrap_two_sided"]] if two else [r["p_bootstrap_one_sided"], r["p_fdr_bh"]])


check_rows("tab:sig5", [
    gsig("ols", "48m_12m", "vol"), gsig("r10", "36m_6m", "texture"), srow("r10r21", "disp"),
    gsig("ols", "36m_6m", "texture"), gsig("r10r21", "48m_12m", "vol"),
    urow("36m_6m"), urow("48m_12m"),
    gsig("r10", "36m_6m", "vol", True), gsig("r10", "36m_12m", "disp", True), srow("r10", "firstorder", True),
    crow("texture"), crow("vol", True), crow("disp", True),
])
# completude: toda inferioridade (IC95 < 0) e superioridade (IC95 > 0, p<.05) de imagem deve estar em tab:sig5
n_sup = sum(int(((d["p_bootstrap_one_sided"] < .05) & (d["ci95_lo"] > 0)).sum()) for d in list(g.values()) + list(SOFT.values()))
n_inf = sum(int((d["ci95_hi"] < 0).sum()) for d in list(g.values()) + list(SOFT.values()))
check_eq("tab:sig5 superioridades uniclasse (5 esperadas)", n_sup, 5)
check_eq("tab:sig5 inferioridades uniclasse (3 esperadas)", n_inf, 3)
cbo = cb[cb["protocol"] != "ols_longcombat − ols"]
check_eq("ComBat: nenhum IC excluindo zero fora de long D−D", int(((cbo["ci95_lo"] > 0) | (cbo["ci95_hi"] < 0)).sum()), 0)

# ================================================================ 11. protocolo em dois níveis
def lvl(d, p="p_bootstrap_one_sided", q="p_fdr_bh", qg=None, lo99="ci99_lo"):
    raw5 = (d[p] < .05) & (d["ci95_lo"] > 0)
    fdr5 = (d[q] < .05) & (d["ci95_lo"] > 0)
    glo5 = ((d[qg] < .05) & (d["ci95_lo"] > 0)).sum() if qg else None
    l99 = d[lo99] if lo99 in d else pd.Series(np.nan, index=d.index)
    raw1 = (d[p] < .01) & (l99 > 0)
    fdr1 = (d[q] < .01) & (l99 > 0)
    return [len(d), int(raw5.sum()), int(fdr5.sum())] + ([int(glo5)] if qg else []) + [int(raw1.sum()), int(fdr1.sum())]


tetos = sens[sens["scope"] == "late_fusion_tetos_48m6m"]
cfx = cf.assign(ci99_lo=[a01.loc[f"vol T1 − {k}", "ci99_lo"] for k in cf.index])
cf_q = apply_bh_fdr(cfx["p_bootstrap_one_sided"].to_numpy())
cfx = cfx.assign(p_fdr_bh=cf_q)
clx = cl.copy()
clx["ci99_lo"] = np.nan
clx.loc["fusion_vs_vol_t1", "ci99_lo"] = fq["ci99_lo"]
cbx = cb.copy()
cbx["ci99_lo"] = np.nan
cbx.loc[(cbx["protocol"] == "ols_longcombat − ols") & (cbx["modality"] == "texture"), "ci99_lo"] = a01.loc["D longComBat − D (textura)", "ci99_lo"]
sfx = pd.concat([SOFT[pr] for pr in SOFT])
sfx["ci99_lo"] = np.nan
sfx.loc[(sfx["comparison"].str.contains("r10_r21|r10r21")) & (sfx.index == "disp"), "ci99_lo"] = a01.loc["soft False: B − T1 (disp)", "ci99_lo"]
two = [
    lvl(uni[uni["contrast"] == CNAME["r10"]], qg="p_fdr_global"),
    lvl(uni[uni["contrast"] == CNAME["r10r21"]], qg="p_fdr_global"),
    lvl(uni[uni["contrast"] == CNAME["ols"]], qg="p_fdr_global"),
    lvl(un, qg="p_fdr_global"),
    lvl(tetos),
    lvl(cfx), lvl(clx), lvl(ex[ex["familia"] == "fusao_demo"]), lvl(cbx), lvl(sfx),
    lvl(ex[ex["familia"] == "disp_template"]),
]
tex_two = []
for r in table_chunk("tab:two_level").split("\\midrule", 2)[2].split("\\\\"):
    parts = r.split("&")
    if len(parts) > 3:
        tex_two.append([int(x) for x in re.findall(r"\d+", "&".join(parts[2:]))])
check_eq("tab:two_level (contagens)", tex_two, two)
check_eq("tab:two_level total testes principal", sum(r[0] for r in two[:5]), 79)

# ================================================================ 12. trajetórias
long_ = pd.read_csv(C / b / "ablation" / "hippocampus" / "vol_long.csv")
long_ = long_[long_["GROUP"].isin(["sMCI", "pMCI"])]
F = "original_shape_MeshVolume"
bil = long_.groupby(["ID_PT", "ID_IMG", "GROUP", "slot", "MRI_DATE"], as_index=False)[F].sum(min_count=2)
tm = visit_times_months(bil)
wd = bil.pivot(index="ID_PT", columns="slot", values=F)[["t0", "t1", "t2"]] * 1e3
pt = wd.join(tm, rsuffix="_m").join(bil.groupby("ID_PT")["GROUP"].first()).dropna()
pt.columns = ["x0", "x1", "x2", "t0", "t1", "t2", "GROUP"]
pt["slope"] = [12 * ols_slope_three(r.t0, r.t1, r.t2, r.x0, r.x1, r.x2) for r in pt.itertuples()]
SLOPE = {gname: pt.loc[pt["GROUP"] == gname, "slope"] for gname in ("sMCI", "pMCI")}
P_MW = stats.mannwhitneyu(SLOPE["sMCI"], SLOPE["pMCI"]).pvalue
check_eq("trajetórias n sMCI/pMCI", (len(SLOPE["sMCI"]), len(SLOPE["pMCI"])), (73, 120))

# ================================================================ 13. coortes, amostra ADNI, splits
FIELDS = ("AGE", "MMSE_SCORE", "ADAS_SCORE", "CDR_SB", "FAQ_SCORE")
rows, ints_exp = [], []
for c in COH:
    d = pd.read_csv(C / c / "adnimerged_longitudinal.csv")
    for gname in ("sMCI", "pMCI"):
        for sl in ("t0", "t1", "t2"):
            s = d[(d["GROUP"] == gname) & (d["slot"] == sl)].drop_duplicates("ID_PT")
            rows.append(sum(([s[f].mean(), s[f].std()] for f in FIELDS), []))
            ints_exp.append((len(s), int((s["SEX"] == "M").sum()), int((s["SEX"] == "F").sum())))
check_rows("tab:cohorts", rows)
ints_tex = [tuple(int(x) for x in m) for m in re.findall(r"&\s*(\d+)\s*&\s*(\d+)/(\d+)\s*&", table_chunk("tab:cohorts"))]
check_eq("tab:cohorts n e sexo M/F", ints_tex, ints_exp)
d48 = pd.read_csv(C / b / "adnimerged_longitudinal.csv")
n_mci_ad = int(d48[(d48["GROUP"] == "pMCI") & (d48["slot"] == "t2") & (d48["DIAG"].astype(str).str.upper().str.contains("AD|DEMENTIA"))]["ID_PT"].nunique())
check_eq("pMCI com AD já em i2 (48m_6m) = 46", n_mci_ad, 46)

allc = pd.concat([pd.read_csv(C / c / "adnimerged_longitudinal.csv") for c in COH]).drop_duplicates("ID_IMG")
MAN = {"GE MEDICAL SYSTEMS": "GE", "SIEMENS": "Siemens", "Philips Medical Systems": "Philips"}
STUDY = {"ADNI~1": "ADNI 1", "ADNI~GO": "ADNI GO", "ADNI~2": "ADNI 2", "Total": None}
for r in table_chunk("tab:adni_study").split("\\\\"):
    parts = [p.strip() for p in r.split("&")]
    key = next((k for k in STUDY if k in parts[0]), None)
    if key is None or len(parts) < 7:
        continue
    s = allc if STUDY[key] is None else allc[allc["STUDY"] == STUDY[key]]
    pts = s.drop_duplicates("ID_PT")
    fs = s["FIELD_STRENGTH"].value_counts()
    exp_ints = [len(s), len(pts), int((pts["SEX"] == "M").sum()), int((pts["SEX"] == "F").sum()),
                int(fs.get(1.5, 0)), int(fs.get(3.0, 0))]
    got_ints = [int(x) for x in re.findall(r"\d+", " ".join([parts[1], parts[2], parts[3], parts[5]]))]
    check_eq(f"tab:adni_study {key} imagens/pacientes/sexo/campo", got_ints, exp_ints)
    got_man = {k: int(v) for k, v in re.findall(r"(\w+) \((\d+)\)", parts[6])}
    exp_man = {MAN[k]: int(v) for k, v in s["MANUFACTURER"].value_counts().items()}
    check_eq(f"tab:adni_study {key} fabricantes", got_man, exp_man)
    ag_ = nums(parts[4])
    check_eq(f"tab:adni_study {key} idade", [close(t, dd, e) for (t, dd), e in zip(ag_, [s["AGE"].mean(), s["AGE"].std()])], [True, True])
nota = BODY[BODY.index("\\label{tab:adni_study}"):]
nota = nota[nota.index("Modelos de equipamento"):nota.index("\\end{flushleft}")].replace("\\_", "_")
for key, study in (("ADNI~1", "ADNI 1"), ("ADNI~GO", "ADNI GO"), ("ADNI~2", "ADNI 2")):
    seg = re.search(re.escape(key) + r":(.*?);\s*(?:ADNI|$)", nota + ";ADNI", re.S).group(1)
    got = {k.strip(): int(v) for k, v in re.findall(r"([\w ]+?) \((\d+)\)", seg)}
    exp_ = {k: int(v) for k, v in allc[allc["STUDY"] == study]["MFG_MODEL"].value_counts().items()}
    check_eq(f"tab:adni_study modelos {key}", got, exp_)

REP = ROOT / "csvs" / "reproducibility" / "splits"
so = pd.read_csv(REP / "outer" / "48m_6m__smci_pmci.csv")
si = pd.read_csv(REP / "inner" / "48m_6m__smci_pmci.csv")


def rng_(s):
    lo, hi = int(s.min()), int(s.max())
    return [lo] if lo == hi else [lo, hi]


def split_row(df, keys, role):
    d = df[df["split"] == role]
    tot = d.groupby(keys).size()
    grp = d.groupby(keys + ["GROUP"]).size().unstack()
    return rng_(tot) + rng_(grp["sMCI"]) + rng_(grp["pMCI"]) + [3 * int(tot.min()), 3 * int(tot.max())]


exp_split = [split_row(so, ["repeat_id", "outer_fold"], "train"), split_row(so, ["repeat_id", "outer_fold"], "test"),
             split_row(si, ["repeat_id", "outer_fold", "inner_fold"], "train"),
             split_row(si, ["repeat_id", "outer_fold", "inner_fold"], "validation")]
got_split = []
for r in table_chunk("tab:split_sizes").split("\\\\"):
    parts = r.split("&")
    if len(parts) == 4 and re.search(r"\d", parts[2]):
        got_split.append([int(x) for x in re.findall(r"\d+", parts[2] + parts[3])])
check_eq("tab:split_sizes", got_split, exp_split)

# ================================================================ 14. números do texto sem fonte
KNOWN: list[float] = []


def add(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    if np.isfinite(f):
        KNOWN.extend([f, -f, 100 * f])


STALE = ("48m12", "q4", "leaky", "main_results", "stability")
for f in T.glob("*.csv"):
    if any(s in f.name for s in STALE):
        continue
    for v in pd.read_csv(f).select_dtypes("number").to_numpy().ravel():
        add(v)
for dct in (AUC, SD):
    for v in dct.values():
        add(v)
for a, s_ in MOD4.values():
    add(a), add(s_)
for (k1, k2), v in A_CB.items():
    add(v)
for v in list(arms.values()) + list(AR.values()) + [a_clin, a_fus, a_fdemo, P_MW] + list(grid.values()):
    add(v)
for gname in SLOPE:
    add(SLOPE[gname].mean())
for c in COH + ("48m_6m_soft_False",):
    for pr in DIRS:
        for m in MODS:
            for col, v in summ(C / c / DIRS[pr] / m / "ablation_summary.csv").items():
                add(v) if isinstance(v, (int, float, np.number)) else None
for fp in grid:
    for col, v in summ(LF / fp / "ablation_summary.csv").items():
        add(v) if isinstance(v, (int, float, np.number)) else None
for c in COH:  # diferenças entre AUCs (texto cita Δ entre representações/coortes)
    for pr in DIRS:
        for m in MODS:
            add(AUC[c, pr, m] - AUC[c, "t1_only", m])
for pr in DIRS:
    for m in MODS:
        add(AUC[b, pr, m] - AUC[S, pr, m])
KNOWN_ARR = np.sort(np.array(KNOWN))


def known(val: float, ndec: int) -> bool:
    tol = 0.5 * 10 ** -ndec + 1e-9
    i = np.searchsorted(KNOWN_ARR, val - tol)
    return i < len(KNOWN_ARR) and KNOWN_ARR[i] <= val + tol


tables_spans = [(m.start(), BODY.index("\\end{table", m.start())) for m in re.finditer(r"\\begin\{(sideways)?table\}", BODY)]
UNKNOWN = []
for m in NUM.finditer(BODY):
    pos = m.start()
    if any(a <= pos <= z for a, z in tables_spans):
        continue
    s = m.group()
    ndec = len(s.split(".")[1])
    if ndec < 2 or known(float(s), ndec):
        continue
    line = BODY.count("\n", 0, pos) + 1
    UNKNOWN.append(f"  linha {line}: {s}  …{BODY[max(0, pos - 50):pos + 20].replace(chr(10), ' ')}…")

# ================================================================ relatório
print(f"\n{N_OK} checagens OK, {len(FAILS)} falhas")
for f_ in FAILS:
    print("  FALHA", f_)
print(f"\n{len(UNKNOWN)} números do texto corrido (≥2 casas) sem correspondência nas fontes — revisar à mão:")
print("\n".join(UNKNOWN))
sys.exit(1 if FAILS else 0)

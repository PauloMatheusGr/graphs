"""Estratos OASIS-3 (DIAG × SEX × década 60-69..80-89) em MNI152.

Fonte única para 2.3/3.1/3.2/pilot: faixa etária, caminho do template e baseline por paciente.
Idade fora de 60-89 usa a década extrema; estrato sem template (N < MIN_N no 2.3) usa a
década disponível mais próxima do mesmo DIAG/SEX. O ref_tag sempre nomeia o template usado.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

OASIS_MNI_DIR = "images/groupwise/references/oasis_mni"
AGE_BINS = ("60-69", "70-79", "80-89")
SLOT_ORDER = {"baseline": 0, "m12": 1, "m24": 2, "t0": 0, "t1": 1, "t2": 2}


def age_bin(age: float) -> str:
    a = min(max(float(age), 60.0), 89.999)
    lo = 60 + 10 * int((a - 60) // 10)
    return f"{lo}-{lo + 9}"


def template_name(diag: str, sex: str, abin: str, n: int) -> str:
    return f"groupwise_SRC-OASIS_DIAG-{diag}_SEX-{sex}_AGE-{abin}_N-{n}_template.nii.gz"


def available(diag: str, sex: str, root: str = OASIS_MNI_DIR) -> dict[str, str]:
    hits = glob.glob(os.path.join(root, f"groupwise_SRC-OASIS_DIAG-{diag}_SEX-{sex}_AGE-*_N-*_template.nii.gz"))
    out = {re.search(r"_AGE-(\d+-\d+)_N-", h).group(1): h for h in hits}
    if len(out) != len(hits):
        raise FileExistsError(f"mais de um template por faixa em {root} para {diag}/{sex}")
    return out


def template_bin(diag: str, sex: str, abin: str, root: str = OASIS_MNI_DIR) -> str:
    av = available(diag, sex, root)
    if not av:
        raise FileNotFoundError(f"nenhum template OASIS {diag}/{sex} em {root}")
    lo = int(abin.split("-")[0])
    return min(av, key=lambda b: (abs(int(b.split("-")[0]) - lo), b))


def template_path(diag: str, sex: str, abin: str, root: str = OASIS_MNI_DIR) -> str:
    return available(diag, sex, root)[template_bin(diag, sex, abin, root)]


def ref_tag(diag: str, sex: str, abin: str, root: str = OASIS_MNI_DIR) -> str:
    return f"OASIS-{diag}_SEX-{sex}_AGE-{template_bin(diag, sex, abin, root)}"


def baseline_by_pt(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """ID_PT → (SEX, década do paciente) da primeira imagem por slot, MRI_DATE, ID_IMG."""
    d = df.copy()
    d["_slot"] = d["slot"].astype(str).str.strip().map(SLOT_ORDER).fillna(99) if "slot" in d else 99
    d["_date"] = pd.to_datetime(d["MRI_DATE"], errors="coerce")
    d = d.sort_values(["ID_PT", "_slot", "_date", "ID_IMG"])
    first = d.groupby("ID_PT", sort=False).head(1)
    return {
        str(r.ID_PT): (str(r.SEX).upper().strip(), age_bin(r.AGE))
        for r in first.itertuples(index=False)
    }


if __name__ == "__main__":
    import tempfile

    assert age_bin(55) == "60-69" and age_bin(69.9) == "60-69" and age_bin(70) == "70-79"
    assert age_bin(85) == "80-89" and age_bin(91) == "80-89" and age_bin(89.99) == "80-89"
    assert {age_bin(a) for a in range(40, 100)} == set(AGE_BINS)
    df = pd.DataFrame({
        "ID_PT": ["p", "p", "q"], "ID_IMG": ["b", "a", "c"], "SEX": ["F", "F", "m"],
        "AGE": [70, 91, 55], "MRI_DATE": ["2010-01-01", "2009-01-01", "2010-01-01"],
        "slot": ["t0", "t1", "t0"],
    })
    assert baseline_by_pt(df) == {"p": ("F", "70-79"), "q": ("M", "60-69")}
    with tempfile.TemporaryDirectory() as root:
        for b, n in (("70-79", 6), ("80-89", 5)):
            open(os.path.join(root, template_name("AD", "M", b, n)), "w").close()
        assert template_bin("AD", "M", "60-69", root) == "70-79"
        assert template_bin("AD", "M", "80-89", root) == "80-89"
        assert ref_tag("AD", "M", "60-69", root) == "OASIS-AD_SEX-M_AGE-70-79"
        assert template_path("AD", "M", "70-79", root).endswith("_N-6_template.nii.gz")
    print("ok: oasis_refs")

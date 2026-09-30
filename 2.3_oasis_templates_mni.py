#!/usr/bin/env python3
"""Templates OASIS-3 em MNI152 com o mesmo protocolo da ADNI.

  preproc/hist_match.py (histogram_match_image2 → MNI) → 2.1 (Rigid → MNI 1 mm) → 2.2 (groupwise SyN)

Estratos DIAG (CN|AD) × SEX × década (60-69, 70-79, 80-89) da planilha oasis3merged.csv, com
idade no exame = AGE (entrada no estudo) + MRI_DATE/365.25. Uma sessão por paciente: a primeira
em 60-89 cuja T1 bruta tem cabeçalho RAS moderno (≤1.25 mm, ≥150 cortes). As sessões legadas
256×256×128 têm orientação errada no cabeçalho e ficam de fora. Entrada = 3-biasfield do
Leandro (extração ANTsXNet + NLM + N4, como a ADNI). N ≤ 20 por estrato (amostra seed 42);
estrato com N < MIN_N não gera template e usa a década vizinha (modules/oasis_refs.py).

  python 2.3_oasis_templates_mni.py select               # csvs/oasis/selected_*.csv
  python 2.3_oasis_templates_mni.py prep [--shard k/n]   # hist-match + rígido → resample_1.0mm_oasis
  python 2.3_oasis_templates_mni.py build DIAG SEX AGE   # groupwise de um estrato (ex.: CN F 70-79)
  python 2.3_oasis_templates_mni.py qc                   # qc.csv / qc.png + geometria = MNI
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import ants
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, "/mnt/study-data/pgirardi/preproc")
import hist_match  # noqa: E402
from oasis_refs import OASIS_MNI_DIR, age_bin, template_name  # noqa: E402

SHEET = "/mnt/study-data/pgirardi/datasets/output/oasis3/oasis3merged.csv"
RAW_DIR = Path("/mnt/databases/mri/oasis/oasis-3")
SRC_DIR = Path("/mnt/databases/mri/oasis/oasis-3/preproc/3-biasfield")
SRC_SUFFIX = "_stripped_nlm_denoised_biascorrected.nii.gz"
MNI = "/mnt/study-data/pgirardi/preproc/atlases/templates/mni152_2009c_template.nii.gz"
SEL_DIR = ROOT / "csvs" / "oasis"
RESAMPLE_DIR = ROOT / "images" / "groupwise" / "resample_1.0mm_oasis"
SUFFIX = "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
OUT_DIR = ROOT / OASIS_MNI_DIR
N_MAX = 20
MIN_N = 5  # ponytail: abaixo disso a média de poucos sujeitos vira anatomia individual; usa a década vizinha
SEED = 42
MIN_NCC_RIGID = 0.15  # ADNI ~0.30; orientação errada dá ~0.09


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modern_src(id_img: str) -> str | None:
    """3-biasfield da primeira run cuja T1 bruta tem cabeçalho RAS moderno; None se só legada."""
    import nibabel as nib

    for src in sorted(SRC_DIR.glob(f"{id_img}_sub-*{SRC_SUFFIX}")):
        raw = RAW_DIR / src.name.replace(SRC_SUFFIX, ".nii.gz")
        if not raw.is_file():
            continue
        h = nib.load(raw)
        if (nib.aff2axcodes(h.affine) == ("R", "A", "S") and max(h.header.get_zooms()[:3]) <= 1.25
                and min(h.shape[:3]) >= 150):
            return str(src)
    return None


def ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def sharpness(img: ants.ANTsImage, mask: ants.ANTsImage) -> float:
    g = ants.iMath(img, "Grad", 1.0).numpy()
    m = mask.numpy() > 0
    return float(g[m].mean() / img.numpy()[m].mean())


def first_modern(g: pd.DataFrame) -> dict | None:
    for r in g.sort_values("MRI_DATE").itertuples(index=False):
        src = modern_src(r.ID_IMG)
        if src:
            return {"ID_IMG": r.ID_IMG, "ID_PT": r.ID_PT, "SEX": r.SEX, "AGE": round(r.age_scan, 2),
                    "DIAG": r.DIAG, "SRC": src}
    return None


def select() -> None:
    from concurrent.futures import ThreadPoolExecutor

    df = pd.read_csv(SHEET)
    df["age_scan"] = df["AGE"] + df["MRI_DATE"] / 365.25
    df = df[df["DIAG"].isin(["CN", "AD"]) & (df["age_scan"] >= 60) & (df["age_scan"] < 90)]
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(first_modern, [g for _, g in df.groupby("ID_PT")]) if r]
    df = pd.DataFrame(rows)
    df["bin"] = df["AGE"].map(age_bin)
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    for old in SEL_DIR.glob("selected_*.csv"):
        old.unlink()
    for (diag, sex, abin), g in df.groupby(["DIAG", "SEX", "bin"]):
        n = min(len(g), N_MAX)
        if n < MIN_N:
            print(f"[skip] {diag} {sex} {abin}: N={n} < {MIN_N} → década vizinha")
            continue
        s = g.sample(n, random_state=SEED).sort_values("ID_PT")
        p = SEL_DIR / f"selected_DIAG-{diag}_SEX-{sex}_AGE-{abin}_N-{n}.csv"
        s[["ID_IMG", "ID_PT", "SEX", "AGE", "DIAG", "SRC"]].to_csv(p, index=False)
        print(f"[ok] {p.name} (disponíveis={len(g)})")


def selected() -> dict[str, str]:
    csvs = sorted(SEL_DIR.glob("selected_*.csv"))
    assert csvs, f"rode 'select' antes: {SEL_DIR}"
    s = pd.concat([pd.read_csv(c) for c in csvs])
    return dict(sorted(zip(s["ID_IMG"], s["SRC"])))


def prep(shard: str | None) -> None:
    ants.config.set_ants_deterministic(True, SEED)
    src_by_id = selected()
    ids = list(src_by_id)
    if shard:
        k, n = map(int, shard.split("/"))
        ids = ids[k::n]
    rs = _load("resample2", "2_resample.py")
    mask_ref, ref_norm, ref_scale = hist_match.load_hist_match_reference(Path(MNI))
    fixed = ants.image_read(MNI)
    fixed_np, fixed_m = fixed.numpy(), ants.get_mask(fixed).numpy() > 0
    RESAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    qc_csv = RESAMPLE_DIR / f"prep_qc_{shard.replace('/', 'of') if shard else 'all'}.csv"
    for i, id_img in enumerate(ids, 1):
        out = RESAMPLE_DIR / f"{id_img}{SUFFIX}"
        if out.is_file():
            print(f"[{i}/{len(ids)}] [SKIP] {out.name}", flush=True)
            continue
        matched = hist_match.histogram_match(Path(src_by_id[id_img]), mask_ref, ref_norm, ref_scale)
        warped = rs.corregistro_rigid_mni(fixed, matched)["warpedmovout"]
        ants.image_write(warped, str(out))
        r = ncc(warped.numpy(), fixed_np, fixed_m)
        pd.DataFrame([{"ID_IMG": id_img, "ncc_mni": r}]).to_csv(
            qc_csv, mode="a", header=not qc_csv.is_file(), index=False)
        print(f"[{i}/{len(ids)}] [OK] {out.name} ncc_mni={r:.3f}", flush=True)


def build(diag: str, sex: str, abin: str) -> None:
    ants.config.set_ants_deterministic(True, SEED)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = os.environ.get("BUILD_THREADS", "2")
    hits = glob.glob(str(SEL_DIR / f"selected_DIAG-{diag}_SEX-{sex}_AGE-{abin}_N-*.csv"))
    assert len(hits) == 1, hits
    ids = pd.read_csv(hits[0])["ID_IMG"].tolist()
    paths = [RESAMPLE_DIR / f"{i}{SUFFIX}" for i in ids]
    missing = [p.name for p in paths if not p.is_file()]
    assert not missing, f"rode 'prep' antes: {missing[:5]}"
    qc_prep = pd.concat([pd.read_csv(f) for f in RESAMPLE_DIR.glob("prep_qc_*.csv")]).set_index("ID_IMG")["ncc_mni"]
    bad = {i: round(qc_prep[i], 3) for i in ids if qc_prep.get(i, 0) < MIN_NCC_RIGID}
    assert not bad, f"rígido → MNI suspeito (NCC < {MIN_NCC_RIGID}): {bad}"
    out = OUT_DIR / template_name(diag, sex, abin, len(ids))
    if out.is_file():
        print(f"[SKIP] {out}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gw = _load("groupwise22", "2.2_groupwise_ants.py")
    tmp_base = ROOT / "images" / "groupwise" / "references" / "_tmp_ants"
    tmp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"oasis_{diag}{sex}{abin}_", dir=tmp_base) as tmp:
        gw._set_tmp_env(Path(tmp))
        template = gw.build_groupwise_template(paths, gw.N_ITER_TEMPLATE, gw.TYPE_OF_TRANSFORM)
    ants.image_write(template, str(out))
    print(f"[OK] {out}", flush=True)


def qc() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mni = ants.image_read(MNI)
    mni_mask = ants.get_mask(mni)
    files = sorted(glob.glob(str(OUT_DIR / "groupwise_SRC-OASIS_*_template.nii.gz")))
    assert files, OUT_DIR
    rows, imgs = [], {"MNI": mni}
    for f in files:
        im = ants.image_read(f)
        assert im.shape == mni.shape and np.allclose(im.spacing, mni.spacing), f
        assert np.allclose(im.origin, mni.origin) and np.allclose(im.direction, mni.direction), f
        m = ants.get_mask(im)
        name = os.path.basename(f).replace("groupwise_SRC-OASIS_", "").replace("_template.nii.gz", "")
        rows.append({"template": name, "brain_mm3": float((m.numpy() > 0).sum()),
                     "ncc_mni": ncc(im.numpy(), mni.numpy(), (m.numpy() > 0) & (mni_mask.numpy() > 0)),
                     "sharpness": sharpness(im, m)})
        imgs[name] = im
    rows.append({"template": "MNI", "sharpness": sharpness(mni, mni_mask)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "qc.csv", index=False)
    print(t.round(4).to_string(index=False))

    j = int(round(mni.origin[1] + 20))  # y = -20 mm (hipocampo), eixo y com direção -1
    fig, ax = plt.subplots(2, len(imgs), figsize=(2.6 * len(imgs), 5.6))
    for c, (name, im) in enumerate(imgs.items()):
        a = im.numpy()
        vmax = np.percentile(a[a > 0], 99.5)
        ax[0, c].imshow(np.rot90(a[:, j, :]), cmap="gray", vmax=vmax)
        ax[1, c].imshow(np.rot90(a[:, :, im.shape[2] // 2 - 10]), cmap="gray", vmax=vmax)
        ax[0, c].set_title(name.replace("_SEX-", " ").replace("DIAG-", ""), fontsize=7)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "qc.png", dpi=70)
    print(f"ok: {len(files)} templates OASIS com geometria do MNI → {OUT_DIR}/qc.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("select", "prep", "build", "qc"))
    p.add_argument("stratum", nargs="*", help="build: DIAG SEX AGE (ex.: CN F 70-79)")
    p.add_argument("--shard", default=None, help="prep: k/n")
    a = p.parse_args()
    if a.cmd == "build":
        assert len(a.stratum) == 3, "build DIAG SEX AGE"
        build(*a.stratum)
    else:
        {"select": select, "prep": lambda: prep(a.shard), "qc": qc}[a.cmd]()

"""Registro rígido da T1 (4-mni-hist-matching) para o MNI e aplicação da mesma transformação
aos rótulos (regions, seg, brain_mask). A transformação de cada imagem é salva em
<output_dir>/transforms/<ID>_rigid.mat e reutilizada: rótulos que ficarem prontos depois
(ex.: parcelação/segmentação ainda em andamento) são levados ao MNI sem novo registro,
que não é determinístico e desalinharia os rótulos da T1 já salva.

Padrões = ADNI-1/GO/2. ADNI-3/4 (validação externa):
  P=/mnt/study-data/pgirardi/datasets_img/adni3-4/preproc
  python 2_resample.py --population csvs/cohorts/all_population_adni34/all_population_adni34.csv \
      --input-dir $P/4-mni-hist-matching --regions-dir $P/5-parcellation/regions \
      --seg-dir $P/6-segmentation --brain-mask-dir $P/1-skull-stripping
"""
import argparse
import os
import shutil
import time
import ants
import pandas as pd

# Lista união (entrada). Imagens/warps continuam em images/ (globais).
COHORT = "all_population"
population_file = f"csvs/cohorts/{COHORT}/all_population_True.csv"

input_dir = "/mnt/databases/mri/adni/preproc/4-mni-hist-matching"
output_dir = "/mnt/study-data/pgirardi/graphs/images/resampled_1.0mm"
ref_mni_img = "/mnt/study-data/pgirardi/datasets_img/atlases/templates/mni152_2009c_template.nii.gz"

# Volumes auxiliares no espaço nativo (labels) e pasta base de saída em MNI
# regions_dir = "/mnt/databases/mri/adni/preproc/5-parcellation/regions"
regions_dir = "/mnt/study-data/pgirardi/insert_to_databases_regions"
seg_dir = "/mnt/databases/mri/adni/preproc/6-segmentation"
brain_mask_dir = "/mnt/databases/mri/adni/preproc/1-skull-stripping"
labels_output_base = "/mnt/study-data/pgirardi/graphs/images"

labels_dir = (
    (regions_dir, "_regions.nii.gz", "regions"),
    (seg_dir, "_seg.nii.gz", "seg"),
    (brain_mask_dir, "_brain_mask.nii.gz", "brain_mask"),
)


def caminho_transform(output_dir, img_id):
    return os.path.join(output_dir, "transforms", f"{img_id}_rigid.mat")


def corregistro_rigid_mni(
    imagem_mni,
    imagem_moving,
    type_of_transform="Rigid",
    interpolator="linear",
):
    return ants.registration(
        fixed=imagem_mni,
        moving=imagem_moving,
        type_of_transform=type_of_transform,
        interpolator=interpolator,
    )


def aplicar_transform_em_labels(imagem_ref_mni, imagem_labels, lista_transformacoes, dst_path):

    imagem_out = ants.apply_transforms(
        fixed=imagem_ref_mni,
        moving=imagem_labels,
        transformlist=lista_transformacoes,
        interpolator="nearestNeighbor",
    )
    pasta = os.path.dirname(dst_path)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    ants.image_write(imagem_out, dst_path)


def _ficheiros_com_prefixo(input_dir, prefix):

    out = []
    try:
        for name in os.listdir(input_dir):
            if not name.startswith(prefix):
                continue
            # Evita falso-positivo: IDs como I416773 começam com I41677,
            # mas não são o mesmo ID. Exigimos um separador após o ID.
            # Ex.: I41677_*.nii.gz OK; I416773_*.nii.gz NÃO.
            if len(name) > len(prefix) and name[len(prefix)] not in ("_", ".", "-"):
                continue
            p = os.path.join(input_dir, name)
            if os.path.isfile(p):
                out.append(p)
    except OSError:
        pass
    return out


def resolver_caminho_imagem(input_dir, img_id):

    candidate = os.path.join(input_dir, img_id)
    # Caso comum neste dataset: existe uma pasta por ID_IMG contendo o NIfTI.
    if os.path.isdir(candidate):
        matches = _ficheiros_com_prefixo(candidate, img_id)
    else:
        if os.path.isfile(candidate):
            return candidate
        matches = _ficheiros_com_prefixo(input_dir, img_id)
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        # Existem múltiplas variantes no disco (ex.: diferentes sufixos do pipeline).
        # Seleciona de forma determinística:
        # 1) prioriza NIfTI
        # 2) prioriza sufixos esperados
        # 3) desempata pelo mtime (mais recente)
        nii = [m for m in matches if m.endswith((".nii.gz", ".nii"))]
        pool = nii if nii else matches

        prefer_suffixes = (
            "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz",
        )

        def _rank(p: str) -> tuple[int, float, str]:
            name = os.path.basename(p)
            suf_rank = next((i for i, suf in enumerate(prefer_suffixes) if name.endswith(suf)), len(prefer_suffixes))
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = -1.0
            # menor suf_rank é melhor; maior mtime é melhor (por isso -mtime)
            return (suf_rank, -mtime, name)

        return sorted(pool, key=_rank)[0]
    return matches[0]


def precisa_algum_label(img_id, labels_output_base=labels_output_base):

    for src_dir, suf, sub in labels_dir:
        src = os.path.join(src_dir, img_id + suf)
        dst = os.path.join(labels_output_base, sub, img_id + suf)
        if os.path.isfile(src) and not os.path.isfile(dst):
            return True
    return False


def salvar_labels_mni(img_id, imagem_ref_mni, lista_tf, prog, labels_output_base=labels_output_base):

    for src_dir, suf, sub in labels_dir:
        src = os.path.join(src_dir, img_id + suf)
        dst = os.path.join(labels_output_base, sub, img_id + suf)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            print(f"{prog} [LABEL SKIP] já existe: {dst}")
            continue
        print(f"{prog} [LABEL RUN] {sub} → {os.path.basename(dst)}")
        t_l = time.perf_counter()
        imagem_aux = ants.image_read(src)
        aplicar_transform_em_labels(imagem_ref_mni, imagem_aux, lista_tf, dst)
        print(f"{prog} [LABEL OK] {sub} em {time.perf_counter() - t_l:.2f} s")


def run_batch(
    population_file=population_file,
    input_dir=input_dir,
    output_dir=output_dir,
    ref_mni_img=ref_mni_img,
    labels_output_base=labels_output_base,
):

    pop = pd.read_csv(population_file, sep=None, engine="python")#.head(10)

    n_total = len(pop)
    print(f"[INFO] Total de imagens (linhas) na população: {n_total}")
    print(f"[INFO] Saída T1 MNI: {output_dir}")
    print(f"[INFO] Saída labels MNI: {labels_output_base}/{{regions,seg,brain_mask}}/")

    os.makedirs(output_dir, exist_ok=True)
    for sub in ("regions", "seg", "brain_mask"):
        os.makedirs(os.path.join(labels_output_base, sub), exist_ok=True)

    fixed = ants.image_read(ref_mni_img)

    t0 = time.perf_counter()

    for k, (_, row) in enumerate(pop.iterrows(), start=1):
        img_id = str(row["ID_IMG"])
        prog = f"[{k}/{n_total}]"

        moving_path = resolver_caminho_imagem(input_dir, img_id)
        if moving_path is None:
            print(f"{prog} [WARN] Não achei arquivo para ID_IMG={img_id} em {input_dir}")
            continue

        exact = os.path.join(input_dir, img_id)
        if not os.path.isfile(exact) and len(_ficheiros_com_prefixo(input_dir, img_id)) > 1:
            print(
                f"{prog} [WARN] Múltiplos matches para {img_id}; usando {os.path.basename(moving_path)}"
            )

        out_img_path = os.path.join(output_dir, os.path.basename(moving_path))
        tf_path = caminho_transform(output_dir, img_id)

        precisa_t1 = not os.path.isfile(out_img_path)
        precisa_labels = precisa_algum_label(img_id, labels_output_base=labels_output_base)

        if not precisa_t1 and not precisa_labels:
            print(f"{prog} [SKIP] T1 e labels já existem para {img_id}")
            continue

        if not precisa_t1 and os.path.isfile(tf_path):
            print(f"{prog} [TF REUSE] {os.path.basename(tf_path)} (sem novo registro)")
            lista_tf = [tf_path]
            imagem_ref_labels = ants.image_read(out_img_path)
        else:
            moving = ants.image_read(moving_path)
            reg = corregistro_rigid_mni(fixed, moving)
            lista_tf = reg["fwdtransforms"]
            if precisa_t1:
                print(f"{prog} [RUN] {os.path.basename(moving_path)} → rigid to MNI")
                os.makedirs(os.path.dirname(tf_path), exist_ok=True)
                shutil.copyfile(lista_tf[0], tf_path)
                ants.image_write(reg["warpedmovout"], out_img_path)
                print(f"{prog} [OK] Salvo: {out_img_path} + {os.path.basename(tf_path)}")
                lista_tf = [tf_path]
                imagem_ref_labels = reg["warpedmovout"]
            else:
                # ponytail: T1 antiga sem .mat salvo (ADNI-1/GO/2) → registra de novo só para os
                # labels, como antes; o rótulo pode ficar levemente desalinhado da T1 já salva.
                print(f"{prog} [WARN] {out_img_path} sem {os.path.basename(tf_path)}: novo registro só para labels")
                imagem_ref_labels = ants.image_read(out_img_path)

        if precisa_labels:
            salvar_labels_mni(img_id, imagem_ref_labels, lista_tf, prog, labels_output_base)

    elapsed = time.perf_counter() - t0
    print(f"[INFO] Concluído: {n_total} linhas processadas no loop em {elapsed:.1f} s.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", default=population_file, help="CSV com a coluna ID_IMG")
    ap.add_argument("--input-dir", default=input_dir, help="T1 (4-mni-hist-matching)")
    ap.add_argument("--regions-dir", default=regions_dir)
    ap.add_argument("--seg-dir", default=seg_dir)
    ap.add_argument("--brain-mask-dir", default=brain_mask_dir)
    ap.add_argument("--output-dir", default=output_dir, help="T1 em MNI + transforms/")
    ap.add_argument("--labels-output-base", default=labels_output_base, help="{regions,seg,brain_mask}/ em MNI")
    a = ap.parse_args()
    labels_dir = (
        (a.regions_dir, "_regions.nii.gz", "regions"),
        (a.seg_dir, "_seg.nii.gz", "seg"),
        (a.brain_mask_dir, "_brain_mask.nii.gz", "brain_mask"),
    )
    run_batch(a.population, a.input_dir, a.output_dir, ref_mni_img, a.labels_output_base)
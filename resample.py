# Groupwise: todos os *.nii.gz no diretório (templates), registrados à referência do pipeline ADNI (MNI)

# import glob
# import os
# import time
# import ants

# input_dir_gw = "/mnt/study-data/pgirardi/groupwise/adni/CN/"
# output_dir_gw = "/mnt/study-data/pgirardi/graphs/images/groupwise"
# ref_adni_img = "/mnt/study-data/pgirardi/preproc/atlases/templates/mni152_2009c_template.nii.gz"

# nii_files = sorted(glob.glob(os.path.join(input_dir_gw, "*.nii.gz")))
# n_total = len(nii_files)
# print(f"[INFO] Total de imagens (*.nii.gz) no diretório: {n_total}")

# if n_total == 0:
#     print("[WARN] Nenhum .nii.gz encontrado em", input_dir_gw)
# else:
#     os.makedirs(output_dir_gw, exist_ok=True)
#     fixed = ants.image_read(ref_adni_img)
#     t0 = time.perf_counter()

#     for k, moving_path in enumerate(nii_files, start=1):
#         prog = f"[{k}/{n_total}]"
#         basename = os.path.basename(moving_path)
#         out_img_path = os.path.join(output_dir_gw, basename)

#         if os.path.isfile(out_img_path):
#             print(f"{prog} [SKIP] Já existe: {out_img_path}")
#             continue

#         print(f"{prog} [RUN] {basename} -> rigid to referência ADNI (MNI)")
#         moving = ants.image_read(moving_path)

#         reg = ants.registration(
#             fixed=fixed,
#             moving=moving,
#             type_of_transform="Rigid",
#             interpolator="linear",
#         )

#         warped = reg["warpedmovout"]
#         ants.image_write(warped, out_img_path)
#         print(f"{prog} [OK] Salvo: {out_img_path}")

#     elapsed = time.perf_counter() - t0
#     print(f"[INFO] Concluído: {n_total} arquivos processados no loop em {elapsed:.1f} s.")

#######################################################################


import os
import time
import ants
import pandas as pd

population_file = "image_data.txt"

input_dir = "/mnt/databases/mri/adni/preproc/4-mni-hist-matching/"
output_dir = "/mnt/study-data/pgirardi/graphs/images/resampled_1.0mm"
ref_mni_img = "/mnt/study-data/pgirardi/preproc/atlases/templates/mni152_2009c_template.nii.gz"

# Volumes auxiliares no espaço nativo (labels) e pasta base de saída em MNI
regions_dir = "/mnt/databases/mri/adni/preproc/5-parcellation/regions"
seg_dir = "/mnt/databases/mri/adni/preproc/6-segmentation"
brain_mask_dir = "/mnt/databases/mri/adni/preproc/1-skull-stripping"
labels_output_base = "/mnt/study-data/pgirardi/graphs/images"

labels_dir = (
    (regions_dir, "_regions.nii.gz", "regions"),
    (seg_dir, "_seg.nii.gz", "seg"),
    (brain_mask_dir, "_brain_mask.nii.gz", "brain_mask"),
)


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

        precisa_t1 = not os.path.isfile(out_img_path)
        precisa_labels = precisa_algum_label(img_id, labels_output_base=labels_output_base)

        if not precisa_t1 and not precisa_labels:
            print(f"{prog} [SKIP] T1 e labels já existem para {img_id}")
            continue

        moving = ants.image_read(moving_path)
        reg = corregistro_rigid_mni(fixed, moving)
        lista_tf = reg["fwdtransforms"]
        warped = reg["warpedmovout"]

        if precisa_t1:
            print(f"{prog} [RUN] {os.path.basename(moving_path)} → rigid to MNI")
            ants.image_write(warped, out_img_path)
            print(f"{prog} [OK] Salvo: {out_img_path}")
            imagem_ref_labels = warped
        else:
            print(f"{prog} [T1 SKIP] já existe: {out_img_path} (registo só para labels)")
            imagem_ref_labels = ants.image_read(out_img_path)

        if precisa_labels:
            salvar_labels_mni(img_id, imagem_ref_labels, lista_tf, prog, labels_output_base)

    elapsed = time.perf_counter() - t0
    print(f"[INFO] Concluído: {n_total} linhas processadas no loop em {elapsed:.1f} s.")


if __name__ == "__main__":
    run_batch()
import os

# Forçar ANTs/ITK a usar este diretório para arquivos temporários (GenericAffine.mat, Warp, InverseWarp)
# Deve ser definido antes de importar ants.
TMP_DIR = "/mnt/study-data/pgirardi/koedam/exps_pgirardi/reproduction/tmp_ants"
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.makedirs(TMP_DIR, exist_ok=True)

import tempfile
tempfile.tempdir = TMP_DIR

import SimpleITK as sitk
import ants
import pandas as pd


def standardize_image(atlas_ref_fname, moving):
    """moving: caminho (str) do atlas ou imagem SimpleITK já carregada (reutilizar em lote)."""
    # Carregar a imagem de referência (imagem fixa = paciente)
    fixed = sitk.ReadImage(atlas_ref_fname, sitk.sitkFloat32)

    # Imagem móvel (atlas): reutilizar em memória se já for SimpleITK
    if isinstance(moving, str):
        moving = sitk.ReadImage(moving, sitk.sitkFloat32)
    else:
        moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Correspondência de histograma entre atlas e imagem fixa
    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(512)
    matcher.SetNumberOfMatchPoints(11)
    matcher.ThresholdAtMeanIntensityOn()
    
    # Executar o histogram matching
    standardized_image = matcher.Execute(moving, fixed)
    
    return standardized_image

def align_atlas_to_image(fix_img_path, atlas_sitk, output_dir, patient_id, mask_ants):
    """atlas_sitk e mask_ants são reutilizados (carregados uma vez no lote)."""
    print("Realizando histogram matching...")
    standardized_image = standardize_image(fix_img_path, atlas_sitk)

    # Salvar a imagem padronizada temporariamente em TMP_DIR
    standardized_image_path = os.path.join(TMP_DIR, f"{patient_id}_stripped_standardized.nii.gz")
    sitk.WriteImage(standardized_image, standardized_image_path)

    try:
        # Carregar imagens no ANTs (fix e padronizada por caso; máscara reutilizada)
        fix_img = ants.image_read(fix_img_path)
        standardized_img = ants.image_read(standardized_image_path)

        # Realizar o registro utilizando a transformação SyN com correlação cruzada
        print("Realizando registro SyN com métrica de correlação cruzada...")
        registration_syn = ants.registration(
            fixed=fix_img,
            moving=standardized_img,
            type_of_transform='SyN',
            # mask_all_stages=True,  # Aplicar máscara em todos os estágios
            # write_composite_transform=True  # Salvar transformação composta
        )

        # print('Aplicando transformação no Atlas')
        # # Aplicar a transformação SyN à imagem padronizada
        # aligned_img = ants.apply_transforms(
        #     fixed=fix_img,
        #     moving=standardized_img,
        #     transformlist=registration_syn['fwdtransforms'],
        #     interpolator='bSpline'
        # )

        print('Aplicando transformação na Mácara')
        # Aplicar a transformação SyN à máscara padronizada
        aligned_mask = ants.apply_transforms(
            fixed=fix_img,
            moving=mask_ants,
            transformlist=registration_syn['fwdtransforms'],
            interpolator='nearestNeighbor'
        )

        # Salvar a máscara registrada
        # aligned_image_path = os.path.join(output_dir, f"{patient_id}_atlas_aligned.nii.gz")
        aligned_mask_path = os.path.join(output_dir, f"{patient_id}_parietal_mask.nii.gz")
        # ants.image_write(aligned_img, aligned_image_path)
        ants.image_write(aligned_mask, aligned_mask_path)

        print(f'Atlas alinhado e salvo em {aligned_mask_path}')
    finally:
        # Remover arquivo temporário padronizado, se existir
        if os.path.exists(standardized_image_path):
            os.remove(standardized_image_path)

mask_path = '/mnt/study-data/pgirardi/koedam/atlas/segmented-parietal-regions-cerebrA-6.nii.gz'


def process_images(input_dir, output_dir, csv_filter_path):
    """Processa apenas as imagens listadas no CSV de filtro (coluna ID_IMG)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Carregar lista de IDs do CSV
    df = pd.read_csv(csv_filter_path)
    if 'ID_IMG' not in df.columns:
        raise ValueError(f"Coluna 'ID_IMG' não encontrada no CSV: {csv_filter_path}")

    target_ids = set(df['ID_IMG'].dropna().astype(str).str.strip())
    print(f"Total de imagens no CSV: {len(target_ids)}")

    atlas_img_path = "/mnt/study-data/pgirardi/koedam/atlas/mni_icbm152_t1_tal_nlin_sym_09c_stripped_antspynet.nii.gz"
    # Carregar atlas e máscara uma vez; reutilizar para todos os casos (otimização I/O)
    print("[process_images] Carregando atlas e máscara parietal em memória...")
    atlas_sitk = sitk.ReadImage(atlas_img_path, sitk.sitkFloat32)
    mask_ants = ants.image_read(mask_path)
    print("[process_images] Atlas e máscara carregados.")

    processed_count = 0
    skipped_count = 0
    missing_count = 0

    total = len(target_ids)

    for idx, img_id in enumerate(sorted(target_ids), start=1):
        # Verificar se máscara já foi gerada para este img_id
        output_mask_path = os.path.join(output_dir, f"{img_id}_parietal_mask.nii.gz")
        if os.path.exists(output_mask_path):
            print(f"[SKIP {idx}/{total}] {img_id}: máscara já existe")
            skipped_count += 1
            continue

        # Montar caminho da imagem fixa pré-processada
        fixed_img_path = os.path.join(
            input_dir,
            f"{img_id}_stripped_nlm_denoised_biascorrected.nii.gz"
        )

        if not os.path.exists(fixed_img_path):
            print(f"[ERROR {idx}/{total}] {img_id}: arquivo não encontrado -> {os.path.basename(fixed_img_path)}")
            missing_count += 1
            continue

        print(f"[{idx}/{total}] Processando {img_id} -> {os.path.basename(fixed_img_path)}")
        try:
            align_atlas_to_image(fixed_img_path, atlas_sitk, output_dir, img_id, mask_ants)
            processed_count += 1
        except Exception as e:
            print(f"[ERROR] {img_id}: falha no processamento - {e}")
            missing_count += 1

    print("\n" + "=" * 60)
    print("Processamento concluído:")
    print(f"  - Processadas: {processed_count}")
    print(f"  - Já existiam: {skipped_count}")
    print(f"  - Não encontradas / com erro: {missing_count}")
    print("=" * 60)


if __name__ == "__main__":
    input_dir = "/mnt/databases/mri/adni/preproc/3-biasfield/"
    output_dir = "/mnt/study-data/pgirardi/koedam/exps_pgirardi/reproduction/parietal_masks"
    csv_filter = "/mnt/study-data/pgirardi/koedam/exps_pgirardi/adnimerged_filtered_defined.csv"

    process_images(input_dir, output_dir, csv_filter)



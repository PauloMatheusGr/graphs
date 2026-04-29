#!/usr/bin/env python3
"""
Extrai features radiomicas por ROI (labels) usando PyRadiomics e salva em CSV (wide):
1 linha por (ID_IMG, ROI), com colunas de metadados + features.

Foco: ser pequeno e simples.
- ROIs problematicas podem ser removidas diretamente em `ROI_TABLE`.
- Por padrao, usa o "default" do PyRadiomics.
- Opcional: use `--params Params.yaml` para reprodutibilidade e controle de features/image types.
- Se o CSV de saida ja existir (e nao usar --overwrite), por padrao **retoma**: pula linhas
  (ID_IMG, roi, side) ja presentes no arquivo (util apos interrupcao). Use --no-resume para
  forcar reprocessamento completo (gera duplicatas no append).
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Defaults (edite aqui se quiser rodar como script, sem CLI)
DEFAULT_LIST = Path("/mnt/study-data/pgirardi/graphs/image_data_teste.txt")
DEFAULT_IMAGE_DIR = Path("/mnt/study-data/pgirardi/graphs/images/resampled_1.0mm")
DEFAULT_IMAGE_SUFFIX = "_stripped_nlm_denoised_biascorrected_mni_template.nii.gz"
DEFAULT_REGIONS_DIR = Path("/mnt/study-data/pgirardi/graphs/images/regions")
DEFAULT_OUT = Path("/mnt/study-data/pgirardi/graphs/csvs/features_radiomic_teste.csv")

# Seleção de features no código (em vez de Params.yaml).
# Referência das classes: https://github.com/AIM-Harvard/pyradiomics/blob/master/docs/features.rst
#
# - Para extrair TUDO como o default do PyRadiomics, deixe ENABLE_FEATURES = None.
# - Para extrair só um subconjunto, defina um dict:
#     - valor "all": habilita todas as features daquela classe
#     - valor [..]: lista de nomes de features daquela classe
#
# Observação: "shape" independe de imagem (usa só a máscara), mas ainda é habilitada aqui.
ENABLE_FEATURES: dict[str, str | list[str]] | None = {
    # "First Order Features" (o seu link aponta para essa seção)
    "firstorder": "all",
    "shape": "all",
    "glcm": "all",
    "glrlm": "all",
    "glszm": "all",
    "ngtdm": "all",
    "gldm": "all",
}

# Image types. Tipicamente, defaults = apenas "Original".
# Para extrair TUDO como o default, deixe ENABLE_IMAGE_TYPES = None.
ENABLE_IMAGE_TYPES: dict[str, dict[str, Any]] | None = {
    "Original": {},
    # Exemplos (custam bem mais tempo):
    # "LoG": {"sigma": [1.0, 2.0, 3.0]},
    # "Wavelet": {},
}

# ROIs mantidas. Se quiser remover mais, apague linhas aqui.
# (entorhinal/parahippocampal L/R ja estao removidas por instabilidade frequente)
ROI_TABLE: tuple[tuple[str, str, int], ...] = (
    ("hippocampus", "L", 17),
    ("hippocampus", "R", 53),
    ("amygdala", "L", 18),
    ("amygdala", "R", 54),
    ("thalamus_proper", "L", 10),
    ("thalamus_proper", "R", 49),
    ("accumbens_area", "L", 26),
    ("accumbens_area", "R", 58),
    ("inf_lateral_ventricle", "L", 5),
    ("inf_lateral_ventricle", "R", 44),
    ("posterior_cingulate", "L", 1023),
    ("posterior_cingulate", "R", 2023),
    ("isthmus_cingulate", "L", 1010),
    ("isthmus_cingulate", "R", 2010),
    ("rostral_anterior_cingulate", "L", 1026),
    ("rostral_anterior_cingulate", "R", 2026),
    ("medial_orbitofrontal", "L", 1014),
    ("medial_orbitofrontal", "R", 2014),
    ("insula", "L", 1035),
    ("insula", "R", 2035),
)


def load_id_imgs(list_path: Path) -> list[str]:
    with list_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID_IMG" not in reader.fieldnames:
            raise SystemExit("CSV precisa ter cabecalho com coluna ID_IMG")
        out: list[str] = []
        seen: set[str] = set()
        for row in reader:
            iid = (row.get("ID_IMG") or "").strip()
            if not iid or iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
    return out


def make_extractor(params: Path | None, *, bin_width: float | None) -> Any:
    try:
        from radiomics import featureextractor  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "Pacote pyradiomics nao encontrado. Instale com: pip install pyradiomics"
        ) from e

    if params is not None:
        return featureextractor.RadiomicsFeatureExtractor(str(params))
    if bin_width is None:
        return featureextractor.RadiomicsFeatureExtractor()
    return featureextractor.RadiomicsFeatureExtractor(binWidth=float(bin_width))


def configure_extractor_in_code(extractor: Any) -> None:
    """
    Controla image types + feature classes/features sem YAML.
    Se ENABLE_* for None, não altera o extractor (mantém defaults do PyRadiomics).
    """
    if ENABLE_IMAGE_TYPES is not None:
        extractor.disableAllImageTypes()
        for image_type, custom_args in ENABLE_IMAGE_TYPES.items():
            extractor.enableImageTypeByName(str(image_type), customArgs=dict(custom_args))

    if ENABLE_FEATURES is not None:
        extractor.disableAllFeatures()
        for cls_name, spec in ENABLE_FEATURES.items():
            if spec == "all":
                extractor.enableFeatureClassByName(str(cls_name))
            else:
                extractor.enableFeaturesByName(**{str(cls_name): list(spec)})


def iter_feature_items(res: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for k, v in res.items():
        if k.startswith("diagnostics_"):
            continue
        yield k, str(v)


def load_existing_csv_state(
    out_path: Path,
) -> tuple[list[str] | None, set[tuple[str, str, str]]]:
    """
    Le cabecalho e chaves ja escritas: (ID_IMG, roi, side) por linha de dados.
    Retorna (header, done) — header None se arquivo vazio/inexistente.
    """
    if not out_path.is_file() or out_path.stat().st_size == 0:
        return None, set()
    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames
        if not fn:
            return None, set()
        header = list(fn)
        need = {"ID_IMG", "roi", "side"}
        if not need.issubset(set(header)):
            logger.warning(
                "CSV existente sem colunas %s; nao sera possivel retomar por chave.",
                need,
            )
            return header, set()
        done: set[tuple[str, str, str]] = set()
        for row in reader:
            iid = (row.get("ID_IMG") or "").strip()
            roi = (row.get("roi") or "").strip()
            side = (row.get("side") or "").strip()
            if iid and roi and side:
                done.add((iid, roi, side))
        return header, done


def main(
    *,
    list_path: Path = DEFAULT_LIST,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    image_suffix: str = DEFAULT_IMAGE_SUFFIX,
    regions_dir: Path = DEFAULT_REGIONS_DIR,
    out_path: Path = DEFAULT_OUT,
    params: Path | None = None,
    bin_width: float | None = None,
    overwrite: bool = False,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("radiomics").setLevel(logging.WARNING)

    id_imgs = load_id_imgs(list_path)
    if not id_imgs:
        raise SystemExit(f"Nenhum ID_IMG encontrado em {list_path}")

    image_dir = image_dir.resolve()
    regions_dir = regions_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params_path = params.resolve() if params is not None else None
    if params_path is not None and not params_path.is_file():
        raise SystemExit(f"Arquivo --params nao encontrado: {params_path}")

    if overwrite and out_path.is_file():
        out_path.unlink()

    extractor = make_extractor(params_path, bin_width=bin_width)
    # Se você não forneceu YAML, use a seleção embutida neste arquivo.
    if params_path is None:
        configure_extractor_in_code(extractor)

    meta_cols = ["ID_IMG", "roi", "side", "label"]
    header: list[str] | None = None
    done_keys: set[tuple[str, str, str]] = set()

    out_exists = out_path.is_file() and out_path.stat().st_size > 0
    if out_exists:
        header, done_keys = load_existing_csv_state(out_path)
        if not header:
            out_exists = False
            done_keys = set()

    if resume and done_keys:
        logger.info(
            "Retomada: %s linhas (ID_IMG, roi, side) ja presentes no CSV.",
            len(done_keys),
        )
    elif out_exists and not resume:
        logger.info(
            "Retomada desligada (--no-resume): sera reprocessado tudo (append pode duplicar)."
        )

    mode = "a" if out_exists else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] | None = None

        for i, id_img in enumerate(id_imgs, start=1):
            if resume and done_keys:
                need = {(id_img, rn, sd) for rn, sd, _ in ROI_TABLE}
                if need.issubset(done_keys):
                    logger.info(
                        "[%s/%s] %s (pulado: ja completo no CSV)",
                        i,
                        len(id_imgs),
                        id_img,
                    )
                    continue

            img_path = image_dir / f"{id_img}{image_suffix}"
            regions_path = regions_dir / f"{id_img}_regions.nii.gz"
            if not img_path.is_file():
                raise SystemExit(f"[{id_img}] arquivo ausente: {img_path}")
            if not regions_path.is_file():
                raise SystemExit(f"[{id_img}] arquivo ausente: {regions_path}")

            logger.info("[%s/%s] %s", i, len(id_imgs), id_img)
            wrote_any = False
            for roi_name, side, label in ROI_TABLE:
                key = (id_img, roi_name, side)
                if resume and key in done_keys:
                    continue

                res = extractor.execute(
                    str(img_path), str(regions_path), label=int(label)
                )
                feat = dict(iter_feature_items(res))
                row: dict[str, str] = {
                    "ID_IMG": id_img,
                    "roi": roi_name,
                    "side": side,
                    "label": str(int(label)),
                    **feat,
                }

                if header is None:
                    feat_cols = sorted([k for k in row.keys() if k not in meta_cols])
                    header = meta_cols + feat_cols
                    writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                    writer.writeheader()
                    logger.info("Header criado (%s features).", len(feat_cols))

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")

                writer.writerow(row)
                wrote_any = True
                if resume:
                    done_keys.add(key)

            # Durabilidade: garante que o que foi escrito para este ID_IMG
            # foi empurrado para o disco antes de seguir para o próximo.
            if wrote_any:
                f.flush()
                os.fsync(f.fileno())

    logger.info("Concluido. CSV=%s", out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extrai features radiomicas (PyRadiomics) por ROI.")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Apaga o CSV de saida e comeca do zero.",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Nao pula (ID_IMG, roi, side) ja no CSV; com append, duplica linhas.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    main(overwrite=args.overwrite, resume=not args.no_resume, verbose=args.verbose)
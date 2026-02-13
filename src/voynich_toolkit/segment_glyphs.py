"""
Step 3: Segmentazione dei singoli glifi.

Partendo dalle regioni di testo, individua e estrae i singoli caratteri/glifi
del Voynichese. Ogni glifo viene normalizzato e catalogato.
"""
import json
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from .config import ToolkitConfig
from .utils import (
    load_grayscale, preprocess_for_text, find_contours_sorted,
    extract_roi, normalize_glyph, print_header, print_step, timer,
)


def extract_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """
    Individua le righe di testo tramite proiezione orizzontale.

    Returns:
        Lista di tuple (y_start, y_end) per ogni riga
    """
    # Proiezione orizzontale: conta i pixel bianchi per riga
    h_proj = np.sum(binary, axis=1) / 255

    # Trova le righe con testo (sopra una soglia)
    threshold = np.max(h_proj) * 0.05
    in_line = h_proj > threshold

    lines = []
    start = None

    for y, active in enumerate(in_line):
        if active and start is None:
            start = y
        elif not active and start is not None:
            if y - start > 8:  # Altezza minima di una riga
                lines.append((start, y))
            start = None

    if start is not None:
        lines.append((start, len(in_line)))

    return lines


def extract_glyphs_from_line(
    line_img: np.ndarray,
    binary_line: np.ndarray,
    config: ToolkitConfig,
) -> list[dict]:
    """
    Estrae i glifi individuali da una singola riga di testo.

    Returns:
        Lista di dict con bbox e immagine normalizzata del glifo
    """
    contours = find_contours_sorted(binary_line, sort_by="left_to_right")
    glyphs = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtra per dimensioni
        if w < config.min_glyph_width or h < config.min_glyph_height:
            continue
        if w > config.max_glyph_width or h > config.max_glyph_height:
            continue
        if cv2.contourArea(contour) < config.min_glyph_area:
            continue

        # Estrai il glifo
        glyph_img = extract_roi(binary_line, (x, y, w, h), padding=config.glyph_padding)

        # Normalizza
        glyph_norm = normalize_glyph(glyph_img, config.glyph_normalize_size)

        glyphs.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": int(cv2.contourArea(contour)),
            "image": glyph_img,
            "normalized": glyph_norm
        })

    return glyphs


@timer
def process_region(
    region_path: Path,
    output_dir: Path,
    config: ToolkitConfig,
) -> list[dict]:
    """
    Processa una regione di testo ed estrae tutti i glifi.

    Returns:
        Lista di informazioni sui glifi estratti
    """
    region_name = region_path.stem
    gray = load_grayscale(region_path)
    binary = preprocess_for_text(gray, config.glyph_binarize_threshold)

    # Trova le righe di testo
    lines = extract_lines(binary)

    all_glyphs = []
    glyph_idx = 0

    for line_num, (y_start, y_end) in enumerate(lines, 1):
        line_img = gray[y_start:y_end, :]
        binary_line = binary[y_start:y_end, :]

        glyphs = extract_glyphs_from_line(line_img, binary_line, config)

        for glyph in glyphs:
            glyph_idx += 1

            # Salva il glifo originale
            glyph_filename = f"{region_name}_g{glyph_idx:04d}.{config.image_format}"
            glyph_path = output_dir / glyph_filename
            cv2.imwrite(str(glyph_path), glyph["image"])

            # Salva il glifo normalizzato
            norm_filename = f"{region_name}_g{glyph_idx:04d}_norm.{config.image_format}"
            norm_path = output_dir / norm_filename
            cv2.imwrite(str(norm_path), glyph["normalized"])

            all_glyphs.append({
                "region": region_name,
                "glyph_id": glyph_idx,
                "line": line_num,
                "x": glyph["x"],
                "y": glyph["y"] + y_start,
                "w": glyph["w"],
                "h": glyph["h"],
                "area": glyph["area"],
                "file": glyph_filename,
                "normalized_file": norm_filename
            })

    return all_glyphs


def create_glyph_atlas(
    glyphs_dir: Path,
    catalog: list[dict],
    glyph_normalize_size: tuple[int, int] = (32, 32),
    image_format: str = "png",
    grid_cols: int = 30,
) -> None:
    """Crea un atlas visuale di tutti i glifi estratti per ispezione rapida."""
    norm_files = [g for g in catalog if "normalized_file" in g]

    if not norm_files:
        return

    cell_size = glyph_normalize_size[0] + 4
    grid_rows = (len(norm_files) + grid_cols - 1) // grid_cols
    atlas_w = grid_cols * cell_size
    atlas_h = grid_rows * cell_size

    atlas = np.ones((atlas_h, atlas_w), dtype=np.uint8) * 30  # Sfondo scuro

    for i, glyph_info in enumerate(norm_files):
        norm_path = glyphs_dir / glyph_info["normalized_file"]
        if not norm_path.exists():
            continue

        glyph = cv2.imread(str(norm_path), cv2.IMREAD_GRAYSCALE)
        if glyph is None:
            continue

        row = i // grid_cols
        col = i % grid_cols
        y = row * cell_size + 2
        x = col * cell_size + 2

        gh, gw = glyph.shape[:2]
        if y + gh <= atlas_h and x + gw <= atlas_w:
            atlas[y:y + gh, x:x + gw] = glyph

    atlas_path = glyphs_dir / f"glyph_atlas.{image_format}"
    cv2.imwrite(str(atlas_path), atlas)
    print(f"  Atlas dei glifi salvato: {atlas_path}")


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per lo step di segmentazione glifi."""
    print_header("VOYNICH TOOLKIT - Step 3: Segmentazione Glifi")
    config.ensure_dirs()

    catalog_path = config.glyphs_dir / "glyphs_catalog.json"

    # Prerequisito: regions_catalog.json
    regions_catalog = config.text_regions_dir / "regions_catalog.json"
    if not regions_catalog.exists():
        raise click.ClickException(
            "Catalogo regioni non trovato! Esegui prima: voynich segment-text"
        )

    if catalog_path.exists() and not force:
        print(f"  Catalogo glifi gia' presente, skip (usa --force per rieseguire)")
        return

    # Trova le regioni di testo (escludendo debug e catalogo)
    regions = sorted([
        f for f in config.text_regions_dir.glob(f"*.{config.image_format}")
        if "debug" not in f.name
    ])

    if not regions:
        raise click.ClickException(
            "Nessuna regione di testo trovata! Esegui prima: voynich segment-text"
        )

    print(f"  Trovate {len(regions)} regioni di testo")

    all_glyphs = []

    for region_path in tqdm(regions, desc="  Segmentazione glifi", unit="reg"):
        glyphs = process_region(region_path, config.glyphs_dir, config)
        all_glyphs.extend(glyphs)

    # Salva il catalogo dei glifi
    with open(catalog_path, "w") as f:
        json.dump(all_glyphs, f, indent=2)

    # Crea atlas visuale
    print_step("Creazione atlas visuale dei glifi...")
    create_glyph_atlas(
        config.glyphs_dir, all_glyphs,
        glyph_normalize_size=config.glyph_normalize_size,
        image_format=config.image_format,
    )

    print(f"\n  Segmentazione glifi completata!")
    print(f"  {len(all_glyphs)} glifi totali estratti")
    print(f"  Catalogo: {catalog_path}")

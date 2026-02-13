"""
Step 2: Segmentazione regioni di testo.

Isola le aree contenenti testo dalle illustrazioni in ogni pagina.
Produce immagini ritagliate delle sole regioni testuali.
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
    extract_roi, print_header, print_step, timer,
)


@timer
def segment_text_regions(
    image_path: Path,
    output_dir: Path,
    min_area: int = 5000,
    binarize_threshold: int = 160,
    dilate_kernel_width: int = 40,
    dilate_kernel_height: int = 15,
    image_format: str = "png",
) -> list[dict]:
    """
    Individua e estrae le regioni di testo da una pagina.

    Strategia:
    1. Preprocessing (blur + binarizzazione adattiva)
    2. Dilatazione per unire caratteri in blocchi di testo
    3. Ricerca contorni dei blocchi
    4. Filtraggio per area e aspect ratio
    5. Estrazione ROI
    """
    page_name = image_path.stem
    gray = load_grayscale(image_path)

    # Preprocessing
    binary = preprocess_for_text(gray, binarize_threshold)

    # Dilatazione: unisce i caratteri vicini in blocchi di testo
    kernel = np.ones((dilate_kernel_height, dilate_kernel_width), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Trova contorni dei blocchi
    contours = find_contours_sorted(dilated, sort_by="top_to_bottom")

    regions = []
    region_idx = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        # Filtra: deve essere abbastanza grande e non troppo stretto/largo
        if area < min_area:
            continue
        if aspect_ratio > 15 or aspect_ratio < 0.1:
            continue
        # Filtra regioni troppo piccole in una dimensione
        if w < 50 or h < 30:
            continue

        region_idx += 1
        roi = extract_roi(gray, (x, y, w, h), padding=10)

        # Salva la regione
        region_filename = f"{page_name}_region_{region_idx:02d}.{image_format}"
        region_path = output_dir / region_filename
        cv2.imwrite(str(region_path), roi)

        regions.append({
            "page": page_name,
            "region": region_idx,
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "aspect_ratio": round(aspect_ratio, 2),
            "file": region_filename
        })

    # Salva anche un'immagine con le regioni evidenziate (debug)
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in regions:
        cv2.rectangle(debug_img,
                       (r["x"], r["y"]),
                       (r["x"] + r["w"], r["y"] + r["h"]),
                       (0, 255, 0), 3)
        cv2.putText(debug_img, f"R{r['region']}",
                    (r["x"], r["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    debug_path = output_dir / f"{page_name}_debug.{image_format}"
    cv2.imwrite(str(debug_path), debug_img)

    return regions


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per lo step di segmentazione testo."""
    print_header("VOYNICH TOOLKIT - Step 2: Segmentazione Testo")
    config.ensure_dirs()

    pages = sorted(config.pages_dir.glob(f"*.{config.image_format}"))
    if not pages:
        raise click.ClickException(
            "Nessuna pagina trovata! Esegui prima: voynich extract"
        )

    catalog_path = config.text_regions_dir / "regions_catalog.json"
    if catalog_path.exists() and not force:
        print(f"  Catalogo regioni gia' presente, skip (usa --force per rieseguire)")
        return

    print(f"  Trovate {len(pages)} pagine da processare")

    all_regions = []

    for page_path in tqdm(pages, desc="  Segmentazione testo", unit="pag"):
        regions = segment_text_regions(
            page_path, config.text_regions_dir,
            min_area=config.min_text_region_area,
            binarize_threshold=config.binarize_threshold,
            dilate_kernel_width=config.dilate_kernel_width,
            dilate_kernel_height=config.dilate_kernel_height,
            image_format=config.image_format,
        )
        all_regions.extend(regions)

    # Salva il catalogo delle regioni
    with open(catalog_path, "w") as f:
        json.dump(all_regions, f, indent=2)

    print(f"\n  Segmentazione completata!")
    print(f"  {len(all_regions)} regioni di testo totali")
    print(f"  Catalogo: {catalog_path}")

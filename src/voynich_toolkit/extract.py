"""
Step 1: Estrazione pagine dal PDF del Manoscritto Voynich.

Converte ogni pagina del PDF in un'immagine PNG ad alta risoluzione.
"""
from pathlib import Path

import click
import fitz  # pymupdf
from tqdm import tqdm

from .config import ToolkitConfig
from .utils import print_header


def extract_pages(pdf_path: Path, output_dir: Path, dpi: int = 300,
                  image_format: str = "png") -> int:
    """
    Estrae tutte le pagine dal PDF come immagini.

    Returns:
        Numero di pagine estratte.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"  PDF aperto: {total} pagine")
    print(f"  Risoluzione: {dpi} DPI")
    print(f"  Output: {output_dir}/")

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    extracted = 0

    for i, page in enumerate(tqdm(doc, desc="  Estrazione pagine", unit="pag")):
        page_num = i + 1
        output_path = output_dir / f"page_{page_num:03d}.{image_format}"

        if output_path.exists():
            continue

        pix = page.get_pixmap(matrix=matrix)
        pix.save(str(output_path))
        extracted += 1

    doc.close()
    return extracted


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per lo step di estrazione."""
    print_header("VOYNICH TOOLKIT - Step 1: Estrazione Pagine")

    if not config.pdf_path.exists():
        raise click.ClickException(
            f"File non trovato: {config.pdf_path}\n"
            f"  Specifica il percorso con --pdf /path/to/voynich.pdf"
        )

    config.ensure_dirs()

    existing = list(config.pages_dir.glob(f"*.{config.image_format}"))
    if existing and not force:
        print(f"  {len(existing)} pagine gia' presenti, skip (usa --force per riestrarre)")
        return

    if force and existing:
        print(f"  --force: rimozione {len(existing)} pagine esistenti...")
        for f in existing:
            f.unlink()

    extracted = extract_pages(
        config.pdf_path, config.pages_dir,
        dpi=config.extract_dpi,
        image_format=config.image_format,
    )

    images = list(config.pages_dir.glob(f"*.{config.image_format}"))
    total_size_mb = sum(f.stat().st_size for f in images) / (1024 * 1024)
    print(f"\n  Estrazione completata! {len(images)} immagini, {total_size_mb:.1f} MB totali")

"""
Voynich Toolkit - Configurazione globale come dataclass.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolkitConfig:
    """Configurazione per tutti gli step del toolkit."""

    # === PERCORSI ===
    pdf_path: Path = Path("voynich.pdf")
    output_dir: Path = Path("output")
    eva_data_dir: Path = Path("eva_data")

    # === ESTRAZIONE PAGINE ===
    extract_dpi: int = 300
    image_format: str = "png"

    # === SEGMENTAZIONE TESTO ===
    binarize_threshold: int = 160
    dilate_kernel_width: int = 40
    dilate_kernel_height: int = 15
    min_text_region_area: int = 5000

    # === SEGMENTAZIONE GLIFI ===
    glyph_binarize_threshold: int = 140
    min_glyph_width: int = 8
    min_glyph_height: int = 10
    max_glyph_width: int = 80
    max_glyph_height: int = 80
    min_glyph_area: int = 100
    glyph_padding: int = 4
    glyph_normalize_size: tuple[int, int] = (32, 32)

    # === ANALISI STATISTICA ===
    top_n_ngrams: int = 30
    max_cluster_glyphs: int = 5000
    cluster_distance_threshold: float = 0.6

    # === EVA ===
    eva_url: str = "http://www.voynich.nu/data/beta/LSI_ivtff_0d.txt"

    # --- Path derivati ---

    @property
    def pages_dir(self) -> Path:
        return self.output_dir / "pages"

    @property
    def text_regions_dir(self) -> Path:
        return self.output_dir / "text_regions"

    @property
    def glyphs_dir(self) -> Path:
        return self.output_dir / "glyphs"

    @property
    def stats_dir(self) -> Path:
        return self.output_dir / "stats"

    @property
    def eva_dir(self) -> Path:
        return self.output_dir / "eva"

    @property
    def lexicon_dir(self) -> Path:
        return self.output_dir / "lexicon"

    @property
    def hebrew_lexicon_path(self) -> Path:
        """Return enriched lexicon if available, else base."""
        enriched = self.lexicon_dir / "lexicon_enriched.json"
        if enriched.exists():
            return enriched
        return self.lexicon_dir / "lexicon.json"

    def ensure_dirs(self) -> None:
        """Crea tutte le directory necessarie."""
        for d in [
            self.pages_dir,
            self.text_regions_dir,
            self.glyphs_dir,
            self.stats_dir,
            self.eva_dir,
            self.eva_data_dir,
            self.lexicon_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_overrides(cls, **kwargs) -> "ToolkitConfig":
        """Crea una config ignorando i valori None (per integrazione con Click)."""
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**filtered)

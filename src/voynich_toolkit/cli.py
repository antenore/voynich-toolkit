"""
CLI unificata per il Voynich Toolkit.

Entry point: voynich
"""
from pathlib import Path

import click

from .config import ToolkitConfig


@click.group()
@click.option("--pdf", type=click.Path(exists=False, path_type=Path), default=None,
              help="Percorso al PDF del Manoscritto Voynich.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Directory di output (default: output/).")
@click.option("--force", is_flag=True, default=False,
              help="Forza la riesecuzione anche se l'output esiste.")
@click.pass_context
def cli(ctx, pdf, output_dir, force):
    """Voynich Manuscript Analysis Toolkit."""
    ctx.ensure_object(dict)
    overrides = {}
    if pdf is not None:
        overrides["pdf_path"] = pdf
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    ctx.obj["config"] = ToolkitConfig.from_overrides(**overrides)
    ctx.obj["force"] = force


@cli.command()
@click.option("--dpi", type=int, default=None,
              help="Risoluzione di estrazione (default: 300).")
@click.pass_context
def extract(ctx, dpi):
    """Step 1: Estrai pagine dal PDF come immagini."""
    from .extract import run

    config = ctx.obj["config"]
    if dpi is not None:
        config.extract_dpi = dpi
    run(config, force=ctx.obj["force"])


@cli.command("segment-text")
@click.option("--min-area", type=int, default=None,
              help="Area minima regione di testo in pixel^2 (default: 5000).")
@click.pass_context
def segment_text(ctx, min_area):
    """Step 2: Segmenta le regioni di testo dalle pagine."""
    from .segment_text import run

    config = ctx.obj["config"]
    if min_area is not None:
        config.min_text_region_area = min_area
    run(config, force=ctx.obj["force"])


@cli.command("segment-glyphs")
@click.option("--glyph-threshold", type=int, default=None,
              help="Soglia binarizzazione glifi (default: 140).")
@click.pass_context
def segment_glyphs(ctx, glyph_threshold):
    """Step 3: Segmenta i singoli glifi dalle regioni di testo."""
    from .segment_glyphs import run

    config = ctx.obj["config"]
    if glyph_threshold is not None:
        config.glyph_binarize_threshold = glyph_threshold
    run(config, force=ctx.obj["force"])


@cli.command()
@click.option("--cluster-threshold", type=float, default=None,
              help="Soglia distanza per clustering (default: 0.6).")
@click.option("--top-n", type=int, default=None,
              help="Numero di n-grammi da mostrare (default: 30).")
@click.pass_context
def stats(ctx, cluster_threshold, top_n):
    """Step 4: Analisi statistica dei glifi."""
    from .statistics import run

    config = ctx.obj["config"]
    if cluster_threshold is not None:
        config.cluster_distance_threshold = cluster_threshold
    if top_n is not None:
        config.top_n_ngrams = top_n
    run(config, force=ctx.obj["force"])


@cli.command()
@click.option("--eva-url", type=str, default=None,
              help="URL per il download della trascrizione EVA.")
@click.pass_context
def eva(ctx, eva_url):
    """Step 5: Confronto con la trascrizione EVA."""
    from .eva_compare import run

    config = ctx.obj["config"]
    if eva_url is not None:
        config.eva_url = eva_url
    run(config, force=ctx.obj["force"])


@cli.command("section-analysis")
@click.pass_context
def section_analysis(ctx):
    """Confronta statistiche tra le sezioni del manoscritto."""
    from .section_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("image-text")
@click.pass_context
def image_text(ctx):
    """Correla features visive delle pagine con features testuali."""
    from .image_text_correlation import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("word-structure")
@click.pass_context
def word_structure(ctx):
    """Analisi struttura delle parole Voynichesi."""
    from .word_structure import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("language-fingerprint")
@click.pass_context
def language_fingerprint(ctx):
    """Fingerprinting linguistico del Voynichese."""
    from .language_fingerprint import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("char-embeddings")
@click.pass_context
def char_embeddings(ctx):
    """Embeddings e analisi co-occorrenza caratteri."""
    from .char_embeddings import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("cipher-hypothesis")
@click.pass_context
def cipher_hypothesis(ctx):
    """Test ipotesi di cifratura e analisi morfologica."""
    from .cipher_hypothesis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("prepare-lexicon")
@click.pass_context
def prepare_lexicon(ctx):
    """Scarica e prepara lessico ebraico medievale per decifrazione."""
    from .prepare_lexicon import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("decipher")
@click.option("--restarts", type=int, default=20,
              help="Numero di restart per la ricerca (default: 20).")
@click.pass_context
def decipher(ctx, restarts):
    """Tentativo di decifrazione vincolata del Voynichese."""
    from .decipher import run

    run(ctx.obj["config"], force=ctx.obj["force"], n_restarts=restarts)


@cli.command("prepare-italian-lexicon")
@click.pass_context
def prepare_italian_lexicon(ctx):
    """Prepara lessico italiano medievale per decifrazione giudeo-italiana."""
    from .prepare_italian_lexicon import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("decipher-italian")
@click.option("--restarts", type=int, default=30,
              help="Numero di restart per la ricerca (default: 30).")
@click.option("--direction", type=click.Choice(["both", "rtl", "ltr"]),
              default="both",
              help="Direzione di lettura da testare (default: both).")
@click.pass_context
def decipher_italian(ctx, restarts, direction):
    """Tentativo di decifrazione giudeo-italiana del Voynichese."""
    from .italian_decipher import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        n_restarts=restarts, direction=direction)


@cli.command("champollion")
@click.option("--direction", type=click.Choice(["both", "rtl", "ltr"]),
              default="both",
              help="Direzione di lettura da testare (default: both).")
@click.option("--min-support", type=int, default=2,
              help="Minimo folii indipendenti per vincolo (default: 2).")
@click.pass_context
def champollion(ctx, direction, min_support):
    """Approccio Champollion: decifra dai nomi di piante."""
    from .champollion import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        direction=direction, min_support=min_support)


@cli.command("fuzzy-decode")
@click.option("--direction", type=click.Choice(["rtl", "ltr"]),
              default=None,
              help="Reading direction (default: from champollion).")
@click.option("--min-word-len", type=int, default=3,
              help="Minimum word length to decode (default: 3).")
@click.option("--max-dist", type=int, default=2,
              help="Maximum Levenshtein distance (default: 2).")
@click.option("--n-candidates", type=int, default=50,
              help="Number of completion candidates to test (default: 50).")
@click.pass_context
def fuzzy_decode(ctx, direction, min_word_len, max_dist, n_candidates):
    """Complete mapping & fuzzy-match decoded text vs Italian lexicon."""
    from .fuzzy_decode import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        direction=direction, min_word_len=min_word_len,
        max_dist=max_dist, n_candidates=n_candidates)


@cli.command("plant-search")
@click.option("--direction", type=click.Choice(["rtl", "ltr"]),
              default=None,
              help="Reading direction (default: from best mapping).")
@click.option("--min-word-len", type=int, default=3,
              help="Minimum word length (default: 3).")
@click.option("--max-dist", type=int, default=2,
              help="Maximum Levenshtein distance (default: 2).")
@click.pass_context
def plant_search(ctx, direction, min_word_len, max_dist):
    """Search for plant names in decoded text (botanical lexicon)."""
    from .plant_search import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        direction=direction, min_word_len=min_word_len,
        max_dist=max_dist)


@cli.command("copyist-errors")
@click.option("--min-freq", type=int, default=2,
              help="Minimum word frequency to consider (default: 2).")
@click.option("--max-dist", type=int, default=2,
              help="Maximum Levenshtein distance (default: 2).")
@click.pass_context
def copyist_errors(ctx, min_freq, max_dist):
    """Analyze potential copyist errors in EVA space."""
    from .copyist_errors import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        min_freq=min_freq, max_dist=max_dist)


@cli.command("hebrew-decode")
@click.option("--direction", type=click.Choice(["rtl", "ltr"]),
              default=None,
              help="Reading direction (default: from champollion).")
@click.option("--n-candidates", type=int, default=243,
              help="Max candidates to test (default: 243).")
@click.pass_context
def hebrew_decode(ctx, direction, n_candidates):
    """Alternative decode path via Hebrew consonantal lexicon."""
    from .hebrew_decode import run

    run(ctx.obj["config"], force=ctx.obj["force"],
        direction=direction, n_candidates=n_candidates)


@cli.command("full-decode")
@click.pass_context
def full_decode(ctx):
    """Decode full manuscript with convergent 16-char mapping."""
    from .full_decode import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("anchor-words")
@click.pass_context
def anchor_words(ctx):
    """Search decoded text for domain anchor words (IT/HE/LA)."""
    from .anchor_words import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("zodiac-test")
@click.pass_context
def zodiac_test(ctx):
    """Test zodiac section against sign/month/planet names."""
    from .zodiac_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("prefix-resolve")
@click.pass_context
def prefix_resolve(ctx):
    """Resolve unknown chars f/i/q via prefix test + brute force."""
    from .prefix_resolve import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("run-all")
@click.pass_context
def run_all(ctx):
    """Esegui tutti gli step in sequenza."""
    ctx.invoke(extract)
    ctx.invoke(segment_text)
    ctx.invoke(segment_glyphs)
    ctx.invoke(stats)
    ctx.invoke(eva)


def main():
    cli()

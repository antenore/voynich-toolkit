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


@cli.command("enrich-lexicon")
@click.pass_context
def enrich_lexicon(ctx):
    """Download Jastrow dict, filter proper nouns, enrich Hebrew lexicon."""
    from .enrich_lexicon import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("cross-language")
@click.pass_context
def cross_language(ctx):
    """Compare Hebrew mapping against Aramaic and random baselines."""
    from .cross_language_baseline import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("section-specificity")
@click.pass_context
def section_specificity(ctx):
    """Analyze domain term concentration across manuscript sections."""
    from .section_specificity import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("validation-summary")
@click.pass_context
def validation_summary(ctx):
    """Aggregate all Phase 8 results into a validation scorecard."""
    from .validation_summary import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("allograph-analysis")
@click.pass_context
def allograph_analysis(ctx):
    """Investigate EVA l/e allography via positional + context analysis."""
    from .allograph_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("allograph-kt-deep")
@click.pass_context
def allograph_kt_deep(ctx):
    """Deep analysis of EVA k/t allography and freed Hebrew slot."""
    from .allograph_kt_deep import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("digraph-analysis")
@click.pass_context
def digraph_analysis(ctx):
    """Investigate EVA digraphs as single Hebrew letter candidates."""
    from .digraph_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("dual-role")
@click.pass_context
def dual_role(ctx):
    """Test position-dependent mapping splits for over-represented letters."""
    from .dual_role_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("mater-lectionis")
@click.pass_context
def mater_lectionis(ctx):
    """Analyze mater lectionis tolerance impact on lexicon matching."""
    from .mater_lectionis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("deep-yl")
@click.pass_context
def deep_yl(ctx):
    """Deep investigation of EVA y (shin) and EVA l (mem)."""
    from .deep_yl_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("allograph-lr-deep")
@click.pass_context
def allograph_lr_deep(ctx):
    """Deep analysis of EVA l/r allography (mem/he)."""
    from .allograph_lr_deep import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("pe-tet-deep")
@click.pass_context
def pe_tet_deep(ctx):
    """Deep investigation of pe and tet medial over-representation."""
    from .pe_tet_investigation import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("phrase-completion")
@click.pass_context
def phrase_completion(ctx):
    """Multi-tier resolution of unknown decoded Hebrew words."""
    from .phrase_completion import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("qof-investigation")
@click.pass_context
def qof_investigation(ctx):
    """Investigate EVA t → qof swap (tet vs qof differential test)."""
    from .qof_investigation import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("digraph-sh-deep")
@click.pass_context
def digraph_sh_deep(ctx):
    """Investigate EVA 'sh' as digraph for tsade/qof/zayin."""
    from .digraph_sh_deep import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("mapping-audit")
@click.pass_context
def mapping_audit(ctx):
    """Per-letter optimality audit of the EVA→Hebrew mapping."""
    from .mapping_audit import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("italian-layer")
@click.pass_context
def italian_layer(ctx):
    """Italian-layer analysis of decoded Hebrew text."""
    from .italian_layer_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("semantic-coherence")
@click.pass_context
def semantic_coherence(ctx):
    """Analyze semantic coherence of decoded text."""
    from .semantic_coherence import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("phrase-reconstruction")
@click.pass_context
def phrase_reconstruction(ctx):
    """Iterative phrase reconstruction from context."""
    from .phrase_reconstruction import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("currier-split")
@click.pass_context
def currier_split(ctx):
    """Test Hebrew mapping on Currier Language A vs B separately."""
    from .currier_split import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("naibbe-test")
@click.pass_context
def naibbe_test(ctx):
    """Test Naibbe-style verbose homophonic cipher hypothesis."""
    from .naibbe_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("judeo-italian")
@click.pass_context
def judeo_italian(ctx):
    """Test Judeo-Italian hypothesis: Italian written in Hebrew script."""
    from .judeo_italian_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("direction-test")
@click.pass_context
def direction_test(ctx):
    """Phase 15 P2a: Test RTL vs LTR reading direction with permutation test."""
    from .direction_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("mapping-audit-honest")
@click.pass_context
def mapping_audit_honest(ctx):
    """Phase 15 P2b: Per-letter audit with honest lexicon (no Sefaria-Corpus)."""
    from .mapping_audit import run

    run(ctx.obj["config"], force=ctx.obj["force"], lexicon_mode="honest")


@cli.command("gimel-tsade")
@click.pass_context
def gimel_tsade(ctx):
    """Phase 15 P2b: Investigate EVA m → tsade swap (honest lexicon)."""
    from .gimel_tsade_investigation import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("scribe-analysis")
@click.pass_context
def scribe_analysis(ctx):
    """Phase 15 P3: Per-scribe (hand) match rate analysis."""
    from .scribe_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("hand1-dive")
@click.pass_context
def hand1_dive(ctx):
    """Phase 16: analisi approfondita Hand 1 (vocab/structure/audit/compare)."""
    from .hand1_deep_dive import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("null-model-test")
@click.pass_context
def null_model_test(ctx):
    """Phase 16B: null model test — il segnale ebraico è più che lunghezza?"""
    from .null_model_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("section-entropy")
@click.pass_context
def section_entropy(ctx):
    """Phase 16C: match rate per sezione, uniformità, profili EVA."""
    from .section_entropy import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("layout-analysis")
@click.pass_context
def layout_analysis(ctx):
    """Phase 17: analisi layout-aware (label vs paragrafo vs circolare)."""
    from .layout_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("meta-analysis")
@click.pass_context
def meta_analysis(ctx):
    """Phase 18: meta-analisi (h2, MATTR, Zipf, tabella comparativa letteratura)."""
    from .meta_analysis import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("cross-analysis")
@click.pass_context
def cross_analysis(ctx):
    """Phase 19: cross-analisi con classificazioni epilectrik/voynich."""
    from .cross_analysis_epilectrik import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("dictalm-validate")
@click.option("--batch-size", default=20, help="Words per API batch")
@click.pass_context
def dictalm_validate(ctx, batch_size):
    """Phase 20: validazione ebraico con DictaLM (Featherless.ai API)."""
    from .dictalm_validation import run

    run(ctx.obj["config"], force=ctx.obj["force"], batch_size=batch_size)


@cli.command("convergence-control")
@click.pass_context
def convergence_control(ctx):
    """Convergence control: hill-climb on shuffled/random vs real text."""
    from .convergence_control import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("cross-validation")
@click.pass_context
def cross_validation(ctx):
    """Cross-validation: Hand 1 train / rest test + random 50/50 splits."""
    from .cross_validation import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("dictalm-calibrate")
@click.option("--batch-size", default=10, help="Words per API batch")
@click.pass_context
def dictalm_calibrate(ctx, batch_size):
    """DictaLM blinded calibration (100 Hebrew + 100 random + 100 Voynich)."""
    from .dictalm_calibration import run

    run(ctx.obj["config"], force=ctx.obj["force"], batch_size=batch_size)


@cli.command("domain-lexicon-test")
@click.pass_context
def domain_lexicon_test(ctx):
    """Domain-specific lexicon test: chi-square + permutation per section."""
    from .domain_lexicon_test import run

    run(ctx.obj["config"], force=ctx.obj["force"])


@cli.command("scribal-error-correction")
@click.pass_context
def scribal_error_correction(ctx):
    """Scribal error correction via visual confusion pairs."""
    from .scribal_error_correction import run

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

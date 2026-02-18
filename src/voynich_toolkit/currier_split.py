"""
Currier A/B split analysis: test Hebrew mapping on each language separately.

Phase 12: The Voynich MS has two statistical "languages" (Currier 1976),
correlating with different scribes (Davis 2020). If the mapping captures
only one scribe's convention, signal should concentrate in one language.

Tests:
  1. Decode + match rate per language (A vs B)
  2. Per-section breakdown within each language
  3. Permutation test per language (200 perms)
  4. Vocabulary overlap A vs B
  5. Herbal sub-analysis (same content type, both languages)
  6. Two-proportion z-test: is the difference A vs B significant?
"""
import json
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import norm

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew, generate_random_lexicon
from .full_decode import FULL_MAPPING, SECTION_NAMES
from .permutation_stats import (
    build_full_mapping,
    make_lexicon_match_scorer,
    permutation_test_mapping,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Corpus splitting
# =====================================================================

def split_corpus_by_language(pages):
    """Split pages into Currier A and B sub-corpora.

    Args:
        pages: list of page dicts from parse_eva_words()

    Returns: dict keyed by language ('A', 'B') with:
        - pages: list of page dicts
        - words: flat list of EVA words
        - sections: Counter of section codes
        - hands: Counter of hand labels
        - n_pages: int
    """
    result = {}
    for lang in ('A', 'B'):
        lang_pages = [p for p in pages if p.get('language') == lang]
        words = []
        sections = Counter()
        hands = Counter()
        for p in lang_pages:
            words.extend(p['words'])
            sec = p.get('section', '?')
            sections[sec] += 1
            hand = p.get('hand', '?')
            hands[hand] += 1
        result[lang] = {
            'pages': lang_pages,
            'words': words,
            'sections': sections,
            'hands': hands,
            'n_pages': len(lang_pages),
        }
    # Also count discarded
    n_unknown = sum(1 for p in pages if p.get('language') not in ('A', 'B'))
    result['_discarded'] = n_unknown
    return result


# =====================================================================
# Decode + match
# =====================================================================

def decode_and_match(eva_words, lexicon_set, min_len=3):
    """Decode EVA words and match against lexicon.

    Returns: dict with n_decoded, n_matched, match_rate,
             n_unique, n_unique_matched, type_match_rate, top_matches
    """
    decoded = []
    for w in eva_words:
        heb = decode_to_hebrew(w)
        if heb and len(heb) >= min_len:
            decoded.append(heb)

    matched = [h for h in decoded if h in lexicon_set]
    types = set(decoded)
    types_matched = {h for h in types if h in lexicon_set}
    freq = Counter(matched)

    return {
        'n_decoded': len(decoded),
        'n_matched': len(matched),
        'match_rate': len(matched) / max(len(decoded), 1),
        'n_unique': len(types),
        'n_unique_matched': len(types_matched),
        'type_match_rate': len(types_matched) / max(len(types), 1),
        'top_matches': freq.most_common(20),
    }


# =====================================================================
# Per-section stats
# =====================================================================

def per_section_stats(pages, lexicon_set, min_len=3):
    """Compute match rate per section within a language sub-corpus.

    Returns: dict keyed by section code with n_pages, n_words,
             n_decoded, n_matched, match_rate
    """
    by_section = {}
    for p in pages:
        sec = p.get('section', '?')
        if sec not in by_section:
            by_section[sec] = {'pages': [], 'words': []}
        by_section[sec]['pages'].append(p)
        by_section[sec]['words'].extend(p['words'])

    results = {}
    for sec, data in sorted(by_section.items()):
        stats = decode_and_match(data['words'], lexicon_set, min_len)
        results[sec] = {
            'n_pages': len(data['pages']),
            'n_words': len(data['words']),
            'n_decoded': stats['n_decoded'],
            'n_matched': stats['n_matched'],
            'match_rate': stats['match_rate'],
        }
    return results


# =====================================================================
# Vocabulary overlap
# =====================================================================

def vocabulary_overlap(words_a, words_b, lexicon_set, min_len=3):
    """Compute vocabulary overlap between A and B decoded types.

    Returns: dict with types_a, types_b, shared, jaccard,
             shared_in_lexicon, shared_lexicon_rate
    """
    types_a = set()
    for w in words_a:
        heb = decode_to_hebrew(w)
        if heb and len(heb) >= min_len:
            types_a.add(heb)

    types_b = set()
    for w in words_b:
        heb = decode_to_hebrew(w)
        if heb and len(heb) >= min_len:
            types_b.add(heb)

    shared = types_a & types_b
    union = types_a | types_b
    jaccard = len(shared) / max(len(union), 1)

    shared_in_lex = {h for h in shared if h in lexicon_set}

    return {
        'types_a': len(types_a),
        'types_b': len(types_b),
        'shared': len(shared),
        'only_a': len(types_a - types_b),
        'only_b': len(types_b - types_a),
        'jaccard': jaccard,
        'shared_in_lexicon': len(shared_in_lex),
        'shared_lexicon_rate': len(shared_in_lex) / max(len(shared), 1),
    }


# =====================================================================
# Two-proportion z-test
# =====================================================================

def two_proportion_ztest(n1_match, n1_total, n2_match, n2_total):
    """Two-proportion z-test: is the match rate difference significant?

    Returns: dict with p1, p2, diff, z_score, p_value, significant_05
    """
    if n1_total == 0 or n2_total == 0:
        return {'p1': 0, 'p2': 0, 'diff': 0, 'z_score': 0,
                'p_value': 1.0, 'significant_05': False}

    p1 = n1_match / n1_total
    p2 = n2_match / n2_total
    p_pool = (n1_match + n2_match) / (n1_total + n2_total)

    if p_pool == 0 or p_pool == 1:
        return {'p1': round(p1, 4), 'p2': round(p2, 4),
                'diff': round(p1 - p2, 4), 'z_score': 0,
                'p_value': 1.0, 'significant_05': False}

    se = (p_pool * (1 - p_pool) * (1/n1_total + 1/n2_total)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        'p1': round(float(p1), 4),
        'p2': round(float(p2), 4),
        'diff': round(float(p1 - p2), 4),
        'z_score': round(float(z), 2),
        'p_value': round(float(p_value), 6),
        'significant_05': bool(p_value < 0.05),
    }


# =====================================================================
# Cross-language comparison per sub-corpus
# =====================================================================

def cross_language_per_corpus(eva_words, hebrew_set, aramaic_set, random_set,
                              min_len=3):
    """Match decoded words against Hebrew, Aramaic, and Random lexicons.

    Returns: dict with per-lexicon match stats and pairwise z-tests.
    """
    # Decode once
    decoded = []
    for w in eva_words:
        heb = decode_to_hebrew(w)
        if heb and len(heb) >= min_len:
            decoded.append(heb)

    n_total = len(decoded)
    results = {}
    for name, lex in [('hebrew', hebrew_set), ('aramaic', aramaic_set),
                      ('random', random_set)]:
        n_match = sum(1 for h in decoded if h in lex)
        rate = n_match / max(n_total, 1)
        results[name] = {
            'n_matched': n_match,
            'n_total': n_total,
            'match_rate': round(float(rate), 4),
            'n_forms': len(lex),
        }

    # Pairwise z-tests: hebrew vs aramaic, hebrew vs random
    comparisons = {}
    heb = results['hebrew']
    for ctrl_name in ('aramaic', 'random'):
        ctrl = results[ctrl_name]
        zt = two_proportion_ztest(
            heb['n_matched'], heb['n_total'],
            ctrl['n_matched'], ctrl['n_total'])
        comparisons[f'hebrew_vs_{ctrl_name}'] = zt

    return {
        'lexicons': results,
        'comparisons': comparisons,
    }


# =====================================================================
# Permutation wrapper
# =====================================================================

def run_permutation(eva_words, lexicon_set, real_mapping, n_perms=200,
                    seed=42):
    """Run permutation test on a sub-corpus.

    Args:
        eva_words: list of raw EVA words
        lexicon_set: set of Hebrew consonantal forms
        real_mapping: augmented mapping (from build_full_mapping)
        n_perms: number of permutations
        seed: random seed

    Returns: dict from permutation_test_mapping
    """
    score_fn = make_lexicon_match_scorer(eva_words, lexicon_set, min_len=3)
    return permutation_test_mapping(score_fn, real_mapping, n_perms, seed)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Currier A/B split analysis."""
    report_path = config.stats_dir / "currier_split.json"
    summary_path = config.stats_dir / "currier_split_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Currier split report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("CURRIER A/B SPLIT ANALYSIS")

    # 1. Parse EVA
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} total words, "
               f"{len(eva_data['pages'])} pages")

    # 2. Split by language
    print_step("Splitting corpus by Currier language...")
    corpus = split_corpus_by_language(eva_data['pages'])
    for lang in ('A', 'B'):
        c = corpus[lang]
        click.echo(f"    Language {lang}: {c['n_pages']} pages, "
                   f"{len(c['words'])} words")
        click.echo(f"      Sections: {dict(c['sections'])}")
        click.echo(f"      Hands: {dict(c['hands'])}")
    click.echo(f"    Discarded (language=?): {corpus['_discarded']} pages")

    # 3. Load lexicon
    print_step("Loading Hebrew lexicon...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    if not enriched_path.exists():
        raise click.ClickException(
            "Enriched lexicon not found. Run: voynich enrich-lexicon")
    with open(enriched_path) as f:
        hlex = json.load(f)
    hebrew_set = set(hlex["all_consonantal_forms"])
    click.echo(f"    Hebrew: {len(hebrew_set):,} consonantal forms")

    # 3b. Load Aramaic lexicon
    aramaic_path = config.lexicon_dir / "aramaic_lexicon.json"
    aramaic_set = set()
    if aramaic_path.exists():
        with open(aramaic_path) as f:
            aram_data = json.load(f)
        aramaic_set = set(aram_data.get("all_consonantal_forms", []))
        click.echo(f"    Aramaic: {len(aramaic_set):,} forms")
    else:
        click.echo("    Aramaic: not available (run enrich-lexicon)")

    # 3c. Generate random lexicon (same length/char distribution as Hebrew)
    random_set = generate_random_lexicon(hebrew_set, seed=42)
    click.echo(f"    Random: {len(random_set):,} forms")

    # 4. Decode + match per language
    print_step("Decoding and matching per language...")
    match_results = {}
    for lang in ('A', 'B'):
        stats = decode_and_match(corpus[lang]['words'], hebrew_set)
        match_results[lang] = stats
        click.echo(f"    Language {lang}: {stats['n_matched']}/{stats['n_decoded']} "
                   f"tokens ({stats['match_rate']*100:.1f}%), "
                   f"{stats['n_unique_matched']}/{stats['n_unique']} types "
                   f"({stats['type_match_rate']*100:.1f}%)")

    # 5. Per-section stats
    print_step("Computing per-section stats...")
    section_results = {}
    for lang in ('A', 'B'):
        sec_stats = per_section_stats(corpus[lang]['pages'], hebrew_set)
        section_results[lang] = sec_stats
        for sec, s in sorted(sec_stats.items()):
            sec_name = SECTION_NAMES.get(sec, sec)
            click.echo(f"    {lang}/{sec_name}: {s['n_matched']}/{s['n_decoded']} "
                       f"({s['match_rate']*100:.1f}%) [{s['n_pages']} pages]")

    # 6. Permutation tests
    print_step("Running permutation tests (200 perms each)...")
    full_map = build_full_mapping(FULL_MAPPING)
    perm_results = {}
    for lang, seed in [('A', 42), ('B', 43)]:
        click.echo(f"    Language {lang}...", nl=False)
        perm = run_permutation(corpus[lang]['words'], hebrew_set,
                               full_map, n_perms=200, seed=seed)
        perm_results[lang] = perm
        click.echo(f" z={perm['z_score']:.2f}, p={perm['p_value']:.4f}")

    # 7. Vocabulary overlap
    print_step("Computing vocabulary overlap...")
    overlap = vocabulary_overlap(corpus['A']['words'],
                                corpus['B']['words'], hebrew_set)
    click.echo(f"    Types A: {overlap['types_a']}, B: {overlap['types_b']}")
    click.echo(f"    Shared: {overlap['shared']} (Jaccard={overlap['jaccard']:.3f})")
    click.echo(f"    Shared in lexicon: {overlap['shared_in_lexicon']} "
               f"({overlap['shared_lexicon_rate']*100:.1f}%)")

    # 8. Two-proportion z-test A vs B
    print_step("Statistical comparison A vs B...")
    a = match_results['A']
    b = match_results['B']
    comparison = two_proportion_ztest(
        a['n_matched'], a['n_decoded'],
        b['n_matched'], b['n_decoded'])
    sig = "*" if comparison['significant_05'] else "ns"
    click.echo(f"    A: {comparison['p1']*100:.1f}% vs B: {comparison['p2']*100:.1f}%")
    click.echo(f"    diff={comparison['diff']*100:+.1f}pp, "
               f"z={comparison['z_score']:.2f}, p={comparison['p_value']:.4f} {sig}")

    # 9. Cross-language comparison per Currier language
    print_step("Cross-language comparison (Hebrew vs Aramaic vs Random)...")
    cross_lang = {}
    if aramaic_set:
        for lang in ('A', 'B'):
            cl = cross_language_per_corpus(
                corpus[lang]['words'], hebrew_set, aramaic_set, random_set)
            cross_lang[lang] = cl
            heb_r = cl['lexicons']['hebrew']['match_rate']
            ara_r = cl['lexicons']['aramaic']['match_rate']
            rnd_r = cl['lexicons']['random']['match_rate']
            click.echo(f"    Language {lang}: Hebrew {heb_r*100:.1f}%, "
                       f"Aramaic {ara_r*100:.1f}%, Random {rnd_r*100:.1f}%")
            hva = cl['comparisons']['hebrew_vs_aramaic']
            hvr = cl['comparisons']['hebrew_vs_random']
            click.echo(f"      vs Aramaic: z={hva['z_score']:.1f}, "
                       f"vs Random: z={hvr['z_score']:.1f}")

        # Compare Aramaic affinity: does B match Aramaic relatively better?
        ara_a = cross_lang['A']['lexicons']['aramaic']['match_rate']
        ara_b = cross_lang['B']['lexicons']['aramaic']['match_rate']
        heb_a = cross_lang['A']['lexicons']['hebrew']['match_rate']
        heb_b = cross_lang['B']['lexicons']['hebrew']['match_rate']
        # Aramaic/Hebrew ratio per language
        ratio_a = ara_a / max(heb_a, 1e-6)
        ratio_b = ara_b / max(heb_b, 1e-6)
        click.echo(f"    Aramaic/Hebrew ratio: A={ratio_a:.3f}, B={ratio_b:.3f}")
        if ratio_b > ratio_a * 1.1:
            click.echo(f"    -> B shows HIGHER relative Aramaic affinity")
        elif ratio_a > ratio_b * 1.1:
            click.echo(f"    -> A shows HIGHER relative Aramaic affinity")
        else:
            click.echo(f"    -> Similar Aramaic affinity in both languages")
    else:
        click.echo("    Skipped (Aramaic lexicon not available)")

    # 10. Herbal sub-analysis (H section exists in both A and B)
    print_step("Herbal sub-analysis (A vs B, same content type)...")
    herbal_a_pages = [p for p in corpus['A']['pages']
                      if p.get('section') == 'H']
    herbal_b_pages = [p for p in corpus['B']['pages']
                      if p.get('section') == 'H']
    herbal_comparison = None
    if herbal_a_pages and herbal_b_pages:
        ha_words = [w for p in herbal_a_pages for w in p['words']]
        hb_words = [w for p in herbal_b_pages for w in p['words']]
        ha_stats = decode_and_match(ha_words, hebrew_set)
        hb_stats = decode_and_match(hb_words, hebrew_set)
        herbal_comparison = {
            'a': {
                'n_pages': len(herbal_a_pages),
                'n_words': len(ha_words),
                **ha_stats,
            },
            'b': {
                'n_pages': len(herbal_b_pages),
                'n_words': len(hb_words),
                **hb_stats,
            },
            'ztest': two_proportion_ztest(
                ha_stats['n_matched'], ha_stats['n_decoded'],
                hb_stats['n_matched'], hb_stats['n_decoded']),
        }
        # Remove non-serializable top_matches tuples
        for side in ('a', 'b'):
            herbal_comparison[side]['top_matches'] = [
                {'word': w, 'count': c}
                for w, c in herbal_comparison[side]['top_matches']
            ]
        hz = herbal_comparison['ztest']
        click.echo(f"    Herbal-A: {len(herbal_a_pages)} pages, "
                   f"{ha_stats['match_rate']*100:.1f}%")
        click.echo(f"    Herbal-B: {len(herbal_b_pages)} pages, "
                   f"{hb_stats['match_rate']*100:.1f}%")
        click.echo(f"    diff={hz['diff']*100:+.1f}pp, "
                   f"z={hz['z_score']:.2f}, p={hz['p_value']:.4f}")
    else:
        click.echo("    Herbal not present in both languages")

    # 10. Save JSON report
    print_step("Saving reports...")

    # Serialize top_matches
    for lang in ('A', 'B'):
        match_results[lang]['top_matches'] = [
            {'word': w, 'count': c}
            for w, c in match_results[lang]['top_matches']
        ]

    # Serialize section counters
    corpus_info = {}
    for lang in ('A', 'B'):
        corpus_info[lang] = {
            'n_pages': corpus[lang]['n_pages'],
            'n_words': len(corpus[lang]['words']),
            'sections': dict(corpus[lang]['sections']),
            'hands': dict(corpus[lang]['hands']),
        }
    corpus_info['discarded_pages'] = corpus['_discarded']

    report = {
        'corpus': corpus_info,
        'match_results': {k: v for k, v in match_results.items()},
        'section_stats': {
            lang: {sec: v for sec, v in stats.items()}
            for lang, stats in section_results.items()
        },
        'permutation_tests': perm_results,
        'vocabulary_overlap': overlap,
        'comparison_a_vs_b': comparison,
        'cross_language': cross_lang if cross_lang else None,
        'herbal_comparison': herbal_comparison,
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    JSON: {report_path}")

    # 11. Save TXT summary
    lines = []
    lines.append("=" * 60)
    lines.append("  CURRIER A/B SPLIT ANALYSIS")
    lines.append("=" * 60)

    lines.append("\n  CORPUS SPLIT")
    lines.append(f"  {'Language':10s} {'Pages':>6s} {'Words':>8s} {'Sections'}")
    lines.append(f"  {'-'*50}")
    for lang in ('A', 'B'):
        ci = corpus_info[lang]
        secs = ', '.join(f"{k}:{v}" for k, v in sorted(ci['sections'].items()))
        lines.append(f"  {lang:10s} {ci['n_pages']:6d} {ci['n_words']:8d} {secs}")
    lines.append(f"  Discarded: {corpus_info['discarded_pages']} pages")

    lines.append(f"\n  TOKEN MATCH RATES")
    lines.append(f"  {'Language':10s} {'Matched':>8s} {'Total':>8s} "
                 f"{'Rate':>8s} {'Types':>8s} {'Type%':>8s}")
    lines.append(f"  {'-'*56}")
    for lang in ('A', 'B'):
        m = match_results[lang]
        lines.append(f"  {lang:10s} {m['n_matched']:8d} {m['n_decoded']:8d} "
                     f"{m['match_rate']*100:7.1f}% "
                     f"{m['n_unique_matched']:8d} "
                     f"{m['type_match_rate']*100:7.1f}%")

    lines.append(f"\n  PERMUTATION TESTS (200 perms)")
    lines.append(f"  {'Language':10s} {'Real':>8s} {'Mean':>8s} "
                 f"{'Std':>8s} {'z':>8s} {'p':>10s}")
    lines.append(f"  {'-'*56}")
    for lang in ('A', 'B'):
        p = perm_results[lang]
        sig = "***" if p['significant_001'] else \
              "**" if p['significant_01'] else \
              "*" if p['significant_05'] else "ns"
        lines.append(f"  {lang:10s} {p['real_score']:8.0f} "
                     f"{p['random_mean']:8.1f} {p['random_std']:8.1f} "
                     f"{p['z_score']:8.2f} {p['p_value']:10.4f} {sig}")

    lines.append(f"\n  PER-SECTION BREAKDOWN")
    lines.append(f"  {'Lang/Sec':14s} {'Pages':>6s} {'Words':>8s} "
                 f"{'Matched':>8s} {'Rate':>8s}")
    lines.append(f"  {'-'*50}")
    for lang in ('A', 'B'):
        for sec, s in sorted(section_results[lang].items()):
            sec_name = SECTION_NAMES.get(sec, sec)[:8]
            label = f"{lang}/{sec_name}"
            lines.append(f"  {label:14s} {s['n_pages']:6d} {s['n_decoded']:8d} "
                         f"{s['n_matched']:8d} {s['match_rate']*100:7.1f}%")

    lines.append(f"\n  VOCABULARY OVERLAP")
    lines.append(f"  Types A: {overlap['types_a']}, B: {overlap['types_b']}")
    lines.append(f"  Shared: {overlap['shared']} (Jaccard={overlap['jaccard']:.3f})")
    lines.append(f"  Only A: {overlap['only_a']}, Only B: {overlap['only_b']}")
    lines.append(f"  Shared in lexicon: {overlap['shared_in_lexicon']} "
                 f"({overlap['shared_lexicon_rate']*100:.1f}%)")

    lines.append(f"\n  A vs B COMPARISON")
    lines.append(f"  A: {comparison['p1']*100:.1f}% vs B: {comparison['p2']*100:.1f}%")
    lines.append(f"  diff = {comparison['diff']*100:+.1f}pp")
    lines.append(f"  z = {comparison['z_score']:.2f}, "
                 f"p = {comparison['p_value']:.4f} "
                 f"{'(significant)' if comparison['significant_05'] else '(not significant)'}")

    if cross_lang:
        lines.append(f"\n  CROSS-LANGUAGE COMPARISON (per Currier language)")
        lines.append(f"  {'Lang':6s} {'Hebrew':>8s} {'Aramaic':>8s} "
                     f"{'Random':>8s} {'H vs A z':>9s} {'H vs R z':>9s}")
        lines.append(f"  {'-'*54}")
        for lang in ('A', 'B'):
            cl = cross_lang[lang]
            h = cl['lexicons']['hebrew']['match_rate']
            ar = cl['lexicons']['aramaic']['match_rate']
            rn = cl['lexicons']['random']['match_rate']
            za = cl['comparisons']['hebrew_vs_aramaic']['z_score']
            zr = cl['comparisons']['hebrew_vs_random']['z_score']
            lines.append(f"  {lang:6s} {h*100:7.1f}% {ar*100:7.1f}% "
                         f"{rn*100:7.1f}% {za:9.1f} {zr:9.1f}")
        ara_a = cross_lang['A']['lexicons']['aramaic']['match_rate']
        ara_b = cross_lang['B']['lexicons']['aramaic']['match_rate']
        heb_a = cross_lang['A']['lexicons']['hebrew']['match_rate']
        heb_b = cross_lang['B']['lexicons']['hebrew']['match_rate']
        ratio_a = ara_a / max(heb_a, 1e-6)
        ratio_b = ara_b / max(heb_b, 1e-6)
        lines.append(f"  Aramaic/Hebrew ratio: A={ratio_a:.3f}, B={ratio_b:.3f}")

    if herbal_comparison:
        hz = herbal_comparison['ztest']
        lines.append(f"\n  HERBAL SUB-ANALYSIS (controlling for content)")
        lines.append(f"  Herbal-A: {herbal_comparison['a']['n_pages']} pages, "
                     f"{herbal_comparison['a']['match_rate']*100:.1f}%")
        lines.append(f"  Herbal-B: {herbal_comparison['b']['n_pages']} pages, "
                     f"{herbal_comparison['b']['match_rate']*100:.1f}%")
        lines.append(f"  diff = {hz['diff']*100:+.1f}pp, "
                     f"z = {hz['z_score']:.2f}, p = {hz['p_value']:.4f}")

    # Verdict
    lines.append(f"\n  {'='*50}")
    a_z = perm_results['A']['z_score']
    b_z = perm_results['B']['z_score']
    a_rate = match_results['A']['match_rate']
    b_rate = match_results['B']['match_rate']

    if comparison['significant_05']:
        higher = 'A' if a_rate > b_rate else 'B'
        lower = 'B' if higher == 'A' else 'A'
        lines.append(f"  VERDICT: Signal significantly STRONGER in Language {higher}")
        lines.append(f"  {higher}: {max(a_rate, b_rate)*100:.1f}% vs "
                     f"{lower}: {min(a_rate, b_rate)*100:.1f}% (p={comparison['p_value']:.4f})")
        if perm_results[higher]['significant_05'] and not perm_results[lower]['significant_05']:
            lines.append(f"  Language {higher} passes permutation test, "
                         f"{lower} does not.")
        elif perm_results[higher]['significant_05'] and perm_results[lower]['significant_05']:
            lines.append(f"  Both languages pass permutation test "
                         f"(A: z={a_z:.1f}, B: z={b_z:.1f})")
    else:
        lines.append(f"  VERDICT: Signal UNIFORM across languages")
        lines.append(f"  A: {a_rate*100:.1f}% (z={a_z:.1f}) vs "
                     f"B: {b_rate*100:.1f}% (z={b_z:.1f})")
        lines.append(f"  No significant difference (p={comparison['p_value']:.4f})")

    lines.append(f"  {'='*50}")

    txt = '\n'.join(lines) + '\n'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    click.echo(f"    TXT: {summary_path}")

    # Console output
    click.echo(f"\n{txt}")

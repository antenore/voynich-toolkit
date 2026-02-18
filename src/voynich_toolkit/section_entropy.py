"""Section entropy analysis: match rate per section, uniformity test, EVA profiles.

Phase 16C: All sections show ~40% match rate with full lexicon (491K), which
looks suspiciously uniform. With the honest lexicon (45K, no Sefaria-Corpus),
the range is 12%–25% — zodiac and astro are much lower. This module quantifies
the inflation artefact, tests uniformity, and characterizes structural
differences across sections.
"""

import json
from collections import Counter

import click
import numpy as np
from scipy.stats import chi2

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew
from .currier_split import per_section_stats, two_proportion_ztest
from .full_decode import SECTION_NAMES
from .mapping_audit import load_honest_lexicon
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Analysis 1: Match rates per section (honest + full)
# =====================================================================

def section_match_rates(pages, honest_lex, full_lex):
    """Compute match rate per section with both lexicons.

    Returns: dict keyed by section code with honest/full rates
    and inflation factor.
    """
    honest_stats = per_section_stats(pages, honest_lex)
    full_stats = per_section_stats(pages, full_lex)

    results = {}
    for sec in sorted(set(honest_stats) | set(full_stats)):
        h = honest_stats.get(sec, {})
        f = full_stats.get(sec, {})
        h_rate = h.get('match_rate', 0)
        f_rate = f.get('match_rate', 0)
        inflation = f_rate / h_rate if h_rate > 0 else 0

        results[sec] = {
            'section_name': SECTION_NAMES.get(sec, sec),
            'n_pages': h.get('n_pages', f.get('n_pages', 0)),
            'n_words': h.get('n_words', f.get('n_words', 0)),
            'n_decoded': h.get('n_decoded', f.get('n_decoded', 0)),
            'honest_matched': h.get('n_matched', 0),
            'honest_rate': round(h_rate, 4),
            'full_matched': f.get('n_matched', 0),
            'full_rate': round(f_rate, 4),
            'inflation_factor': round(inflation, 2),
        }

    return results


# =====================================================================
# Analysis 2: Chi-square uniformity test
# =====================================================================

def uniformity_test(section_stats, lexicon_key='honest'):
    """Chi-square test for uniform match rate across sections.

    H0: all sections have the same match rate.
    Returns chi2 statistic, p-value, df, and standardized residuals.
    """
    rate_key = f'{lexicon_key}_rate'
    matched_key = f'{lexicon_key}_matched'

    sections = []
    observed = []
    totals = []
    for sec, s in sorted(section_stats.items()):
        n_dec = s.get('n_decoded', 0)
        if n_dec < 10:  # skip tiny sections
            continue
        sections.append(sec)
        observed.append(s[matched_key])
        totals.append(n_dec)

    total_matched = sum(observed)
    total_decoded = sum(totals)
    if total_decoded == 0:
        return {'chi2': 0, 'p_value': 1, 'df': 0, 'residuals': {}}

    pooled_rate = total_matched / total_decoded

    # Expected matches per section under H0
    expected = [pooled_rate * t for t in totals]

    # Chi-square statistic
    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected)
                    if e > 0)
    df = len(sections) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df) if df > 0 else 1.0

    # Standardized residuals
    residuals = {}
    for sec, o, e in zip(sections, observed, expected):
        if e > 0:
            residuals[sec] = round((o - e) / e ** 0.5, 2)

    return {
        'chi2': round(float(chi2_stat), 2),
        'p_value': round(float(p_value), 6),
        'df': df,
        'pooled_rate': round(pooled_rate, 4),
        'residuals': residuals,
        'significant_05': bool(p_value < 0.05),
    }


# =====================================================================
# Analysis 3: EVA structural profile per section
# =====================================================================

def section_eva_profile(pages):
    """Compute EVA structural profile per section.

    Per section: avg word length, type/token ratio, hapax ratio,
    top-5 EVA bigrams.
    """
    by_section = {}
    for p in pages:
        sec = p.get('section', '?')
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p['words'])

    results = {}
    for sec, words in sorted(by_section.items()):
        if len(words) < 10:
            continue
        lengths = [len(w) for w in words]
        types = set(words)
        hapax = sum(1 for w, c in Counter(words).items() if c == 1)

        # EVA bigrams
        bigram_counter = Counter()
        for w in words:
            for i in range(len(w) - 1):
                bigram_counter[w[i:i+2]] += 1

        results[sec] = {
            'section_name': SECTION_NAMES.get(sec, sec),
            'n_words': len(words),
            'n_types': len(types),
            'avg_word_length': round(np.mean(lengths), 2),
            'type_token_ratio': round(len(types) / len(words), 4),
            'hapax_ratio': round(hapax / len(types), 4) if types else 0,
            'top5_bigrams': [
                {'bigram': bg, 'count': c}
                for bg, c in bigram_counter.most_common(5)
            ],
        }

    return results


# =====================================================================
# Analysis 4: Gloss profile per section
# =====================================================================

def section_gloss_profile(pages, honest_lex, form_to_gloss):
    """Top-5 glossed words per section + Jaccard overlap."""
    by_section = {}
    for p in pages:
        sec = p.get('section', '?')
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p['words'])

    section_matched_types = {}
    section_top_glosses = {}

    for sec, words in sorted(by_section.items()):
        if len(words) < 10:
            continue
        decoded_freq = Counter()
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= 3 and heb in honest_lex:
                decoded_freq[heb] += 1

        section_matched_types[sec] = set(decoded_freq.keys())

        # Top-5 with glosses
        top5 = []
        for heb, count in decoded_freq.most_common(5):
            gloss = form_to_gloss.get(heb, '')
            top5.append({'hebrew': heb, 'count': count, 'gloss': gloss})
        section_top_glosses[sec] = top5

    # Jaccard overlap between all section pairs
    secs = sorted(section_matched_types.keys())
    jaccard = {}
    for i, s1 in enumerate(secs):
        for s2 in secs[i+1:]:
            t1 = section_matched_types[s1]
            t2 = section_matched_types[s2]
            union = t1 | t2
            inter = t1 & t2
            j = len(inter) / len(union) if union else 0
            jaccard[f'{s1}-{s2}'] = round(j, 3)

    return {
        'top_glosses': section_top_glosses,
        'jaccard_overlap': jaccard,
    }


# =====================================================================
# Summary formatter
# =====================================================================

def format_summary(match_rates, uniformity_honest, uniformity_full,
                   eva_profiles, gloss_profiles):
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("  SECTION ENTROPY ANALYSIS — Phase 16C")
    lines.append("=" * 70)

    # Match rates table
    lines.append("\n  MATCH RATES PER SECTION (honest 45K vs full 491K)")
    lines.append(f"  {'Sec':>4s} {'Name':>14s} {'Pages':>5s} {'Decoded':>7s} "
                 f"{'Honest%':>8s} {'Full%':>7s} {'Inflate':>7s}")
    lines.append("  " + "-" * 60)
    for sec in sorted(match_rates, key=lambda s: -match_rates[s]['honest_rate']):
        s = match_rates[sec]
        lines.append(
            f"  {sec:>4s} {s['section_name']:>14s} {s['n_pages']:5d} "
            f"{s['n_decoded']:7d} {s['honest_rate']*100:7.1f}% "
            f"{s['full_rate']*100:6.1f}% {s['inflation_factor']:6.1f}x"
        )

    # Uniformity tests
    lines.append(f"\n  UNIFORMITY TEST (chi-square, H0: same rate everywhere)")
    for label, u in [('Honest 45K', uniformity_honest),
                     ('Full 491K', uniformity_full)]:
        sig = "SIGNIFICANT" if u['significant_05'] else "not significant"
        lines.append(f"  {label}: chi2={u['chi2']:.1f}, df={u['df']}, "
                     f"p={u['p_value']:.6f} ({sig})")
        lines.append(f"    Pooled rate: {u['pooled_rate']*100:.1f}%")
        residual_str = ', '.join(
            f"{sec}={r:+.1f}" for sec, r in sorted(u['residuals'].items(),
                                                    key=lambda x: -abs(x[1]))
        )
        lines.append(f"    Residuals: {residual_str}")

    # EVA profiles
    lines.append(f"\n  EVA STRUCTURAL PROFILES")
    lines.append(f"  {'Sec':>4s} {'Words':>7s} {'Types':>6s} {'AvgLen':>6s} "
                 f"{'TTR':>6s} {'Hapax%':>7s} {'Top bigrams'}")
    lines.append("  " + "-" * 70)
    for sec in sorted(eva_profiles, key=lambda s: -eva_profiles[s]['n_words']):
        p = eva_profiles[sec]
        bg_str = ' '.join(b['bigram'] for b in p['top5_bigrams'][:3])
        lines.append(
            f"  {sec:>4s} {p['n_words']:7d} {p['n_types']:6d} "
            f"{p['avg_word_length']:6.2f} {p['type_token_ratio']:6.3f} "
            f"{p['hapax_ratio']*100:6.1f}% {bg_str}"
        )

    # Gloss profiles
    lines.append(f"\n  TOP GLOSSES PER SECTION (honest lexicon)")
    for sec, glosses in sorted(gloss_profiles['top_glosses'].items()):
        if not glosses:
            continue
        sec_name = SECTION_NAMES.get(sec, sec)
        gloss_parts = []
        for g in glosses[:5]:
            gloss_text = g['gloss'][:20] if g['gloss'] else '?'
            gloss_parts.append(f"{g['hebrew']}({gloss_text},{g['count']})")
        lines.append(f"  {sec} ({sec_name}): {', '.join(gloss_parts)}")

    # Jaccard overlap
    lines.append(f"\n  JACCARD OVERLAP (matched types between sections)")
    jac = gloss_profiles['jaccard_overlap']
    for pair in sorted(jac, key=lambda p: -jac[p]):
        lines.append(f"    {pair}: {jac[pair]:.3f}")

    # Verdict
    lines.append(f"\n  {'='*60}")
    if uniformity_honest['significant_05'] and not uniformity_full['significant_05']:
        lines.append("  VERDICT: Honest lexicon reveals NON-UNIFORM sections")
        lines.append("  Full lexicon masks differences (inflation artefact)")
    elif uniformity_honest['significant_05'] and uniformity_full['significant_05']:
        lines.append("  VERDICT: Both lexicons show NON-UNIFORM sections")
    elif not uniformity_honest['significant_05']:
        lines.append("  VERDICT: Sections are statistically UNIFORM")
    lines.append(f"  {'='*60}")

    return '\n'.join(lines) + '\n'


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Section entropy analysis."""
    report_path = config.stats_dir / "section_entropy.json"
    summary_path = config.stats_dir / "section_entropy_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Section entropy report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("SECTION ENTROPY ANALYSIS")

    # 1. Parse EVA
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} total words, "
               f"{len(eva_data['pages'])} pages")

    # 2. Load lexicons
    print_step("Loading lexicons...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    if not enriched_path.exists():
        raise click.ClickException(
            "Enriched lexicon not found. Run: voynich enrich-lexicon")
    with open(enriched_path) as f:
        lex_data = json.load(f)
    full_lex = set(lex_data["all_consonantal_forms"])
    form_to_gloss = lex_data.get("form_to_gloss", {})
    click.echo(f"    Full: {len(full_lex):,} forms")

    honest_lex, _ = load_honest_lexicon(config)
    click.echo(f"    Honest: {len(honest_lex):,} forms")

    pages = eva_data['pages']

    # 3. Match rates per section
    print_step("Computing match rates per section...")
    match_rates = section_match_rates(pages, honest_lex, full_lex)
    for sec in sorted(match_rates, key=lambda s: -match_rates[s]['honest_rate']):
        s = match_rates[sec]
        click.echo(f"    {sec} ({s['section_name']}): "
                   f"honest={s['honest_rate']*100:.1f}%, "
                   f"full={s['full_rate']*100:.1f}%, "
                   f"inflate={s['inflation_factor']:.1f}x "
                   f"[{s['n_pages']} pages]")

    # 4. Uniformity tests
    print_step("Running uniformity tests (chi-square)...")
    uniformity_honest = uniformity_test(match_rates, 'honest')
    uniformity_full = uniformity_test(match_rates, 'full')
    sig_h = "***" if uniformity_honest['p_value'] < 0.001 else \
            "*" if uniformity_honest['significant_05'] else "ns"
    sig_f = "***" if uniformity_full['p_value'] < 0.001 else \
            "*" if uniformity_full['significant_05'] else "ns"
    click.echo(f"    Honest: chi2={uniformity_honest['chi2']:.1f}, "
               f"p={uniformity_honest['p_value']:.6f} {sig_h}")
    click.echo(f"    Full:   chi2={uniformity_full['chi2']:.1f}, "
               f"p={uniformity_full['p_value']:.6f} {sig_f}")

    # 5. EVA profiles
    print_step("Computing EVA structural profiles...")
    eva_profiles = section_eva_profile(pages)
    for sec in sorted(eva_profiles, key=lambda s: -eva_profiles[s]['n_words']):
        p = eva_profiles[sec]
        click.echo(f"    {sec}: {p['n_words']} words, "
                   f"avglen={p['avg_word_length']:.2f}, "
                   f"TTR={p['type_token_ratio']:.3f}, "
                   f"hapax={p['hapax_ratio']*100:.1f}%")

    # 6. Gloss profiles
    print_step("Computing gloss profiles...")
    gloss_profiles = section_gloss_profile(pages, honest_lex, form_to_gloss)
    for sec, glosses in sorted(gloss_profiles['top_glosses'].items()):
        if glosses:
            top = glosses[0]
            click.echo(f"    {sec}: top={top['hebrew']} "
                       f"({top['gloss'][:25]}, n={top['count']})")

    # 7. Pairwise z-tests (largest vs smallest section)
    print_step("Pairwise comparisons...")
    sorted_secs = sorted(match_rates,
                         key=lambda s: match_rates[s]['honest_rate'],
                         reverse=True)
    # Filter to sections with enough data
    sorted_secs = [s for s in sorted_secs
                   if match_rates[s]['n_decoded'] >= 50]
    pairwise = {}
    if len(sorted_secs) >= 2:
        best_sec = sorted_secs[0]
        worst_sec = sorted_secs[-1]
        b = match_rates[best_sec]
        w = match_rates[worst_sec]
        ztest = two_proportion_ztest(
            b['honest_matched'], b['n_decoded'],
            w['honest_matched'], w['n_decoded'])
        pairwise['best_vs_worst'] = {
            'best': best_sec,
            'worst': worst_sec,
            **ztest,
        }
        click.echo(f"    {best_sec} vs {worst_sec}: "
                   f"diff={ztest['diff']*100:+.1f}pp, "
                   f"z={ztest['z_score']:.2f}, p={ztest['p_value']:.6f}")

    # 8. Save JSON
    print_step("Saving reports...")
    report = {
        'match_rates': match_rates,
        'uniformity_honest': uniformity_honest,
        'uniformity_full': uniformity_full,
        'eva_profiles': eva_profiles,
        'gloss_profiles': gloss_profiles,
        'pairwise': pairwise,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    JSON: {report_path}")

    # 9. Save TXT summary
    summary = format_summary(match_rates, uniformity_honest, uniformity_full,
                             eva_profiles, gloss_profiles)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    click.echo(f"    TXT: {summary_path}")

    click.echo(f"\n{summary}")

"""
Per-scribe (hand) analysis: Hebrew match rate per scribal hand.

Phase 15 P3: Davis (2020) identified 5 scribes whose hands correlate
with Currier A/B languages. The Currier split showed Language A is
significantly stronger (+7pp). This module asks whether that difference
is driven by a specific scribe or is uniform across scribes.

Tests:
  1. Match rate per hand (1,2,3,4,5,X,Y,?) — full + honest lexicon
  2. Permutation test for hands with ≥2000 decoded tokens
  3. Two-proportion z-test: each hand vs Hand 1 (reference)
  4. Per-section breakdown within each major hand
  5. Within-content comparison: Herbal pages by hand
  6. Within-content comparison: Astronomical pages by hand
  7. Pairwise comparison matrix between all large hands

Key hypothesis: if the A>B signal is a genuine scribal effect,
Hand 1 should show significantly higher match rate than Hand 2,
even when controlling for content type.

Outputs:
  scribe_analysis.json        — full report
  scribe_analysis_summary.txt — human-readable
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .currier_split import (
    decode_and_match,
    per_section_stats,
    two_proportion_ztest,
    run_permutation,
)
from .full_decode import FULL_MAPPING, SECTION_NAMES
from .mapping_audit import load_honest_lexicon
from .permutation_stats import build_full_mapping
from .utils import print_header, print_step
from .word_structure import parse_eva_words


MIN_WORDS_STATS = 200        # minimum words for basic stats
MIN_WORDS_PERM = 2000        # minimum decoded tokens for permutation test
N_PERMS = 200
SEED = 42

# Named hands (from Davis 2020, with '?' for unattributed)
HAND_NAMES = {
    "1": "Hand-1 (A)",
    "2": "Hand-2 (B)",
    "3": "Hand-3 (B)",
    "4": "Hand-4 (A)",
    "5": "Hand-5 (B)",
    "X": "Hand-X (B?)",
    "Y": "Hand-Y (B?)",
    "?": "Unknown",
}


# =====================================================================
# Corpus splitting
# =====================================================================

def split_corpus_by_hand(pages: list[dict]) -> dict:
    """Split pages into sub-corpora by scribal hand.

    Returns: dict keyed by hand label (str) with:
        - pages: list of page dicts
        - words: flat list of EVA words
        - sections: Counter of section codes
        - languages: Counter of Currier language labels
        - n_pages: int
    """
    result: dict[str, dict] = {}
    for p in pages:
        hand = p.get("hand", "?")
        if hand not in result:
            result[hand] = {
                "pages": [],
                "words": [],
                "sections": Counter(),
                "languages": Counter(),
                "n_pages": 0,
            }
        result[hand]["pages"].append(p)
        result[hand]["words"].extend(p["words"])
        result[hand]["sections"][p.get("section", "?")] += 1
        result[hand]["languages"][p.get("language", "?")] += 1
        result[hand]["n_pages"] += 1

    return result


# =====================================================================
# Content-controlled comparisons
# =====================================================================

def section_by_hand(pages: list[dict], section_code: str,
                    lexicon_set: set, min_pages: int = 2) -> dict:
    """Match rate per hand restricted to one section type.

    Args:
        pages: all corpus pages
        section_code: e.g. 'H' (herbal), 'S' (astronomical)
        lexicon_set: Hebrew consonantal forms
        min_pages: minimum pages for a hand to be included

    Returns: dict keyed by hand label with stats
    """
    by_hand: dict[str, list] = {}
    for p in pages:
        if p.get("section") != section_code:
            continue
        hand = p.get("hand", "?")
        by_hand.setdefault(hand, []).extend(p["words"])

    results = {}
    for hand, words in by_hand.items():
        n_pages = sum(1 for p in pages
                      if p.get("section") == section_code
                      and p.get("hand") == hand)
        if n_pages < min_pages:
            continue
        stats = decode_and_match(words, lexicon_set)
        results[hand] = {
            "n_pages": n_pages,
            "n_words": len(words),
            **stats,
        }

    return results


# =====================================================================
# Pairwise comparison table
# =====================================================================

def pairwise_comparison(hand_stats: dict, reference_hand: str = "1") -> dict:
    """Two-proportion z-test: each hand vs reference hand.

    Returns: dict keyed by hand label with z-test results.
    """
    ref = hand_stats.get(reference_hand)
    if not ref:
        return {}

    comparisons = {}
    for hand, stats in hand_stats.items():
        if hand == reference_hand:
            continue
        zt = two_proportion_ztest(
            ref["n_matched"], ref["n_decoded"],
            stats["n_matched"], stats["n_decoded"],
        )
        comparisons[hand] = zt

    return comparisons


# =====================================================================
# Formatting
# =====================================================================

def _fmt_rate(r):
    return f"{r*100:.1f}%"


def format_summary(corpus_by_hand, match_full, match_honest,
                   perm_results, pairwise, herbal_by_hand,
                   astro_by_hand, section_by_hand_full) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("  SCRIBE ANALYSIS — Per-hand Hebrew Match Rate")
    lines.append("=" * 68)

    # ── Corpus overview ──
    lines.append("\n── Corpus Overview ──")
    lines.append(f"  {'Hand':>8s}  {'Name':20s}  {'Pages':>5s}  "
                 f"{'Words':>7s}  {'Langs':20s}  {'Sections'}")
    lines.append("  " + "-" * 80)
    for hand in sorted(corpus_by_hand.keys()):
        c = corpus_by_hand[hand]
        langs = ', '.join(f"{l}:{n}" for l, n in
                          sorted(c["languages"].items(), key=lambda x: -x[1]))
        secs = ', '.join(f"{s}:{n}" for s, n in
                         sorted(c["sections"].items(), key=lambda x: -x[1]))
        lines.append(f"  {hand:>8s}  {HAND_NAMES.get(hand,'?'):20s}  "
                     f"{c['n_pages']:5d}  {len(c['words']):7,}  {langs:20s}  {secs}")

    # ── Match rates ──
    lines.append(f"\n── Match Rates (token, min {MIN_WORDS_STATS} words) ──")
    lines.append(f"  {'Hand':>8s}  {'Full 491K':>10s}  {'Honest 46K':>11s}  "
                 f"{'N dec.':>7s}  {'Perm z':>7s}  {'p':>7s}")
    lines.append("  " + "-" * 60)
    for hand in sorted(corpus_by_hand.keys()):
        if len(corpus_by_hand[hand]["words"]) < MIN_WORDS_STATS:
            continue
        mf = match_full.get(hand, {})
        mh = match_honest.get(hand, {})
        perm = perm_results.get(hand, {})
        pz = f"{perm['z_score']:7.2f}" if perm else "    n/a"
        pp = f"{perm['p_value']:7.4f}" if perm else "    n/a"
        sig = ""
        if perm:
            sig = ("***" if perm.get("significant_001") else
                   "**" if perm.get("significant_01") else
                   "*" if perm.get("significant_05") else "ns")
        lines.append(f"  {hand:>8s}  {_fmt_rate(mf.get('match_rate',0)):>10s}  "
                     f"{_fmt_rate(mh.get('match_rate',0)):>11s}  "
                     f"{mf.get('n_decoded',0):>7,}  {pz}  {pp} {sig}")

    # ── Pairwise vs Hand 1 ──
    if pairwise:
        lines.append(f"\n── Pairwise Comparison vs Hand 1 (reference) ──")
        lines.append(f"  {'Hand':>8s}  {'H1 rate':>9s}  {'Rate':>9s}  "
                     f"{'Diff':>8s}  {'z':>7s}  {'p':>8s}  Sig")
        lines.append("  " + "-" * 62)
        ref_rate = match_full.get("1", {}).get("match_rate", 0)
        for hand in sorted(pairwise.keys()):
            zt = pairwise[hand]
            sig = "*" if zt["significant_05"] else "ns"
            rate = match_full.get(hand, {}).get("match_rate", 0)
            lines.append(f"  {hand:>8s}  {_fmt_rate(ref_rate):>9s}  "
                         f"{_fmt_rate(rate):>9s}  "
                         f"{zt['diff']*100:>+7.1f}pp  "
                         f"{zt['z_score']:>7.2f}  "
                         f"{zt['p_value']:>8.4f}  {sig}")

    # ── Herbal section (H) by hand ──
    if herbal_by_hand:
        lines.append(f"\n── Herbal Section (H) by Hand ──")
        lines.append(f"  {'Hand':>8s}  {'Pages':>5s}  {'Rate':>9s}  {'N dec':>7s}")
        lines.append("  " + "-" * 36)
        for hand in sorted(herbal_by_hand.keys()):
            s = herbal_by_hand[hand]
            lines.append(f"  {hand:>8s}  {s['n_pages']:>5d}  "
                         f"{_fmt_rate(s['match_rate']):>9s}  "
                         f"{s['n_decoded']:>7,}")
        # z-test hand 1 vs hand 2 in herbal
        if "1" in herbal_by_hand and "2" in herbal_by_hand:
            h1 = herbal_by_hand["1"]
            h2 = herbal_by_hand["2"]
            zt = two_proportion_ztest(h1["n_matched"], h1["n_decoded"],
                                      h2["n_matched"], h2["n_decoded"])
            lines.append(f"\n  Herbal H1 vs H2: diff={zt['diff']*100:+.1f}pp  "
                         f"z={zt['z_score']:.2f}  p={zt['p_value']:.4f}")

    # ── Astronomical section (S) by hand ──
    if astro_by_hand:
        lines.append(f"\n── Astronomical Section (S) by Hand ──")
        lines.append(f"  {'Hand':>8s}  {'Pages':>5s}  {'Rate':>9s}  {'N dec':>7s}")
        lines.append("  " + "-" * 36)
        for hand in sorted(astro_by_hand.keys()):
            s = astro_by_hand[hand]
            lines.append(f"  {hand:>8s}  {s['n_pages']:>5d}  "
                         f"{_fmt_rate(s['match_rate']):>9s}  "
                         f"{s['n_decoded']:>7,}")

    # ── Per-section for Hand 1 and Hand 2 ──
    for hand in ["1", "2"]:
        sec_data = section_by_hand_full.get(hand)
        if not sec_data:
            continue
        lines.append(f"\n── Section Breakdown — {HAND_NAMES.get(hand, hand)} ──")
        lines.append(f"  {'Section':10s}  {'Pages':>5s}  {'Decoded':>8s}  "
                     f"{'Matched':>8s}  {'Rate':>8s}")
        lines.append("  " + "-" * 46)
        for sec, s in sorted(sec_data.items()):
            sec_name = SECTION_NAMES.get(sec, sec)[:8]
            lines.append(f"  {sec_name:10s}  {s['n_pages']:5d}  "
                         f"{s['n_decoded']:8d}  {s['n_matched']:8d}  "
                         f"{s['match_rate']*100:7.1f}%")

    lines.append(f"\n{'=' * 68}")
    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Per-scribe (hand) Hebrew match rate analysis."""
    report_path = config.stats_dir / "scribe_analysis.json"
    summary_path = config.stats_dir / "scribe_analysis_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Scribe analysis report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 15 P3 — Per-Scribe (Hand) Analysis")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Load lexicons
    print_step("Loading lexicons...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    if not enriched_path.exists():
        raise click.ClickException("Enriched lexicon not found. Run: voynich enrich-lexicon")
    with open(enriched_path) as f:
        hlex = json.load(f)
    full_lex = set(hlex["all_consonantal_forms"])
    honest_lex, _ = load_honest_lexicon(config)
    click.echo(f"    Full: {len(full_lex):,} forms | Honest: {len(honest_lex):,} forms")

    # 3. Split by hand
    print_step("Splitting corpus by scribal hand...")
    corpus = split_corpus_by_hand(pages)
    for hand in sorted(corpus.keys()):
        c = corpus[hand]
        langs = dict(c["languages"])
        secs = dict(c["sections"])
        click.echo(f"    Hand {hand:>2s} ({HAND_NAMES.get(hand,'?')}): "
                   f"{c['n_pages']} pages, {len(c['words']):,} words  "
                   f"langs={langs}  secs={secs}")

    # 4. Match rates — full and honest lexicon
    print_step("Computing match rates (full + honest lexicon)...")
    match_full = {}
    match_honest = {}
    for hand, c in corpus.items():
        if len(c["words"]) < MIN_WORDS_STATS:
            click.echo(f"    Hand {hand}: skipped (<{MIN_WORDS_STATS} words)")
            continue
        mf = decode_and_match(c["words"], full_lex)
        mh = decode_and_match(c["words"], honest_lex)
        match_full[hand] = mf
        match_honest[hand] = mh
        click.echo(f"    Hand {hand}: full={mf['match_rate']*100:.1f}%  "
                   f"honest={mh['match_rate']*100:.1f}%  "
                   f"({mf['n_decoded']:,} decoded)")

    # 5. Permutation tests for large hands
    print_step(f"Permutation tests (≥{MIN_WORDS_PERM} decoded tokens, {N_PERMS} perms)...")
    full_map = build_full_mapping(FULL_MAPPING)
    perm_results = {}
    seeds = {"1": SEED, "2": SEED+1, "3": SEED+2, "4": SEED+3,
             "5": SEED+4, "X": SEED+5, "Y": SEED+6, "?": SEED+7}
    for hand, mf in match_full.items():
        if mf["n_decoded"] < MIN_WORDS_PERM:
            click.echo(f"    Hand {hand}: skipped perm test (<{MIN_WORDS_PERM} decoded)")
            continue
        click.echo(f"    Hand {hand}...", nl=False)
        perm = run_permutation(corpus[hand]["words"], full_lex,
                               full_map, n_perms=N_PERMS, seed=seeds.get(hand, SEED))
        perm_results[hand] = perm
        sig = ("***" if perm["significant_001"] else
               "**" if perm["significant_01"] else
               "*" if perm["significant_05"] else "ns")
        click.echo(f" z={perm['z_score']:.2f}  p={perm['p_value']:.4f}  {sig}")

    # 6. Pairwise z-tests vs Hand 1
    print_step("Pairwise comparison vs Hand 1 (reference)...")
    pairwise = pairwise_comparison(match_full, reference_hand="1")
    for hand, zt in sorted(pairwise.items()):
        sig = "*" if zt["significant_05"] else "ns"
        click.echo(f"    H1 vs H{hand}: diff={zt['diff']*100:+.1f}pp  "
                   f"z={zt['z_score']:.2f}  p={zt['p_value']:.4f}  {sig}")

    # 7. Content-controlled analysis — Herbal (H)
    print_step("Herbal section (H) by hand...")
    herbal = section_by_hand(pages, "H", full_lex, min_pages=2)
    for hand, s in sorted(herbal.items()):
        click.echo(f"    Hand {hand}: {s['n_pages']} pages  "
                   f"{s['match_rate']*100:.1f}%  ({s['n_decoded']:,} decoded)")
    if "1" in herbal and "2" in herbal:
        h1, h2 = herbal["1"], herbal["2"]
        zt = two_proportion_ztest(h1["n_matched"], h1["n_decoded"],
                                  h2["n_matched"], h2["n_decoded"])
        click.echo(f"    Herbal H1 vs H2: diff={zt['diff']*100:+.1f}pp  "
                   f"z={zt['z_score']:.2f}  p={zt['p_value']:.4f}")

    # 8. Content-controlled analysis — Astronomical (S)
    print_step("Astronomical section (S) by hand...")
    astro = section_by_hand(pages, "S", full_lex, min_pages=2)
    for hand, s in sorted(astro.items()):
        click.echo(f"    Hand {hand}: {s['n_pages']} pages  "
                   f"{s['match_rate']*100:.1f}%  ({s['n_decoded']:,} decoded)")

    # 9. Per-section breakdown for major hands
    print_step("Per-section breakdown for Hands 1 and 2...")
    section_by_hand_full = {}
    for hand in ["1", "2"]:
        if hand not in corpus:
            continue
        sec_stats = per_section_stats(corpus[hand]["pages"], full_lex)
        section_by_hand_full[hand] = sec_stats
        for sec, s in sorted(sec_stats.items()):
            sec_name = SECTION_NAMES.get(sec, sec)
            click.echo(f"    Hand {hand}/{sec_name}: "
                       f"{s['match_rate']*100:.1f}% ({s['n_pages']} pages)")

    # 10. Save report
    print_step("Saving reports...")

    # Serialize top_matches
    def _ser(stats_dict):
        out = {}
        for hand, s in stats_dict.items():
            d = dict(s)
            if "top_matches" in d:
                d["top_matches"] = [{"word": w, "count": c}
                                    for w, c in d["top_matches"]]
            out[hand] = d
        return out

    report = {
        "corpus": {
            hand: {
                "n_pages": c["n_pages"],
                "n_words": len(c["words"]),
                "sections": dict(c["sections"]),
                "languages": dict(c["languages"]),
            }
            for hand, c in corpus.items()
        },
        "match_full": _ser(match_full),
        "match_honest": _ser(match_honest),
        "permutation_tests": perm_results,
        "pairwise_vs_hand1": pairwise,
        "herbal_by_hand": _ser(herbal),
        "astro_by_hand": _ser(astro),
        "section_by_hand": {
            hand: sec_stats
            for hand, sec_stats in section_by_hand_full.items()
        },
        "min_words_stats": MIN_WORDS_STATS,
        "min_words_perm": MIN_WORDS_PERM,
        "n_perms": N_PERMS,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(
        corpus, match_full, match_honest, perm_results,
        pairwise, herbal, astro, section_by_hand_full,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT: {summary_path}")

    click.echo(f"\n{summary}")

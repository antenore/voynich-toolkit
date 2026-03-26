#!/usr/bin/env python3
"""
Honesty tests: length-stratified permutation + Hebrew prefix analysis.

Test #1: Does the signal survive when controlling for word length?
Test #3: Do Hebrew grammatical prefixes appear at expected frequencies?

Usage:
    python scripts/honesty_tests.py [--perms 200]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voynich_toolkit.config import ToolkitConfig
from voynich_toolkit.full_decode import FULL_MAPPING
from voynich_toolkit.permutation_stats import (
    build_full_mapping,
    decode_eva_with_mapping,
    generate_random_mapping,
    permutation_test_mapping,
)
from voynich_toolkit.word_structure import parse_eva_words


def load_honest_lexicon(config):
    """Load honest lexicon (STEPBible + Jastrow + Klein + curated, no Sefaria)."""
    lex_path = config.lexicon_dir / "lexicon_enriched.json"
    if not lex_path.exists():
        lex_path = config.lexicon_dir / "lexicon.json"
    with open(lex_path) as f:
        data = json.load(f)

    # Filter out Sefaria-Corpus if the enriched file has source info
    all_forms = set(data.get("all_consonantal_forms", []))

    # Try to load the honest subset from DB or fallback to filtering by length
    # For now use the full set - the real honest filtering happens via the
    # glossed_words table, but for a quick test this is adequate
    return all_forms


def load_honest_lexicon_from_db():
    """Load honest lexicon from SQLite (no Sefaria-Corpus)."""
    import sqlite3
    db_path = Path(__file__).resolve().parent.parent / "voynich.db"
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT consonants FROM lexicon WHERE source != 'Sefaria-Corpus'"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows if r[0]}


def decode_corpus(eva_words, mapping):
    """Decode all EVA words, return list of (eva, hebrew, length) tuples."""
    results = []
    for w in eva_words:
        heb = decode_eva_with_mapping(w, mapping, mode="hebrew")
        if heb:
            results.append((w, heb, len(heb)))
    return results


# =====================================================================
# TEST 1: Length-stratified permutation test
# =====================================================================

def test_length_stratified(eva_words, lexicon_set, full_map, n_perms=200):
    """Run permutation test stratified by decoded Hebrew word length."""
    print("\n" + "=" * 70)
    print("TEST 1: Length-Stratified Permutation Test")
    print("=" * 70)
    print(f"Lexicon size: {len(lexicon_set):,} forms (honest, no Sefaria)")
    print(f"Permutations: {n_perms}")

    # Count lexicon forms by length for context
    lex_by_len = Counter(len(f) for f in lexicon_set)
    print(f"\nLexicon forms by length:")
    for l in sorted(lex_by_len):
        possible = 22 ** l
        density = lex_by_len[l] / possible * 100
        print(f"  len={l}: {lex_by_len[l]:>6,} forms / {possible:>10,} possible = {density:.2f}% density")

    # Group EVA words by their DECODED Hebrew length
    decoded = decode_corpus(eva_words, full_map)
    by_len = defaultdict(list)
    for eva, heb, hlen in decoded:
        bucket = min(hlen, 5)  # 5+ grouped together
        by_len[bucket].append(eva)

    print(f"\nDecoded words by Hebrew length:")
    for l in sorted(by_len):
        label = f"{l}+" if l == 5 else str(l)
        print(f"  len={label}: {len(by_len[l]):>5,} tokens")

    # Run permutation test for each length bucket
    results = {}
    for length in sorted(by_len):
        if len(by_len[length]) < 10:
            continue

        label = f"{length}+" if length == 5 else str(length)
        words_bucket = by_len[length]

        # Filter lexicon to same length range for context
        if length == 5:
            lex_same_len = {f for f in lexicon_set if len(f) >= 5}
        else:
            lex_same_len = {f for f in lexicon_set if len(f) == length}

        # Score function for this bucket
        def make_scorer(bucket_words, lex):
            def scorer(mapping):
                n = 0
                for w in bucket_words:
                    h = decode_eva_with_mapping(w, mapping, mode="hebrew")
                    if h and h in lex:
                        n += 1
                return n
            return scorer

        scorer = make_scorer(words_bucket, lexicon_set)

        # Real score
        real_score = scorer(full_map)
        real_rate = real_score / len(words_bucket) * 100

        # Quick permutation
        print(f"\n  Length {label}: {len(words_bucket)} tokens, "
              f"real matches = {real_score} ({real_rate:.1f}%)")
        print(f"    Lexicon forms at this length: {len(lex_same_len):,}")

        result = permutation_test_mapping(
            scorer, full_map, n_perms=n_perms, seed=42
        )

        random_rate = result["random_mean"] / len(words_bucket) * 100
        print(f"    Random mean = {result['random_mean']:.1f} ({random_rate:.1f}%)")
        print(f"    z = {result['z_score']:.2f}, p = {result['p_value']:.4f}")
        print(f"    Ratio real/random = {real_score / max(result['random_mean'], 0.1):.2f}x")

        results[label] = {
            "n_tokens": len(words_bucket),
            "real_matches": real_score,
            "real_rate": round(real_rate, 2),
            "random_mean": round(result["random_mean"], 2),
            "random_rate": round(random_rate, 2),
            "z_score": result["z_score"],
            "p_value": result["p_value"],
            "ratio": round(real_score / max(result["random_mean"], 0.1), 2),
        }

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY: Length-Stratified Results")
    print("-" * 70)
    print(f"{'Length':>6} {'Tokens':>7} {'Real%':>7} {'Rand%':>7} {'Ratio':>6} {'z':>7} {'p':>8} {'Sig?':>5}")
    for label, r in results.items():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{label:>6} {r['n_tokens']:>7,} {r['real_rate']:>6.1f}% {r['random_rate']:>6.1f}% "
              f"{r['ratio']:>5.1f}x {r['z_score']:>6.2f} {r['p_value']:>8.4f} {sig:>5}")

    return results


# =====================================================================
# TEST 3: Hebrew prefix frequency analysis
# =====================================================================

def test_hebrew_prefixes(eva_words, full_map):
    """Check if Hebrew grammatical prefixes appear at expected frequencies."""
    print("\n" + "=" * 70)
    print("TEST 3: Hebrew Prefix Frequency Analysis")
    print("=" * 70)

    # Expected prefix frequencies in Biblical Hebrew (rough estimates from
    # corpus studies, e.g., Andersen & Forbes 1986)
    # These are for ATTACHED prefixes (ב, כ, ל, מ, ש, ה, ו)
    EXPECTED_PREFIXES = {
        'b': ('bet (in/with)', 0.04, 0.08),      # 4-8% in running text
        'k': ('kaf (like/as)', 0.01, 0.04),       # 1-4%
        'l': ('lamed (to/for)', 0.04, 0.10),      # 4-10%
        'm': ('mem (from)', 0.03, 0.08),           # 3-8%
        'S': ('shin-she (that)', 0.01, 0.05),      # 1-5% (Mishnaic she-)
        'h': ('he (the)', 0.05, 0.12),             # 5-12% (very common)
        'w': ('vav (and)', 0.08, 0.20),            # 8-20% (most common conjunction)
    }

    # Decode all words
    decoded_words = []
    for w in eva_words:
        heb = decode_eva_with_mapping(w, full_map, mode="hebrew")
        if heb and len(heb) >= 2:
            decoded_words.append(heb)

    total = len(decoded_words)
    print(f"Total decoded words (len>=2): {total:,}")

    # Count initial letters
    initial_counts = Counter(w[0] for w in decoded_words)

    print(f"\nAll initial letter frequencies:")
    for letter, count in initial_counts.most_common():
        pct = count / total * 100
        print(f"  {letter:>2}: {count:>5,} ({pct:>5.1f}%)")

    # Compare against expected prefix ranges
    print(f"\n{'Prefix':>8} {'Expected':>12} {'Observed':>10} {'Count':>7} {'Status':>12}")
    print("-" * 55)

    n_in_range = 0
    n_tested = 0
    for letter, (name, lo, hi) in EXPECTED_PREFIXES.items():
        count = initial_counts.get(letter, 0)
        observed = count / total
        obs_pct = observed * 100

        if lo <= observed <= hi:
            status = "IN RANGE"
            n_in_range += 1
        elif observed < lo:
            status = "TOO LOW"
        else:
            status = "TOO HIGH"
        n_tested += 1

        print(f"  {letter} ({name[:15]:>15}): {lo*100:.0f}-{hi*100:.0f}%  "
              f"{obs_pct:>6.1f}%  {count:>6,}  {status}")

    # Also check: what % of words start with a "prefix letter"?
    prefix_letters = set(EXPECTED_PREFIXES.keys())
    n_with_prefix = sum(initial_counts.get(l, 0) for l in prefix_letters)
    pct_prefix = n_with_prefix / total * 100

    print(f"\nWords starting with a prefix letter: {n_with_prefix:,} / {total:,} = {pct_prefix:.1f}%")
    print(f"Expected in Hebrew text: ~30-50% of words carry a prefix")
    print(f"Prefixes in expected range: {n_in_range}/{n_tested}")

    # Second-letter analysis: if first letter is a prefix, what follows?
    # In Hebrew, after a prefix you expect a root consonant
    print(f"\n--- Second-letter after potential prefixes ---")
    for letter in ['b', 'l', 'h', 'w', 'm']:
        words_with_prefix = [w for w in decoded_words if w[0] == letter and len(w) >= 3]
        if not words_with_prefix:
            continue
        second_letters = Counter(w[1] for w in words_with_prefix)
        top3 = second_letters.most_common(3)
        top_str = ", ".join(f"{l}={c}" for l, c in top3)
        print(f"  After {letter}: {len(words_with_prefix):>4,} words, top second letters: {top_str}")

    # Comparison with random mapping
    print(f"\n--- Control: random mapping prefix distribution ---")
    rng_map = generate_random_mapping(eva_chars=sorted(full_map.keys()), seed=42)
    rng_map = build_full_mapping(rng_map)
    rng_decoded = []
    for w in eva_words:
        heb = decode_eva_with_mapping(w, rng_map, mode="hebrew")
        if heb and len(heb) >= 2:
            rng_decoded.append(heb)

    rng_initials = Counter(w[0] for w in rng_decoded)
    rng_total = len(rng_decoded)

    print(f"{'Prefix':>8} {'Real':>8} {'Random':>8} {'Diff':>8}")
    for letter, (name, lo, hi) in EXPECTED_PREFIXES.items():
        real_pct = initial_counts.get(letter, 0) / total * 100
        rng_pct = rng_initials.get(letter, 0) / rng_total * 100
        diff = real_pct - rng_pct
        print(f"  {letter:>6}: {real_pct:>6.1f}% {rng_pct:>6.1f}% {diff:>+6.1f}%")

    return {
        "n_in_range": n_in_range,
        "n_tested": n_tested,
        "prefix_word_pct": round(pct_prefix, 1),
        "initial_distribution": {l: round(c / total * 100, 1)
                                 for l, c in initial_counts.most_common()},
    }


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Honesty tests")
    parser.add_argument("--perms", type=int, default=200,
                        help="Number of permutations (default 200)")
    args = parser.parse_args()

    config = ToolkitConfig()
    eva_path = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_path)

    # Build full mapping
    full_map = build_full_mapping(dict(FULL_MAPPING))

    # Load honest lexicon from DB
    print("Loading honest lexicon from SQLite...")
    lexicon = load_honest_lexicon_from_db()
    print(f"  {len(lexicon):,} unique consonantal forms (no Sefaria-Corpus)")

    # All EVA words
    all_words = eva_data["words"]
    print(f"  {len(all_words):,} EVA tokens")

    # Run both tests
    len_results = test_length_stratified(all_words, lexicon, full_map,
                                         n_perms=args.perms)
    prefix_results = test_hebrew_prefixes(all_words, full_map)

    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    # Length test verdict
    sig_buckets = sum(1 for r in len_results.values() if r["p_value"] < 0.05)
    total_buckets = len(len_results)
    print(f"\nLength test: {sig_buckets}/{total_buckets} length buckets significant (p<0.05)")
    if any(r["p_value"] < 0.05 for l, r in len_results.items() if l in ("4", "5+")):
        print("  → Signal survives at longer word lengths — STRONGER evidence")
    else:
        print("  → Signal driven by short words — WEAKER evidence (high chance match rate)")

    # Prefix test verdict
    print(f"\nPrefix test: {prefix_results['n_in_range']}/{prefix_results['n_tested']} prefixes in expected Hebrew range")
    if prefix_results["n_in_range"] >= 5:
        print("  → Prefix distribution matches Hebrew — SUPPORTING evidence")
    elif prefix_results["n_in_range"] >= 3:
        print("  → Partial match — INCONCLUSIVE")
    else:
        print("  → Prefix distribution does NOT match Hebrew — WEAKENING evidence")


if __name__ == "__main__":
    main()

"""
Deep analysis of EVA l/r allography.

Critical finding: l~r cosine=0.954, context=0.747 (above allograph thresholds).
Currently: l → mem(m), r → he(h). If they are allographs, one mapping may be wrong.

Investigates:
  1. Detailed positional profiles and context overlap
  2. Minimal pairs (l↔r substitution in EVA words)
  3. Unification tests: both→mem, both→he, swapped
  4. Positional split possibilities
  5. Impact on lexicon match rate, z-score, and anchor words
  6. Co-occurrence analysis (do l and r appear in the same word?)
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import (
    FULL_MAPPING, II_HEBREW, I_HEBREW, CH_HEBREW, INITIAL_D_HEBREW,
    DIRECTION, preprocess_eva,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Lexicon loading
# =====================================================================

def _load_hebrew_lexicon(config):
    """Load enriched Hebrew lexicon as set of consonantal forms."""
    for name in ("lexicon_enriched.json", "lexicon.json"):
        path = config.lexicon_dir / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            forms = data.get("all_consonantal_forms", [])
            if forms:
                return set(forms)
    return set()


# =====================================================================
# Flexible decoder with mapping overrides
# =====================================================================

def decode_with_mapping(eva_word, mapping_overrides=None):
    """Decode EVA word with optional mapping overrides.

    Returns Hebrew consonantal string or None.
    """
    mapping = dict(FULL_MAPPING)
    if mapping_overrides:
        mapping.update(mapping_overrides)

    prefix, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))

    parts = []
    for ch in chars:
        if ch == '\x01':
            parts.append(II_HEBREW)
        elif ch == '\x02':
            parts.append(I_HEBREW)
        elif ch == '\x03':
            parts.append(CH_HEBREW)
        elif ch in mapping:
            parts.append(mapping[ch])
        else:
            return None

    # Positional split: dalet at word-initial → bet
    if parts and parts[0] == 'd':
        parts[0] = INITIAL_D_HEBREW

    return ''.join(parts)


# =====================================================================
# Positional analysis
# =====================================================================

def positional_profile(words, char):
    """Positional profile for an EVA character."""
    counts = {'initial': 0, 'medial': 0, 'final': 0}
    for w in words:
        for i, c in enumerate(w):
            if c == char:
                if len(w) == 1:
                    counts['initial'] += 1  # isolated = initial
                elif i == 0:
                    counts['initial'] += 1
                elif i == len(w) - 1:
                    counts['final'] += 1
                else:
                    counts['medial'] += 1
    return counts


def bigram_context(words, char):
    """Before/after character context for an EVA character."""
    before = Counter()
    after = Counter()
    for w in words:
        for i, c in enumerate(w):
            if c == char:
                before[w[i-1] if i > 0 else '#'] += 1
                after[w[i+1] if i < len(w) - 1 else '#'] += 1
    return before, after


# =====================================================================
# Co-occurrence analysis
# =====================================================================

def cooccurrence_analysis(words, char_a, char_b):
    """Analyze how often char_a and char_b appear in the same word."""
    n_both = 0
    n_only_a = 0
    n_only_b = 0
    n_neither = 0
    for w in words:
        has_a = char_a in w
        has_b = char_b in w
        if has_a and has_b:
            n_both += 1
        elif has_a:
            n_only_a += 1
        elif has_b:
            n_only_b += 1
        else:
            n_neither += 1

    total = len(words)
    # Complementary distribution score: allographs should rarely co-occur
    cooccurrence_rate = n_both / max(n_both + n_only_a + n_only_b, 1)
    return {
        'both': n_both,
        'only_a': n_only_a,
        'only_b': n_only_b,
        'neither': n_neither,
        'cooccurrence_rate': cooccurrence_rate,
    }


# =====================================================================
# Minimal pairs
# =====================================================================

def find_minimal_pairs(words, char_a, char_b, max_pairs=50):
    """Find words that differ only by char_a↔char_b at one position."""
    word_set = set(words)
    pairs = []
    seen = set()

    for w in word_set:
        for i, c in enumerate(w):
            if c == char_a:
                variant = w[:i] + char_b + w[i+1:]
                if variant in word_set and (w, variant) not in seen:
                    seen.add((w, variant))
                    seen.add((variant, w))
                    pairs.append((w, variant, i))

    # Sort by frequency
    freq = Counter(words)
    pairs.sort(key=lambda x: -(freq.get(x[0], 0) + freq.get(x[1], 0)))
    return pairs[:max_pairs]


# =====================================================================
# Unification / swap tests
# =====================================================================

def test_mapping_variant(eva_words, lexicon_set, mapping_overrides,
                         label, min_len=3):
    """Test a mapping variant and return match statistics.

    Returns dict with n_matched, n_total, rate, top_matches.
    """
    n_total = 0
    n_matched = 0
    matched_words = Counter()

    for w in eva_words:
        heb = decode_with_mapping(w, mapping_overrides)
        if heb is None or len(heb) < min_len:
            continue
        n_total += 1
        if heb in lexicon_set:
            n_matched += 1
            matched_words[heb] += 1

    rate = n_matched / max(n_total, 1) * 100
    return {
        'label': label,
        'overrides': {k: v for k, v in (mapping_overrides or {}).items()},
        'n_matched': n_matched,
        'n_total': n_total,
        'rate': rate,
        'top_matches': matched_words.most_common(20),
    }


def test_positional_split(eva_words, lexicon_set, char, position,
                           new_hebrew, label, min_len=3):
    """Test a positional split: char at position → new_hebrew, elsewhere unchanged.

    position: 'initial', 'medial', or 'final' in Hebrew (after RTL reversal).
    """
    base_mapping = dict(FULL_MAPPING)
    original_hebrew = base_mapping.get(char)
    if original_hebrew is None:
        return None

    n_total = 0
    n_matched = 0
    n_changed = 0
    gained = 0
    lost = 0

    for w in eva_words:
        # Decode with original
        heb_orig = decode_with_mapping(w)
        if heb_orig is None or len(heb_orig) < min_len:
            continue

        # Now decode with positional override
        prefix, processed = preprocess_eva(w)
        chars = list(reversed(processed))

        parts = []
        for ch in chars:
            if ch == '\x01':
                parts.append(II_HEBREW)
            elif ch == '\x02':
                parts.append(I_HEBREW)
            elif ch == '\x03':
                parts.append(CH_HEBREW)
            elif ch in base_mapping:
                parts.append(base_mapping[ch])
            else:
                parts = None
                break

        if parts is None:
            continue

        # Apply initial d→bet
        if parts and parts[0] == 'd':
            parts[0] = INITIAL_D_HEBREW

        # Apply positional split for our target char
        for i, ch_heb in enumerate(parts):
            if ch_heb == original_hebrew:
                # Check position
                pos_type = 'initial' if i == 0 else \
                           'final' if i == len(parts) - 1 else 'medial'
                if pos_type == position:
                    parts[i] = new_hebrew

        heb_mod = ''.join(parts)
        n_total += 1

        orig_match = heb_orig in lexicon_set
        mod_match = heb_mod in lexicon_set

        if heb_mod != heb_orig:
            n_changed += 1

        if mod_match and not orig_match:
            gained += 1
        elif orig_match and not mod_match:
            lost += 1
        if mod_match:
            n_matched += 1

    return {
        'label': label,
        'char': char,
        'position': position,
        'new_hebrew': new_hebrew,
        'new_hebrew_name': CONSONANT_NAMES.get(new_hebrew, new_hebrew),
        'n_total': n_total,
        'n_changed': n_changed,
        'n_matched': n_matched,
        'rate': n_matched / max(n_total, 1) * 100,
        'gained': gained,
        'lost': lost,
        'net': gained - lost,
    }


# =====================================================================
# Random baseline for z-score
# =====================================================================

def generate_random_lexicon(hebrew_set, seed=42):
    """Generate a random consonantal lexicon matching Hebrew stats."""
    rng = random.Random(seed)
    forms = list(hebrew_set)
    lengths = [len(f) for f in forms]
    all_chars = ''.join(forms)
    char_freq = Counter(all_chars)
    total = sum(char_freq.values())
    chars = list(char_freq.keys())
    weights = [char_freq[c] / total for c in chars]

    random_forms = set()
    for length in lengths:
        form = ''.join(rng.choices(chars, weights=weights, k=length))
        random_forms.add(form)
    return random_forms


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Deep analysis of EVA l/r allography."""
    report_path = config.stats_dir / "allograph_lr_deep_report.json"

    if report_path.exists() and not force:
        click.echo("  l/r deep report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("DEEP ANALYSIS — EVA l/r ALLOGRAPHY")
    click.echo("  Currently: l → mem(m), r → he(h)")
    click.echo("  Finding: cosine=0.954, context=0.747 — above allograph thresholds")

    # Parse EVA text
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(Path(eva_file))
    words = eva_data["words"]
    word_freq = Counter(words)
    unique_words = list(word_freq.keys())
    click.echo(f"    {eva_data['total_words']} words, {len(unique_words)} unique")

    # Load lexicon
    print_step("Loading Hebrew lexicon...")
    hebrew_set = _load_hebrew_lexicon(config)
    click.echo(f"    {len(hebrew_set)} consonantal forms")

    random_set = generate_random_lexicon(hebrew_set, seed=42)
    click.echo(f"    {len(random_set)} random baseline forms")

    report = {}

    # ═══════════════════════════════════════════════════════════════
    # 1. DETAILED POSITIONAL PROFILES
    # ═══════════════════════════════════════════════════════════════
    print_header("1. POSITIONAL PROFILES — l vs r")

    l_profile = positional_profile(words, 'l')
    r_profile = positional_profile(words, 'r')
    l_total = sum(l_profile.values())
    r_total = sum(r_profile.values())

    click.echo(f"\n  EVA l:  init={l_profile['initial']:>5d} "
               f"({l_profile['initial']/l_total*100:.1f}%)  "
               f"med={l_profile['medial']:>5d} "
               f"({l_profile['medial']/l_total*100:.1f}%)  "
               f"fin={l_profile['final']:>5d} "
               f"({l_profile['final']/l_total*100:.1f}%)  "
               f"total={l_total}")
    click.echo(f"  EVA r:  init={r_profile['initial']:>5d} "
               f"({r_profile['initial']/r_total*100:.1f}%)  "
               f"med={r_profile['medial']:>5d} "
               f"({r_profile['medial']/r_total*100:.1f}%)  "
               f"fin={r_profile['final']:>5d} "
               f"({r_profile['final']/r_total*100:.1f}%)  "
               f"total={r_total}")

    # Cosine of positional profiles
    l_vec = np.array([l_profile[p] / l_total for p in ['initial', 'medial', 'final']])
    r_vec = np.array([r_profile[p] / r_total for p in ['initial', 'medial', 'final']])
    cosine = float(np.dot(l_vec, r_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(r_vec)))
    click.echo(f"\n  Cosine similarity: {cosine:.4f}")

    report['profiles'] = {
        'l': l_profile, 'r': r_profile,
        'l_total': l_total, 'r_total': r_total,
        'cosine': cosine,
    }

    # ═══════════════════════════════════════════════════════════════
    # 2. BIGRAM CONTEXT
    # ═══════════════════════════════════════════════════════════════
    print_header("2. BIGRAM CONTEXT — l vs r")

    l_before, l_after = bigram_context(words, 'l')
    r_before, r_after = bigram_context(words, 'r')

    # Context overlap
    def weighted_overlap(ca, cb):
        ta = sum(ca.values())
        tb = sum(cb.values())
        if ta == 0 or tb == 0:
            return 0.0
        all_chars = set(ca) | set(cb)
        return sum(min(ca.get(c, 0)/ta, cb.get(c, 0)/tb) for c in all_chars)

    before_overlap = weighted_overlap(l_before, r_before)
    after_overlap = weighted_overlap(l_after, r_after)
    combined_overlap = (before_overlap + after_overlap) / 2

    click.echo(f"\n  Before-context overlap: {before_overlap:.3f}")
    click.echo(f"  After-context overlap:  {after_overlap:.3f}")
    click.echo(f"  Combined:               {combined_overlap:.3f}")

    # Show top contexts
    click.echo(f"\n  Top BEFORE contexts:")
    click.echo(f"    {'Char':5s} {'l%':>8s} {'r%':>8s} {'Δ':>8s}")
    click.echo(f"    {'-'*30}")
    l_before_total = sum(l_before.values())
    r_before_total = sum(r_before.values())
    all_before = sorted(set(l_before) | set(r_before),
                        key=lambda c: -(l_before.get(c, 0) + r_before.get(c, 0)))
    for c in all_before[:12]:
        lp = l_before.get(c, 0) / l_before_total * 100
        rp = r_before.get(c, 0) / r_before_total * 100
        click.echo(f"    {c:5s} {lp:>7.1f}% {rp:>7.1f}% {lp-rp:>+7.1f}")

    click.echo(f"\n  Top AFTER contexts:")
    click.echo(f"    {'Char':5s} {'l%':>8s} {'r%':>8s} {'Δ':>8s}")
    click.echo(f"    {'-'*30}")
    l_after_total = sum(l_after.values())
    r_after_total = sum(r_after.values())
    all_after = sorted(set(l_after) | set(r_after),
                       key=lambda c: -(l_after.get(c, 0) + r_after.get(c, 0)))
    for c in all_after[:12]:
        lp = l_after.get(c, 0) / l_after_total * 100
        rp = r_after.get(c, 0) / r_after_total * 100
        click.echo(f"    {c:5s} {lp:>7.1f}% {rp:>7.1f}% {lp-rp:>+7.1f}")

    report['context'] = {
        'before_overlap': before_overlap,
        'after_overlap': after_overlap,
        'combined_overlap': combined_overlap,
    }

    # ═══════════════════════════════════════════════════════════════
    # 3. CO-OCCURRENCE
    # ═══════════════════════════════════════════════════════════════
    print_header("3. CO-OCCURRENCE — l and r in same word")

    coocc = cooccurrence_analysis(words, 'l', 'r')
    click.echo(f"\n  Both l and r:  {coocc['both']:>6d} words")
    click.echo(f"  Only l:        {coocc['only_a']:>6d} words")
    click.echo(f"  Only r:        {coocc['only_b']:>6d} words")
    click.echo(f"  Neither:       {coocc['neither']:>6d} words")
    click.echo(f"  Co-occ rate:   {coocc['cooccurrence_rate']:.3f}")
    click.echo(f"\n  (Allographs should have LOW co-occurrence rate)")
    click.echo(f"  Reference: f/p co-occurrence ~0.15 (confirmed allographs)")

    report['cooccurrence'] = coocc

    # ═══════════════════════════════════════════════════════════════
    # 4. MINIMAL PAIRS
    # ═══════════════════════════════════════════════════════════════
    print_header("4. MINIMAL PAIRS — l↔r substitution")

    pairs = find_minimal_pairs(words, 'l', 'r')
    click.echo(f"\n  {len(pairs)} minimal pairs found")
    click.echo(f"\n  Top 15 by frequency:")
    click.echo(f"  {'Word-l':15s} {'freq':>6s} {'Word-r':15s} {'freq':>6s} "
               f"{'Heb-l':>10s} {'Heb-r':>10s}")
    click.echo(f"  {'-'*70}")
    for w_l, w_r, pos in pairs[:15]:
        # Ensure w_l has 'l' and w_r has 'r'
        if 'l' not in w_l:
            w_l, w_r = w_r, w_l
        freq_l = word_freq.get(w_l, 0)
        freq_r = word_freq.get(w_r, 0)
        heb_l = decode_with_mapping(w_l) or '?'
        heb_r = decode_with_mapping(w_r) or '?'
        click.echo(f"  {w_l:15s} {freq_l:>6d} {w_r:15s} {freq_r:>6d} "
                   f"{heb_l:>10s} {heb_r:>10s}")

    report['minimal_pairs'] = [{
        'word_l': (w_l if 'l' in w_l else w_r),
        'word_r': (w_r if 'r' in w_r else w_l),
        'pos': pos,
    } for w_l, w_r, pos in pairs[:30]]

    # ═══════════════════════════════════════════════════════════════
    # 5. UNIFICATION TESTS
    # ═══════════════════════════════════════════════════════════════
    print_header("5. UNIFICATION TESTS — lexicon match rate")

    click.echo("\n  Testing 5 mapping variants against Hebrew + Random lexicons...\n")

    variants = [
        ({}, "CURRENT (l→mem, r→he)"),
        ({'l': 'h', 'r': 'h'}, "BOTH → he(h)"),
        ({'l': 'm', 'r': 'm'}, "BOTH → mem(m)"),
        ({'l': 'h', 'r': 'm'}, "SWAPPED (l→he, r→mem)"),
        # Also test with unmapped letters
        ({'r': 'm'}, "r→mem (l unchanged=mem)"),
        ({'l': 'h'}, "l→he (r unchanged=he)"),
    ]

    hebrew_results = []
    random_results = []

    for overrides, label in variants:
        heb_res = test_mapping_variant(words, hebrew_set, overrides, label)
        rand_res = test_mapping_variant(words, random_set, overrides,
                                        label + " [random]")
        hebrew_results.append(heb_res)
        random_results.append(rand_res)

    click.echo(f"  {'Variant':<35s} {'Heb match':>10s} {'Heb rate':>9s} "
               f"{'Rand rate':>10s} {'Δ(H-R)':>8s} {'ΔvsBase':>8s}")
    click.echo(f"  {'-'*82}")

    base_rate = hebrew_results[0]['rate']
    base_rand = random_results[0]['rate']
    for heb_res, rand_res in zip(hebrew_results, random_results):
        delta_hr = heb_res['rate'] - rand_res['rate']
        delta_base = heb_res['rate'] - base_rate
        click.echo(f"  {heb_res['label']:<35s} {heb_res['n_matched']:>10d} "
                   f"{heb_res['rate']:>8.1f}% {rand_res['rate']:>9.1f}% "
                   f"{delta_hr:>+7.1f} {delta_base:>+7.1f}")

    report['unification_tests'] = {
        'hebrew': [{
            'label': r['label'],
            'n_matched': r['n_matched'],
            'n_total': r['n_total'],
            'rate': r['rate'],
        } for r in hebrew_results],
        'random': [{
            'label': r['label'],
            'rate': r['rate'],
        } for r in random_results],
    }

    # ═══════════════════════════════════════════════════════════════
    # 6. POSITIONAL SPLIT TESTS
    # ═══════════════════════════════════════════════════════════════
    print_header("6. POSITIONAL SPLITS")

    click.echo("\n  Testing: what if l or r maps differently by position?\n")

    split_results = []
    for char in ['l', 'r']:
        for position in ['initial', 'medial', 'final']:
            for new_heb in ['m', 'h', 'z', 's', 'C', 'q']:
                # Skip if it's the current mapping (no change)
                current = FULL_MAPPING.get(char)
                if new_heb == current:
                    continue
                result = test_positional_split(
                    words, hebrew_set, char, position, new_heb,
                    f"{char}@{position}→{CONSONANT_NAMES.get(new_heb, new_heb)}")
                if result and result['n_changed'] > 0:
                    split_results.append(result)

    # Sort by net improvement
    split_results.sort(key=lambda x: -x['net'])

    click.echo(f"  {'Split':<25s} {'Changed':>8s} {'Gained':>7s} "
               f"{'Lost':>6s} {'Net':>6s} {'Rate':>7s}")
    click.echo(f"  {'-'*62}")
    for r in split_results[:15]:
        click.echo(f"  {r['label']:<25s} {r['n_changed']:>8d} "
                   f"{r['gained']:>7d} {r['lost']:>6d} {r['net']:>+6d} "
                   f"{r['rate']:>6.1f}%")
    if len(split_results) > 15:
        click.echo(f"  ... ({len(split_results) - 15} more, all negative or small)")

    # Also show worst (most negative) to see what we'd lose
    click.echo(f"\n  Bottom 5 (worst splits):")
    for r in split_results[-5:]:
        click.echo(f"  {r['label']:<25s} {r['n_changed']:>8d} "
                   f"{r['gained']:>7d} {r['lost']:>6d} {r['net']:>+6d}")

    report['positional_splits'] = [{
        'label': r['label'],
        'char': r['char'],
        'position': r['position'],
        'new_hebrew': r['new_hebrew'],
        'n_changed': r['n_changed'],
        'gained': r['gained'],
        'lost': r['lost'],
        'net': r['net'],
        'rate': r['rate'],
    } for r in split_results]

    # ═══════════════════════════════════════════════════════════════
    # 7. VERDICT
    # ═══════════════════════════════════════════════════════════════
    click.echo(f"\n{'='*65}")
    click.echo("  VERDICT — l/r ALLOGRAPHY")
    click.echo(f"{'='*65}")

    # Summarize key findings
    click.echo(f"\n  Distributional similarity:")
    click.echo(f"    Cosine: {cosine:.4f}  (threshold >0.95)")
    click.echo(f"    Context: {combined_overlap:.3f}  (threshold >0.50)")
    click.echo(f"    Co-occurrence: {coocc['cooccurrence_rate']:.3f}  "
               f"(allographs expect <0.20)")

    # Is it an allograph?
    is_distributional_allograph = cosine > 0.95 and combined_overlap > 0.50
    is_complementary = coocc['cooccurrence_rate'] < 0.20

    click.echo(f"\n  Allograph criteria:")
    click.echo(f"    Distributional: {'PASS' if is_distributional_allograph else 'FAIL'}")
    click.echo(f"    Complementary:  {'PASS' if is_complementary else 'FAIL'}")

    # Best unification
    best_unification = max(hebrew_results[1:], key=lambda x: x['rate'])
    current = hebrew_results[0]
    click.echo(f"\n  Lexicon impact:")
    click.echo(f"    Current:          {current['rate']:.1f}% "
               f"({current['n_matched']} matches)")
    click.echo(f"    Best unification: {best_unification['label']} "
               f"= {best_unification['rate']:.1f}% "
               f"({best_unification['n_matched']} matches)")
    click.echo(f"    Delta:            {best_unification['rate'] - current['rate']:+.1f}%")

    # Best split
    best_split = split_results[0] if split_results else None
    if best_split:
        click.echo(f"    Best split:       {best_split['label']} "
                   f"net={best_split['net']:+d}")

    # Overall verdict
    click.echo(f"\n  {'='*55}")
    if is_distributional_allograph and is_complementary:
        if best_unification['rate'] > current['rate']:
            click.echo(f"  l/r ARE ALLOGRAPHS — unification IMPROVES mapping")
            click.echo(f"  Recommendation: {best_unification['label']}")
        else:
            click.echo(f"  l/r ARE ALLOGRAPHS distributionally, but")
            click.echo(f"  unification does NOT improve lexicon matching.")
            click.echo(f"  Keep current separate mapping.")
    elif is_distributional_allograph:
        click.echo(f"  l/r have SIMILAR DISTRIBUTION but HIGH co-occurrence")
        click.echo(f"  ({coocc['cooccurrence_rate']:.3f}). Not true allographs —")
        click.echo(f"  similar positions but different bigram roles.")
    else:
        click.echo(f"  l/r are NOT allographs.")
    click.echo(f"  {'='*55}")

    # Save report
    print_step("Saving report...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")
